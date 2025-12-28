"""Core Document Agent implementation using LangChain 1.2.0."""

import asyncio
import functools
import time
import uuid
import hashlib
import logging
from typing import Dict, List, Any, Optional

from src.core.executors import get_executors

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelRetryMiddleware,
    ToolRetryMiddleware,
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
    PIIMiddleware,
)
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
import json
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .config import DocumentAgentConfig
from .tools import create_document_tools
from src.utils.timer_utils import elapsed_ms

# Base agent and shared utilities
from src.agents.core.base_agent import BaseAgent
from src.agents.core.token_utils import calculate_token_usage

# Tool selection imports
from src.agents.core.middleware import LLMToolSelector, QueryClassifier, QueryIntent
from .schemas import (
    DocumentRequest,
    DocumentResponse,
    GeneratedContent,
    GenerationOptions,
    FAQ,
    Question,
    TokenUsage,
    SessionInfo
)

logger = logging.getLogger(__name__)


# NOTE: RateLimiter and SessionManager classes have been moved to shared modules:
# - src/agents/core/rate_limiter.py
# - src/agents/core/session_manager.py


SYSTEM_PROMPT = """You are a document analysis assistant with two operational modes.

## MODE 1: RAG Search (Question Answering)
Use when user asks QUESTIONS about document content:
- "What are the payment terms?"
- "Find information about X"
- "Tell me about Y"
- "Who is responsible for..."
- "When does the contract expire?"

For RAG Search:
→ Use ONLY the rag_search tool
→ Consider conversation history for follow-up questions
→ Always cite your sources in responses
→ Support semantic, keyword, and hybrid search modes

## MODE 2: Content Generation
Use when user wants to CREATE new content:
- "Generate a summary"
- "Create FAQs"
- "Make comprehension questions"
- "Summarize this document"
- "Extract key points"

For Content Generation:
→ FIRST use document_loader to get the document content
→ THEN use the appropriate generation tool(s)
→ FINALLY use content_persist to save all generated content

## Tool Selection Rules:
1. NEVER use rag_search for content generation requests
2. NEVER use generation tools for Q&A requests
3. For ambiguous requests, prefer RAG search (more conversational)
4. The system pre-filters available tools - use what's provided

## Available Tools:
{tools}

## Quality Guidelines:
- Generate accurate, document-grounded content only - never make up information
- For FAQs, focus on commonly asked questions with clear, helpful answers
- For questions, create a mix of easy, medium, and hard difficulty levels
- If the document cannot be found, inform the user clearly"""


# Template for dynamic SYSTEM_PROMPT with filtered tools
def get_system_prompt(tool_names: list) -> str:
    """Get system prompt with available tools listed."""
    return SYSTEM_PROMPT.format(tools=", ".join(tool_names))


class DocumentAgent(BaseAgent):
    """AI-powered document analysis agent using LangGraph.

    Inherits from BaseAgent for shared functionality:
    - Session management
    - Rate limiting
    - Memory (short-term and long-term)
    - Audit logging
    """

    def __init__(self, config: Optional[DocumentAgentConfig] = None):
        """Initialize Document Agent.

        Args:
            config: Agent configuration. Uses defaults if not provided.
        """
        # Use default config if not provided
        config = config or DocumentAgentConfig()

        # Initialize base agent (session manager, rate limiter, memory, audit)
        super().__init__(config)

        # Initialize LLM (document-specific)
        self.llm = self._init_llm()

        # Initialize tools (document-specific)
        self.tools = self._init_tools()

        # Build tool name lookup for quick access
        self.tools_by_name = {tool.name: tool for tool in self.tools}

        # Initialize tool selection components (document-specific)
        self.query_classifier = None
        self.tool_selector = None
        if self.config.enable_tool_selection:
            self._init_tool_selection()

        # Build middleware list from config (document-specific)
        self.middleware_list = self._build_middleware()

        # Initialize checkpointer for conversation memory (document-specific)
        self.checkpointer = MemorySaver()

        # Create agent
        self.agent = self._create_agent()

        logger.info(f"Document Agent initialized with model: {self.config.gemini_model}")

    def _build_middleware(self) -> List:
        """Build LangChain middleware list from config."""
        middleware = []

        if not self.config.enable_middleware:
            logger.info("Middleware disabled via configuration")
            return middleware

        try:
            # Model retry with exponential backoff
            middleware.append(
                ModelRetryMiddleware(
                    max_retries=self.config.model_retry_max_attempts,
                    backoff_factor=2.0,
                    initial_delay=1.0,
                )
            )

            # Tool retry
            middleware.append(
                ToolRetryMiddleware(max_retries=self.config.tool_retry_max_attempts)
            )

            # Call limits to prevent runaway loops
            middleware.append(
                ModelCallLimitMiddleware(run_limit=self.config.model_call_limit)
            )
            middleware.append(
                ToolCallLimitMiddleware(run_limit=self.config.tool_call_limit)
            )

            # PII detection if enabled
            if self.config.enable_pii_detection:
                pii_strategy = self.config.pii_strategy
                middleware.extend([
                    PIIMiddleware("email", strategy=pii_strategy, apply_to_input=True),
                    PIIMiddleware("credit_card", strategy=pii_strategy, apply_to_input=True),
                ])

            logger.info(f"Built {len(middleware)} middleware components")
            return middleware

        except Exception as e:
            logger.warning(f"Failed to build middleware: {e}")
            return []

    # _init_memory() is inherited from BaseAgent

    def _init_tools(self) -> List:
        """Initialize agent tools."""
        return create_document_tools(self.config)

    def _get_agent_type(self) -> str:
        """Get agent type identifier for audit/memory."""
        return "document"

    def _init_llm(self):
        """Initialize the language model using init_chat_model with token tracking callback."""
        if not self.config.google_api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable is required. "
                "Get your API key from https://aistudio.google.com/app/apikey"
            )

        # Use shared callback creation from BaseAgent
        callbacks = self._create_token_tracking_callback("document_agent")

        llm = init_chat_model(
            model=self.config.gemini_model,
            model_provider="google_genai",
            temperature=self.config.temperature,
            api_key=self.config.google_api_key,
            timeout=60,  # Prevent hanging requests
            max_retries=2,
            callbacks=callbacks if callbacks else None,
        )

        logger.info(f"Initialized LLM: {self.config.gemini_model}")
        return llm

    def _init_tool_selection(self):
        """Initialize query classifier and tool selector."""
        try:
            # Initialize query classifier with LLM fallback
            self.query_classifier = QueryClassifier(
                use_llm_fallback=True,
                llm_model=self.config.tool_selector_model,
                llm_provider="google_genai",
                api_key=self.config.google_api_key
            )

            # Initialize LLM-based tool selector
            self.tool_selector = LLMToolSelector(
                model=self.config.tool_selector_model,
                provider="google_genai",
                max_tools=self.config.tool_selector_max_tools,
                api_key=self.config.google_api_key
            )

            logger.info(
                f"Tool selection enabled: classifier + selector "
                f"(model={self.config.tool_selector_model}, max_tools={self.config.tool_selector_max_tools})"
            )

        except Exception as e:
            logger.warning(f"Failed to initialize tool selection: {e}. Using all tools.")
            self.query_classifier = None
            self.tool_selector = None

    def _get_tools_for_query(self, query: str, context: dict) -> list:
        """
        Get relevant tools based on query intent.

        Uses two-stage filtering:
        1. QueryClassifier determines intent (RAG vs Generation)
        2. LLMToolSelector narrows down within the category

        Args:
            query: User's query string
            context: Context dict with document_name, has_parsed_path, etc.

        Returns:
            List of relevant tools for this query
        """
        # If tool selection disabled, return all tools
        if not self.config.enable_tool_selection or not self.query_classifier:
            return self.tools

        try:
            # Stage 1: Classify query intent
            intent = self.query_classifier.classify(query, context)
            logger.debug(f"Query intent: {intent.value} for: {query[:50]}...")

            # Map intent to tool subsets
            if intent == QueryIntent.RAG_SEARCH:
                # RAG search only needs the rag_search tool
                candidate_tools = [self.tools_by_name.get('rag_search')]
                candidate_tools = [t for t in candidate_tools if t is not None]

            elif intent == QueryIntent.CONTENT_GENERATION:
                # Content generation needs loader, generators, and persist
                tool_names = [
                    'document_loader',
                    'summary_generator',
                    'faq_generator',
                    'question_generator',
                    'content_persist'
                ]
                candidate_tools = [
                    self.tools_by_name.get(name)
                    for name in tool_names
                    if self.tools_by_name.get(name) is not None
                ]

            elif intent == QueryIntent.DOCUMENT_LOAD:
                # Just document loading
                candidate_tools = [self.tools_by_name.get('document_loader')]
                candidate_tools = [t for t in candidate_tools if t is not None]

            elif intent == QueryIntent.CONVERSATIONAL:
                # Conversational query (about previous questions, memory, etc.)
                # Return empty list - let LLM answer from conversation history
                logger.info("Conversational query - no tools needed, using memory")
                return []

            else:
                # MIXED intent - use all tools and let LLMToolSelector narrow down
                candidate_tools = self.tools

            # Stage 2: Use LLMToolSelector to further narrow if we have many tools
            if self.tool_selector and len(candidate_tools) > self.config.tool_selector_max_tools:
                selected = self.tool_selector.select_tools(query, candidate_tools, context=None)
                logger.info(
                    f"Tool selection: {intent.value} -> {len(candidate_tools)} candidates -> "
                    f"{len(selected)} selected: {[t.name for t in selected]}"
                )
                return selected

            logger.info(
                f"Tool selection: {intent.value} -> {len(candidate_tools)} tools: "
                f"{[t.name for t in candidate_tools]}"
            )
            return candidate_tools

        except Exception as e:
            logger.warning(f"Tool selection failed: {e}. Using all tools.")
            return self.tools

    def _create_agent(self):
        """Create the agent using LangChain 1.2.0 with built-in middleware and checkpointer."""
        agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=SYSTEM_PROMPT,
            middleware=self.middleware_list if self.middleware_list else None,
            checkpointer=self.checkpointer  # Enable automatic conversation memory
        )

        logger.info(f"Created agent with LangChain 1.2.0 ({len(self.middleware_list)} middleware, checkpointer=MemorySaver)")
        return agent

    # _init_audit_logging() is inherited from BaseAgent

    async def process_request(self, request: DocumentRequest) -> DocumentResponse:
        """
        Process a document request and return generated content.

        Args:
            request: Document request with document name and query

        Returns:
            Document response with generated content
        """
        start_time = time.time()

        try:
            session = self.session_manager.get_or_create_session(request.session_id)

            # Note: LangChain middleware handles call limits per-run automatically

            if not self.rate_limiter.is_allowed(session.session_id):
                retry_after = self.rate_limiter.get_retry_after(session.session_id)
                return DocumentResponse(
                    success=False,
                    message=f"Rate limit exceeded. Please try again in {retry_after} seconds.",
                    document_name=request.document_name,
                    session_id=session.session_id,
                    processing_time_ms=elapsed_ms(start_time)
                )

            # Note: PII detection is handled by LangChain PIIMiddleware automatically
            query_context = f"{request.query}||{request.document_name}||{request.options}"
            query_hash = hashlib.md5(query_context.encode()).hexdigest()

            cached_response = self.session_manager.get_cached_response(session.session_id, query_hash)
            if cached_response:
                logger.info(f"Returning cached response for query hash {query_hash}")
                processing_time = elapsed_ms(start_time)
                cached_response["processing_time_ms"] = processing_time
                return DocumentResponse(**cached_response)

            session.query_count += 1
            if request.document_name not in session.documents_processed:
                session.documents_processed.append(request.document_name)

            # Note: Short-term memory is now handled automatically by the checkpointer
            # via thread_id in config. No manual chat_history management needed.

            # Get long-term context if user_id provided (cross-session context)
            long_term_context = ""
            if self.long_term_memory and request.user_id:
                long_term_context = self.long_term_memory.get_relevant_context(
                    request.user_id, request.query
                )

            context = self._prepare_context(request, long_term_context)

            try:
                async with asyncio.timeout(self.config.timeout_seconds):
                    agent_result = await self._execute_agent(context, request)
            except asyncio.TimeoutError:
                processing_time = elapsed_ms(start_time)
                logger.error(f"Agent execution timed out after {self.config.timeout_seconds}s")
                return DocumentResponse(
                    success=False,
                    message=f"Request timed out after {self.config.timeout_seconds} seconds.",
                    document_name=request.document_name,
                    session_id=session.session_id,
                    processing_time_ms=processing_time
                )
            # Note: Call limits and PII detection are handled by LangChain middleware

            response_text = agent_result.get('response', '')

            token_usage = self._calculate_token_usage(
                request.query,
                response_text
            )

            self.session_manager.update_session(
                session.session_id,
                total_tokens_used=session.total_tokens_used + token_usage.total_tokens,
                total_processing_time_ms=session.total_processing_time_ms + elapsed_ms(start_time)
            )

            processing_time = elapsed_ms(start_time)

            # Log audit event in background to avoid blocking and event loop issues
            if self.audit_logger:
                self._log_audit_event_async(request, processing_time)

            response_data = DocumentResponse(
                success=True,
                message="Document processed successfully",
                response_text=response_text,  # Include agent's response for RAG chat
                document_name=request.document_name,
                source_path=agent_result.get('source_path'),
                content=agent_result.get('content'),
                document_metadata=agent_result.get('metadata'),
                tools_used=agent_result.get('tools_used', []),
                token_usage=token_usage,
                session_id=session.session_id,
                processing_time_ms=processing_time,
                persisted=agent_result.get('persisted', False),
                database_id=agent_result.get('database_id'),
                output_file_path=agent_result.get('output_file_path')
            )

            # Note: Short-term memory is handled automatically by the checkpointer.
            # The agent state (including all messages) is persisted per thread_id.

            cache_data = response_data.model_dump()
            cache_data.pop('timestamp', None)
            self.session_manager.cache_response(session.session_id, query_hash, cache_data)

            return response_data

        except Exception as e:
            processing_time = elapsed_ms(start_time)
            logger.error(f"Error processing document request: {e}", exc_info=True)

            return DocumentResponse(
                success=False,
                message=f"Error processing request: {str(e)}",
                document_name=request.document_name,
                session_id=request.session_id or str(uuid.uuid4()),
                processing_time_ms=processing_time
            )

    def _prepare_context(
        self, request: DocumentRequest, long_term_context: str = ""
    ) -> str:
        """Prepare context string for the agent."""
        context_parts = [
            f"Document: {request.document_name}",
            f"Parsed File Path: {request.parsed_file_path}",
            f"User Request: {request.query}",
        ]

        if request.options:
            if request.options.num_faqs:
                context_parts.append(f"Generate {request.options.num_faqs} FAQs")
            if request.options.num_questions:
                context_parts.append(f"Generate {request.options.num_questions} questions")
            if request.options.summary_max_words:
                context_parts.append(f"Summary max words: {request.options.summary_max_words}")

        # Include long-term context if available
        if long_term_context:
            context_parts.append("")
            context_parts.append(long_term_context)

        return "\n".join(context_parts)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((TimeoutError, ConnectionError, OSError)),
        reraise=True
    )
    async def _execute_agent(
        self, context: str, request: DocumentRequest
    ) -> Dict[str, Any]:
        """Execute the agent with the given context and request.

        Conversation history is handled automatically by the checkpointer via thread_id.
        Token tracking is handled via thread-local usage context.
        """
        # Import usage context for token tracking
        try:
            from src.core.usage.context import usage_context
            USAGE_CONTEXT_AVAILABLE = True
        except ImportError:
            USAGE_CONTEXT_AVAILABLE = False
            usage_context = None

        session_id = request.session_id or "default"

        # Set up usage context for token tracking (if available)
        ctx_manager = None
        if USAGE_CONTEXT_AVAILABLE and usage_context and request.organization_id:
            ctx_manager = usage_context(
                org_id=request.organization_id,
                feature="document_agent",
                user_id=request.user_id,
                session_id=session_id,
            )
            logger.debug(f"Token tracking context set for org {request.organization_id}")

        try:
            # Enter usage context if available
            if ctx_manager:
                ctx_manager.__enter__()

            return await self._execute_agent_inner(context, request, session_id)

        finally:
            # Exit usage context
            if ctx_manager:
                ctx_manager.__exit__(None, None, None)

    async def _execute_agent_inner(
        self, context: str, request: DocumentRequest, session_id: str
    ) -> Dict[str, Any]:
        """Inner agent execution (called within usage context)."""
        try:
            input_text = f"{context}\n\nPlease process this document and fulfill the request."

            # Create message - checkpointer handles history automatically via thread_id
            message = HumanMessage(content=input_text)

            # Config with thread_id for checkpointer to manage conversation history
            config = {
                "configurable": {
                    "thread_id": session_id  # Enables automatic conversation continuity
                }
            }

            # Get relevant tools for this query
            query_context = {
                "document_name": request.document_name,
                "has_parsed_path": bool(request.parsed_file_path),
                "organization_name": getattr(request, 'organization_id', None)
            }
            relevant_tools = self._get_tools_for_query(request.query, query_context)

            # Handle conversational queries (no tools needed - just LLM)
            if not relevant_tools:
                logger.info("Conversational query - invoking LLM directly without tools")

                # Build messages with conversation history from short-term memory
                messages_to_send = []
                if self.short_term_memory and session_id:
                    history = self.short_term_memory.get_messages(session_id)
                    if history:
                        messages_to_send.extend(history)
                        logger.info(f"Including {len(history)} messages from conversation history")

                # Add current message
                messages_to_send.append(message)

                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    get_executors().agent_executor,
                    functools.partial(
                        self.agent.invoke,
                        {"messages": messages_to_send},
                        config
                    )
                )
            # Create dynamic agent with filtered tools if tool selection is enabled
            elif self.config.enable_tool_selection and relevant_tools != self.tools:
                dynamic_prompt = get_system_prompt([t.name for t in relevant_tools])
                dynamic_agent = create_agent(
                    model=self.llm,
                    tools=relevant_tools,
                    system_prompt=dynamic_prompt,
                    middleware=self.middleware_list if self.middleware_list else None,
                    checkpointer=self.checkpointer  # Use same checkpointer for filtered agents
                )
                logger.debug(f"Executing agent with {len(relevant_tools)} filtered tools")
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    get_executors().agent_executor,
                    functools.partial(
                        dynamic_agent.invoke,
                        {"messages": [message]},  # Only current message - checkpointer handles history
                        config
                    )
                )
            else:
                # Use default agent with all tools
                logger.debug("Executing agent with all tools")
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    get_executors().agent_executor,
                    functools.partial(
                        self.agent.invoke,
                        {"messages": [message]},  # Only current message - checkpointer handles history
                        config
                    )
                )

            # LangGraph agent returns dict with 'messages' key
            response_text = ""
            all_messages = []
            if result:
                if isinstance(result, dict):
                    if "messages" in result and result["messages"]:
                        all_messages = result["messages"]
                        last_message = all_messages[-1]
                        if hasattr(last_message, 'content'):
                            content = last_message.content
                            # Handle complex content structures (list of content blocks)
                            if isinstance(content, list):
                                text_parts = []
                                for item in content:
                                    if isinstance(item, dict) and item.get('type') == 'text':
                                        text_parts.append(item.get('text', ''))
                                    elif isinstance(item, str):
                                        text_parts.append(item)
                                response_text = ''.join(text_parts)
                            elif isinstance(content, str):
                                response_text = content
                            else:
                                response_text = str(content)
                        else:
                            response_text = str(last_message)
                    else:
                        response_text = result.get("output", str(result))
                else:
                    response_text = str(result)

            # Parse result including tool outputs from all messages
            parsed_result = self._parse_agent_result(response_text, all_messages)
            logger.info(f"Agent execution completed, response length: {len(response_text)} chars")

            # Save to short-term memory for conversation continuity
            if self.short_term_memory and session_id:
                self.short_term_memory.add_human_message(session_id, request.query)
                if response_text:
                    self.short_term_memory.add_ai_message(session_id, response_text)
                logger.debug(f"Saved conversation to short-term memory for session {session_id}")

            return parsed_result

        except Exception as e:
            logger.error(f"Error executing agent: {e}", exc_info=True)
            return {
                "response": f"Error during processing: {str(e)}",
                "content": None
            }

    def _parse_agent_result(self, response_text: str, messages: List = None) -> Dict[str, Any]:
        """Parse the agent's response to extract structured content from tool outputs."""
        result = {
            "response": response_text,
            "content": None,
            "source_path": None,
            "metadata": None,
            "tools_used": [],
            "persisted": False
        }

        if not messages:
            return result

        # Extract content from tool messages
        summary = None
        faqs = None
        questions = None
        tools_used = []

        for msg in messages:
            # Check if it's a ToolMessage (contains tool output)
            if isinstance(msg, ToolMessage):
                tool_name = getattr(msg, 'name', '') or ''

                # Try to parse the tool output as JSON
                try:
                    content = msg.content if hasattr(msg, 'content') else str(msg)
                    tool_output = json.loads(content) if isinstance(content, str) else content

                    # Create ToolUsage entry with parsed output
                    tool_success = isinstance(tool_output, dict) and tool_output.get('success', False)
                    tools_used.append({
                        "tool_name": tool_name,
                        "input_data": {},  # Not available from ToolMessage
                        "output_data": tool_output if isinstance(tool_output, dict) else {"raw": str(tool_output)},
                        "execution_time_ms": 0.0,  # Not tracked at this level
                        "success": tool_success,
                        "error_message": tool_output.get('error') if isinstance(tool_output, dict) else None
                    })

                    if tool_success:
                        # Extract summary from summary_generator tool
                        if 'summary' in tool_output and tool_output['summary']:
                            summary = tool_output['summary']
                            logger.debug(f"Extracted summary: {len(summary)} chars")

                        # Extract FAQs from faq_generator tool
                        if 'faqs' in tool_output and tool_output['faqs']:
                            faqs = [
                                FAQ(question=f['question'], answer=f['answer'])
                                for f in tool_output['faqs']
                                if isinstance(f, dict) and 'question' in f and 'answer' in f
                            ]
                            logger.debug(f"Extracted {len(faqs)} FAQs")

                        # Extract questions from question_generator tool
                        if 'questions' in tool_output and tool_output['questions']:
                            questions = [
                                Question(
                                    question=q['question'],
                                    expected_answer=q.get('expected_answer', ''),
                                    difficulty=q.get('difficulty', 'medium')
                                )
                                for q in tool_output['questions']
                                if isinstance(q, dict) and 'question' in q
                            ]
                            logger.debug(f"Extracted {len(questions)} questions")

                        # Extract source path from document loader tool
                        if 'source_path' in tool_output:
                            result['source_path'] = tool_output['source_path']

                except (json.JSONDecodeError, TypeError, KeyError) as e:
                    logger.debug(f"Could not parse tool output as structured content: {e}")
                    # Still add ToolUsage entry for failed parse
                    tools_used.append({
                        "tool_name": tool_name,
                        "input_data": {},
                        "output_data": None,
                        "execution_time_ms": 0.0,
                        "success": False,
                        "error_message": str(e)
                    })
                    continue

        # Build GeneratedContent if we have any content
        if summary or faqs or questions:
            result['content'] = GeneratedContent(
                summary=summary,
                faqs=faqs,
                questions=questions
            )
            logger.info(f"Parsed content: summary={bool(summary)}, faqs={len(faqs) if faqs else 0}, questions={len(questions) if questions else 0}")

        result['tools_used'] = tools_used
        return result

    def _calculate_token_usage(self, input_text: str, output_text: str) -> TokenUsage:
        """Calculate token usage using shared utility."""
        estimate = calculate_token_usage(
            input_text, output_text, model=self.config.gemini_model
        )
        return TokenUsage(
            prompt_tokens=estimate.prompt_tokens,
            completion_tokens=estimate.completion_tokens,
            total_tokens=estimate.total_tokens,
            estimated_cost_usd=estimate.estimated_cost_usd
        )

    def _log_audit_event_async(self, request: DocumentRequest, processing_time: float):
        """Log query for audit trail via centralized audit queue."""
        if not self.audit_logger:
            return

        try:
            from src.agents.core.audit_queue import enqueue_audit_event

            enqueue_audit_event(
                event_type="document_agent_query",
                file_name=request.document_name,
                organization_id=request.organization_id,
                details={
                    "session_id": request.session_id,
                    "document_name": request.document_name,
                    "parsed_file_path": request.parsed_file_path,
                    "query": request.query,
                    "processing_time_ms": processing_time,
                    "success": True
                }
            )
        except Exception as e:
            logger.warning(f"Failed to enqueue audit event: {e}")

    # Convenience methods for direct generation

    async def generate_summary(
        self,
        document_name: str,
        parsed_file_path: str,
        max_words: Optional[int] = None,
        organization_id: Optional[str] = None
    ) -> str:
        """
        Generate a summary for a document.

        Args:
            document_name: Name of the document
            parsed_file_path: GCS path to parsed document (e.g., 'Acme corp/parsed/invoices/Sample1.md')
            max_words: Maximum words for summary (uses config default if not provided)
            organization_id: Organization ID for multi-tenant isolation

        Returns:
            Generated summary text
        """
        options = GenerationOptions(
            summary_max_words=max_words or self.config.summary_max_words
        )
        request = DocumentRequest(
            document_name=document_name,
            parsed_file_path=parsed_file_path,
            query=f"Generate a summary of this document (max {options.summary_max_words} words)",
            options=options,
            organization_id=organization_id
        )
        response = await self.process_request(request)

        if response.success and response.content and response.content.summary:
            return response.content.summary
        return response.message

    async def generate_faqs(
        self,
        document_name: str,
        parsed_file_path: str,
        num_faqs: Optional[int] = None,
        organization_id: Optional[str] = None
    ) -> List[FAQ]:
        """
        Generate FAQs for a document.

        Args:
            document_name: Name of the document
            parsed_file_path: GCS path to parsed document (e.g., 'Acme corp/parsed/invoices/Sample1.md')
            num_faqs: Number of FAQs (uses config default if not provided)
            organization_id: Organization ID for multi-tenant isolation

        Returns:
            List of FAQ objects
        """
        options = GenerationOptions(
            num_faqs=num_faqs or self.config.default_num_faqs
        )
        request = DocumentRequest(
            document_name=document_name,
            parsed_file_path=parsed_file_path,
            query=f"Generate {options.num_faqs} FAQs from this document",
            options=options,
            organization_id=organization_id
        )
        response = await self.process_request(request)

        if response.success and response.content and response.content.faqs:
            return response.content.faqs
        return []

    async def generate_questions(
        self,
        document_name: str,
        parsed_file_path: str,
        num_questions: Optional[int] = None,
        organization_id: Optional[str] = None
    ) -> List[Question]:
        """
        Generate comprehension questions for a document.

        Args:
            document_name: Name of the document
            parsed_file_path: GCS path to parsed document (e.g., 'Acme corp/parsed/invoices/Sample1.md')
            num_questions: Number of questions (uses config default if not provided)
            organization_id: Organization ID for multi-tenant isolation

        Returns:
            List of Question objects
        """
        options = GenerationOptions(
            num_questions=num_questions or self.config.default_num_questions
        )
        request = DocumentRequest(
            document_name=document_name,
            parsed_file_path=parsed_file_path,
            query=f"Generate {options.num_questions} comprehension questions from this document",
            options=options,
            organization_id=organization_id
        )
        response = await self.process_request(request)

        if response.success and response.content and response.content.questions:
            return response.content.questions
        return []

    async def generate_all(
        self,
        document_name: str,
        parsed_file_path: str,
        options: Optional[GenerationOptions] = None,
        organization_id: Optional[str] = None
    ) -> GeneratedContent:
        """
        Generate summary, FAQs, and questions for a document.

        Args:
            document_name: Name of the document
            parsed_file_path: GCS path to parsed document (e.g., 'Acme corp/parsed/invoices/Sample1.md')
            options: Generation options (uses config defaults if not provided)
            organization_id: Organization ID for multi-tenant isolation

        Returns:
            GeneratedContent with summary, FAQs, and questions
        """
        opts = options or GenerationOptions(
            num_faqs=self.config.default_num_faqs,
            num_questions=self.config.default_num_questions,
            summary_max_words=self.config.summary_max_words
        )

        request = DocumentRequest(
            document_name=document_name,
            parsed_file_path=parsed_file_path,
            query="Generate a summary, FAQs, and comprehension questions for this document",
            options=opts,
            organization_id=organization_id
        )
        response = await self.process_request(request)

        if response.success and response.content:
            return response.content
        return GeneratedContent()

    async def chat(
        self,
        query: str,
        organization_name: str,
        session_id: Optional[str] = None,
        folder_filter: Optional[str] = None,
        file_filter: Optional[str] = None,
        search_mode: str = "hybrid",
        organization_id: Optional[str] = None
    ) -> DocumentResponse:
        """
        Conversational RAG - chat with documents.

        Uses short-term memory for conversation history, enabling follow-up
        questions and contextual responses.

        Args:
            query: User's question or search query
            organization_name: Organization name for store lookup
            session_id: Optional session ID for conversation continuity
            folder_filter: Optional folder name to filter search
            file_filter: Optional file name to filter search
            search_mode: Search mode - 'semantic', 'keyword', or 'hybrid'
            organization_id: Organization ID for multi-tenant isolation

        Returns:
            DocumentResponse with answer and citations
        """
        # Build the query with search context
        search_context = f"Search documents for: {query}"
        if folder_filter:
            search_context += f" (filter by folder: {folder_filter})"
        if file_filter:
            search_context += f" (filter by file: {file_filter})"
        search_context += f" [organization: {organization_name}, mode: {search_mode}]"

        request = DocumentRequest(
            document_name="rag_search",  # Placeholder for RAG operations
            parsed_file_path="rag_search",  # Placeholder for RAG operations
            query=search_context,
            session_id=session_id,
            organization_id=organization_id,
        )

        return await self.process_request(request)

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the agent and its components."""
        try:
            # Get base health status (sessions, rate limiter, memory, audit)
            base_status = self._get_base_health_status()

            # Check LLM status
            llm_status = "healthy" if self.llm else "unhealthy"

            return {
                "status": "healthy" if llm_status == "healthy" else "degraded",
                "components": {
                    "llm": llm_status,
                    "model": self.config.gemini_model,
                    "audit_logging": base_status["audit_logging"],
                    "short_term_memory": "enabled" if self.short_term_memory else "disabled",
                    "long_term_memory": "enabled" if self.long_term_memory else "disabled"
                },
                "sessions": base_status["sessions"],
                "rate_limiter": base_status["rate_limiter"],
                "memory": {
                    **base_status["memory"],
                    "checkpointer": "MemorySaver (in-memory)"
                },
                "middleware": {
                    "enabled": len(self.middleware_list) > 0,
                    "components": len(self.middleware_list)
                },
                "config": {
                    "default_num_faqs": self.config.default_num_faqs,
                    "default_num_questions": self.config.default_num_questions,
                    "summary_max_words": self.config.summary_max_words
                }
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown agent resources gracefully.

        Note: Audit queue is a shared singleton managed at the app level.
        This method only cleans up agent-specific resources.

        Args:
            wait: If True, wait for pending tasks to complete.
                  If False, cancel pending tasks immediately.
        """
        logger.info(f"Shutting down DocumentAgent (wait={wait})")

        # Cleanup base agent resources (sessions, rate limiter)
        self._cleanup_resources()

        logger.info("DocumentAgent shutdown complete")

    # end_session() is inherited from BaseAgent
    # _save_conversation_summary() is inherited from BaseAgent
