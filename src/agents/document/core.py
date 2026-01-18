"""Core Document Agent implementation using LangChain 1.2.0."""

import asyncio
import time
import uuid
import hashlib
import logging
from typing import Dict, List, Any, Optional

from src.agents.core.concurrency import AgentConcurrency

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelRetryMiddleware,
    ToolRetryMiddleware,
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
    PIIMiddleware,
)
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .config import DocumentAgentConfig
from .tools import create_document_tools
from .context import rag_filter_context
from src.utils.timer_utils import elapsed_ms
from src.constants import Timeouts

# Base agent and shared utilities
from src.agents.core.base_agent import BaseAgent

# Tool selection imports
from .tool_selection import ToolSelectionManager, bind_rag_filters
from .result_parser import AgentResultParser, calculate_agent_token_usage
from .schemas import (
    DocumentRequest,
    DocumentResponse,
    GeneratedContent,
    GenerationOptions,
    FAQ,
    Question,
    TokenUsage,
    SessionInfo,
    DirectGenerationResult,
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

        # Initialize tool selection manager (document-specific)
        self.tool_selection_manager = ToolSelectionManager(
            tools=self.tools,
            config=self.config,
            api_key=self.config.openai_api_key
        )

        # Initialize result parser for extracting structured content
        self.result_parser = AgentResultParser()

        # Build middleware list from config (document-specific)
        self.middleware_list = self._build_middleware()

        # Initialize checkpointer for conversation memory (document-specific)
        self.checkpointer = MemorySaver()

        # Create agent
        self.agent = self._create_agent()

        logger.info(f"Document Agent initialized with model: {self.config.openai_model}")

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
        if not self.config.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required.")

        # Use shared callback creation from BaseAgent
        callbacks = self._create_token_tracking_callback("document_agent")

        llm = init_chat_model(
            model=self.config.openai_model,
            model_provider="openai",
            temperature=self.config.temperature,
            api_key=self.config.openai_api_key,
            use_responses_api=True,  # Required for gpt-5-nano
            timeout=Timeouts.LLM_EXECUTION,
            max_retries=2,
            callbacks=callbacks if callbacks else None,
        )

        logger.info(f"Initialized LLM: {self.config.openai_model}")
        return llm

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

    async def process_request(
        self, request: DocumentRequest, rag_only: bool = False
    ) -> DocumentResponse:
        """
        Process a document request and return generated content.

        Args:
            request: Document request with document name and query
            rag_only: If True, skip tool selection and use only rag_search tool.
                      This optimizes RAG chat requests by ~2-3 seconds.

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
                    agent_result = await self._execute_agent(context, request, rag_only=rag_only)
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

            token_usage = calculate_agent_token_usage(
                request.query,
                response_text,
                model=self.config.openai_model
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
                citations=agent_result.get('citations', []),  # RAG search citations
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
        self, context: str, request: DocumentRequest, rag_only: bool = False
    ) -> Dict[str, Any]:
        """Execute the agent with the given context and request.

        Conversation history is handled automatically by the checkpointer via thread_id.
        Token tracking is handled via thread-local usage context.

        Args:
            context: Prepared context string for the agent
            request: Document request with document name and query
            rag_only: If True, skip tool selection and use only rag_search tool
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

            return await self._execute_agent_inner(context, request, session_id, rag_only=rag_only)

        finally:
            # Exit usage context
            if ctx_manager:
                ctx_manager.__exit__(None, None, None)

    async def _execute_agent_inner(
        self, context: str, request: DocumentRequest, session_id: str, rag_only: bool = False
    ) -> Dict[str, Any]:
        """Inner agent execution (called within usage context).

        Args:
            context: Prepared context string for the agent
            request: Document request with document name and query
            session_id: Session ID for conversation continuity
            rag_only: If True, skip tool selection and use only rag_search tool
        """
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
            # PERF: Skip tool selection for RAG-only requests (saves ~2-3 seconds)
            if rag_only:
                # Direct RAG tool lookup - no LLM calls needed
                rag_tool = self.tools_by_name.get('rag_search')
                relevant_tools = [rag_tool] if rag_tool else self.tools
                logger.info(f"RAG-only mode: skipping tool selection, using rag_search directly")
            else:
                # Full tool selection via QueryClassifier + LLMToolSelector
                query_context = {
                    "document_name": request.document_name,
                    "has_parsed_path": bool(request.parsed_file_path),
                    "organization_name": getattr(request, 'organization_id', None)
                }
                relevant_tools = self.tool_selection_manager.get_tools_for_query(
                    request.query, query_context
                )

            # Bind filters to RAG tool if present in request (ensures correct cache scoping)
            filters_bound = False
            if request.file_filter or request.folder_filter:
                relevant_tools = bind_rag_filters(
                    relevant_tools,
                    file_filter=request.file_filter,
                    folder_filter=request.folder_filter,
                )
                filters_bound = True
                logger.debug(
                    f"Bound RAG filters: file={request.file_filter}, folder={request.folder_filter}"
                )

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

                # Use native async invocation with semaphore-based concurrency control
                # Context propagation for token tracking is automatic with ainvoke()
                async with AgentConcurrency.get_semaphore():
                    result = await self.agent.ainvoke(
                        {"messages": messages_to_send},
                        config
                    )
            # Create dynamic agent with filtered tools if:
            # - Filters are bound (to ensure bound RAG tool is used), OR
            # - Tool selection is enabled and filtered tools differ from default
            elif filters_bound or (self.tool_selection_manager.enabled and relevant_tools != self.tools):
                dynamic_prompt = get_system_prompt([t.name for t in relevant_tools])
                dynamic_agent = create_agent(
                    model=self.llm,
                    tools=relevant_tools,
                    system_prompt=dynamic_prompt,
                    middleware=self.middleware_list if self.middleware_list else None,
                    checkpointer=self.checkpointer  # Use same checkpointer for filtered agents
                )
                logger.debug(f"Executing agent with {len(relevant_tools)} filtered tools")
                # Use native async invocation with semaphore-based concurrency control
                async with AgentConcurrency.get_semaphore():
                    result = await dynamic_agent.ainvoke(
                        {"messages": [message]},  # Only current message - checkpointer handles history
                        config
                    )
            else:
                # Use default agent with all tools
                logger.debug("Executing agent with all tools")
                # Use native async invocation with semaphore-based concurrency control
                async with AgentConcurrency.get_semaphore():
                    result = await self.agent.ainvoke(
                        {"messages": [message]},  # Only current message - checkpointer handles history
                        config
                    )

            # LangGraph agent returns dict with 'messages' key
            response_text = ""
            all_messages = []
            if result:
                if isinstance(result, dict):
                    if "messages" in result and result["messages"]:
                        all_messages = result["messages"]
                        # Debug: Log message types to trace tool outputs
                        msg_types = [type(m).__name__ for m in all_messages]
                        logger.info(f"Agent returned {len(all_messages)} messages: {msg_types}")
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
            parsed_result = self.result_parser.parse(response_text, all_messages)
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
        # Use direct invocation for better performance (bypasses ReAct agent)
        if self.config.use_direct_invocation:
            result = await self.generate_summary_direct(
                document_name=document_name,
                parsed_file_path=parsed_file_path,
                max_words=max_words,
                organization_id=organization_id,
            )
            if result.success and result.summary:
                return result.summary
            logger.warning(f"Direct summary generation failed: {result.error}")
            return result.error or "Summary generation failed"

        # Fall back to ReAct agent pattern
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

        # Debug: Log response structure to trace summary extraction
        logger.info(
            f"generate_summary response: success={response.success}, "
            f"has_content={response.content is not None}, "
            f"has_summary={response.content.summary is not None if response.content else False}, "
            f"summary_len={len(response.content.summary) if response.content and response.content.summary else 0}"
        )

        if response.success and response.content and response.content.summary:
            return response.content.summary
        logger.warning(f"generate_summary falling back to message: {response.message[:100]}...")
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
        # Use direct invocation for better performance (bypasses ReAct agent)
        if self.config.use_direct_invocation:
            result = await self.generate_faqs_direct(
                document_name=document_name,
                parsed_file_path=parsed_file_path,
                num_faqs=num_faqs,
                organization_id=organization_id,
            )
            if result.success and result.faqs:
                # Convert dict list to FAQ objects
                return [FAQ(question=f['question'], answer=f['answer']) for f in result.faqs]
            logger.warning(f"Direct FAQ generation failed: {result.error}")
            return []

        # Fall back to ReAct agent pattern
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
        # Use direct invocation for better performance (bypasses ReAct agent)
        if self.config.use_direct_invocation:
            result = await self.generate_questions_direct(
                document_name=document_name,
                parsed_file_path=parsed_file_path,
                num_questions=num_questions,
                organization_id=organization_id,
            )
            if result.success and result.questions:
                # Convert dict list to Question objects
                return [
                    Question(
                        question=q['question'],
                        expected_answer=q.get('expected_answer'),
                        difficulty=q.get('difficulty'),
                    )
                    for q in result.questions
                ]
            logger.warning(f"Direct question generation failed: {result.error}")
            return []

        # Fall back to ReAct agent pattern
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
        # Use direct invocation for better performance (bypasses ReAct agent)
        if self.config.use_direct_invocation:
            result = await self.generate_all_direct(
                document_name=document_name,
                parsed_file_path=parsed_file_path,
                options=options,
                organization_id=organization_id,
            )
            if result.success:
                # Convert to GeneratedContent
                faqs = None
                if result.faqs:
                    faqs = [FAQ(question=f['question'], answer=f['answer']) for f in result.faqs]

                questions = None
                if result.questions:
                    questions = [
                        Question(
                            question=q['question'],
                            expected_answer=q.get('expected_answer'),
                            difficulty=q.get('difficulty'),
                        )
                        for q in result.questions
                    ]

                return GeneratedContent(
                    summary=result.summary,
                    faqs=faqs,
                    questions=questions,
                )
            logger.warning(f"Direct generate_all failed: {result.error}")
            return GeneratedContent()

        # Fall back to ReAct agent pattern
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
        file_filter: Optional[List[str]] = None,
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
            file_filter: Optional list of file names to filter search
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
            files_str = ", ".join(file_filter) if len(file_filter) > 1 else file_filter[0]
            search_context += f" (filter by file(s): {files_str})"
        search_context += f" [organization: {organization_name}, mode: {search_mode}]"

        request = DocumentRequest(
            document_name="rag_search",  # Placeholder for RAG operations
            parsed_file_path="rag_search",  # Placeholder for RAG operations
            query=search_context,
            session_id=session_id,
            organization_id=organization_id,
            file_filter=file_filter,      # Pass structured filter for cache scoping
            folder_filter=folder_filter,  # Pass structured filter for cache scoping
        )

        # PERF: Skip tool selection for RAG chat - we know only rag_search is needed
        # This saves ~2-3 seconds by avoiding QueryClassifier + LLMToolSelector LLM calls
        return await self.process_request(request, rag_only=True)

    async def chat_stream(
        self,
        query: str,
        organization_name: str,
        session_id: Optional[str] = None,
        folder_filter: Optional[str] = None,
        file_filter: Optional[List[str]] = None,
        search_mode: str = "hybrid",
        organization_id: Optional[str] = None,
        include_tool_events: bool = True,
    ):
        """
        Streaming conversational RAG - chat with documents using SSE.

        Yields events as they occur:
        - status: Progress updates
        - tool_start: Tool execution beginning
        - tool_end: Tool execution complete
        - token: Individual LLM tokens
        - citations: Source citations
        - usage: Token usage statistics
        - done: Stream complete

        Args:
            query: User's question or search query
            organization_name: Organization name for store lookup
            session_id: Optional session ID for conversation continuity
            folder_filter: Optional folder name to filter search
            file_filter: Optional list of file names to filter search
            search_mode: Search mode - 'semantic', 'keyword', or 'hybrid'
            organization_id: Organization ID for multi-tenant isolation
            include_tool_events: Whether to include tool_start/tool_end events

        Yields:
            Dict with event type and data
        """
        import json

        # Build the query with search context
        search_context = f"Search documents for: {query}"
        if folder_filter:
            search_context += f" (filter by folder: {folder_filter})"
        if file_filter:
            files_str = ", ".join(file_filter) if len(file_filter) > 1 else file_filter[0]
            search_context += f" (filter by file(s): {files_str})"
        search_context += f" [organization: {organization_name}, mode: {search_mode}]"

        session_id = session_id or str(uuid.uuid4())

        # Yield initial status
        yield {"event": "status", "message": "Starting RAG search..."}

        # Get the RAG tool directly (skip tool selection for streaming too)
        rag_tool = self.tools_by_name.get('rag_search')
        relevant_tools = [rag_tool] if rag_tool else self.tools

        # Bind filters to RAG tool if present
        if file_filter or folder_filter:
            relevant_tools = bind_rag_filters(
                relevant_tools,
                file_filter=file_filter,
                folder_filter=folder_filter,
            )

        # Create dynamic agent for streaming
        dynamic_prompt = get_system_prompt([t.name for t in relevant_tools])
        dynamic_agent = create_agent(
            model=self.llm,
            tools=relevant_tools,
            system_prompt=dynamic_prompt,
            middleware=self.middleware_list if self.middleware_list else None,
            checkpointer=self.checkpointer
        )

        config = {"configurable": {"thread_id": session_id}}
        message = HumanMessage(content=search_context)

        # Track state during streaming
        accumulated_text = ""
        citations = []
        input_tokens = 0
        output_tokens = 0
        tool_start_time = None

        try:
            async with AgentConcurrency.get_semaphore():
                # Use astream_events for granular streaming
                async for event in dynamic_agent.astream_events(
                    {"messages": [message]},
                    config,
                    version="v2",
                ):
                    event_kind = event.get("event")

                    # Tool start event
                    if event_kind == "on_tool_start" and include_tool_events:
                        tool_name = event.get("name", "unknown")
                        tool_start_time = time.time()
                        yield {
                            "event": "tool_start",
                            "tool_name": tool_name,
                            "status": "executing",
                        }

                    # Tool end event
                    elif event_kind == "on_tool_end" and include_tool_events:
                        tool_name = event.get("name", "unknown")
                        duration_ms = elapsed_ms(tool_start_time) if tool_start_time else 0

                        # Extract citations from RAG tool output
                        output = event.get("data", {}).get("output", "")
                        if tool_name == "rag_search" and output:
                            try:
                                result = json.loads(output) if isinstance(output, str) else output
                                citations = result.get("citations", [])
                            except (json.JSONDecodeError, AttributeError):
                                pass

                        yield {
                            "event": "tool_end",
                            "tool_name": tool_name,
                            "status": "complete",
                            "duration_ms": duration_ms,
                            "citations_count": len(citations),
                        }

                    # Token streaming from LLM
                    elif event_kind == "on_chat_model_stream":
                        chunk = event.get("data", {}).get("chunk")
                        if chunk and hasattr(chunk, "content"):
                            content = chunk.content
                            # Handle both string and list content types
                            if content:
                                if isinstance(content, str):
                                    token = content
                                elif isinstance(content, list):
                                    # Extract text from list of content blocks
                                    token = "".join(
                                        item.get("text", "") if isinstance(item, dict) else str(item)
                                        for item in content
                                    )
                                else:
                                    token = str(content)

                                if token:
                                    accumulated_text += token
                                    output_tokens += 1  # Approximate token count
                                    yield {
                                        "event": "token",
                                        "token": token,
                                        "accumulated": accumulated_text,
                                    }

                    # LLM start - capture input tokens
                    elif event_kind == "on_chat_model_start":
                        # Yield status for first LLM call after tool completes
                        if citations:
                            yield {"event": "status", "message": "Generating answer..."}

                    # LLM end - capture usage if available
                    elif event_kind == "on_chat_model_end":
                        response = event.get("data", {}).get("output")
                        if response and hasattr(response, "usage_metadata"):
                            usage = response.usage_metadata
                            if usage:
                                input_tokens = getattr(usage, "input_tokens", 0)
                                output_tokens = getattr(usage, "output_tokens", output_tokens)

            # Yield citations
            if citations:
                yield {
                    "event": "citations",
                    "citations": citations,
                }

            # Yield usage
            yield {
                "event": "usage",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }

            # Yield done
            yield {
                "event": "done",
                "session_id": session_id,
                "success": True,
            }

        except Exception as e:
            logger.error(f"Streaming chat error: {e}", exc_info=True)
            yield {
                "event": "error",
                "error": str(e),
                "recoverable": False,
            }

    # =========================================================================
    # Direct Tool Invocation Methods (Bypass ReAct Agent)
    # =========================================================================

    async def _load_document_direct(
        self,
        document_name: str,
        parsed_file_path: str,
    ) -> tuple[bool, str, Optional[str]]:
        """Load document content directly using DocumentLoaderTool.

        Args:
            document_name: Name of the document
            parsed_file_path: GCS path to parsed document

        Returns:
            Tuple of (success, content, content_hash)
        """
        import json
        from src.core.executors import get_executors
        from src.utils.async_utils import run_in_executor_with_context
        from .tools.base import compute_content_hash

        loader_tool = self.tools_by_name.get('document_loader')
        if not loader_tool:
            logger.error("document_loader tool not found")
            return False, "Document loader tool not available", None

        try:
            # Run loader in executor (it's sync internally for GCS)
            result_json = await run_in_executor_with_context(
                get_executors().io_executor,
                loader_tool._run,
                document_name,
                parsed_file_path,
            )

            result = json.loads(result_json)
            if not result.get('success'):
                return False, result.get('error', 'Unknown error loading document'), None

            content = result.get('content', '')
            content_hash = compute_content_hash(content)

            logger.info(
                f"Direct document load: {document_name} "
                f"({len(content)} chars, hash={content_hash[:8]}...)"
            )
            return True, content, content_hash

        except Exception as e:
            logger.error(f"Direct document load failed: {e}")
            return False, str(e), None

    async def generate_summary_direct(
        self,
        document_name: str,
        parsed_file_path: str,
        max_words: Optional[int] = None,
        organization_id: Optional[str] = None,
        persist: bool = True,
    ) -> DirectGenerationResult:
        """Generate summary using direct tool invocation (bypasses ReAct agent).

        This method provides ~75% latency improvement over the ReAct agent pattern
        by calling tools directly instead of going through the agent decision loop.

        Args:
            document_name: Name of the document
            parsed_file_path: GCS path to parsed document
            max_words: Maximum words for summary
            organization_id: Organization ID for multi-tenant isolation
            persist: Whether to persist the result to GCS/database

        Returns:
            DirectGenerationResult with summary
        """
        import json
        start_time = time.time()

        # Load document
        success, content, content_hash = await self._load_document_direct(
            document_name, parsed_file_path
        )
        if not success:
            return DirectGenerationResult(
                success=False,
                error=content,  # content contains error message on failure
                processing_time_ms=elapsed_ms(start_time),
            )

        # Get summary tool and invoke directly
        summary_tool = self.tools_by_name.get('summary_generator')
        if not summary_tool:
            return DirectGenerationResult(
                success=False,
                error="Summary generator tool not available",
                processing_time_ms=elapsed_ms(start_time),
            )

        try:
            # Use async method if available, otherwise sync
            summary_max_words = max_words or self.config.summary_max_words
            result_json = await summary_tool._arun(
                content=content,
                document_name=document_name,
                max_words=summary_max_words,
                organization_id=organization_id,
            )

            result = json.loads(result_json)
            if not result.get('success'):
                return DirectGenerationResult(
                    success=False,
                    error=result.get('error', 'Summary generation failed'),
                    processing_time_ms=elapsed_ms(start_time),
                )

            summary = result.get('summary')
            cached = result.get('cached', False)

            # Persist if requested and not cached (cached = already persisted)
            output_paths = None
            persisted = False
            if persist and not cached:
                persist_result = await self._persist_content_direct(
                    document_name=document_name,
                    parsed_file_path=parsed_file_path,
                    summary=summary,
                    content_hash=content_hash,
                    organization_id=organization_id,
                )
                persisted = persist_result.get('success', False)
                output_paths = persist_result.get('output_file_paths')

            processing_time = elapsed_ms(start_time)
            logger.info(
                f"Direct summary generation: {document_name} "
                f"({processing_time:.1f}ms, cached={cached})"
            )

            return DirectGenerationResult(
                success=True,
                summary=summary,
                processing_time_ms=processing_time,
                cached=cached,
                persisted=persisted,
                output_file_paths=output_paths,
                content_hash=content_hash,
            )

        except Exception as e:
            logger.error(f"Direct summary generation failed: {e}")
            return DirectGenerationResult(
                success=False,
                error=str(e),
                processing_time_ms=elapsed_ms(start_time),
            )

    async def generate_faqs_direct(
        self,
        document_name: str,
        parsed_file_path: str,
        num_faqs: Optional[int] = None,
        organization_id: Optional[str] = None,
        persist: bool = True,
    ) -> DirectGenerationResult:
        """Generate FAQs using direct tool invocation (bypasses ReAct agent).

        Args:
            document_name: Name of the document
            parsed_file_path: GCS path to parsed document
            num_faqs: Number of FAQs to generate
            organization_id: Organization ID for multi-tenant isolation
            persist: Whether to persist the result to GCS/database

        Returns:
            DirectGenerationResult with FAQs
        """
        import json
        start_time = time.time()

        # Load document
        success, content, content_hash = await self._load_document_direct(
            document_name, parsed_file_path
        )
        if not success:
            return DirectGenerationResult(
                success=False,
                error=content,
                processing_time_ms=elapsed_ms(start_time),
            )

        # Get FAQ tool and invoke directly
        faq_tool = self.tools_by_name.get('faq_generator')
        if not faq_tool:
            return DirectGenerationResult(
                success=False,
                error="FAQ generator tool not available",
                processing_time_ms=elapsed_ms(start_time),
            )

        try:
            faq_count = num_faqs or self.config.default_num_faqs
            result_json = await faq_tool._arun(
                content=content,
                document_name=document_name,
                num_faqs=faq_count,
                organization_id=organization_id,
            )

            result = json.loads(result_json)
            if not result.get('success'):
                return DirectGenerationResult(
                    success=False,
                    error=result.get('error', 'FAQ generation failed'),
                    processing_time_ms=elapsed_ms(start_time),
                )

            faqs = result.get('faqs', [])
            cached = result.get('cached', False)

            # Persist if requested and not cached
            output_paths = None
            persisted = False
            if persist and not cached:
                persist_result = await self._persist_content_direct(
                    document_name=document_name,
                    parsed_file_path=parsed_file_path,
                    faqs=json.dumps({'faqs': faqs}),
                    content_hash=content_hash,
                    organization_id=organization_id,
                )
                persisted = persist_result.get('success', False)
                output_paths = persist_result.get('output_file_paths')

            processing_time = elapsed_ms(start_time)
            logger.info(
                f"Direct FAQ generation: {document_name} "
                f"({len(faqs)} FAQs, {processing_time:.1f}ms, cached={cached})"
            )

            return DirectGenerationResult(
                success=True,
                faqs=faqs,
                processing_time_ms=processing_time,
                cached=cached,
                persisted=persisted,
                output_file_paths=output_paths,
                content_hash=content_hash,
            )

        except Exception as e:
            logger.error(f"Direct FAQ generation failed: {e}")
            return DirectGenerationResult(
                success=False,
                error=str(e),
                processing_time_ms=elapsed_ms(start_time),
            )

    async def generate_questions_direct(
        self,
        document_name: str,
        parsed_file_path: str,
        num_questions: Optional[int] = None,
        organization_id: Optional[str] = None,
        persist: bool = True,
    ) -> DirectGenerationResult:
        """Generate questions using direct tool invocation (bypasses ReAct agent).

        Args:
            document_name: Name of the document
            parsed_file_path: GCS path to parsed document
            num_questions: Number of questions to generate
            organization_id: Organization ID for multi-tenant isolation
            persist: Whether to persist the result to GCS/database

        Returns:
            DirectGenerationResult with questions
        """
        import json
        start_time = time.time()

        # Load document
        success, content, content_hash = await self._load_document_direct(
            document_name, parsed_file_path
        )
        if not success:
            return DirectGenerationResult(
                success=False,
                error=content,
                processing_time_ms=elapsed_ms(start_time),
            )

        # Get question tool and invoke directly
        question_tool = self.tools_by_name.get('question_generator')
        if not question_tool:
            return DirectGenerationResult(
                success=False,
                error="Question generator tool not available",
                processing_time_ms=elapsed_ms(start_time),
            )

        try:
            question_count = num_questions or self.config.default_num_questions
            result_json = await question_tool._arun(
                content=content,
                document_name=document_name,
                num_questions=question_count,
                organization_id=organization_id,
            )

            result = json.loads(result_json)
            if not result.get('success'):
                return DirectGenerationResult(
                    success=False,
                    error=result.get('error', 'Question generation failed'),
                    processing_time_ms=elapsed_ms(start_time),
                )

            questions = result.get('questions', [])
            cached = result.get('cached', False)

            # Persist if requested and not cached
            output_paths = None
            persisted = False
            if persist and not cached:
                persist_result = await self._persist_content_direct(
                    document_name=document_name,
                    parsed_file_path=parsed_file_path,
                    questions=json.dumps({'questions': questions}),
                    content_hash=content_hash,
                    organization_id=organization_id,
                )
                persisted = persist_result.get('success', False)
                output_paths = persist_result.get('output_file_paths')

            processing_time = elapsed_ms(start_time)
            logger.info(
                f"Direct question generation: {document_name} "
                f"({len(questions)} questions, {processing_time:.1f}ms, cached={cached})"
            )

            return DirectGenerationResult(
                success=True,
                questions=questions,
                processing_time_ms=processing_time,
                cached=cached,
                persisted=persisted,
                output_file_paths=output_paths,
                content_hash=content_hash,
            )

        except Exception as e:
            logger.error(f"Direct question generation failed: {e}")
            return DirectGenerationResult(
                success=False,
                error=str(e),
                processing_time_ms=elapsed_ms(start_time),
            )

    async def generate_all_direct(
        self,
        document_name: str,
        parsed_file_path: str,
        options: Optional[GenerationOptions] = None,
        organization_id: Optional[str] = None,
        persist: bool = True,
    ) -> DirectGenerationResult:
        """Generate all content types using direct tool invocation with parallel execution.

        This method:
        1. Loads the document once
        2. Runs summary, FAQ, and question generation in parallel
        3. Persists all content in a single operation

        Args:
            document_name: Name of the document
            parsed_file_path: GCS path to parsed document
            options: Generation options (num_faqs, num_questions, summary_max_words)
            organization_id: Organization ID for multi-tenant isolation
            persist: Whether to persist the result to GCS/database

        Returns:
            DirectGenerationResult with summary, FAQs, and questions
        """
        import json
        start_time = time.time()

        # Load document once
        success, content, content_hash = await self._load_document_direct(
            document_name, parsed_file_path
        )
        if not success:
            return DirectGenerationResult(
                success=False,
                error=content,
                processing_time_ms=elapsed_ms(start_time),
            )

        # Get tools
        summary_tool = self.tools_by_name.get('summary_generator')
        faq_tool = self.tools_by_name.get('faq_generator')
        question_tool = self.tools_by_name.get('question_generator')

        if not all([summary_tool, faq_tool, question_tool]):
            missing = []
            if not summary_tool:
                missing.append('summary_generator')
            if not faq_tool:
                missing.append('faq_generator')
            if not question_tool:
                missing.append('question_generator')
            return DirectGenerationResult(
                success=False,
                error=f"Missing tools: {', '.join(missing)}",
                processing_time_ms=elapsed_ms(start_time),
            )

        # Get options
        opts = options or GenerationOptions()
        summary_max_words = opts.summary_max_words or self.config.summary_max_words
        num_faqs = opts.num_faqs or self.config.default_num_faqs
        num_questions = opts.num_questions or self.config.default_num_questions

        try:
            # Run all generators in parallel using asyncio.gather
            summary_coro = summary_tool._arun(
                content=content,
                document_name=document_name,
                max_words=summary_max_words,
                organization_id=organization_id,
            )
            faq_coro = faq_tool._arun(
                content=content,
                document_name=document_name,
                num_faqs=num_faqs,
                organization_id=organization_id,
            )
            question_coro = question_tool._arun(
                content=content,
                document_name=document_name,
                num_questions=num_questions,
                organization_id=organization_id,
            )

            # Execute in parallel
            results = await asyncio.gather(
                summary_coro, faq_coro, question_coro,
                return_exceptions=True
            )

            # Parse results
            summary = None
            faqs = None
            questions = None
            any_cached = False
            errors = []

            # Summary result
            if isinstance(results[0], Exception):
                errors.append(f"Summary: {results[0]}")
            else:
                summary_result = json.loads(results[0])
                if summary_result.get('success'):
                    summary = summary_result.get('summary')
                    if summary_result.get('cached'):
                        any_cached = True
                else:
                    errors.append(f"Summary: {summary_result.get('error')}")

            # FAQ result
            if isinstance(results[1], Exception):
                errors.append(f"FAQs: {results[1]}")
            else:
                faq_result = json.loads(results[1])
                if faq_result.get('success'):
                    faqs = faq_result.get('faqs', [])
                    if faq_result.get('cached'):
                        any_cached = True
                else:
                    errors.append(f"FAQs: {faq_result.get('error')}")

            # Question result
            if isinstance(results[2], Exception):
                errors.append(f"Questions: {results[2]}")
            else:
                question_result = json.loads(results[2])
                if question_result.get('success'):
                    questions = question_result.get('questions', [])
                    if question_result.get('cached'):
                        any_cached = True
                else:
                    errors.append(f"Questions: {question_result.get('error')}")

            # Persist all content together if requested and any content was generated
            output_paths = None
            persisted = False
            if persist and (summary or faqs or questions) and not any_cached:
                persist_result = await self._persist_content_direct(
                    document_name=document_name,
                    parsed_file_path=parsed_file_path,
                    summary=summary,
                    faqs=json.dumps({'faqs': faqs}) if faqs else None,
                    questions=json.dumps({'questions': questions}) if questions else None,
                    content_hash=content_hash,
                    organization_id=organization_id,
                )
                persisted = persist_result.get('success', False)
                output_paths = persist_result.get('output_file_paths')

            processing_time = elapsed_ms(start_time)
            success = bool(summary or faqs or questions)

            logger.info(
                f"Direct generate_all: {document_name} "
                f"(summary={'yes' if summary else 'no'}, "
                f"faqs={len(faqs) if faqs else 0}, "
                f"questions={len(questions) if questions else 0}, "
                f"{processing_time:.1f}ms)"
            )

            return DirectGenerationResult(
                success=success,
                summary=summary,
                faqs=faqs,
                questions=questions,
                processing_time_ms=processing_time,
                cached=any_cached,
                persisted=persisted,
                output_file_paths=output_paths,
                content_hash=content_hash,
                error='; '.join(errors) if errors and not success else None,
            )

        except Exception as e:
            logger.error(f"Direct generate_all failed: {e}")
            return DirectGenerationResult(
                success=False,
                error=str(e),
                processing_time_ms=elapsed_ms(start_time),
            )

    async def _persist_content_direct(
        self,
        document_name: str,
        parsed_file_path: str,
        summary: Optional[str] = None,
        faqs: Optional[str] = None,
        questions: Optional[str] = None,
        content_hash: Optional[str] = None,
        organization_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Persist generated content directly using ContentPersistTool.

        Args:
            document_name: Name of the document
            parsed_file_path: GCS path to parsed document
            summary: Generated summary text
            faqs: FAQs as JSON string
            questions: Questions as JSON string
            content_hash: Hash of source content for cache validation
            organization_id: Organization ID for multi-tenant isolation

        Returns:
            Dict with success status and output_file_paths
        """
        import json
        from src.core.executors import get_executors
        from src.utils.async_utils import run_in_executor_with_context

        persist_tool = self.tools_by_name.get('content_persist')
        if not persist_tool:
            return {'success': False, 'error': 'Content persist tool not available'}

        try:
            result_json = await run_in_executor_with_context(
                get_executors().io_executor,
                persist_tool._run,
                document_name,
                parsed_file_path,
                summary,
                faqs,
                questions,
                content_hash,
                organization_id,
            )

            return json.loads(result_json)

        except Exception as e:
            logger.error(f"Direct content persist failed: {e}")
            return {'success': False, 'error': str(e)}

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
                    "model": self.config.openai_model,
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
