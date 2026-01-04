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
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .config import DocumentAgentConfig
from .tools import create_document_tools
from .context import rag_filter_context
from src.utils.timer_utils import elapsed_ms
from src.utils.async_utils import run_in_executor_with_context

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
            timeout=300,  # 5 minutes for complex generation tasks
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

            # Get relevant tools for this query via tool selection manager
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

                # Use run_in_executor_with_context to propagate usage context
                # for token tracking in the callback handler
                result = await run_in_executor_with_context(
                    get_executors().agent_executor,
                    self.agent.invoke,
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
                # Use run_in_executor_with_context to propagate usage context
                # for token tracking in the callback handler
                result = await run_in_executor_with_context(
                    get_executors().agent_executor,
                    dynamic_agent.invoke,
                    {"messages": [message]},  # Only current message - checkpointer handles history
                    config
                )
            else:
                # Use default agent with all tools
                logger.debug("Executing agent with all tools")
                # Use run_in_executor_with_context to propagate usage context
                # for token tracking in the callback handler
                result = await run_in_executor_with_context(
                    get_executors().agent_executor,
                    self.agent.invoke,
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
            file_filter=file_filter,      # Pass structured filter for cache scoping
            folder_filter=folder_filter,  # Pass structured filter for cache scoping
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
