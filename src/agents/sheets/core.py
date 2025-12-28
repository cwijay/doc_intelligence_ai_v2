"""Core Sheets Agent implementation using LangChain 1.2.0."""

import asyncio
import time
import uuid
import hashlib
import logging
import os
from collections import OrderedDict
from typing import Dict, List, Any, Optional
import pandas as pd

from concurrent.futures import ThreadPoolExecutor

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelRetryMiddleware,
    ToolRetryMiddleware,
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
)
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver

from .config import SheetsAgentConfig
from .gcs_loader import validate_file_path as gcs_validate_file_path
from src.utils.timer_utils import elapsed_ms
from src.utils.gcs_utils import is_gcs_path
from src.agents.core.audit_queue import enqueue_audit_event

# Base agent and shared utilities
from src.agents.core.base_agent import BaseAgent
from src.agents.core.token_utils import calculate_token_usage

# Background executor for non-blocking audit logging
_audit_executor: Optional[ThreadPoolExecutor] = None
from .tools import (
    FilePreviewTool,
    CrossFileQueryTool,
    SmartAnalysisTool,
)
from .schemas import (
    ChatRequest,
    ChatResponse,
    FileMetadata,
    ToolUsage,
    TokenUsage,
    SessionInfo
)

logger = logging.getLogger(__name__)


# NOTE: RateLimiter class has been moved to src/agents/core/rate_limiter.py


class FileCache:
    """Manages cached file data to avoid redundant reads using efficient OrderedDict LRU."""

    def __init__(self, max_size: int = 50):
        self.cache: OrderedDict[str, pd.DataFrame] = OrderedDict()
        self.max_size = max_size
        self._hits = 0
        self._misses = 0

    def get(self, file_path: str) -> Optional[pd.DataFrame]:
        """Get cached DataFrame if available, moving to end (most recently used)."""
        if file_path in self.cache:
            # Move to end to mark as recently used
            self.cache.move_to_end(file_path)
            self._hits += 1
            logger.debug(f"Cache hit for {file_path}")
            return self.cache[file_path].copy()
        self._misses += 1
        return None

    def put(self, file_path: str, df: pd.DataFrame):
        """Cache DataFrame with LRU eviction using OrderedDict."""
        if file_path in self.cache:
            # Update existing entry and move to end
            self.cache.move_to_end(file_path)
            self.cache[file_path] = df.copy()
        else:
            # Evict oldest if at capacity
            if len(self.cache) >= self.max_size:
                oldest_file, _ = self.cache.popitem(last=False)
                logger.debug(f"Evicted {oldest_file} from cache")

            self.cache[file_path] = df.copy()
        logger.debug(f"Cached {file_path} (shape: {df.shape})")

    def clear(self):
        """Clear all cached data."""
        self.cache.clear()
        self._hits = 0
        self._misses = 0
        logger.debug("File cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_percent": round(hit_rate, 2)
        }


# NOTE: SessionManager class has been moved to src/agents/core/session_manager.py


SYSTEM_PROMPT = """You are an expert Excel and data analysis assistant.

CRITICAL RULES:
1. For single file queries: Call smart_analysis ONCE, then answer directly from its output. Do NOT call any other tools.
2. For multi-file queries (2+ files): Use cross_file_query for SQL-based analysis across files.
3. ONE tool call is sufficient for most queries. Never call the same tool twice.

The smart_analysis tool provides complete file preview + query analysis in a single call.
After receiving tool output, synthesize the findings and provide a clear, business-friendly answer.

Do NOT:
- Call multiple tools for the same file
- Call file_preview separately (smart_analysis includes preview)
- Make redundant queries on data you already have"""


class SheetsAgent(BaseAgent):
    """AI-powered Excel/Sheets analysis agent using LangGraph.

    Inherits from BaseAgent for shared functionality:
    - Session management
    - Rate limiting
    - Memory (short-term and long-term)
    - Audit logging
    """

    def __init__(self, config: SheetsAgentConfig):
        """Initialize Sheets Agent.

        Args:
            config: Agent configuration
        """
        # File cache is sheets-specific (not in BaseAgent)
        self.file_cache = FileCache(max_size=50)

        # Initialize base agent (session manager, rate limiter, memory, audit)
        super().__init__(config)

        # Initialize LLM (sheets-specific)
        self.llm = self._init_llm()

        # Initialize tools (sheets-specific)
        self.tools = self._init_tools()

        # Build middleware list from config (LangChain 1.2.0 built-in middleware)
        self.middleware_list = self._build_middleware()

        # Initialize checkpointer for conversation memory
        self.checkpointer = MemorySaver()

        # Create agent with middleware and checkpointer
        self.agent = self._create_agent()

        logger.info(f"Sheets Agent initialized with model: {self.config.openai_model}")

    def _init_llm(self):
        """Initialize the language model using init_chat_model with token tracking callback."""
        if not self.config.openai_api_key:
            raise ValueError("OpenAI API key is required")

        # Use shared callback creation from BaseAgent
        callbacks = self._create_token_tracking_callback("sheets_agent")

        llm = init_chat_model(
            model=self.config.openai_model,
            model_provider="openai",
            temperature=self.config.temperature,
            api_key=self.config.openai_api_key,
            use_responses_api=True,  # Required for gpt-5.1-codex-mini
            timeout=60,  # Prevent hanging requests
            max_retries=2,
            callbacks=callbacks if callbacks else None,
        )

        logger.info(f"Initialized LLM: {self.config.openai_model}")
        return llm

    def _init_tools(self) -> List:
        """Initialize agent tools with focused tool sets for efficiency.

        Single-file queries use only SmartAnalysisTool (prevents redundant tool calls).
        Multi-file queries use CrossFileQueryTool + FilePreviewTool.
        """
        # Primary tool for single-file analysis (with caching)
        self.single_file_tools = [
            SmartAnalysisTool(config=self.config, file_cache=self.file_cache),
        ]

        # Tools for multi-file analysis (with caching)
        self.multi_file_tools = [
            CrossFileQueryTool(config=self.config, file_cache=self.file_cache),
            FilePreviewTool(config=self.config, file_cache=self.file_cache),
        ]

        # All tools (for health checks and reference)
        all_tools = self.single_file_tools + self.multi_file_tools

        logger.info(f"Initialized {len(all_tools)} tools (single-file: {len(self.single_file_tools)}, multi-file: {len(self.multi_file_tools)})")
        return all_tools

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

            logger.info(f"Built {len(middleware)} middleware components")
            return middleware

        except Exception as e:
            logger.warning(f"Failed to build middleware: {e}")
            return []

    def _create_agent(self, tools: List = None):
        """Create the agent using LangChain 1.2.0 with built-in middleware and checkpointer.

        Args:
            tools: Optional list of tools. If None, uses default single-file tools.
        """
        tools_to_use = tools if tools is not None else self.single_file_tools
        agent = create_agent(
            model=self.llm,
            tools=tools_to_use,
            system_prompt=SYSTEM_PROMPT,
            middleware=self.middleware_list if self.middleware_list else None,
            checkpointer=self.checkpointer  # Enable automatic conversation memory
        )

        logger.info(f"Created agent with {len(tools_to_use)} tools ({len(self.middleware_list)} middleware)")
        return agent

    # _init_audit_logging() is inherited from BaseAgent

    def _get_agent_type(self) -> str:
        """Get agent type identifier for audit/memory."""
        return "sheets"

    async def process_chat(self, request: ChatRequest) -> ChatResponse:
        """
        Process a chat request and return analysis results.

        Args:
            request: Chat request with file paths and query

        Returns:
            Chat response with analysis results
        """
        start_time = time.time()

        try:
            # Get or create session
            session = self.session_manager.get_or_create_session(request.session_id)

            # Check rate limit
            if not self.rate_limiter.is_allowed(session.session_id):
                retry_after = self.rate_limiter.get_retry_after(session.session_id)
                return ChatResponse(
                    success=False,
                    message=f"Rate limit exceeded. Please try again in {retry_after} seconds.",
                    files_processed=[],
                    tools_used=[],
                    session_id=session.session_id,
                    total_processing_time_ms=elapsed_ms(start_time)
                )

            # Create query hash for caching
            query_context = f"{request.query}||{','.join(sorted(request.file_paths))}"
            query_hash = hashlib.md5(query_context.encode()).hexdigest()

            # Check for cached response
            cached_response = self.session_manager.get_cached_response(session.session_id, query_hash)
            if cached_response:
                logger.info(f"Returning cached response for query hash {query_hash}")
                processing_time = elapsed_ms(start_time)
                cached_response["total_processing_time_ms"] = processing_time
                return ChatResponse(**cached_response)

            # Validate files
            validation_results = self._validate_files(request.file_paths)

            # Update session with current files
            session.files_in_context = request.file_paths
            session.query_count += 1

            # Get chat history from short-term memory
            chat_history = []
            if self.short_term_memory:
                chat_history = self.short_term_memory.get_messages(session.session_id)

            # Get long-term context if user_id provided
            long_term_context = ""
            if self.long_term_memory and request.user_id:
                long_term_context = self.long_term_memory.get_relevant_context(
                    request.user_id, request.query
                )

            # Prepare context for the agent
            context = self._prepare_context(request, validation_results, long_term_context)

            # Dynamic tool selection based on file count
            # Single file: Use only SmartAnalysisTool (prevents LLM from making redundant calls)
            # Multiple files: Use CrossFileQueryTool + FilePreviewTool
            if len(request.file_paths) == 1:
                agent_to_use = self.agent  # Default single-file agent
                logger.info(f"Using single-file agent (SmartAnalysisTool only)")
            else:
                # Create multi-file agent on demand
                agent_to_use = self._create_agent(self.multi_file_tools)
                logger.info(f"Using multi-file agent ({len(self.multi_file_tools)} tools)")

            # Execute agent with timeout enforcement
            try:
                async with asyncio.timeout(self.config.timeout_seconds):
                    agent_result = await self._execute_agent(context, request.query, chat_history, request, agent_to_use)
            except asyncio.TimeoutError:
                processing_time = elapsed_ms(start_time)
                logger.error(f"Agent execution timed out after {self.config.timeout_seconds}s")
                return ChatResponse(
                    success=False,
                    message=f"Request timed out after {self.config.timeout_seconds} seconds. Please try a simpler query.",
                    files_processed=[],
                    tools_used=[],
                    session_id=session.session_id,
                    total_processing_time_ms=processing_time
                )

            # Extract tool usage information
            tools_used = self._extract_tool_usage(agent_result)

            # Calculate token usage
            token_usage = self._calculate_token_usage(request.query, agent_result.get('response', ''))

            # Update session tracking
            self.session_manager.update_session(
                session.session_id,
                total_tokens_used=session.total_tokens_used + token_usage.total_tokens,
                total_processing_time_ms=session.total_processing_time_ms + elapsed_ms(start_time)
            )

            # Prepare file metadata
            files_metadata = [
                FileMetadata(
                    file_path=file_path,
                    file_type=self._get_file_extension(file_path),
                    processing_time_ms=50.0
                )
                for file_path in request.file_paths
            ]

            processing_time = elapsed_ms(start_time)

            # Log audit event if available
            if self.audit_logger:
                self._log_audit_event(request, agent_result, processing_time)

            response_data = ChatResponse(
                success=True,
                message="Analysis completed successfully",
                response=agent_result.get('response', 'Analysis completed'),
                files_processed=files_metadata,
                tools_used=tools_used,
                token_usage=token_usage,
                session_id=session.session_id,
                total_processing_time_ms=processing_time,
                data=agent_result.get('data')
            )

            # Save to short-term memory
            if self.short_term_memory:
                self.short_term_memory.add_human_message(
                    session.session_id, request.query
                )
                self.short_term_memory.add_ai_message(
                    session.session_id, agent_result.get('response', '')
                )

            # Cache the response
            cache_data = response_data.model_dump()
            cache_data.pop('timestamp', None)
            self.session_manager.cache_response(session.session_id, query_hash, cache_data)

            return response_data

        except Exception as e:
            processing_time = elapsed_ms(start_time)
            logger.error(f"Error processing chat request: {e}")

            return ChatResponse(
                success=False,
                message=f"Error processing request: {str(e)}",
                files_processed=[],
                tools_used=[],
                session_id=request.session_id or str(uuid.uuid4()),
                total_processing_time_ms=processing_time
            )

    def _validate_files(self, file_paths: List[str]) -> Dict[str, Any]:
        """Validate that files exist and are accessible (supports both local and GCS paths)."""
        validation_results = {
            "valid_files": [],
            "invalid_files": [],
            "total_size_mb": 0
        }

        for file_path in file_paths:
            try:
                # Use GCS-aware validation
                is_valid, error = gcs_validate_file_path(file_path)

                if is_valid:
                    file_info = {"path": file_path}

                    # Get file size for local files only (skip for GCS)
                    if not is_gcs_path(file_path) and os.path.exists(file_path):
                        file_size = os.path.getsize(file_path)
                        file_info["size_mb"] = file_size / (1024 * 1024)
                        validation_results["total_size_mb"] += file_size / (1024 * 1024)

                    validation_results["valid_files"].append(file_info)
                else:
                    validation_results["invalid_files"].append({
                        "path": file_path,
                        "error": error or "Validation failed"
                    })
            except Exception as e:
                validation_results["invalid_files"].append({
                    "path": file_path,
                    "error": str(e)
                })

        return validation_results

    def _prepare_context(
        self, request: ChatRequest, validation_results: Dict, long_term_context: str = ""
    ) -> str:
        """Prepare context string for the agent."""
        query_lower = request.query.lower()
        analysis_hints = []

        if any(term in query_lower for term in ['revenue', 'sales', 'profit', 'total', 'sum', 'amount']):
            analysis_hints.append("This appears to be a revenue/financial analysis query")

        if any(term in query_lower for term in ['q1', 'q2', 'q3', 'q4', 'quarter', 'fy25', 'fy24', 'fiscal']):
            analysis_hints.append("This involves quarterly or fiscal year analysis")

        if any(term in query_lower for term in ['trend', 'over time', 'monthly', 'yearly', 'growth']):
            analysis_hints.append("This requires trend analysis over time periods")

        context_parts = [
            f"User Query: {request.query}",
            f"Files to analyze: {', '.join(request.file_paths)}",
        ]

        if analysis_hints:
            context_parts.append(f"Query Analysis: {'; '.join(analysis_hints)}")

        if validation_results["invalid_files"]:
            invalid_files_info = [
                f"{item['path']}: {item['error']}"
                for item in validation_results["invalid_files"]
            ]
            context_parts.append(f"Invalid files: {'; '.join(invalid_files_info)}")

        if validation_results["total_size_mb"] > self.config.max_file_size_mb:
            context_parts.append(f"Warning: Total file size ({validation_results['total_size_mb']:.1f}MB) exceeds recommended limit")

        # Include long-term context if available
        if long_term_context:
            context_parts.append("")
            context_parts.append(long_term_context)

        return "\n".join(context_parts)

    async def _execute_agent(
        self, context: str, query: str, chat_history: List = None, request: Any = None, agent: Any = None
    ) -> Dict[str, Any]:
        """Execute the agent with the given context and query.

        Args:
            context: Prepared context string
            query: User query
            chat_history: Previous messages
            request: Original request object
            agent: Optional agent to use (for dynamic tool selection)

        Note: Retry logic is now handled by ModelRetryMiddleware (LangChain built-in).
        Token tracking is handled via thread-local usage context.
        """
        # Import usage context for token tracking
        try:
            from src.core.usage.context import usage_context
            USAGE_CONTEXT_AVAILABLE = True
        except ImportError:
            USAGE_CONTEXT_AVAILABLE = False
            usage_context = None

        # Extract org_id from request if available
        org_id = getattr(request, 'organization_id', None) if request else None
        user_id = getattr(request, 'user_id', None) if request else None
        session_id = getattr(request, 'session_id', None) if request else None

        # Set up usage context for token tracking (if available)
        ctx_manager = None
        if USAGE_CONTEXT_AVAILABLE and usage_context and org_id:
            ctx_manager = usage_context(
                org_id=org_id,
                feature="sheets_agent",
                user_id=user_id,
                session_id=session_id,
            )
            logger.debug(f"Token tracking context set for org {org_id}")

        try:
            # Enter usage context if available
            if ctx_manager:
                ctx_manager.__enter__()

            return await self._execute_agent_inner(context, query, session_id or "default", chat_history, agent)

        finally:
            # Exit usage context
            if ctx_manager:
                ctx_manager.__exit__(None, None, None)

    async def _execute_agent_inner(
        self, context: str, query: str, session_id: str, chat_history: List = None, agent: Any = None
    ) -> Dict[str, Any]:
        """Inner agent execution (called within usage context).

        Args:
            agent: Optional agent to use. If None, uses self.agent (default single-file agent).
        """
        try:
            message = HumanMessage(content=f"{context}\n\nPlease analyze the data and answer: {query}")

            # Build messages list with chat history
            messages = list(chat_history) if chat_history else []
            messages.append(message)

            # Use provided agent or default
            agent_to_use = agent if agent is not None else self.agent

            logger.debug(f"Executing LangGraph agent with thread_id: {session_id}")
            result = await asyncio.to_thread(
                agent_to_use.invoke,
                {"messages": messages},
                {"configurable": {"thread_id": session_id}}
            )
            logger.debug(f"Agent result type: {type(result)}")

            response_text = ""
            if result:
                if isinstance(result, dict):
                    if "messages" in result and result["messages"]:
                        last_message = result["messages"][-1]
                        if hasattr(last_message, 'content'):
                            response_text = str(last_message.content)
                        logger.debug(f"Extracted response from messages: {len(response_text)} chars")
                    elif "output" in result:
                        response_text = str(result["output"])
                        logger.debug(f"Extracted response from output: {len(response_text)} chars")
                elif hasattr(result, 'content'):
                    response_text = str(result.content)
                else:
                    response_text = str(result)

            final_response = response_text or "Analysis completed successfully"
            logger.info(f"Agent execution completed, response length: {len(final_response)} chars")

            return {
                "response": final_response,
                "data": None
            }

        except Exception as e:
            logger.error(f"Error executing agent: {e}", exc_info=True)
            return {
                "response": f"I apologize, but I encountered an error while analyzing your data: {str(e)}. Please try again or rephrase your question.",
                "data": None
            }

    def _extract_tool_usage(self, agent_result: Dict) -> List[ToolUsage]:
        """Extract tool usage information from agent result."""
        return [
            ToolUsage(
                tool_name="smart_analysis",
                input_data={"file_path": "analyzed"},
                execution_time_ms=100.0,
                success=True
            )
        ]

    def _calculate_token_usage(self, input_text: str, output_text: str) -> TokenUsage:
        """Calculate token usage using shared utility."""
        estimate = calculate_token_usage(
            input_text, output_text, model=self.config.openai_model
        )
        return TokenUsage(
            prompt_tokens=estimate.prompt_tokens,
            completion_tokens=estimate.completion_tokens,
            total_tokens=estimate.total_tokens,
            estimated_cost_usd=estimate.estimated_cost_usd
        )

    def _get_file_extension(self, file_path: str) -> str:
        """Get file extension from file path."""
        return file_path.split('.')[-1].lower()

    def _log_audit_event(self, request: ChatRequest, result: Dict, processing_time: float):
        """Log query to audit database using async-safe audit queue."""
        try:
            enqueue_audit_event(
                event_type="sheets_agent_query",
                details={
                    "session_id": request.session_id,
                    "query": request.query,
                    "file_paths": request.file_paths,
                    "processing_time_ms": processing_time,
                    "success": True
                },
                organization_id=request.organization_id
            )
        except Exception as e:
            logger.warning(f"Failed to enqueue audit event: {e}")

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the agent and its components."""
        try:
            # Get base health status (sessions, rate limiter, memory, audit)
            base_status = self._get_base_health_status()

            # Check LLM status
            openai_status = "healthy" if self.llm else "unhealthy"

            return {
                "status": "healthy" if openai_status == "healthy" else "degraded",
                "components": {
                    "api": "healthy",
                    "openai": openai_status,
                    "duckdb": "healthy",
                    "audit_logging": base_status["audit_logging"],
                    "short_term_memory": "enabled" if self.short_term_memory else "disabled",
                    "long_term_memory": "enabled" if self.long_term_memory else "disabled"
                },
                "sessions": base_status["sessions"],
                "cache": self.file_cache.get_stats(),
                "rate_limiter": base_status["rate_limiter"],
                "memory": base_status["memory"]
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "components": {
                    "api": "healthy",
                    "openai": "unknown",
                    "duckdb": "unknown",
                    "audit_logging": "unknown"
                }
            }

    # end_session() is inherited from BaseAgent
    # _save_conversation_summary() is inherited from BaseAgent

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown agent resources gracefully.

        This should be called before closing database connections to ensure
        all pending audit logs are written and resources are cleaned up.

        Args:
            wait: If True, wait for pending tasks to complete.
                  If False, cancel pending tasks immediately.
        """
        global _audit_executor

        logger.info(f"Shutting down SheetsAgent (wait={wait})")

        # Shutdown audit executor if exists
        if _audit_executor:
            logger.info("Shutting down audit executor")
            _audit_executor.shutdown(wait=wait, cancel_futures=not wait)
            logger.info("Audit executor shutdown complete")

        # Clean up DuckDB connection pool (sheets-specific)
        try:
            from .tools import get_duckdb_pool
            pool = get_duckdb_pool()
            pool.close_all()
            logger.info("DuckDB connection pool closed")
        except Exception as e:
            logger.warning(f"Failed to close DuckDB pool: {e}")

        # Cleanup base agent resources (sessions, rate limiter)
        self._cleanup_resources()

        # Clear file cache (sheets-specific)
        self.file_cache.clear()

        logger.info("SheetsAgent shutdown complete")
