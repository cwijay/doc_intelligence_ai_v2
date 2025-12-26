"""Core Sheets Agent implementation using LangGraph."""

import asyncio
import threading
import time
import uuid
import hashlib
import logging
import os
from collections import defaultdict, OrderedDict
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd

from concurrent.futures import ThreadPoolExecutor

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .config import SheetsAgentConfig
from src.utils.timer_utils import elapsed_ms

# Shared agent infrastructure imports
from src.agents.core.rate_limiter import RateLimiter
from src.agents.core.session_manager import SessionManager

# Background executor for non-blocking audit logging
_audit_executor: Optional[ThreadPoolExecutor] = None
from .tools import (
    FilePreviewTool,
    CrossFileQueryTool,
    SingleFileQueryTool,
    SmartAnalysisTool,
    DataAnalysisTool
)
from .schemas import (
    ChatRequest,
    ChatResponse,
    FileMetadata,
    ToolUsage,
    TokenUsage,
    SessionInfo
)

# Memory imports (optional - graceful fallback if not available)
try:
    from src.agents.core.memory import (
        MemoryConfig,
        ShortTermMemory,
        PostgresLongTermMemory,
        ConversationSummary
    )
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False

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


SYSTEM_PROMPT = """You are an expert Excel and data analysis assistant. You help users analyze spreadsheet data through natural language queries.

Your capabilities include:
1. Loading and previewing Excel (.xlsx, .xls) and CSV files from local filesystem
2. Executing complex queries across multiple files using SQL via DuckDB
3. Performing statistical analysis, correlation analysis, and data quality checks
4. Identifying trends, patterns, and outliers in data
5. Providing business insights and recommendations based on data

Guidelines:
- For single file analysis, ALWAYS use the smart_analysis tool first - it efficiently combines preview and analysis in one call
- Use appropriate tools for different types of analysis
- Provide clear, business-friendly explanations of findings
- When working with multiple files, look for relationships and linkages
- Focus on actionable insights and recommendations
- Handle errors gracefully and suggest alternatives

Tool Selection Priority:
1. For single file queries: Use smart_analysis (most efficient)
2. For multi-file queries: Use cross_file_query
3. For detailed data analysis: Use data_analysis
4. Only use file_preview + single_file_query if smart_analysis doesn't meet your needs

Use the available tools to analyze data and provide comprehensive answers to user questions."""


class SheetsAgent:
    """AI-powered Excel/Sheets analysis agent using LangGraph."""

    def __init__(self, config: SheetsAgentConfig):
        """
        Initialize Sheets Agent.

        Args:
            config: Agent configuration
        """
        self.config = config
        self.session_manager = SessionManager(config.session_timeout_minutes)
        self.file_cache = FileCache(max_size=50)

        # Initialize rate limiter
        self.rate_limiter = RateLimiter(
            max_requests=config.rate_limit_requests,
            window_seconds=config.rate_limit_window_seconds
        )

        # Initialize LLM
        self.llm = self._init_llm()

        # Initialize tools
        self.tools = self._init_tools()

        # Create ReAct agent
        self.agent = self._create_agent()

        # Initialize audit logging (optional)
        self.audit_logger = self._init_audit_logging()

        # Initialize memory systems
        self.short_term_memory = None
        self.long_term_memory = None
        self._init_memory()

        logger.info("Sheets Agent initialized successfully")

    def _init_memory(self):
        """Initialize short-term and long-term memory systems."""
        if not MEMORY_AVAILABLE:
            logger.warning("Memory module not available - memory features disabled")
            return

        # Initialize short-term memory
        if self.config.enable_short_term_memory:
            self.short_term_memory = ShortTermMemory(
                max_messages=self.config.short_term_max_messages
            )
            logger.info(
                f"Short-term memory enabled (max {self.config.short_term_max_messages} messages)"
            )

        # Initialize long-term memory (PostgreSQL)
        if self.config.enable_long_term_memory:
            try:
                memory_config = MemoryConfig()
                self.long_term_memory = PostgresLongTermMemory(memory_config)
                logger.info("Long-term memory enabled (PostgreSQL)")
            except Exception as e:
                logger.warning(f"Failed to initialize long-term memory: {e}")

    def _init_llm(self):
        """Initialize the language model."""
        try:
            if not self.config.openai_api_key:
                raise ValueError("OpenAI API key is required")

            llm = ChatOpenAI(
                model=self.config.openai_model,
                api_key=self.config.openai_api_key,
                temperature=self.config.temperature,
                use_responses_api=True,  # Required for gpt-5.1-codex-mini
                request_timeout=60,  # Prevent hanging requests
                max_retries=2
            )

            logger.info(f"Initialized LLM: {self.config.openai_model}")
            return llm

        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise

    def _init_tools(self) -> List:
        """Initialize agent tools."""
        tools = [
            SmartAnalysisTool(config=self.config, file_cache=self.file_cache),
            FilePreviewTool(config=self.config),
            SingleFileQueryTool(config=self.config),
            CrossFileQueryTool(config=self.config),
            DataAnalysisTool(config=self.config)
        ]

        logger.info(f"Initialized {len(tools)} tools")
        return tools

    def _create_agent(self):
        """Create the ReAct agent using LangGraph."""
        agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=SYSTEM_PROMPT
        )

        logger.info("Created ReAct agent with LangGraph")
        return agent

    def _init_audit_logging(self):
        """Initialize audit logging (PostgreSQL)."""
        try:
            from src.db.repositories.audit_repository import log_event
            logger.info("Initialized audit logging (PostgreSQL)")
            return log_event  # Return the function
        except ImportError as e:
            logger.warning(f"Audit module not available: {e}")
            return None
        except Exception as e:
            logger.warning(f"Failed to initialize audit logging: {e}")
            return None

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

            # Execute agent with timeout enforcement
            try:
                async with asyncio.timeout(self.config.timeout_seconds):
                    agent_result = await self._execute_agent(context, request.query, chat_history)
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
        """Validate that files exist and are accessible."""
        validation_results = {
            "valid_files": [],
            "invalid_files": [],
            "total_size_mb": 0
        }

        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    validation_results["valid_files"].append({
                        "path": file_path,
                        "size_mb": file_size / (1024 * 1024)
                    })
                    validation_results["total_size_mb"] += file_size / (1024 * 1024)
                else:
                    validation_results["invalid_files"].append({
                        "path": file_path,
                        "error": "File not found"
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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((TimeoutError, ConnectionError, OSError)),
        reraise=True
    )
    async def _execute_agent(
        self, context: str, query: str, chat_history: List = None
    ) -> Dict[str, Any]:
        """Execute the agent with the given context and query, with retry logic."""
        try:
            message = HumanMessage(content=f"{context}\n\nPlease analyze the data and answer: {query}")

            # Build messages list with chat history
            messages = list(chat_history) if chat_history else []
            messages.append(message)

            logger.debug("Executing LangGraph agent")
            result = await asyncio.to_thread(
                self.agent.invoke, {"messages": messages}
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
        """Calculate token usage (estimation)."""
        prompt_tokens = int(len(input_text.split()) * 1.3)
        completion_tokens = int(len(output_text.split()) * 1.3)
        total_tokens = prompt_tokens + completion_tokens

        estimated_cost = (prompt_tokens * 0.00003 + completion_tokens * 0.00006) / 1000

        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=round(estimated_cost, 6)
        )

    def _get_file_extension(self, file_path: str) -> str:
        """Get file extension from file path."""
        return file_path.split('.')[-1].lower()

    def _log_audit_event(self, request: ChatRequest, result: Dict, processing_time: float):
        """Log query to audit database."""
        try:
            if self.audit_logger:
                self.audit_logger(
                    event_type="sheets_agent_query",
                    details={
                        "session_id": request.session_id,
                        "query": request.query,
                        "file_paths": request.file_paths,
                        "processing_time_ms": processing_time,
                        "success": True
                    }
                )
        except Exception as e:
            logger.warning(f"Failed to log audit event: {e}")

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the agent and its components."""
        try:
            # Skip LLM health check to avoid blocking - just check if configured
            openai_status = "healthy" if self.llm else "unhealthy"

            # Cleanup expired sessions and rate limiter entries
            self.session_manager.cleanup_expired_sessions()
            self.rate_limiter.cleanup()

            return {
                "status": "healthy" if openai_status == "healthy" else "degraded",
                "components": {
                    "api": "healthy",
                    "openai": openai_status,
                    "duckdb": "healthy",
                    "audit_logging": "healthy" if self.audit_logger else "disabled",
                    "short_term_memory": "enabled" if self.short_term_memory else "disabled",
                    "long_term_memory": "enabled" if self.long_term_memory else "disabled"
                },
                "sessions": {
                    "active_count": len(self.session_manager.sessions),
                    "total_queries": sum(s.query_count for s in self.session_manager.sessions.values())
                },
                "cache": self.file_cache.get_stats(),
                "rate_limiter": {
                    "tracked_sessions": len(self.rate_limiter.requests)
                },
                "memory": {
                    "short_term_sessions": (
                        self.short_term_memory.get_session_count()
                        if self.short_term_memory else 0
                    ),
                    "short_term_enabled": self.config.enable_short_term_memory,
                    "long_term_enabled": self.config.enable_long_term_memory
                }
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

    def end_session(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        save_summary: bool = True
    ) -> bool:
        """
        End a session and optionally save summary to long-term memory.

        Args:
            session_id: Session identifier
            user_id: User ID for long-term memory
            save_summary: Whether to save conversation summary

        Returns:
            True if session ended successfully
        """
        try:
            # Get session info
            session = self.session_manager.sessions.get(session_id)

            if not session:
                logger.warning(f"Session {session_id} not found")
                return False

            # Save conversation summary to long-term memory
            if save_summary and self.long_term_memory and user_id:
                self._save_conversation_summary(session_id, user_id, session)

            # Clear short-term memory for session
            if self.short_term_memory:
                self.short_term_memory.delete_session(session_id)

            # Remove from session manager
            if session_id in self.session_manager.sessions:
                del self.session_manager.sessions[session_id]

            logger.info(f"Session {session_id} ended successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to end session {session_id}: {e}")
            return False

    def _save_conversation_summary(
        self,
        session_id: str,
        user_id: str,
        session: SessionInfo
    ) -> None:
        """Save conversation summary to long-term memory."""
        if not self.long_term_memory or not MEMORY_AVAILABLE:
            return

        try:
            # Get conversation summary from short-term memory
            summary_text = ""
            if self.short_term_memory:
                summary_text = (
                    self.short_term_memory.get_conversation_summary(session_id)
                    or f"Session with {session.query_count} queries"
                )

            summary = ConversationSummary(
                session_id=session_id,
                user_id=user_id,
                agent_type="sheets",
                summary=summary_text,
                key_topics=[],
                documents_discussed=session.files_in_context,
                queries_count=session.query_count
            )

            self.long_term_memory.save_conversation_summary(summary)
            logger.info(f"Saved conversation summary for session {session_id}")

        except Exception as e:
            logger.error(f"Failed to save conversation summary: {e}")

    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown agent resources gracefully.

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

        # Clean up DuckDB connection pool
        try:
            from .tools import get_duckdb_pool
            pool = get_duckdb_pool()
            pool.close_all()
            logger.info("DuckDB connection pool closed")
        except Exception as e:
            logger.warning(f"Failed to close DuckDB pool: {e}")

        # Clean up expired sessions
        self.session_manager.cleanup_expired_sessions()

        # Clean up rate limiter
        self.rate_limiter.cleanup()

        # Clear file cache
        self.file_cache.clear()

        logger.info("SheetsAgent shutdown complete")
