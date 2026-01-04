"""Core Extractor Agent implementation for structured data extraction.

Uses LangChain structured output to extract data from documents
based on user-defined schemas.
"""

import asyncio
import functools
import json
import logging
import time
import uuid
from typing import Dict, List, Any, Optional

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from .config import ExtractorAgentConfig
from .tools import create_extractor_tools, load_schema_from_gcs, list_schemas_from_gcs, invalidate_schema_cache
from .schemas import (
    ExtractionRequest,
    ExtractionResponse,
    AnalyzeFieldsRequest,
    AnalyzeFieldsResponse,
    GenerateSchemaRequest,
    GenerateSchemaResponse,
    ExtractDataRequest,
    ExtractDataResponse,
    DiscoveredField,
    TokenUsage
)
from src.agents.core.base_agent import BaseAgent
from src.agents.core.token_utils import calculate_token_usage
from src.utils.timer_utils import elapsed_ms

# Try to import executor pools
try:
    from src.core.executors import get_executors
    EXECUTORS_AVAILABLE = True
except ImportError:
    EXECUTORS_AVAILABLE = False

# Try to import usage context for automatic token tracking
try:
    from src.core.usage import usage_context
    USAGE_CONTEXT_AVAILABLE = True
except ImportError:
    USAGE_CONTEXT_AVAILABLE = False
    usage_context = None

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a document extraction assistant that helps users extract structured data from documents.

You have three specialized tools:

1. **field_analyzer**: Analyze a document to discover all extractable fields
   - Use when user wants to analyze a document's structure
   - Returns field names, types, sample values, and locations (header/line_item/footer)

2. **schema_generator**: Generate a JSON schema from selected fields
   - Use when user has selected fields and wants to create an extraction template
   - Saves schema to GCS for reuse

3. **data_extractor**: Extract data from a document using a schema
   - Use when user has a schema (generated or from template) and wants to extract data
   - Returns structured data matching the schema

## Workflow:
1. First, analyze the document to discover fields
2. Present fields to user for selection
3. Generate schema from selected fields (optionally save as template)
4. Extract data using the schema

## Guidelines:
- Be thorough when analyzing documents - identify all possible fields
- For invoices, look for: vendor info, customer info, dates, line items, totals, taxes
- For contracts, look for: parties, dates, terms, signatures, amounts
- For receipts, look for: vendor, items, amounts, payment method, date
- Always include confidence scores to help users understand extraction reliability
"""


class ExtractorAgent(BaseAgent):
    """AI-powered document extraction agent using LangChain.

    Provides three main capabilities:
    1. Field Analysis - Discover extractable fields from documents
    2. Schema Generation - Create and save extraction templates
    3. Data Extraction - Extract structured data using schemas

    Inherits from BaseAgent for shared functionality:
    - Session management
    - Rate limiting
    - Memory (short-term and long-term)
    - Audit logging
    """

    def __init__(self, config: Optional[ExtractorAgentConfig] = None):
        """Initialize Extractor Agent.

        Args:
            config: Agent configuration. Uses defaults if not provided.
        """
        config = config or ExtractorAgentConfig()
        super().__init__(config)

        # Initialize LLM
        self.llm = self._init_llm()

        # Initialize tools
        self.tools = self._init_tools()
        self.tools_by_name = {tool.name: tool for tool in self.tools}

        # Initialize checkpointer for conversation memory
        self.checkpointer = MemorySaver()

        # Create agent (simplified - we'll invoke tools directly for extraction)
        self.agent = self._create_agent()

        logger.info(f"Extractor Agent initialized with model: {self.config.openai_model}")

    def _init_llm(self):
        """Initialize the language model."""
        if not self.config.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required."
            )

        callbacks = self._create_token_tracking_callback("extractor_agent")

        llm = init_chat_model(
            model=self.config.openai_model,
            model_provider="openai",
            temperature=self.config.temperature,
            api_key=self.config.openai_api_key,
            timeout=self.config.extraction_timeout_seconds,
            max_retries=2,
            callbacks=callbacks if callbacks else None,
        )

        logger.info(f"Initialized LLM: {self.config.openai_model}")
        return llm

    def _init_tools(self) -> List:
        """Initialize agent tools."""
        return create_extractor_tools(self.config)

    def _get_agent_type(self) -> str:
        """Get agent type identifier for audit/memory."""
        return "extractor"

    def _create_agent(self):
        """Create the agent (simplified for extraction use case)."""
        # For extraction, we invoke tools directly rather than through
        # a full agent loop, so this is mainly for compatibility
        from langchain.agents import create_agent

        agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=SYSTEM_PROMPT,
            checkpointer=self.checkpointer
        )

        logger.info("Created Extractor Agent")
        return agent

    # ==========================================================================
    # Main API Methods
    # ==========================================================================

    async def analyze_fields(
        self,
        content: str,
        document_name: str,
        document_type_hint: Optional[str] = None,
        organization_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> AnalyzeFieldsResponse:
        """Analyze a document to discover extractable fields.

        Args:
            content: Parsed document content (markdown)
            document_name: Name of the document
            document_type_hint: Optional hint about document type
            organization_id: Organization ID for multi-tenancy
            session_id: Optional session ID for tracking

        Returns:
            AnalyzeFieldsResponse with discovered fields
        """
        start_time = time.time()

        try:
            # Get or create session
            session = self.session_manager.get_or_create_session(session_id)

            # Check rate limit
            if not self.rate_limiter.is_allowed(session.session_id):
                retry_after = self.rate_limiter.get_retry_after(session.session_id)
                return AnalyzeFieldsResponse(
                    success=False,
                    document_name=document_name,
                    error=f"Rate limit exceeded. Try again in {retry_after} seconds.",
                    processing_time_ms=elapsed_ms(start_time)
                )

            # Get the field analyzer tool
            tool = self.tools_by_name.get('field_analyzer')
            if not tool:
                return AnalyzeFieldsResponse(
                    success=False,
                    document_name=document_name,
                    error="Field analyzer tool not available",
                    processing_time_ms=elapsed_ms(start_time)
                )

            # Set up usage context for token tracking (non-blocking via callback handler)
            ctx_manager = None
            if USAGE_CONTEXT_AVAILABLE and usage_context and organization_id:
                ctx_manager = usage_context(
                    org_id=organization_id,
                    feature="extractor_agent",
                    session_id=session.session_id,
                )
                ctx_manager.__enter__()

            try:
                # Execute tool in thread pool
                result_json = await self._run_tool_async(
                    tool,
                    content=content,
                    document_name=document_name,
                    document_type_hint=document_type_hint,
                    organization_id=organization_id
                )
            finally:
                if ctx_manager:
                    ctx_manager.__exit__(None, None, None)

            result = json.loads(result_json)
            processing_time = elapsed_ms(start_time)

            # Calculate token usage for this analysis
            token_usage = self._calculate_token_usage(content, result_json)

            if result.get("success"):
                # Convert to DiscoveredField objects
                fields = [
                    DiscoveredField(**f) for f in result.get("fields", [])
                ]
                line_item_fields = None
                if result.get("line_item_fields"):
                    line_item_fields = [
                        DiscoveredField(**f) for f in result["line_item_fields"]
                    ]

                header_count = len(fields) if fields else 0
                line_item_count = len(line_item_fields) if line_item_fields else 0

                # Log enhanced audit event
                self._log_audit_event_async(
                    "field_analysis",
                    document_name,
                    organization_id,
                    {
                        "fields_discovered": result.get("total_fields", 0),
                        "header_fields": header_count,
                        "line_item_fields": line_item_count,
                        "document_type": result.get("document_type"),
                        "has_line_items": result.get("has_line_items", False),
                        "processing_time_ms": processing_time,
                        "parallel_analysis": result.get("parallel_analysis", False),
                        "session_id": session.session_id,
                        "total_tokens": token_usage.total_tokens,
                        "estimated_cost_usd": token_usage.estimated_cost_usd
                    }
                )

                return AnalyzeFieldsResponse(
                    success=True,
                    document_name=document_name,
                    document_type=result.get("document_type", "unknown"),
                    fields=fields,
                    has_line_items=result.get("has_line_items", False),
                    line_item_fields=line_item_fields,
                    processing_time_ms=processing_time,
                    session_id=session.session_id,
                    token_usage=token_usage
                )
            else:
                # Log failure event
                self._log_audit_event_async(
                    "field_analysis_failed",
                    document_name,
                    organization_id,
                    {
                        "error": result.get("error", "Unknown error"),
                        "processing_time_ms": processing_time
                    }
                )
                return AnalyzeFieldsResponse(
                    success=False,
                    document_name=document_name,
                    error=result.get("error", "Unknown error"),
                    processing_time_ms=processing_time
                )

        except Exception as e:
            logger.error(f"Field analysis failed: {e}", exc_info=True)
            # Log failure event
            self._log_audit_event_async(
                "field_analysis_failed",
                document_name,
                organization_id,
                {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "processing_time_ms": elapsed_ms(start_time)
                }
            )
            return AnalyzeFieldsResponse(
                success=False,
                document_name=document_name,
                error=str(e),
                processing_time_ms=elapsed_ms(start_time)
            )

    async def generate_schema(
        self,
        selected_fields: List[Dict[str, Any]],
        template_name: str,
        document_type: str,
        organization_id: str,
        folder_name: Optional[str] = None,
        save_to_gcs: bool = True,
        session_id: Optional[str] = None
    ) -> GenerateSchemaResponse:
        """Generate a JSON schema from selected fields.

        Args:
            selected_fields: List of field definitions to include
            template_name: Name for the schema template
            document_type: Type of document (invoice, contract, etc.)
            organization_id: Organization ID for scoping
            save_to_gcs: Whether to save schema to GCS
            session_id: Optional session ID for tracking

        Returns:
            GenerateSchemaResponse with generated schema
        """
        start_time = time.time()

        try:
            session = self.session_manager.get_or_create_session(session_id)

            if not self.rate_limiter.is_allowed(session.session_id):
                retry_after = self.rate_limiter.get_retry_after(session.session_id)
                return GenerateSchemaResponse(
                    success=False,
                    template_name=template_name,
                    error=f"Rate limit exceeded. Try again in {retry_after} seconds.",
                    processing_time_ms=elapsed_ms(start_time)
                )

            tool = self.tools_by_name.get('schema_generator')
            if not tool:
                return GenerateSchemaResponse(
                    success=False,
                    template_name=template_name,
                    error="Schema generator tool not available",
                    processing_time_ms=elapsed_ms(start_time)
                )

            # Set up usage context for token tracking (non-blocking via callback handler)
            ctx_manager = None
            if USAGE_CONTEXT_AVAILABLE and usage_context and organization_id:
                ctx_manager = usage_context(
                    org_id=organization_id,
                    feature="extractor_agent",
                    session_id=session.session_id,
                )
                ctx_manager.__enter__()

            try:
                result_json = await self._run_tool_async(
                    tool,
                    selected_fields=selected_fields,
                    template_name=template_name,
                    document_type=document_type,
                    organization_id=organization_id,
                    save_to_gcs=save_to_gcs
                )
            finally:
                if ctx_manager:
                    ctx_manager.__exit__(None, None, None)

            result = json.loads(result_json)

            # Calculate token usage for schema generation
            input_text = json.dumps(selected_fields)
            token_usage = self._calculate_token_usage(input_text, result_json)

            if not result.get("success"):
                return GenerateSchemaResponse(
                    success=False,
                    template_name=template_name,
                    error=result.get("error", "Unknown error"),
                    processing_time_ms=elapsed_ms(start_time),
                    token_usage=token_usage
                )

            # Save to GCS from async context (this works correctly)
            gcs_uri = None
            if save_to_gcs and result.get("schema"):
                try:
                    from src.storage.config import get_storage
                    from src.db.repositories.extraction_repository import get_organization_name

                    storage = get_storage()
                    schema_content = json.dumps(result["schema"], indent=2, ensure_ascii=False)

                    # Resolve org_id to org_name for path construction
                    org_name = await get_organization_name(organization_id)
                    if not org_name:
                        logger.warning(f"Could not resolve org_name for {organization_id}, using org_id as fallback")
                        org_name = organization_id

                    # Build GCS path: {org_name}/schema/{folder_name}/{template_name}.json
                    safe_name = template_name.strip().replace(' ', '_').lower()
                    filename = f"{safe_name}.json"
                    if folder_name:
                        directory = f"{org_name}/schema/{folder_name}"
                    else:
                        directory = f"{org_name}/schema"

                    gcs_uri = await storage.save(
                        content=schema_content,
                        filename=filename,
                        directory=directory,
                        use_prefix=False  # Use org path directly, no demo_docs prefix
                    )
                    logger.info(f"Schema saved to GCS: {gcs_uri}")

                    # Invalidate cache for this template
                    invalidate_schema_cache(organization_id, template_name)

                except Exception as e:
                    logger.error(f"Failed to save schema to GCS: {e}")
                    return GenerateSchemaResponse(
                        success=False,
                        template_name=template_name,
                        error=f"Schema generated but failed to save to GCS: {e}",
                        processing_time_ms=elapsed_ms(start_time)
                    )

            processing_time = elapsed_ms(start_time)

            # Log enhanced audit event
            self._log_audit_event_async(
                "schema_generated",
                template_name,
                organization_id,
                {
                    "document_type": document_type,
                    "header_fields_count": result.get("header_fields_count", 0),
                    "line_item_fields_count": result.get("line_item_fields_count", 0),
                    "total_field_count": result.get("header_fields_count", 0) + result.get("line_item_fields_count", 0),
                    "saved_to_gcs": bool(gcs_uri),
                    "gcs_uri": gcs_uri,
                    "processing_time_ms": processing_time,
                    "session_id": session.session_id,
                    "total_tokens": token_usage.total_tokens,
                    "estimated_cost_usd": token_usage.estimated_cost_usd
                }
            )

            return GenerateSchemaResponse(
                success=True,
                template_name=result.get("template_name"),
                document_type=result.get("document_type"),
                schema_definition=result.get("schema"),
                gcs_uri=gcs_uri,
                processing_time_ms=processing_time,
                session_id=session.session_id,
                token_usage=token_usage
            )

        except Exception as e:
            logger.error(f"Schema generation failed: {e}", exc_info=True)
            # Log failure event
            self._log_audit_event_async(
                "schema_generation_failed",
                template_name,
                organization_id,
                {
                    "document_type": document_type,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "processing_time_ms": elapsed_ms(start_time)
                }
            )
            return GenerateSchemaResponse(
                success=False,
                template_name=template_name,
                error=str(e),
                processing_time_ms=elapsed_ms(start_time)
            )

    async def extract_data(
        self,
        content: str,
        schema: Dict[str, Any],
        document_name: str,
        organization_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> ExtractDataResponse:
        """Extract structured data from a document using a schema.

        Args:
            content: Parsed document content
            schema: JSON schema for extraction
            document_name: Name of the document
            organization_id: Organization ID for multi-tenancy
            session_id: Optional session ID for tracking

        Returns:
            ExtractDataResponse with extracted data
        """
        start_time = time.time()

        try:
            session = self.session_manager.get_or_create_session(session_id)

            if not self.rate_limiter.is_allowed(session.session_id):
                retry_after = self.rate_limiter.get_retry_after(session.session_id)
                return ExtractDataResponse(
                    success=False,
                    document_name=document_name,
                    error=f"Rate limit exceeded. Try again in {retry_after} seconds.",
                    processing_time_ms=elapsed_ms(start_time)
                )

            tool = self.tools_by_name.get('data_extractor')
            if not tool:
                return ExtractDataResponse(
                    success=False,
                    document_name=document_name,
                    error="Data extractor tool not available",
                    processing_time_ms=elapsed_ms(start_time)
                )

            result_json = await self._run_tool_async(
                tool,
                content=content,
                schema_definition=schema,
                document_name=document_name,
                organization_id=organization_id
            )

            result = json.loads(result_json)
            processing_time = elapsed_ms(start_time)

            if result.get("success"):
                # Calculate token usage
                token_usage = self._calculate_token_usage(content, str(result.get("extracted_data", {})))

                # Generate extraction job ID
                extraction_job_id = str(uuid.uuid4())

                # Token usage is tracked via @track_tokens decorator on API endpoint

                # Log enhanced audit event
                self._log_audit_event_async(
                    "data_extracted",
                    document_name,
                    organization_id,
                    {
                        "extraction_job_id": extraction_job_id,
                        "schema_title": result.get("schema_title"),
                        "fields_extracted": result.get("extracted_field_count", 0),
                        "prompt_tokens": token_usage.prompt_tokens,
                        "completion_tokens": token_usage.completion_tokens,
                        "total_tokens": token_usage.total_tokens,
                        "estimated_cost_usd": token_usage.estimated_cost_usd,
                        "processing_time_ms": processing_time,
                        "session_id": session.session_id
                    }
                )

                return ExtractDataResponse(
                    success=True,
                    extraction_job_id=extraction_job_id,
                    document_name=document_name,
                    schema_title=result.get("schema_title"),
                    extracted_data=result.get("extracted_data"),
                    extracted_field_count=result.get("extracted_field_count", 0),
                    token_usage=token_usage,
                    processing_time_ms=processing_time,
                    session_id=session.session_id
                )
            else:
                # Log failure event
                self._log_audit_event_async(
                    "data_extraction_failed",
                    document_name,
                    organization_id,
                    {
                        "error": result.get("error", "Unknown error"),
                        "processing_time_ms": processing_time
                    }
                )
                return ExtractDataResponse(
                    success=False,
                    document_name=document_name,
                    error=result.get("error", "Unknown error"),
                    processing_time_ms=processing_time
                )

        except Exception as e:
            logger.error(f"Data extraction failed: {e}", exc_info=True)
            # Log failure event
            self._log_audit_event_async(
                "data_extraction_failed",
                document_name,
                organization_id,
                {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "processing_time_ms": elapsed_ms(start_time)
                }
            )
            return ExtractDataResponse(
                success=False,
                document_name=document_name,
                error=str(e),
                processing_time_ms=elapsed_ms(start_time)
            )

    # ==========================================================================
    # Template Management
    # ==========================================================================

    async def list_templates(self, organization_id: str) -> List[Dict[str, Any]]:
        """List all extraction templates for an organization.

        Args:
            organization_id: Organization ID

        Returns:
            List of template metadata
        """
        return await list_schemas_from_gcs(organization_id)

    async def get_template(
        self,
        organization_id: str,
        template_name: str,
        folder_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get a specific template by name.

        Args:
            organization_id: Organization ID (UUID)
            template_name: Template name
            folder_name: Optional folder name where schema is stored

        Returns:
            Schema dict if found, None otherwise
        """
        return await load_schema_from_gcs(organization_id, template_name, folder_name)

    # ==========================================================================
    # Helper Methods
    # ==========================================================================

    async def _run_tool_async(self, tool, **kwargs) -> str:
        """Run a tool asynchronously in the executor pool.

        Args:
            tool: The tool to run
            **kwargs: Tool arguments

        Returns:
            Tool result as JSON string
        """
        if EXECUTORS_AVAILABLE:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                get_executors().agent_executor,
                functools.partial(tool._run, **kwargs)
            )
        else:
            # Fallback to running in current thread
            return tool._run(**kwargs)

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

    def _log_audit_event_async(
        self,
        event_type: str,
        file_name: str,
        organization_id: Optional[str],
        details: Dict[str, Any]
    ):
        """Log audit event asynchronously."""
        if not self.audit_logger:
            return

        try:
            from src.agents.core.audit_queue import enqueue_audit_event

            enqueue_audit_event(
                event_type=f"extractor_{event_type}",
                file_name=file_name,
                organization_id=organization_id,
                details=details
            )
        except Exception as e:
            logger.warning(f"Failed to enqueue audit event: {e}")

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the agent."""
        try:
            base_status = self._get_base_health_status()
            llm_status = "healthy" if self.llm else "unhealthy"

            return {
                "status": "healthy" if llm_status == "healthy" else "degraded",
                "components": {
                    "llm": llm_status,
                    "model": self.config.openai_model,
                    "tools": [t.name for t in self.tools],
                    "audit_logging": base_status["audit_logging"],
                },
                "sessions": base_status["sessions"],
                "rate_limiter": base_status["rate_limiter"],
                "memory": base_status["memory"],
                "config": {
                    "max_fields_to_analyze": self.config.max_fields_to_analyze,
                    "extraction_timeout_seconds": self.config.extraction_timeout_seconds,
                }
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown agent resources gracefully."""
        logger.info(f"Shutting down ExtractorAgent (wait={wait})")
        self._cleanup_resources()
        logger.info("ExtractorAgent shutdown complete")
