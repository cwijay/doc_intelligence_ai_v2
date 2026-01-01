"""Field analyzer tool using LLM structured output.

Analyzes document content to discover all extractable fields.
Uses parallel analysis for header fields and line item fields for faster processing.
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional, Type, Tuple

from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

from ..config import ExtractorAgentConfig
from .base import (
    FieldAnalyzerInput,
    FieldAnalysisResult,
    HeaderFieldsResult,
    LineItemFieldsResult,
    DiscoveredField,
    truncate_content,
    compute_content_hash
)
from src.utils.timer_utils import elapsed_ms

logger = logging.getLogger(__name__)

# Enable/disable parallel analysis
ENABLE_PARALLEL_ANALYSIS = True


class FieldAnalyzerTool(BaseTool):
    """Tool to analyze document content and discover extractable fields.

    Uses LLM with structured output to identify all fields that can be
    extracted from a document (invoice, contract, receipt, etc.).

    Supports parallel analysis mode where header and line item analysis
    run concurrently for ~40% faster processing.
    """

    name: str = "field_analyzer"
    description: str = """Analyze document content to discover all extractable fields.
    Takes parsed document content and optional document type hint.
    Returns a list of discovered fields with their names, types, sample values, and locations."""
    args_schema: Type[BaseModel] = FieldAnalyzerInput

    config: ExtractorAgentConfig = Field(default_factory=ExtractorAgentConfig)
    llm: Optional[Any] = None

    def _get_llm(self):
        """Get or create LLM instance (OpenAI)."""
        if self.llm is None:
            self.llm = init_chat_model(
                model=self.config.openai_model,
                model_provider="openai",
                temperature=self.config.temperature,
                api_key=self.config.openai_api_key,
            )
            logger.info(f"FieldAnalyzer using OpenAI: {self.config.openai_model}")
        return self.llm

    def _build_header_prompt(self, content: str, type_hint: str) -> str:
        """Build prompt for header/footer field analysis."""
        return f"""Analyze this document and identify HEADER and FOOTER fields only.

{type_hint}

For each field, provide:
- field_name: Machine-readable name in snake_case (e.g., invoice_number, total_amount)
- display_name: Human-readable label (e.g., "Invoice Number", "Total Amount")
- data_type: One of: string, number, date, currency, boolean
- sample_value: An example value extracted from the document (if visible)
- confidence: Extraction confidence from 0.0 to 1.0
- location: "header", "footer", or "body" (NOT "line_item")
- required: Whether the field appears mandatory (true/false)

Focus on:
- Document identifiers (invoice number, PO number, reference ID)
- Dates (invoice date, due date, delivery date)
- Parties (vendor name, customer name, addresses, phones)
- Summary amounts (total, subtotal, tax, discount)
- Payment terms, notes, signatures

Also determine:
- Document type (invoice, contract, receipt, purchase_order, etc.)
- Whether the document has line items (a table of products/services)

Document content:
{content}
"""

    def _build_line_item_prompt(self, content: str, type_hint: str) -> str:
        """Build prompt for line item field analysis."""
        return f"""Analyze this document and identify LINE ITEM fields only.

{type_hint}

Line items are repeated rows in a table (products, services, items purchased).

For each LINE ITEM field, provide:
- field_name: Machine-readable name in snake_case (e.g., line_description, line_quantity)
- display_name: Human-readable label (e.g., "Description", "Quantity")
- data_type: One of: string, number, date, currency, boolean
- sample_value: An example value from ONE line item
- confidence: Extraction confidence from 0.0 to 1.0
- location: MUST be "line_item"
- required: Whether the field appears in every line item

Common line item fields:
- Description/name of item
- Quantity
- Unit of measure
- Unit price
- Line total/amount
- Item code/SKU
- Tax per line

If no line items exist, return an empty fields list.

Document content:
{content}
"""

    def _analyze_headers(self, llm, content: str, type_hint: str) -> HeaderFieldsResult:
        """Analyze header/footer fields (parallel task 1)."""
        prompt = self._build_header_prompt(content, type_hint)
        structured_llm = llm.with_structured_output(HeaderFieldsResult)
        return structured_llm.invoke(prompt)

    def _analyze_line_items(self, llm, content: str, type_hint: str) -> LineItemFieldsResult:
        """Analyze line item fields (parallel task 2)."""
        prompt = self._build_line_item_prompt(content, type_hint)
        structured_llm = llm.with_structured_output(LineItemFieldsResult)
        return structured_llm.invoke(prompt)

    def _run_parallel(
        self,
        content: str,
        type_hint: str
    ) -> Tuple[HeaderFieldsResult, LineItemFieldsResult]:
        """Run header and line item analysis in parallel."""
        llm = self._get_llm()

        with ThreadPoolExecutor(max_workers=2) as executor:
            header_future = executor.submit(
                self._analyze_headers, llm, content, type_hint
            )
            line_item_future = executor.submit(
                self._analyze_line_items, llm, content, type_hint
            )

            header_result = header_future.result()
            line_item_result = line_item_future.result()

        return header_result, line_item_result

    def _run_sequential(
        self,
        content: str,
        type_hint: str
    ) -> Tuple[HeaderFieldsResult, LineItemFieldsResult]:
        """Run analysis sequentially (fallback mode)."""
        llm = self._get_llm()
        header_result = self._analyze_headers(llm, content, type_hint)
        line_item_result = self._analyze_line_items(llm, content, type_hint)
        return header_result, line_item_result

    def _run(
        self,
        content: str,
        document_name: str = "",
        document_type_hint: Optional[str] = None,
        organization_id: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Analyze document content to discover extractable fields.

        Uses parallel analysis for header and line item fields for faster processing.
        """
        start_time = time.time()

        # Truncate content if too long
        truncated_content = truncate_content(content, max_chars=12000)
        content_hash = compute_content_hash(content)

        # Build type hint
        type_hint = f"Document type hint: {document_type_hint}" if document_type_hint else "Document type: auto-detect"

        try:
            # Run parallel or sequential analysis
            if ENABLE_PARALLEL_ANALYSIS:
                logger.debug("Running parallel field analysis")
                header_result, line_item_result = self._run_parallel(
                    truncated_content, type_hint
                )
            else:
                logger.debug("Running sequential field analysis")
                header_result, line_item_result = self._run_sequential(
                    truncated_content, type_hint
                )

            duration_ms = elapsed_ms(start_time)

            # Handle None results from structured output
            if header_result is None:
                logger.warning("Header analysis returned None, using empty result")
                header_fields = []
                document_type = "unknown"
                header_has_line_items = False
            else:
                header_fields = header_result.model_dump().get("fields", [])
                document_type = header_result.document_type or "unknown"
                header_has_line_items = header_result.has_line_items

            if line_item_result is None:
                logger.warning("Line item analysis returned None, using empty result")
                line_item_fields = []
            else:
                line_item_fields = line_item_result.model_dump().get("fields", [])

            # Ensure line item fields have correct location
            for field in line_item_fields:
                field["location"] = "line_item"

            total_fields = len(header_fields) + len(line_item_fields)
            has_line_items = header_has_line_items or len(line_item_fields) > 0

            logger.info(
                f"Field analysis complete: {total_fields} fields "
                f"({len(header_fields)} header, {len(line_item_fields)} line item) "
                f"in {duration_ms:.1f}ms"
                f"{' [parallel]' if ENABLE_PARALLEL_ANALYSIS else ''}"
            )

            return json.dumps({
                "success": True,
                "document_name": document_name,
                "document_type": document_type,
                "fields": header_fields,
                "has_line_items": has_line_items,
                "line_item_fields": line_item_fields if line_item_fields else None,
                "total_fields": total_fields,
                "processing_time_ms": duration_ms,
                "content_hash": content_hash,
                "parallel_analysis": ENABLE_PARALLEL_ANALYSIS
            }, default=str)

        except Exception as e:
            logger.error(f"Field analysis failed: {e}")
            return json.dumps({
                "success": False,
                "error": str(e),
                "processing_time_ms": elapsed_ms(start_time)
            })
