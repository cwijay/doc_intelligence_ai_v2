"""Data extractor tool using LLM structured output.

Extracts structured data from documents using a provided JSON schema.
"""

import json
import logging
import time
from typing import Any, Optional, Type, Dict

from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

from ..config import ExtractorAgentConfig
from .base import (
    DataExtractorInput,
    schema_to_pydantic,
    truncate_content,
    compute_content_hash,
    format_extraction_result
)
from src.utils.timer_utils import elapsed_ms

logger = logging.getLogger(__name__)


class DataExtractorTool(BaseTool):
    """Tool to extract structured data from documents using a schema.

    Uses LLM with structured output (dynamic Pydantic models) to extract
    data according to the provided JSON schema.
    """

    name: str = "data_extractor"
    description: str = """Extract structured data from document content using a provided schema.
    Takes document content and JSON schema.
    Returns extracted data matching the schema structure."""
    args_schema: Type[BaseModel] = DataExtractorInput

    config: ExtractorAgentConfig = Field(default_factory=ExtractorAgentConfig)
    llm: Optional[Any] = None
    fallback_llm: Optional[Any] = None

    def _get_llm(self):
        """Get or create primary LLM instance (gpt-5-nano)."""
        if self.llm is None:
            self.llm = init_chat_model(
                model=self.config.openai_model,
                model_provider="openai",
                temperature=self.config.temperature,
                api_key=self.config.openai_api_key,
            )
            logger.info(f"Using primary LLM: {self.config.openai_model}")
        return self.llm

    def _get_fallback_llm(self):
        """Get or create fallback LLM instance (gpt-4o-mini)."""
        if self.fallback_llm is None:
            self.fallback_llm = init_chat_model(
                model=self.config.openai_fallback_model,
                model_provider="openai",
                temperature=self.config.temperature,
                api_key=self.config.openai_api_key,
            )
            logger.info(f"Using fallback LLM: {self.config.openai_fallback_model}")
        return self.fallback_llm

    def _run(
        self,
        content: str,
        schema_definition: Dict[str, Any],
        document_name: str = "",
        organization_id: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Extract structured data from document using schema."""
        start_time = time.time()
        schema = schema_definition  # Use local alias for cleaner code

        try:
            # Validate schema
            if not schema or not schema.get("properties"):
                return json.dumps({
                    "success": False,
                    "error": "Invalid schema: must have 'properties' defined"
                })

            # Truncate content if too long
            truncated_content = truncate_content(content, max_chars=15000)
            content_hash = compute_content_hash(content)

            # Debug logging for content
            if len(content) > len(truncated_content):
                logger.info(f"Content truncated: {len(content)} -> {len(truncated_content)} chars")
            else:
                logger.info(f"Content ready for extraction: {len(content)} chars")
            if len(truncated_content.strip()) < 100:
                logger.warning(f"Content may be too short: '{truncated_content[:200]}...'")

            # Get schema metadata
            schema_title = schema.get("title", "Extraction Schema")
            schema_properties = schema.get("properties", {})

            # Build extraction prompt
            prompt = self._build_extraction_prompt(
                content=truncated_content,
                schema=schema
            )

            # Create dynamic Pydantic model from schema
            try:
                extraction_model = schema_to_pydantic(schema, "DynamicExtraction")
            except Exception as e:
                logger.error(f"Failed to create dynamic model from schema: {e}")
                return json.dumps({
                    "success": False,
                    "error": f"Invalid schema structure: {str(e)}"
                })

            # Get LLM and invoke with structured output
            llm = self._get_llm()
            # Use include_raw=True to access raw response even when parsing fails
            structured_llm = llm.with_structured_output(extraction_model, include_raw=True)

            try:
                response = structured_llm.invoke(prompt)

                # Debug logging for response structure
                logger.debug(f"Response keys: {list(response.keys()) if response else 'None'}")
                logger.debug(f"Parsed result: {response.get('parsed')}")
                if response.get('parsing_error'):
                    logger.debug(f"Parsing error: {response.get('parsing_error')}")

                # Response is dict with 'raw', 'parsed', and 'parsing_error' keys
                if response.get('parsed'):
                    # Successful parsing - use the Pydantic model
                    extracted_data = response['parsed'].model_dump()
                else:
                    # Parsing failed - extract from raw response
                    raw_message = response.get('raw')
                    extracted_data = self._extract_from_raw_response(raw_message, schema)
                    # If raw extraction failed, try fallback model
                    if extracted_data is None:
                        logger.info("Raw extraction failed, trying fallback extraction")
                        fallback_llm = self._get_fallback_llm()
                        extracted_data = self._fallback_extraction(fallback_llm, prompt, schema)

            except Exception as e:
                logger.warning(f"Structured extraction failed, trying fallback: {e}")
                # Fallback: try regular extraction with fallback model
                fallback_llm = self._get_fallback_llm()
                extracted_data = self._fallback_extraction(fallback_llm, prompt, schema)

            duration_ms = elapsed_ms(start_time)

            # Count extracted fields
            extracted_count = self._count_extracted_fields(extracted_data)

            logger.info(
                f"Data extraction complete: {extracted_count} fields "
                f"from {document_name} in {duration_ms:.1f}ms"
            )

            return json.dumps({
                "success": True,
                "document_name": document_name,
                "schema_title": schema_title,
                "extracted_data": extracted_data,
                "extracted_field_count": extracted_count,
                "processing_time_ms": duration_ms,
                "content_hash": content_hash
            }, default=str)

        except Exception as e:
            logger.error(f"Data extraction failed: {e}")
            return json.dumps({
                "success": False,
                "error": str(e),
                "processing_time_ms": elapsed_ms(start_time)
            })

    def _build_extraction_prompt(self, content: str, schema: Dict[str, Any]) -> str:
        """Build the extraction prompt with schema context."""
        schema_description = self._describe_schema(schema)

        return f"""Extract structured data from this document according to the schema below.

SCHEMA:
{schema_description}

INSTRUCTIONS:
1. Extract ALL fields defined in the schema from the document
2. For fields not found in the document, set them to null
3. For line_items (array fields), extract each row/item as a separate object
4. Use the exact field names from the schema
5. Convert values to the appropriate types (numbers, dates as ISO strings, etc.)
6. For currency/amount fields, extract just the numeric value
7. Be precise and accurate - extract only what is explicitly stated

DOCUMENT CONTENT:
{content}

Extract the data according to the schema:"""

    def _describe_schema(self, schema: Dict[str, Any]) -> str:
        """Generate a human-readable description of the schema."""
        lines = []
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        for name, prop in properties.items():
            prop_type = prop.get("type", "string")
            description = prop.get("description", name)
            req_marker = "(required)" if name in required else "(optional)"

            if prop_type == "array":
                items = prop.get("items", {})
                if items.get("type") == "object":
                    item_props = items.get("properties", {})
                    item_desc = ", ".join(item_props.keys())
                    lines.append(f"- {name}: array of objects with: {item_desc} {req_marker}")
                else:
                    lines.append(f"- {name}: array of {items.get('type', 'string')} {req_marker}")
            else:
                lines.append(f"- {name}: {prop_type} - {description} {req_marker}")

        return "\n".join(lines)

    def _extract_from_raw_response(
        self,
        raw_message,
        schema: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Extract data from raw LLM response when structured parsing fails.

        Handles the 'parameters' wrapper that some LLMs return in tool calls.
        Returns None if extraction fails, so caller can try fallback.
        """
        # Debug logging for raw message structure
        logger.debug(f"Raw message type: {type(raw_message)}")
        if hasattr(raw_message, 'content'):
            content_preview = str(raw_message.content)[:500] if raw_message.content else 'None'
            logger.debug(f"Raw content preview: {content_preview}")
        if hasattr(raw_message, 'tool_calls'):
            logger.debug(f"Tool calls: {raw_message.tool_calls}")
        if hasattr(raw_message, 'additional_kwargs'):
            logger.debug(f"Additional kwargs keys: {list(raw_message.additional_kwargs.keys())}")

        try:
            # Check for tool_calls in the raw message
            if hasattr(raw_message, 'tool_calls') and raw_message.tool_calls:
                tool_call = raw_message.tool_calls[0]
                # Tool call args contain the data, possibly wrapped in 'parameters'
                args = tool_call.get('args', {}) if isinstance(tool_call, dict) else getattr(tool_call, 'args', {})

                if isinstance(args, dict):
                    # Unwrap 'parameters' if present
                    if 'parameters' in args:
                        return args['parameters']
                    return args

            # Check for additional_kwargs with function_call
            if hasattr(raw_message, 'additional_kwargs'):
                kwargs = raw_message.additional_kwargs
                if 'function_call' in kwargs:
                    func_call = kwargs['function_call']
                    args_str = func_call.get('arguments', '{}')
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                    if 'parameters' in args:
                        return args['parameters']
                    return args

            # Try to parse content directly
            if hasattr(raw_message, 'content') and raw_message.content:
                content = raw_message.content
                if isinstance(content, str):
                    # Try JSON parsing
                    data = json.loads(content)
                    if 'parameters' in data:
                        return data['parameters']
                    return data
                elif isinstance(content, dict):
                    if 'parameters' in content:
                        return content['parameters']
                    return content

            logger.warning("Could not extract data from raw response")
            return None  # Return None to trigger fallback

        except Exception as e:
            logger.warning(f"Failed to extract from raw response: {e}")
            return None  # Return None to trigger fallback

    def _fallback_extraction(
        self,
        llm,
        prompt: str,
        schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fallback extraction using JSON parsing."""
        fallback_prompt = prompt + "\n\nRespond with a valid JSON object only, no other text."

        response = llm.invoke(fallback_prompt)

        # Safely extract string content from response
        raw_content = response.content
        if isinstance(raw_content, list):
            # Handle list of content blocks - extract text from each
            response_text = ''.join(
                str(item.get('text', item) if isinstance(item, dict) else item)
                for item in raw_content
            )
        elif isinstance(raw_content, dict):
            # Handle dict response - may contain the data directly
            if 'text' in raw_content:
                response_text = str(raw_content['text'])
            else:
                # Try to use it directly as extracted data
                return raw_content
        else:
            response_text = str(raw_content) if raw_content else ""

        # Try to parse JSON from response
        try:
            # Clean up response if needed
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]

            return json.loads(response_text.strip())
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse fallback JSON: {e}")
            # Return empty structure based on schema
            return self._empty_from_schema(schema)

    def _empty_from_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Create an empty data structure matching the schema."""
        result = {}
        for name, prop in schema.get("properties", {}).items():
            prop_type = prop.get("type", "string")
            if prop_type == "array":
                result[name] = []
            elif prop_type == "object":
                result[name] = {}
            else:
                result[name] = None
        return result

    def _count_extracted_fields(self, data: Dict[str, Any]) -> int:
        """Count non-null extracted fields."""
        count = 0
        for key, value in data.items():
            if value is not None:
                if isinstance(value, list):
                    count += len(value)
                elif isinstance(value, dict):
                    count += self._count_extracted_fields(value)
                else:
                    count += 1
        return count
