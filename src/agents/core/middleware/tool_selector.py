"""
LLM Tool Selector middleware.

Pre-filters tools based on query relevance using a lightweight LLM.
Reduces token consumption and improves tool selection accuracy.

Aligned with LangChain's LLMToolSelectorMiddleware patterns:
- Uses structured output for reliable tool name extraction
- Supports always_include for critical tools
- Supports custom system_prompt

Reference: https://docs.langchain.com/oss/python/langchain/middleware/built-in#llm-tool-selector
"""

import logging
import os
from typing import List, Optional

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain.chat_models import init_chat_model

logger = logging.getLogger(__name__)

# Default model for tool selection (can be overridden via env var)
# Using gpt-5.2 for better tool selection accuracy
DEFAULT_TOOL_SELECTOR_MODEL = os.getenv("TOOL_SELECTOR_MODEL", "gpt-5.2-2025-12-11")

# Default system prompt for tool selection
DEFAULT_SYSTEM_PROMPT = """You are a tool selection assistant. Given a user query, select the most relevant tools from the available options.

Analyze the query carefully:
- For FAQs/FAQ generation, select faq_generator
- For questions/quiz/comprehension questions, select question_generator
- For summary/summarize requests, select summary_generator
- For search/find/lookup queries, select rag_search

Return only the tool names that are most relevant for the query."""


class ToolSelectionResult(BaseModel):
    """Schema for structured tool selection output."""
    selected_tools: List[str] = Field(
        description="List of tool names to use for this query. Use exact tool names from the available tools list."
    )


class LLMToolSelector:
    """
    Pre-filters tools based on query relevance using a lightweight LLM.

    This middleware helps the agent find the right tool by narrowing down
    the available tools based on the user's query before the main agent runs.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        provider: str = "openai",
        max_tools: int = 3,
        always_include: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize the tool selector.

        Aligned with LangChain's LLMToolSelectorMiddleware interface.

        Args:
            model: Model to use for tool selection (should be fast/cheap).
                   Defaults to TOOL_SELECTOR_MODEL env var or gpt-5.2.
            provider: Model provider
            max_tools: Maximum tools to return per query (excluding always_include)
            always_include: Tool names to always include regardless of selection.
                          These don't count against max_tools limit.
            system_prompt: Custom instructions for the selection model.
                          Uses DEFAULT_SYSTEM_PROMPT if not specified.
            api_key: Optional API key override
        """
        self.model = model or DEFAULT_TOOL_SELECTOR_MODEL
        self.provider = provider
        self.max_tools = max_tools
        self.always_include = always_include or []
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self._llm = None
        self._structured_llm = None
        self.api_key = api_key

    @property
    def llm(self):
        """Lazy initialization of the LLM."""
        if self._llm is None:
            kwargs = {
                "model": self.model,
                "model_provider": self.provider,
                "temperature": 0,
            }
            # use_responses_api only required for specific models like gpt-5-nano
            if "nano" in self.model.lower():
                kwargs["use_responses_api"] = True
            if self.api_key:
                kwargs["api_key"] = self.api_key
            self._llm = init_chat_model(**kwargs)
        return self._llm

    @property
    def structured_llm(self):
        """Lazy initialization of the structured output LLM."""
        if self._structured_llm is None:
            self._structured_llm = self.llm.with_structured_output(ToolSelectionResult)
        return self._structured_llm

    def select_tools(
        self,
        query: str,
        available_tools: List[BaseTool],
        context: Optional[str] = None,
        force_selection: bool = False,
        select_count: Optional[int] = None
    ) -> List[BaseTool]:
        """
        Select most relevant tools for the given query using structured output.

        Args:
            query: User's query
            available_tools: List of all available tools
            context: Optional additional context
            force_selection: If True, always use LLM selection even with few tools
            select_count: Override number of tools to select (default: self.max_tools)

        Returns:
            List of selected tools (subset of available_tools)
        """
        num_to_select = select_count or self.max_tools

        # Build lookup maps
        tools_by_name = {tool.name: tool for tool in available_tools}
        tool_names = list(tools_by_name.keys())

        # Separate always_include tools from selectable tools
        always_included = []
        selectable_tools = []
        for tool in available_tools:
            if tool.name in self.always_include:
                always_included.append(tool)
            else:
                selectable_tools.append(tool)

        # If no selectable tools, just return always_included
        if not selectable_tools:
            logger.debug("No selectable tools, returning always_include only")
            return always_included

        # If few selectable tools AND not forcing selection, return all
        if len(selectable_tools) <= num_to_select and not force_selection:
            logger.debug(
                f"Only {len(selectable_tools)} selectable tools, returning all with always_include"
            )
            return always_included + selectable_tools

        # Build tool descriptions for the prompt
        tool_info = "\n".join([
            f"- {tool.name}: {tool.description}"
            for tool in selectable_tools
        ])

        prompt = f"""{self.system_prompt}

User Query: "{query}"

Available Tools (select {num_to_select}):
{tool_info}

Select the {num_to_select} most relevant tool(s) for this query."""

        if context:
            prompt = f"Context: {context}\n\n{prompt}"

        try:
            # Try structured output first
            result = self.structured_llm.invoke(prompt)

            # Match selected tool names against available tools
            selected = []
            for name in result.selected_tools:
                # Try exact match first
                if name in tools_by_name:
                    selected.append(tools_by_name[name])
                else:
                    # Try case-insensitive match
                    name_lower = name.lower().replace("-", "_").replace(" ", "_")
                    for tool_name, tool in tools_by_name.items():
                        if tool_name.lower().replace("-", "_") == name_lower:
                            selected.append(tool)
                            break

            # Limit to requested count
            selected = selected[:num_to_select]

            # If no matches from structured output, fall back to text parsing
            if not selected:
                logger.warning(
                    f"Structured output returned no matches: {result.selected_tools}. "
                    "Falling back to text parsing."
                )
                selected = self._select_tools_text_fallback(
                    query, selectable_tools, num_to_select, context
                )

            # Combine with always_include tools
            final_selection = always_included + selected

            logger.info(
                f"Selected {len(selected)} tools from {len(selectable_tools)} "
                f"(+{len(always_included)} always_include): {[t.name for t in final_selection]}"
            )
            return final_selection

        except Exception as e:
            logger.warning(f"Structured tool selection failed: {e}. Using text fallback.")
            selected = self._select_tools_text_fallback(
                query, selectable_tools, num_to_select, context
            )
            return always_included + selected

    def _select_tools_text_fallback(
        self,
        query: str,
        available_tools: List[BaseTool],
        num_to_select: int,
        context: Optional[str] = None
    ) -> List[BaseTool]:
        """
        Fallback text-based tool selection when structured output fails.

        Args:
            query: User's query
            available_tools: List of selectable tools
            num_to_select: Number of tools to select
            context: Optional context

        Returns:
            List of selected tools
        """
        tool_info = "\n".join([
            f"- {tool.name}: {tool.description}"
            for tool in available_tools
        ])

        prompt = f"""{self.system_prompt}

User Query: "{query}"

Available Tools:
{tool_info}

Select the {num_to_select} most relevant tool(s). Return ONLY tool names, one per line:"""

        if context:
            prompt = f"Context: {context}\n\n{prompt}"

        try:
            response = self.llm.invoke(prompt)
            content = response.content
            if isinstance(content, list):
                response_text = "".join(
                    block.get("text", str(block)) if isinstance(block, dict) else str(block)
                    for block in content
                ).strip()
            else:
                response_text = content.strip()

            # Parse response to get tool names
            selected_names = []
            for line in response_text.split("\n"):
                name = line.strip().lower()
                for prefix in ["-", "*", "•", "1.", "2.", "3.", "4.", "5."]:
                    if name.startswith(prefix):
                        name = name[len(prefix):].strip()
                name = name.replace("-", "_").replace(" ", "_")
                if name:
                    selected_names.append(name)

            # Match against available tools
            selected = []
            for tool in available_tools:
                tool_name_normalized = tool.name.lower().replace("-", "_")
                if tool_name_normalized in selected_names:
                    selected.append(tool)
                    if len(selected) >= num_to_select:
                        break

            if not selected:
                # Partial matching fallback
                for tool in available_tools:
                    tool_name_normalized = tool.name.lower().replace("-", "_")
                    for name in selected_names:
                        if name in tool_name_normalized or tool_name_normalized in name:
                            selected.append(tool)
                            break
                    if len(selected) >= num_to_select:
                        break

            if not selected:
                logger.warning(
                    f"Text fallback returned no matches for: {selected_names}. "
                    "Using all selectable tools."
                )
                return available_tools[:num_to_select]

            return selected[:num_to_select]

        except Exception as e:
            logger.error(f"Text fallback also failed: {e}. Using first {num_to_select} tools.")
            return available_tools[:num_to_select]

    def get_tool_descriptions(self, tools: List[BaseTool]) -> str:
        """Get formatted tool descriptions for logging/debugging."""
        return "\n".join([
            f"- {tool.name}: {tool.description[:100]}..."
            for tool in tools
        ])
