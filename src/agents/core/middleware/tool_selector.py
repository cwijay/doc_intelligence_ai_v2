"""
LLM Tool Selector middleware.

Pre-filters tools based on query relevance using a lightweight LLM.
Reduces token consumption and improves tool selection accuracy.
"""

import logging
import os
from typing import List, Optional

from langchain_core.tools import BaseTool
from langchain.chat_models import init_chat_model

logger = logging.getLogger(__name__)

# Default model for tool selection (can be overridden via env var)
DEFAULT_TOOL_SELECTOR_MODEL = os.getenv("TOOL_SELECTOR_MODEL", "gemini-3-flash-preview")


class LLMToolSelector:
    """
    Pre-filters tools based on query relevance using a lightweight LLM.

    This middleware helps the agent find the right tool by narrowing down
    the available tools based on the user's query before the main agent runs.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        provider: str = "google_genai",
        max_tools: int = 3,
        api_key: Optional[str] = None
    ):
        """
        Initialize the tool selector.

        Args:
            model: Model to use for tool selection (should be fast/cheap).
                   Defaults to TOOL_SELECTOR_MODEL env var or gemini-3-flash-preview.
            provider: Model provider
            max_tools: Maximum tools to return per query
            api_key: Optional API key override
        """
        self.model = model or DEFAULT_TOOL_SELECTOR_MODEL
        self.provider = provider
        self.max_tools = max_tools
        self._llm = None
        self.api_key = api_key

    @property
    def llm(self):
        """Lazy initialization of the LLM."""
        if self._llm is None:
            kwargs = {
                "model": self.model,
                "model_provider": self.provider,
                "temperature": 0
            }
            if self.api_key:
                kwargs["api_key"] = self.api_key
            self._llm = init_chat_model(**kwargs)
        return self._llm

    def select_tools(
        self,
        query: str,
        available_tools: List[BaseTool],
        context: Optional[str] = None
    ) -> List[BaseTool]:
        """
        Select most relevant tools for the given query.

        Args:
            query: User's query
            available_tools: List of all available tools
            context: Optional additional context

        Returns:
            List of selected tools (subset of available_tools)
        """
        # If we have few tools, return all of them
        if len(available_tools) <= self.max_tools:
            logger.debug(
                f"Only {len(available_tools)} tools available, returning all"
            )
            return available_tools

        # Build tool descriptions for the prompt
        tool_info = "\n".join([
            f"- {tool.name}: {tool.description}"
            for tool in available_tools
        ])

        prompt = f"""You are a tool selection assistant. Given a user query, select the most relevant tools.

User Query: "{query}"

Available Tools:
{tool_info}

Instructions:
1. Analyze the query to understand what the user wants to accomplish
2. Select the {self.max_tools} most relevant tools for this query
3. Return ONLY the tool names, one per line, no explanations

Selected Tools:"""

        if context:
            prompt = f"Context: {context}\n\n{prompt}"

        try:
            response = self.llm.invoke(prompt)
            # Handle both string and list content formats (Gemini returns list)
            content = response.content
            if isinstance(content, list):
                # Extract text from content blocks (Gemini format)
                response_text = "".join(
                    block.get("text", str(block)) if isinstance(block, dict) else str(block)
                    for block in content
                ).strip()
            else:
                response_text = content.strip()

            # Parse the response to get tool names
            selected_names = []
            for line in response_text.split("\n"):
                # Clean up the line
                name = line.strip().lower()
                # Remove common prefixes/bullets
                for prefix in ["-", "*", "•", "1.", "2.", "3.", "4.", "5."]:
                    if name.startswith(prefix):
                        name = name[len(prefix):].strip()
                # Normalize underscores/hyphens
                name = name.replace("-", "_").replace(" ", "_")
                if name:
                    selected_names.append(name)

            # Match against available tools
            selected = []
            for tool in available_tools:
                tool_name_normalized = tool.name.lower().replace("-", "_")
                if tool_name_normalized in selected_names:
                    selected.append(tool)

            # Fallback: if no matches, try partial matching
            if not selected:
                for tool in available_tools:
                    tool_name_normalized = tool.name.lower().replace("-", "_")
                    for name in selected_names:
                        if name in tool_name_normalized or tool_name_normalized in name:
                            selected.append(tool)
                            break

            # Final fallback: return all tools if selection completely failed
            if not selected:
                logger.warning(
                    f"Tool selection returned no matches for names: {selected_names}. "
                    "Using all tools."
                )
                return available_tools

            logger.info(
                f"Selected {len(selected)} tools from {len(available_tools)}: "
                f"{[t.name for t in selected]}"
            )
            return selected

        except Exception as e:
            logger.warning(f"Tool selection failed: {e}. Using all tools.")
            return available_tools

    def get_tool_descriptions(self, tools: List[BaseTool]) -> str:
        """Get formatted tool descriptions for logging/debugging."""
        return "\n".join([
            f"- {tool.name}: {tool.description[:100]}..."
            for tool in tools
        ])
