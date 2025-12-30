"""Tool selection manager for DocumentAgent.

Provides intelligent tool filtering based on query intent using
two-stage filtering: QueryClassifier + LLMToolSelector.
"""

import logging
from typing import List, Optional, Dict, Any

from src.agents.core.middleware import LLMToolSelector, QueryClassifier, QueryIntent

logger = logging.getLogger(__name__)


class ToolSelectionManager:
    """Manages intelligent tool selection based on query intent.

    Uses two-stage filtering:
    1. QueryClassifier determines intent (RAG vs Generation)
    2. LLMToolSelector narrows down within the category
    """

    def __init__(
        self,
        tools: List,
        config,
        api_key: str,
    ):
        """Initialize the tool selection manager.

        Args:
            tools: List of all available tools
            config: Agent configuration with tool selection settings
            api_key: API key for LLM-based selection
        """
        self.tools = tools
        self.config = config
        self.tools_by_name = {tool.name: tool for tool in tools}

        self.query_classifier: Optional[QueryClassifier] = None
        self.tool_selector: Optional[LLMToolSelector] = None

        if config.enable_tool_selection:
            self._init_components(api_key)

    def _init_components(self, api_key: str):
        """Initialize query classifier and tool selector."""
        try:
            # Initialize query classifier with LLM fallback
            self.query_classifier = QueryClassifier(
                use_llm_fallback=True,
                llm_model=self.config.tool_selector_model,
                llm_provider="google_genai",
                api_key=api_key
            )

            # Initialize LLM-based tool selector
            self.tool_selector = LLMToolSelector(
                model=self.config.tool_selector_model,
                provider="google_genai",
                max_tools=self.config.tool_selector_max_tools,
                api_key=api_key
            )

            logger.info(
                f"Tool selection enabled: classifier + selector "
                f"(model={self.config.tool_selector_model}, max_tools={self.config.tool_selector_max_tools})"
            )

        except Exception as e:
            logger.warning(f"Failed to initialize tool selection: {e}. Using all tools.")
            self.query_classifier = None
            self.tool_selector = None

    @property
    def enabled(self) -> bool:
        """Check if tool selection is enabled and initialized."""
        return self.config.enable_tool_selection and self.query_classifier is not None

    def get_tools_for_query(self, query: str, context: Dict[str, Any]) -> List:
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
        if not self.enabled:
            return self.tools

        try:
            # Stage 1: Classify query intent
            intent = self.query_classifier.classify(query, context)
            logger.debug(f"Query intent: {intent.value} for: {query[:50]}...")

            # Map intent to tool subsets
            candidate_tools = self._get_tools_for_intent(intent)

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

    def _get_tools_for_intent(self, intent: QueryIntent) -> List:
        """Get candidate tools based on query intent."""
        if intent == QueryIntent.RAG_SEARCH:
            # RAG search only needs the rag_search tool
            candidate_tools = [self.tools_by_name.get('rag_search')]
            return [t for t in candidate_tools if t is not None]

        elif intent == QueryIntent.CONTENT_GENERATION:
            # Content generation needs loader, generators, and persist
            tool_names = [
                'document_loader',
                'summary_generator',
                'faq_generator',
                'question_generator',
                'content_persist'
            ]
            return [
                self.tools_by_name.get(name)
                for name in tool_names
                if self.tools_by_name.get(name) is not None
            ]

        elif intent == QueryIntent.DOCUMENT_LOAD:
            # Just document loading
            candidate_tools = [self.tools_by_name.get('document_loader')]
            return [t for t in candidate_tools if t is not None]

        elif intent == QueryIntent.CONVERSATIONAL:
            # Conversational query - no tools needed
            logger.info("Conversational query - no tools needed, using memory")
            return []

        else:
            # MIXED intent - use all tools
            return self.tools


def bind_rag_filters(
    tools: List,
    file_filter: Optional[str] = None,
    folder_filter: Optional[str] = None,
) -> List:
    """
    Bind filter values to the RAG search tool for correct cache scoping.

    When a request targets a specific document or folder, the RAG tool
    needs these filters to ensure the semantic cache correctly scopes
    queries. This prevents cross-document cache hits.

    Args:
        tools: List of tools (may include RAG search tool)
        file_filter: File name to bind to RAG tool
        folder_filter: Folder name to bind to RAG tool

    Returns:
        Modified tools list with bound RAG tool (if present)
    """
    if not file_filter and not folder_filter:
        return tools

    from .tools.rag_search import RAGSearchTool

    modified_tools = []
    for tool in tools:
        if isinstance(tool, RAGSearchTool):
            # Create a new RAG tool with bound filters
            bound_tool = tool.with_bound_filters(
                file_filter=file_filter,
                folder_filter=folder_filter,
            )
            modified_tools.append(bound_tool)
            logger.debug(f"Bound filters to RAG tool: file={file_filter}, folder={folder_filter}")
        else:
            modified_tools.append(tool)

    return modified_tools
