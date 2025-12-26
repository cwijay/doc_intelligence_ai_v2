"""
Query Intent Classifier for tool routing.

Classifies user queries to determine primary intent, enabling
intelligent pre-filtering of tools before agent execution.
"""

import logging
import re
from enum import Enum
from typing import Dict, List, Optional

from langchain.chat_models import init_chat_model

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """User query intent categories."""
    RAG_SEARCH = "rag_search"           # Q&A, search, find info
    CONTENT_GENERATION = "generation"    # Summary, FAQs, questions
    DOCUMENT_LOAD = "document_load"      # Explicit load request
    CONVERSATIONAL = "conversational"    # Memory/conversation-related (no tools needed)
    MIXED = "mixed"                       # Multiple intents


class QueryClassifier:
    """
    Classifies user queries to determine primary intent.

    Uses a two-stage approach:
    1. Rule-based pattern matching (fast, no API call)
    2. LLM-based classification (fallback for ambiguous cases)
    """

    # Keywords that indicate conversation/memory-related queries (check first!)
    CONVERSATIONAL_PATTERNS = [
        r"\bprevious\s+query\b", r"\bprevious\s+question\b",
        r"\bwhat\s+did\s+i\s+ask\b", r"\bwhat\s+i\s+asked\b",
        r"\bmy\s+last\s+question\b", r"\bmy\s+previous\b",
        r"\bdo\s+you\s+remember\b", r"\bremember\s+when\s+i\b",
        r"\bearlier\s+i\s+(asked|said)\b", r"\bbefore\s+i\s+(asked|said)\b",
        r"\bour\s+conversation\b", r"\bour\s+chat\b",
        r"\bwhat\s+we\s+(talked|discussed)\b",
        r"\bwhat\s+was\s+my\b",
    ]

    # Keywords that strongly indicate RAG search intent
    RAG_PATTERNS = [
        r"\bwhat\b", r"\bwho\b", r"\bwhen\b", r"\bwhere\b", r"\bwhy\b", r"\bhow\b",
        r"\bfind\b", r"\bsearch\b", r"\blook\s+for\b", r"\btell\s+me\s+about\b",
        r"\bexplain\b", r"\bdescribe\b", r"\banswer\b", r"\bquery\b",
        r"\bshow\s+me\b", r"\bget\s+info\b", r"\binformation\s+about\b",
        r"\bcan\s+you\s+tell\b", r"\bdo\s+you\s+know\b"
    ]

    # Keywords that strongly indicate content generation intent
    GENERATION_PATTERNS = [
        r"\bgenerate\b", r"\bcreate\b", r"\bsummarize\b", r"\bsummary\b",
        r"\bfaq\b", r"\bfaqs\b", r"\bquestions?\b", r"\bquiz\b",
        r"\bextract\b", r"\bproduce\b", r"\bmake\b", r"\bbuild\b",
        r"\bwrite\b", r"\bcompose\b", r"\bdraft\b",
        r"\bcomprehension\s+questions\b", r"\bstudy\s+questions\b"
    ]

    # Keywords that indicate document loading
    LOAD_PATTERNS = [
        r"\bload\b", r"\bopen\b", r"\bread\b", r"\bfetch\b",
        r"\bget\s+document\b", r"\bget\s+file\b"
    ]

    def __init__(
        self,
        use_llm_fallback: bool = True,
        llm_model: Optional[str] = None,
        llm_provider: str = "google_genai",
        api_key: Optional[str] = None
    ):
        """
        Initialize the query classifier.

        Args:
            use_llm_fallback: Whether to use LLM for ambiguous cases
            llm_model: Model for LLM fallback (uses classifier's own model if needed)
            llm_provider: Model provider
            api_key: Optional API key override
        """
        self.use_llm_fallback = use_llm_fallback
        self.llm_model = llm_model
        self.llm_provider = llm_provider
        self.api_key = api_key
        self._llm = None

        # Compile patterns for efficiency
        self._conv_compiled = [re.compile(p, re.IGNORECASE) for p in self.CONVERSATIONAL_PATTERNS]
        self._rag_compiled = [re.compile(p, re.IGNORECASE) for p in self.RAG_PATTERNS]
        self._gen_compiled = [re.compile(p, re.IGNORECASE) for p in self.GENERATION_PATTERNS]
        self._load_compiled = [re.compile(p, re.IGNORECASE) for p in self.LOAD_PATTERNS]

    @property
    def llm(self):
        """Lazy initialization of the LLM for fallback classification."""
        if self._llm is None and self.use_llm_fallback and self.llm_model:
            kwargs = {
                "model": self.llm_model,
                "model_provider": self.llm_provider,
                "temperature": 0
            }
            if self.api_key:
                kwargs["api_key"] = self.api_key
            self._llm = init_chat_model(**kwargs)
        return self._llm

    def classify(
        self,
        query: str,
        context: Optional[Dict] = None
    ) -> QueryIntent:
        """
        Classify a query to determine user intent.

        Args:
            query: User's query string
            context: Optional context dict with keys like:
                - document_name: Name of document if specified
                - has_parsed_path: Whether a parsed file path was provided
                - organization_name: Organization for RAG search

        Returns:
            QueryIntent enum value
        """
        if not query or not query.strip():
            logger.warning("Empty query received, defaulting to MIXED")
            return QueryIntent.MIXED

        query_lower = query.lower().strip()
        context = context or {}

        # Stage 1: Rule-based classification
        intent = self._rule_based_classify(query_lower, context)

        if intent != QueryIntent.MIXED:
            logger.debug(f"Rule-based classification: {intent.value} for query: {query[:50]}...")
            return intent

        # Stage 2: LLM fallback for ambiguous cases
        if self.use_llm_fallback and self.llm:
            intent = self._llm_classify(query, context)
            logger.debug(f"LLM classification: {intent.value} for query: {query[:50]}...")
            return intent

        # Default to RAG search (more conversational/natural)
        logger.debug(f"Default classification: RAG_SEARCH for ambiguous query: {query[:50]}...")
        return QueryIntent.RAG_SEARCH

    def _rule_based_classify(
        self,
        query: str,
        context: Dict
    ) -> QueryIntent:
        """
        Classify using rule-based pattern matching.

        Returns MIXED if patterns are ambiguous.
        """
        # Check conversational patterns FIRST (highest priority)
        # These are about the conversation itself, not document content
        conv_score = sum(1 for p in self._conv_compiled if p.search(query))
        if conv_score >= 1:
            logger.info(f"Conversational query detected: {query[:50]}...")
            return QueryIntent.CONVERSATIONAL

        # Count matches for each intent
        rag_score = sum(1 for p in self._rag_compiled if p.search(query))
        gen_score = sum(1 for p in self._gen_compiled if p.search(query))
        load_score = sum(1 for p in self._load_compiled if p.search(query))

        # Context-based boosting
        if context.get("has_parsed_path") and context.get("document_name"):
            # Specific document context suggests generation
            gen_score += 1
        if context.get("organization_name") and not context.get("document_name"):
            # Org-wide search context suggests RAG
            rag_score += 1

        # Strong generation signals
        if gen_score >= 2 and gen_score > rag_score:
            return QueryIntent.CONTENT_GENERATION

        # Strong RAG signals
        if rag_score >= 2 and rag_score > gen_score:
            return QueryIntent.RAG_SEARCH

        # Explicit load request
        if load_score >= 1 and rag_score == 0 and gen_score == 0:
            return QueryIntent.DOCUMENT_LOAD

        # Single strong indicator
        if gen_score == 1 and rag_score == 0:
            return QueryIntent.CONTENT_GENERATION
        if rag_score == 1 and gen_score == 0:
            return QueryIntent.RAG_SEARCH

        # Ambiguous - needs LLM or default
        return QueryIntent.MIXED

    def _llm_classify(
        self,
        query: str,
        context: Dict
    ) -> QueryIntent:
        """
        Classify using LLM for ambiguous cases.
        """
        prompt = f"""Classify this user query into one category:

Query: "{query}"

Categories:
1. RAG_SEARCH - User wants to find/search for information or ask questions about documents
2. CONTENT_GENERATION - User wants to generate new content (summary, FAQs, questions)
3. DOCUMENT_LOAD - User wants to load/open a specific document

Respond with ONLY the category name (RAG_SEARCH, CONTENT_GENERATION, or DOCUMENT_LOAD):"""

        try:
            response = self.llm.invoke(prompt)
            response_text = response.content.strip().upper()

            if "CONTENT_GENERATION" in response_text or "GENERATION" in response_text:
                return QueryIntent.CONTENT_GENERATION
            elif "DOCUMENT_LOAD" in response_text or "LOAD" in response_text:
                return QueryIntent.DOCUMENT_LOAD
            elif "RAG_SEARCH" in response_text or "RAG" in response_text or "SEARCH" in response_text:
                return QueryIntent.RAG_SEARCH
            else:
                logger.warning(f"LLM returned unexpected classification: {response_text}")
                return QueryIntent.RAG_SEARCH  # Default to conversational

        except Exception as e:
            logger.warning(f"LLM classification failed: {e}, defaulting to RAG_SEARCH")
            return QueryIntent.RAG_SEARCH

    def get_intent_description(self, intent: QueryIntent) -> str:
        """Get human-readable description of an intent."""
        descriptions = {
            QueryIntent.RAG_SEARCH: "Search and answer questions about documents",
            QueryIntent.CONTENT_GENERATION: "Generate new content (summary, FAQs, questions)",
            QueryIntent.DOCUMENT_LOAD: "Load a specific document",
            QueryIntent.CONVERSATIONAL: "Memory/conversation-related query (no tools needed)",
            QueryIntent.MIXED: "Ambiguous intent - multiple possible actions"
        }
        return descriptions.get(intent, "Unknown intent")
