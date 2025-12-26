"""
Memory data models for Document and Sheets agents.

Provides schemas for:
- User preferences (persistent across sessions)
- Conversation summaries (historical context)
- Generic memory entries
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class UserPreferences(BaseModel):
    """User-specific preferences stored in long-term memory."""

    user_id: str = Field(..., description="Unique user identifier")
    organization_id: Optional[str] = Field(
        default=None,
        description="Organization ID for multi-tenancy"
    )
    preferred_language: Optional[str] = Field(
        default="en",
        description="Preferred language for responses"
    )
    preferred_summary_length: Optional[int] = Field(
        default=500,
        description="Preferred summary length in words"
    )
    preferred_faq_count: Optional[int] = Field(
        default=5,
        description="Preferred number of FAQs to generate"
    )
    preferred_question_count: Optional[int] = Field(
        default=10,
        description="Preferred number of questions to generate"
    )
    custom_settings: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional custom settings"
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="When preferences were first created"
    )
    updated_at: datetime = Field(
        default_factory=datetime.now,
        description="When preferences were last updated"
    )


class ConversationSummary(BaseModel):
    """Summary of past conversations for long-term memory."""

    session_id: str = Field(..., description="Session identifier")
    organization_id: Optional[str] = Field(
        default=None,
        description="Organization ID for multi-tenancy"
    )
    user_id: str = Field(..., description="User identifier")
    agent_type: str = Field(
        ...,
        description="Type of agent ('document' or 'sheets')"
    )
    summary: str = Field(..., description="Summary of the conversation")
    key_topics: List[str] = Field(
        default_factory=list,
        description="Key topics discussed"
    )
    documents_discussed: List[str] = Field(
        default_factory=list,
        description="Documents referenced in conversation"
    )
    queries_count: int = Field(
        default=0,
        description="Number of queries in session"
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="When summary was created"
    )


class MemoryEntry(BaseModel):
    """Generic memory entry for long-term storage."""

    id: Optional[str] = Field(
        default=None,
        description="Entry ID"
    )
    organization_id: Optional[str] = Field(
        default=None,
        description="Organization ID for multi-tenancy"
    )
    namespace: str = Field(
        ...,
        description="Memory namespace ('preferences' or 'summaries')"
    )
    key: str = Field(
        ...,
        description="Entry key (user_id or session_id)"
    )
    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Memory data payload"
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="When entry was created"
    )
    updated_at: datetime = Field(
        default_factory=datetime.now,
        description="When entry was last updated"
    )


class ConversationMessage(BaseModel):
    """Single message in conversation history."""

    role: str = Field(
        ...,
        description="Message role ('human', 'ai', 'system')"
    )
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When message was sent"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional message metadata"
    )
