"""Common schema models shared across API endpoints."""

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List

from pydantic import BaseModel, Field


# =============================================================================
# Enums for Type Safety
# =============================================================================

class HealthStatusEnum(str, Enum):
    """Health status values for service components."""
    healthy = "healthy"
    unhealthy = "unhealthy"
    degraded = "degraded"


class DifficultyLevel(str, Enum):
    """Difficulty levels for generated questions."""
    easy = "easy"
    medium = "medium"
    hard = "hard"


class JobStatus(str, Enum):
    """Processing job status values."""
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class ContentType(str, Enum):
    """Generated content type values."""
    summary = "summary"
    faqs = "faqs"
    questions = "questions"


# =============================================================================
# Common Response Models
# =============================================================================

class TokenUsage(BaseModel):
    """Token usage statistics."""
    prompt_tokens: int = Field(default=0, example=150)
    completion_tokens: int = Field(default=0, example=89)
    total_tokens: int = Field(default=0, example=239)
    estimated_cost_usd: Optional[float] = Field(default=None, example=0.000024)


class ToolUsage(BaseModel):
    """Information about tools used during processing."""
    tool_name: str = Field(..., description="Name of the tool used")
    input_data: Dict[str, Any] = Field(..., description="Input provided to the tool")
    output_data: Optional[Dict[str, Any]] = Field(None, description="Output from the tool")
    execution_time_ms: float = Field(..., description="Tool execution time")
    success: bool = Field(..., description="Whether tool execution was successful")
    error_message: Optional[str] = Field(None, description="Error message if tool failed")


class ErrorResponse(BaseModel):
    """Standard error response."""
    success: bool = Field(default=False, example=False)
    error: str = Field(..., example="Document not found")
    message: Optional[str] = Field(default=None, example="The requested document 'Sample1.md' could not be found")
    details: Optional[List[Dict[str, Any]]] = None
    request_id: Optional[str] = Field(default=None, example="req_abc123")


class SuccessResponse(BaseModel):
    """Standard success response."""
    success: bool = True
    message: Optional[str] = None


class PaginationParams(BaseModel):
    """Pagination parameters."""
    limit: int = Field(default=20, ge=1, le=100, example=20)
    offset: int = Field(default=0, ge=0, example=0)


class PaginatedResponse(BaseModel):
    """Paginated response wrapper."""
    items: List[Any]
    total: int
    limit: int
    offset: int
    has_more: bool


class SessionInfo(BaseModel):
    """Session information."""
    session_id: str
    user_id: Optional[str] = None
    created_at: datetime
    last_activity: datetime
    expires_at: datetime
    query_count: int = 0
    total_tokens_used: int = 0
    total_processing_time_ms: float = 0


class HealthStatus(BaseModel):
    """Service health status."""
    status: HealthStatusEnum = Field(default=HealthStatusEnum.healthy, example="healthy")
    version: str = Field(default="3.0.0", example="3.0.0")
    timestamp: datetime
    components: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
