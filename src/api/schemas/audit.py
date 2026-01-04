"""Audit and Analytics API schemas."""

from datetime import datetime
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field


# =============================================================================
# Request Models
# =============================================================================

class LogEventRequest(BaseModel):
    """Request to log a custom audit event."""
    event_type: str = Field(..., description="Type of event", example="document_viewed")
    file_name: Optional[str] = Field(default=None, example="Sample1.md")
    document_hash: Optional[str] = Field(default=None, example="abc123def456")
    details: Dict[str, Any] = Field(default_factory=dict, example={"user_id": "user_123", "action": "view"})


# =============================================================================
# Response Models
# =============================================================================

class ProcessingJob(BaseModel):
    """A processing job record."""
    id: str = Field(..., example="job_abc123")
    organization_id: Optional[str] = Field(default=None, description="Organization ID for multi-tenancy")
    document_hash: str = Field(..., example="sha256_abc123def456")
    file_name: str = Field(..., example="Sample1.md")
    model: str = Field(..., example="gemini-3-flash-preview")
    complexity: str = Field(default="normal", example="normal")
    status: str = Field(..., example="completed")
    started_at: datetime
    completed_at: Optional[datetime] = None
    cached: bool = Field(default=False, example=False)
    output_path: Optional[str] = Field(default=None, example="generated/Sample1_generated.json")
    duration_ms: Optional[int] = Field(default=None, example=2345)
    error_message: Optional[str] = None


class DocumentRecord(BaseModel):
    """A document record."""
    organization_id: Optional[str] = Field(default=None, description="Organization ID for multi-tenancy")
    file_hash: Optional[str] = Field(default="", example="sha256_abc123def456")
    storage_path: Optional[str] = Field(default="", example="parsed/Sample1.md")
    filename: Optional[str] = Field(default="", example="Sample1.md")
    file_size: Optional[int] = Field(default=0, example=15234)
    created_at: Optional[datetime] = None


class DocumentGeneration(BaseModel):
    """A document generation record."""
    id: str = Field(..., example="gen_abc123")
    organization_id: Optional[str] = Field(default=None, description="Organization ID for multi-tenancy")
    document_name: str = Field(..., example="Sample1.md")
    document_hash: Optional[str] = Field(default=None, example="sha256_abc123def456")
    source_path: Optional[str] = Field(default=None, example="parsed/Sample1.md")
    generation_type: str = Field(..., example="summary")
    content: Dict[str, Any] = Field(default_factory=dict, description="Generated content")
    options: Dict[str, Any] = Field(default_factory=dict, description="Generation options used")
    model: str = Field(..., example="gemini-3-flash-preview")
    processing_time_ms: Optional[float] = Field(default=None, example=1234.56)
    session_id: Optional[str] = Field(default=None, example="sess_abc123")
    created_at: datetime
    content_preview: Optional[str] = Field(None, description="Preview of generated content", example="This document covers the key aspects of...")


class AuditEvent(BaseModel):
    """An audit trail event."""
    id: str = Field(..., example="evt_abc123")
    organization_id: Optional[str] = Field(default=None, description="Organization ID for multi-tenancy")
    created_at: Optional[datetime] = Field(default=None, description="When the event was created")
    event_type: Optional[str] = Field(default="unknown", example="generation_completed")
    document_hash: Optional[str] = Field(default=None, example="sha256_abc123def456")
    file_name: Optional[str] = Field(default=None, example="Sample1.md")
    job_id: Optional[str] = Field(default=None, example="job_abc123")
    details: Dict[str, Any] = Field(default_factory=dict, example={"generation_type": "summary", "word_count": 245})


# =============================================================================
# List Responses
# =============================================================================

class ListJobsResponse(BaseModel):
    """Response for listing jobs."""
    success: bool = Field(..., example=True)
    jobs: List[ProcessingJob] = Field(default_factory=list)
    total: int = Field(default=0, example=25)
    limit: int = Field(..., example=20)
    offset: int = Field(..., example=0)
    error: Optional[str] = None


class GetJobResponse(BaseModel):
    """Response for getting a single job."""
    success: bool = Field(..., example=True)
    job: Optional[ProcessingJob] = None
    error: Optional[str] = None


class ListDocumentsResponse(BaseModel):
    """Response for listing documents."""
    success: bool = Field(..., example=True)
    documents: List[DocumentRecord] = Field(default_factory=list)
    total: int = Field(default=0, example=15)
    limit: int = Field(..., example=20)
    offset: int = Field(..., example=0)
    error: Optional[str] = None


class GetDocumentResponse(BaseModel):
    """Response for getting a document with its jobs."""
    success: bool = Field(..., example=True)
    document: Optional[DocumentRecord] = None
    jobs: List[ProcessingJob] = Field(default_factory=list)
    generations: List[DocumentGeneration] = Field(default_factory=list)
    error: Optional[str] = None


class ListGenerationsResponse(BaseModel):
    """Response for listing generations."""
    success: bool = Field(..., example=True)
    generations: List[DocumentGeneration] = Field(default_factory=list)
    total: int = Field(default=0, example=42)
    limit: int = Field(..., example=20)
    offset: int = Field(..., example=0)
    error: Optional[str] = None


class AuditTrailResponse(BaseModel):
    """Response for audit trail."""
    success: bool = Field(..., example=True)
    events: List[AuditEvent] = Field(default_factory=list)
    total: int = Field(default=0, example=150)
    limit: int = Field(..., example=20)
    offset: int = Field(..., example=0)
    error: Optional[str] = None


class LogEventResponse(BaseModel):
    """Response for logging an event."""
    success: bool = Field(..., example=True)
    event_id: Optional[str] = Field(default=None, example="evt_abc123")
    error: Optional[str] = None


# =============================================================================
# Analytics/Dashboard
# =============================================================================

class DashboardStats(BaseModel):
    """Dashboard statistics."""
    total_documents: int = Field(default=0, example=15)
    total_jobs: int = Field(default=0, example=42)
    total_generations: int = Field(default=0, example=89)
    jobs_by_status: Dict[str, int] = Field(default_factory=dict, example={"completed": 38, "processing": 2, "failed": 2})
    generations_by_type: Dict[str, int] = Field(default_factory=dict, example={"summary": 30, "faqs": 29, "questions": 30})
    cache_hit_rate: float = Field(default=0.0, example=0.42)
    avg_processing_time_ms: float = Field(default=0.0, example=1523.45)


class DashboardResponse(BaseModel):
    """Response for dashboard data."""
    success: bool = Field(..., example=True)
    stats: Optional[DashboardStats] = None
    recent_jobs: List[ProcessingJob] = Field(default_factory=list)
    recent_generations: List[DocumentGeneration] = Field(default_factory=list)
    error: Optional[str] = None


# =============================================================================
# Activity Timeline
# =============================================================================


class ActivityTimelineItem(BaseModel):
    """A single activity item for the timeline."""
    id: str = Field(..., example="evt_abc123")
    timestamp: Optional[datetime] = Field(default=None, description="When the activity occurred")
    timestamp_ago: Optional[str] = Field(default="unknown", description="Relative time", example="5 minutes ago")
    event_type: Optional[str] = Field(default="unknown", description="Type of activity", example="generation_completed")
    title: Optional[str] = Field(default="Activity", description="Display title", example="Summary Generated")
    description: Optional[str] = Field(default=None, example="Generated summary for Sample1.md")
    file_name: Optional[str] = Field(default=None, example="Sample1.md")
    document_hash: Optional[str] = Field(default=None, example="sha256_abc123")
    status: Optional[str] = Field(default=None, example="completed")
    status_color: Optional[str] = Field(default=None, description="Suggested UI color", example="green")
    icon: Optional[str] = Field(default=None, description="Suggested icon name", example="file-text")


class ActivityTimelineResponse(BaseModel):
    """Response for activity timeline endpoint."""
    success: bool = Field(..., example=True)
    activities: List[ActivityTimelineItem] = Field(default_factory=list)
    total: int = Field(default=0, example=150)
    limit: int = Field(..., example=50)
    offset: int = Field(..., example=0)
    has_more: bool = Field(default=False, description="Whether more results are available")
    start_date: Optional[datetime] = Field(default=None, description="Filter start date used")
    end_date: Optional[datetime] = Field(default=None, description="Filter end date used")
    error: Optional[str] = None
