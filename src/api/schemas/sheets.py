"""Sheets Agent API schemas."""

from datetime import datetime
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from .common import TokenUsage, HealthStatusEnum


# =============================================================================
# Request Models
# =============================================================================

class SheetsAnalyzeRequest(BaseModel):
    """Request for sheets analysis."""
    file_paths: List[str] = Field(..., description="List of file paths to analyze", example=["data/sales_2024.xlsx"])
    query: str = Field(..., description="Analysis query in natural language", example="What is the total revenue by region?")
    session_id: Optional[str] = Field(None, description="Session ID for context", example="sess_abc123")
    user_id: Optional[str] = Field(None, description="User ID", example="user_123")
    options: Optional[Dict[str, Any]] = None


class SheetsPreviewRequest(BaseModel):
    """Request to preview a file."""
    file_path: str = Field(..., description="Path to the file to preview", example="data/sales_2024.xlsx")
    rows: int = Field(default=10, ge=1, le=100, description="Number of rows to preview", example=10)
    sheet_name: Optional[str] = Field(None, description="Sheet name for Excel files", example="Sheet1")


# =============================================================================
# Response Models
# =============================================================================

class FileMetadata(BaseModel):
    """Metadata about a processed file."""
    file_path: str = Field(..., example="data/sales_2024.xlsx")
    file_type: str = Field(..., example="xlsx")
    size_bytes: Optional[int] = Field(default=None, example=52480)
    rows: Optional[int] = Field(default=None, example=1500)
    columns: Optional[int] = Field(default=None, example=12)
    column_names: Optional[List[str]] = Field(default=None, example=["Date", "Region", "Revenue", "Units"])
    sheet_names: Optional[List[str]] = Field(default=None, example=["Sales", "Summary"])
    processing_time_ms: Optional[float] = Field(default=None, example=45.23)


class ToolUsage(BaseModel):
    """Information about tool usage during analysis."""
    tool_name: str = Field(..., example="SingleFileQueryTool")
    input_summary: Optional[str] = Field(default=None, example="SELECT SUM(revenue) FROM sales GROUP BY region")
    output_summary: Optional[str] = Field(default=None, example="Query returned 4 rows")
    execution_time_ms: float = Field(..., example=123.45)
    success: bool = Field(default=True, example=True)
    error_message: Optional[str] = None


class SheetsAnalyzeResponse(BaseModel):
    """Response for sheets analysis."""
    success: bool = Field(..., example=True)
    message: Optional[str] = Field(default=None, example="Analysis completed successfully")
    response: Optional[str] = Field(default=None, example="The total revenue by region is: North: $1.2M, South: $890K...")
    files_processed: List[FileMetadata] = Field(default_factory=list)
    tools_used: List[ToolUsage] = Field(default_factory=list)
    token_usage: Optional[TokenUsage] = None
    session_id: Optional[str] = Field(default=None, example="sess_abc123")
    processing_time_ms: float = Field(..., example=2345.67)
    error: Optional[str] = None


class SheetsPreviewResponse(BaseModel):
    """Response for file preview."""
    success: bool = Field(..., example=True)
    file_info: Optional[FileMetadata] = None
    preview_data: Optional[List[Dict[str, Any]]] = Field(default=None, example=[{"Date": "2024-01-01", "Revenue": 15000}])
    dtypes: Optional[Dict[str, str]] = Field(default=None, example={"Date": "datetime64", "Revenue": "float64"})
    error: Optional[str] = None


class SheetsHealthResponse(BaseModel):
    """Health response for sheets agent."""
    status: HealthStatusEnum = Field(..., example="healthy")
    agent_ready: bool = Field(..., example=True)
    duckdb_ready: bool = Field(..., example=True)
    tools_available: List[str] = Field(..., example=["SmartAnalysisTool", "FilePreviewTool", "SingleFileQueryTool"])
    cache_stats: Optional[Dict[str, Any]] = Field(default=None, example={"hits": 45, "misses": 12})
