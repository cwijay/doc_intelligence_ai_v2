"""Pydantic schemas for Intelligence API endpoints.

These schemas define the request/response models for the business intelligence
report generation endpoints.
"""

from datetime import date, datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.agents.report.schemas import (
    ReportType,
    ReportStatus,
    OutputFormat,
    DateRange,
    ReportOptions,
    Insight,
    ChartData,
    ReportSummary,
)


class CreateReportRequest(BaseModel):
    """Request to create a new report."""
    folder_id: str = Field(..., description="Folder ID containing documents to analyze")
    report_type: ReportType = Field(..., description="Type of report to generate")
    date_range_start: Optional[date] = Field(None, description="Start date for filtering (inclusive)")
    date_range_end: Optional[date] = Field(None, description="End date for filtering (inclusive)")
    include_charts: bool = Field(default=True, description="Include charts in the report")
    include_recommendations: bool = Field(default=True, description="Include AI recommendations")
    output_format: OutputFormat = Field(default=OutputFormat.PDF, description="Primary output format")
    additional_formats: List[OutputFormat] = Field(default=[], description="Additional output formats to generate")
    max_insights: int = Field(default=5, ge=1, le=20, description="Maximum insights to generate")
    currency_symbol: str = Field(default="$", description="Currency symbol for monetary values")

    def to_date_range(self) -> Optional[DateRange]:
        """Convert to DateRange object if dates are provided."""
        if self.date_range_start and self.date_range_end:
            return DateRange(start=self.date_range_start, end=self.date_range_end)
        return None

    def to_options(self) -> ReportOptions:
        """Convert to ReportOptions object."""
        return ReportOptions(
            include_charts=self.include_charts,
            include_recommendations=self.include_recommendations,
            output_format=self.output_format,
            additional_formats=self.additional_formats,
            max_insights=self.max_insights,
            currency_symbol=self.currency_symbol,
        )


class CreateReportResponse(BaseModel):
    """Response after creating a report job."""
    success: bool
    report_id: str
    status: ReportStatus
    message: str
    estimated_time_seconds: Optional[int] = None


class ReportInfo(BaseModel):
    """Summary information about a report."""
    id: str
    organization_id: str
    folder_id: str
    report_type: ReportType
    status: ReportStatus
    document_count: int
    extracted_record_count: int
    created_at: datetime
    completed_at: Optional[datetime] = None
    processing_time_ms: Optional[int] = None
    error_message: Optional[str] = None

    class Config:
        from_attributes = True


class ReportDetailResponse(BaseModel):
    """Detailed report response including all data."""
    success: bool
    report: ReportInfo
    summary: Optional[ReportSummary] = None
    insights: List[Insight] = []
    charts: List[ChartData] = []
    pdf_path: Optional[str] = None
    excel_path: Optional[str] = None
    json_path: Optional[str] = None
    progress_percentage: Optional[float] = None


class ListReportsResponse(BaseModel):
    """Response for listing reports."""
    success: bool
    reports: List[ReportInfo]
    total: int
    limit: int
    offset: int


class DownloadReportResponse(BaseModel):
    """Response for report download request."""
    success: bool
    download_url: str
    format: str
    expires_in_seconds: int = 3600


class QuickAnalyzeRequest(BaseModel):
    """Request for quick analysis without full report."""
    folder_id: str = Field(..., description="Folder ID containing documents")
    analysis_type: str = Field(
        default="summary",
        description="Type of analysis: summary, trends, anomalies, top_vendors, top_categories"
    )
    date_range_start: Optional[date] = None
    date_range_end: Optional[date] = None
    limit: int = Field(default=10, ge=1, le=100, description="Number of results")

    def to_date_range(self) -> Optional[DateRange]:
        """Convert to DateRange object if dates are provided."""
        if self.date_range_start and self.date_range_end:
            return DateRange(start=self.date_range_start, end=self.date_range_end)
        return None


class QuickAnalyzeResponse(BaseModel):
    """Response for quick analysis."""
    success: bool
    analysis_type: str
    results: Dict[str, Any]
    insights: List[Insight] = []
    processing_time_ms: int


class DashboardDataRequest(BaseModel):
    """Request for dashboard data."""
    folder_id: str
    report_type: ReportType = Field(default=ReportType.EXPENSE_SUMMARY)
    date_range_start: Optional[date] = None
    date_range_end: Optional[date] = None

    def to_date_range(self) -> Optional[DateRange]:
        if self.date_range_start and self.date_range_end:
            return DateRange(start=self.date_range_start, end=self.date_range_end)
        return None


class DashboardMetric(BaseModel):
    """A single dashboard metric."""
    id: str
    label: str
    value: Any
    format: str  # currency, number, percent, text
    currency: Optional[str] = None
    icon: Optional[str] = None


class DashboardChart(BaseModel):
    """Dashboard chart data."""
    chart_type: str
    title: str
    data: Dict[str, Any]


class DashboardTable(BaseModel):
    """Dashboard table data."""
    id: str
    title: str
    columns: List[Dict[str, Any]]
    rows: List[Dict[str, Any]]


class DashboardDataResponse(BaseModel):
    """Response for dashboard data request."""
    success: bool
    summary_metrics: List[DashboardMetric]
    charts: List[DashboardChart]
    tables: List[DashboardTable]
    insights: List[Insight]
    generated_at: datetime


class ComparePeriodsRequest(BaseModel):
    """Request to compare two time periods."""
    folder_id: str
    report_type: ReportType = Field(default=ReportType.EXPENSE_SUMMARY)
    period1_start: date
    period1_end: date
    period2_start: date
    period2_end: date


class PeriodComparison(BaseModel):
    """Comparison between two periods."""
    metric: str
    period1_value: float
    period2_value: float
    change_amount: float
    change_percentage: float
    trend: str  # "up", "down", "stable"


class ComparePeriodsResponse(BaseModel):
    """Response for period comparison."""
    success: bool
    period1_label: str
    period2_label: str
    comparisons: List[PeriodComparison]
    insights: List[Insight]
    processing_time_ms: int
