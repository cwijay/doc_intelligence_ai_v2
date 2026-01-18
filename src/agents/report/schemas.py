"""Pydantic schemas for Report Agent requests and responses."""

from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ReportType(str, Enum):
    """Available report types for business intelligence."""
    EXPENSE_SUMMARY = "expense_summary"
    VENDOR_ANALYSIS = "vendor_analysis"
    INVOICE_RECONCILIATION = "invoice_reconciliation"
    CASH_FLOW_PROJECTION = "cash_flow_projection"
    SPEND_TRENDS = "spend_trends"
    TAX_PREPARATION = "tax_preparation"


class ReportStatus(str, Enum):
    """Report generation status."""
    PENDING = "pending"
    PROCESSING = "processing"
    EXTRACTING = "extracting"
    AGGREGATING = "aggregating"
    ANALYZING = "analyzing"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


class OutputFormat(str, Enum):
    """Available output formats for reports."""
    PDF = "pdf"
    EXCEL = "excel"
    JSON = "json"
    MARKDOWN = "markdown"


class DateRange(BaseModel):
    """Date range for report filtering."""
    start: date = Field(..., description="Start date (inclusive)")
    end: date = Field(..., description="End date (inclusive)")

    def to_dict(self) -> Dict[str, str]:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
        }


class ReportOptions(BaseModel):
    """Options for report generation."""
    include_charts: bool = Field(default=True, description="Include charts in report")
    include_raw_data: bool = Field(default=False, description="Include raw extracted data")
    include_recommendations: bool = Field(default=True, description="Include AI-generated recommendations")
    output_format: OutputFormat = Field(default=OutputFormat.PDF, description="Primary output format")
    additional_formats: List[OutputFormat] = Field(default=[], description="Additional output formats")
    chart_types: List[str] = Field(default=["pie", "bar", "line"], description="Chart types to generate")
    max_insights: int = Field(default=5, ge=1, le=20, description="Maximum number of insights to generate")
    currency_symbol: str = Field(default="$", description="Currency symbol for monetary values")
    locale: str = Field(default="en_US", description="Locale for number/date formatting")


class CategoryBreakdown(BaseModel):
    """Breakdown of amounts by category."""
    category: str
    amount: float
    percentage: float
    count: int


class VendorBreakdown(BaseModel):
    """Breakdown of amounts by vendor."""
    vendor: str
    amount: float
    percentage: float
    invoice_count: int
    avg_amount: float


class MonthlyTrend(BaseModel):
    """Monthly trend data point."""
    month: str  # YYYY-MM format
    amount: float
    count: int
    change_percentage: Optional[float] = None


class Insight(BaseModel):
    """AI-generated insight from report analysis."""
    category: str  # e.g., "spending", "trend", "anomaly", "recommendation"
    title: str
    description: str
    severity: Optional[str] = None  # "info", "warning", "critical"
    data_points: Optional[Dict[str, Any]] = None


class ChartData(BaseModel):
    """Chart data and metadata."""
    chart_type: str  # "pie", "bar", "line", "area"
    title: str
    data: Dict[str, Any]
    gcs_path: Optional[str] = None  # Path to rendered chart image


class ReportSummary(BaseModel):
    """Summary metrics for a report."""
    total_amount: float
    document_count: int
    record_count: int
    vendor_count: int
    category_count: int
    date_range: DateRange
    top_category: Optional[str] = None
    top_vendor: Optional[str] = None
    avg_transaction_amount: Optional[float] = None


class ExpenseSummaryData(BaseModel):
    """Data structure for expense summary report."""
    summary: ReportSummary
    by_category: List[CategoryBreakdown]
    by_vendor: List[VendorBreakdown]
    monthly_trends: List[MonthlyTrend]
    top_expenses: List[Dict[str, Any]]
    insights: List[Insight]
    charts: List[ChartData]


class VendorAnalysisData(BaseModel):
    """Data structure for vendor analysis report."""
    summary: ReportSummary
    vendor_rankings: List[VendorBreakdown]
    vendor_trends: Dict[str, List[MonthlyTrend]]
    payment_terms_analysis: Optional[Dict[str, Any]] = None
    concentration_metrics: Dict[str, Any]
    insights: List[Insight]
    charts: List[ChartData]


class ReconciliationMatch(BaseModel):
    """A matched PO-Invoice pair."""
    po_number: Optional[str]
    invoice_number: str
    vendor: str
    po_amount: Optional[float]
    invoice_amount: float
    discrepancy: Optional[float]
    match_status: str  # "matched", "amount_mismatch", "missing_po", "missing_invoice"


class InvoiceReconciliationData(BaseModel):
    """Data structure for invoice reconciliation report."""
    summary: Dict[str, Any]
    matched_pairs: List[ReconciliationMatch]
    unmatched_invoices: List[Dict[str, Any]]
    unmatched_pos: List[Dict[str, Any]]
    discrepancies: List[ReconciliationMatch]
    insights: List[Insight]


class Report(BaseModel):
    """Complete report object."""
    id: str
    organization_id: str
    folder_id: str
    report_type: ReportType
    status: ReportStatus
    date_range: DateRange
    options: ReportOptions
    document_count: int
    extracted_record_count: int
    summary: Optional[ReportSummary] = None
    insights: List[Insight] = []
    charts: List[ChartData] = []
    pdf_path: Optional[str] = None
    excel_path: Optional[str] = None
    json_path: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    processing_time_ms: Optional[int] = None

    # Type-specific data (only one will be populated based on report_type)
    expense_data: Optional[ExpenseSummaryData] = None
    vendor_data: Optional[VendorAnalysisData] = None
    reconciliation_data: Optional[InvoiceReconciliationData] = None


class ReportRequest(BaseModel):
    """Request to generate a report."""
    folder_id: str = Field(..., description="Folder containing documents to analyze")
    report_type: ReportType = Field(..., description="Type of report to generate")
    date_range: Optional[DateRange] = Field(None, description="Date range filter (optional)")
    options: Optional[ReportOptions] = Field(default_factory=ReportOptions, description="Report options")


class ReportResponse(BaseModel):
    """Response from report generation."""
    success: bool
    report_id: str
    status: ReportStatus
    message: str
    estimated_time_seconds: Optional[int] = None


class ReportStatusResponse(BaseModel):
    """Response for report status check."""
    success: bool
    report: Report
    progress_percentage: Optional[float] = None


class ListReportsResponse(BaseModel):
    """Response for listing reports."""
    success: bool
    reports: List[Report]
    total: int
    limit: int
    offset: int


class AnalyzeRequest(BaseModel):
    """Request for quick analysis without full report."""
    folder_id: str
    analysis_type: str = Field(default="summary", description="Type of analysis: summary, trends, anomalies")
    date_range: Optional[DateRange] = None
    limit: int = Field(default=10, ge=1, le=100, description="Number of results to return")


class AnalyzeResponse(BaseModel):
    """Response for quick analysis."""
    success: bool
    analysis_type: str
    results: Dict[str, Any]
    insights: List[Insight]
    processing_time_ms: int


class DashboardData(BaseModel):
    """Data structure for frontend dashboard display."""
    summary_metrics: List[Dict[str, Any]]
    charts: List[ChartData]
    tables: List[Dict[str, Any]]
    insights: List[Insight]
    generated_at: datetime
