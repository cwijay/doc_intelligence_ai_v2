"""Report Agent package for Business Intelligence Report Generation.

This agent generates business intelligence reports from extracted document data:
- Expense summaries
- Vendor analysis
- Invoice reconciliation
- Cash flow projections
- Spend trends
"""

from .core import ReportAgent, get_report_agent, shutdown_report_agent
from .config import ReportAgentConfig
from .schemas import (
    ReportType,
    ReportStatus,
    OutputFormat,
    DateRange,
    ReportOptions,
    Report,
    ReportRequest,
    ReportResponse,
    ReportStatusResponse,
    ListReportsResponse,
    AnalyzeRequest,
    AnalyzeResponse,
    DashboardData,
    Insight,
    ChartData,
    ReportSummary,
    CategoryBreakdown,
    VendorBreakdown,
    MonthlyTrend,
    ExpenseSummaryData,
    VendorAnalysisData,
    InvoiceReconciliationData,
)

__all__ = [
    # Agent
    "ReportAgent",
    "get_report_agent",
    "shutdown_report_agent",
    # Config
    "ReportAgentConfig",
    # Schemas - Enums
    "ReportType",
    "ReportStatus",
    "OutputFormat",
    # Schemas - Core Models
    "DateRange",
    "ReportOptions",
    "Report",
    "ReportSummary",
    "Insight",
    "ChartData",
    # Schemas - API
    "ReportRequest",
    "ReportResponse",
    "ReportStatusResponse",
    "ListReportsResponse",
    "AnalyzeRequest",
    "AnalyzeResponse",
    "DashboardData",
    # Schemas - Data Models
    "CategoryBreakdown",
    "VendorBreakdown",
    "MonthlyTrend",
    "ExpenseSummaryData",
    "VendorAnalysisData",
    "InvoiceReconciliationData",
]
