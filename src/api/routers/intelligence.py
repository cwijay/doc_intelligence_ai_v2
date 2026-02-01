"""
Intelligence API router for Business Intelligence Reports.

Endpoints for generating and managing business intelligence reports:
- Create reports (expense summary, vendor analysis, invoice reconciliation)
- Get report status and details
- List reports
- Quick analysis (dashboard data)
- Download reports (PDF, Excel)
"""

import asyncio
import logging
import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Path, BackgroundTasks

from ..dependencies import get_org_id
from ..schemas.intelligence import (
    CreateReportRequest,
    CreateReportResponse,
    ReportInfo,
    ReportDetailResponse,
    ListReportsResponse,
    DownloadReportResponse,
    QuickAnalyzeRequest,
    QuickAnalyzeResponse,
    DashboardDataRequest,
    DashboardDataResponse,
    DashboardMetric,
    DashboardChart,
    DashboardTable,
    ComparePeriodsRequest,
    ComparePeriodsResponse,
)
from src.agents.report import (
    get_report_agent,
    ReportType,
    ReportStatus,
)
from src.db.repositories import report_repository
from src.storage import get_storage

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# REPORT CREATION
# =============================================================================


async def _generate_report_background(
    report_id: str,
    org_id: str,
    folder_id: str,
    report_type: ReportType,
    date_range,
    options,
    folder_name: Optional[str] = None,
):
    """Background task to generate a report."""
    agent = get_report_agent()

    try:
        # Update status to processing
        await report_repository.update_report_status(report_id, "processing")

        # Generate the report
        report = await agent.generate_report(
            org_id=org_id,
            folder_id=folder_id,
            report_type=report_type,
            date_range=date_range,
            options=options,
            folder_name=folder_name,
        )

        # Update database with results
        await report_repository.update_report_data(
            report_id=report_id,
            document_count=report.document_count,
            extracted_record_count=report.extracted_record_count,
            summary=report.summary.model_dump() if report.summary else None,
            insights=[i.model_dump() for i in report.insights] if report.insights else None,
            charts=[c.model_dump() for c in report.charts] if report.charts else None,
            pdf_path=report.pdf_path,
            excel_path=report.excel_path,
            json_path=report.json_path,
        )

        # Update status
        await report_repository.update_report_status(
            report_id=report_id,
            status=report.status.value,
            error_message=report.error_message,
            processing_time_ms=report.processing_time_ms,
        )

        logger.info(f"Report {report_id} generation completed: {report.status.value}")

    except Exception as e:
        logger.error(f"Report {report_id} generation failed: {e}", exc_info=True)
        await report_repository.update_report_status(
            report_id=report_id,
            status="failed",
            error_message=str(e),
        )


@router.post(
    "/reports",
    response_model=CreateReportResponse,
    operation_id="createReport",
    summary="Create a business intelligence report",
    tags=["Intelligence Reports"],
)
async def create_report(
    request: CreateReportRequest,
    background_tasks: BackgroundTasks,
    org_id: str = Depends(get_org_id),
):
    """
    Create a new business intelligence report.

    **Report Types:**
    - `expense_summary`: Analyze expenses by category, vendor, and time
    - `vendor_analysis`: Vendor spend rankings and concentration analysis
    - `invoice_reconciliation`: Match invoices with purchase orders
    - `spend_trends`: Historical spending trend analysis
    - `cash_flow_projection`: Cash flow based on AR/AP
    - `tax_preparation`: Expense categorization for tax reporting

    **Workflow:**
    1. Creates a report job (returns immediately)
    2. Background task loads extracted data from folder
    3. Runs DuckDB analysis queries
    4. Generates AI insights
    5. Creates charts and renders PDF/Excel

    Poll `GET /reports/{report_id}` for status updates.
    """
    # Create report record in database
    date_range = request.to_date_range()
    options = request.to_options()

    report_dict = await report_repository.create_report(
        organization_id=org_id,
        folder_id=request.folder_id,
        report_type=request.report_type.value,
        date_range_start=str(date_range.start) if date_range else None,
        date_range_end=str(date_range.end) if date_range else None,
        options=options.model_dump(),
    )

    report_id = report_dict.get("id") or str(report_dict)

    # Start background generation
    background_tasks.add_task(
        _generate_report_background,
        report_id=report_id,
        org_id=org_id,
        folder_id=request.folder_id,
        report_type=request.report_type,
        date_range=date_range,
        options=options,
        folder_name=request.folder_name,
    )

    # Estimate time based on report type
    estimated_time = {
        ReportType.EXPENSE_SUMMARY: 30,
        ReportType.VENDOR_ANALYSIS: 25,
        ReportType.INVOICE_RECONCILIATION: 45,
        ReportType.SPEND_TRENDS: 20,
        ReportType.CASH_FLOW_PROJECTION: 35,
        ReportType.TAX_PREPARATION: 40,
    }.get(request.report_type, 30)

    return CreateReportResponse(
        success=True,
        report_id=report_id,
        status=ReportStatus.PENDING,
        message=f"Report generation started. Poll GET /reports/{report_id} for status.",
        estimated_time_seconds=estimated_time,
    )


# =============================================================================
# REPORT RETRIEVAL
# =============================================================================


@router.get(
    "/reports/{report_id}",
    response_model=ReportDetailResponse,
    operation_id="getReport",
    summary="Get report details",
    tags=["Intelligence Reports"],
)
async def get_report(
    report_id: str = Path(..., description="Report ID"),
    org_id: str = Depends(get_org_id),
):
    """
    Get details of a report including status, summary, insights, and download paths.

    **Status values:**
    - `pending`: Report job created, not yet started
    - `extracting`: Loading extracted data from folder
    - `aggregating`: Aggregating data into dataset
    - `analyzing`: Running DuckDB analysis queries
    - `generating`: Creating insights and charts
    - `completed`: Report ready for download
    - `failed`: Report generation failed
    """
    report_dict = await report_repository.get_report_by_org(report_id, org_id)

    if not report_dict:
        raise HTTPException(status_code=404, detail="Report not found")

    # Build response
    report_info = ReportInfo(
        id=report_dict["id"],
        organization_id=report_dict["organization_id"],
        folder_id=report_dict["folder_id"],
        report_type=ReportType(report_dict["report_type"]),
        status=ReportStatus(report_dict["status"]),
        document_count=report_dict.get("document_count", 0),
        extracted_record_count=report_dict.get("extracted_record_count", 0),
        created_at=report_dict["created_at"],
        completed_at=report_dict.get("completed_at"),
        processing_time_ms=report_dict.get("processing_time_ms"),
        error_message=report_dict.get("error_message"),
    )

    # Parse stored JSON fields
    from src.agents.report.schemas import ReportSummary, Insight, ChartData

    summary = None
    if report_dict.get("summary"):
        try:
            summary = ReportSummary(**report_dict["summary"])
        except Exception:
            pass

    insights = []
    if report_dict.get("insights"):
        try:
            insights = [Insight(**i) for i in report_dict["insights"]]
        except Exception:
            pass

    charts = []
    if report_dict.get("charts"):
        try:
            charts = [ChartData(**c) for c in report_dict["charts"]]
        except Exception:
            pass

    # Calculate progress
    progress = None
    status_progress = {
        "pending": 0,
        "processing": 10,
        "extracting": 20,
        "aggregating": 40,
        "analyzing": 60,
        "generating": 80,
        "completed": 100,
        "failed": None,
    }
    progress = status_progress.get(report_dict["status"])

    return ReportDetailResponse(
        success=True,
        report=report_info,
        summary=summary,
        insights=insights,
        charts=charts,
        pdf_path=report_dict.get("pdf_path"),
        excel_path=report_dict.get("excel_path"),
        json_path=report_dict.get("json_path"),
        progress_percentage=progress,
    )


@router.get(
    "/reports",
    response_model=ListReportsResponse,
    operation_id="listReports",
    summary="List reports",
    tags=["Intelligence Reports"],
)
async def list_reports(
    status: Optional[str] = Query(None, description="Filter by status"),
    report_type: Optional[str] = Query(None, description="Filter by report type"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    org_id: str = Depends(get_org_id),
):
    """List all reports for the organization with optional filters."""
    reports_list = await report_repository.list_reports(
        organization_id=org_id,
        status=status,
        report_type=report_type,
        limit=limit,
        offset=offset,
    )

    total = await report_repository.count_reports(
        organization_id=org_id,
        status=status,
        report_type=report_type,
    )

    reports = [
        ReportInfo(
            id=r["id"],
            organization_id=r["organization_id"],
            folder_id=r["folder_id"],
            report_type=ReportType(r["report_type"]),
            status=ReportStatus(r["status"]),
            document_count=r.get("document_count", 0),
            extracted_record_count=r.get("extracted_record_count", 0),
            created_at=r["created_at"],
            completed_at=r.get("completed_at"),
            processing_time_ms=r.get("processing_time_ms"),
            error_message=r.get("error_message"),
        )
        for r in reports_list
    ]

    return ListReportsResponse(
        success=True,
        reports=reports,
        total=total,
        limit=limit,
        offset=offset,
    )


# =============================================================================
# REPORT DOWNLOAD
# =============================================================================


@router.get(
    "/reports/{report_id}/download",
    response_model=DownloadReportResponse,
    operation_id="downloadReport",
    summary="Get download URL for report",
    tags=["Intelligence Reports"],
)
async def download_report(
    report_id: str = Path(..., description="Report ID"),
    format: str = Query("pdf", description="Format: pdf, excel, json"),
    org_id: str = Depends(get_org_id),
):
    """
    Get a signed download URL for the report.

    **Formats:**
    - `pdf`: PDF report with charts and tables
    - `excel`: Excel workbook with multiple sheets
    - `json`: Raw JSON data for integration
    """
    report_dict = await report_repository.get_report_by_org(report_id, org_id)

    if not report_dict:
        raise HTTPException(status_code=404, detail="Report not found")

    if report_dict["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Report not ready for download (status: {report_dict['status']})"
        )

    # Get the appropriate path
    path_map = {
        "pdf": report_dict.get("pdf_path"),
        "excel": report_dict.get("excel_path"),
        "json": report_dict.get("json_path"),
    }

    file_path = path_map.get(format.lower())
    if not file_path:
        raise HTTPException(
            status_code=404,
            detail=f"Report not available in {format} format"
        )

    # Generate signed URL
    storage = get_storage()

    try:
        # Parse GCS path
        if file_path.startswith("gs://"):
            blob_path = file_path.split("/", 3)[-1]
        else:
            blob_path = file_path

        download_url = await storage.generate_signed_url(
            blob_path=blob_path,
            expiration_minutes=60,
            method="GET",
        )

        return DownloadReportResponse(
            success=True,
            download_url=download_url,
            format=format.lower(),
            expires_in_seconds=3600,
        )

    except Exception as e:
        logger.error(f"Failed to generate download URL: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate download URL")


# =============================================================================
# QUICK ANALYSIS / DASHBOARD
# =============================================================================


@router.post(
    "/analyze",
    response_model=QuickAnalyzeResponse,
    operation_id="quickAnalyze",
    summary="Quick analysis without full report",
    tags=["Intelligence Analysis"],
)
async def quick_analyze(
    request: QuickAnalyzeRequest,
    org_id: str = Depends(get_org_id),
):
    """
    Run a quick analysis on folder data without generating a full report.

    **Analysis Types:**
    - `summary`: Basic metrics (totals, counts, top categories/vendors)
    - `trends`: Monthly spending trends
    - `anomalies`: Unusual transactions or patterns
    - `top_vendors`: Top vendors by spend
    - `top_categories`: Top categories by spend
    """
    start_time = time.time()
    agent = get_report_agent()

    try:
        # Load and analyze data
        date_range = request.to_date_range()
        extracted_data = await agent._load_extracted_data(org_id, request.folder_id, date_range)

        if not extracted_data.get("records"):
            return QuickAnalyzeResponse(
                success=False,
                analysis_type=request.analysis_type,
                results={"error": "No data found in folder"},
                insights=[],
                processing_time_ms=int((time.time() - start_time) * 1000),
            )

        # Aggregate and analyze
        from src.agents.report.schemas import ReportOptions
        df = await agent._aggregate_data(extracted_data, ReportType.EXPENSE_SUMMARY)
        analysis = await agent._run_analysis(df, ReportType.EXPENSE_SUMMARY, ReportOptions())

        # Build results based on analysis type
        results = {}
        if request.analysis_type == "summary":
            results = {
                "total_amount": analysis.get("total_amount", 0),
                "record_count": analysis.get("record_count", 0),
                "vendor_count": analysis.get("vendor_count", 0),
                "top_categories": analysis.get("by_category", [])[:request.limit],
                "top_vendors": analysis.get("by_vendor", [])[:request.limit],
            }
        elif request.analysis_type == "trends":
            results = {
                "monthly_trends": analysis.get("monthly_trends", []),
            }
        elif request.analysis_type == "top_vendors":
            results = {
                "top_vendors": analysis.get("by_vendor", [])[:request.limit],
            }
        elif request.analysis_type == "top_categories":
            results = {
                "top_categories": analysis.get("by_category", [])[:request.limit],
            }
        else:
            results = analysis

        # Generate quick insights
        insights = await agent._generate_insights(
            analysis,
            ReportType.EXPENSE_SUMMARY,
            ReportOptions(max_insights=3),
        )

        return QuickAnalyzeResponse(
            success=True,
            analysis_type=request.analysis_type,
            results=results,
            insights=insights,
            processing_time_ms=int((time.time() - start_time) * 1000),
        )

    except Exception as e:
        logger.error(f"Quick analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/dashboard",
    response_model=DashboardDataResponse,
    operation_id="getDashboardData",
    summary="Get dashboard data for UI display",
    tags=["Intelligence Analysis"],
)
async def get_dashboard_data(
    request: DashboardDataRequest,
    org_id: str = Depends(get_org_id),
):
    """
    Get data formatted for dashboard/UI display.

    Returns:
    - Summary metrics (total spend, document count, etc.)
    - Chart data (pie, bar, line charts)
    - Table data (category breakdown, vendor breakdown)
    - Quick insights
    """
    agent = get_report_agent()

    try:
        date_range = request.to_date_range()
        dashboard_data = await agent.get_dashboard_data(
            org_id=org_id,
            folder_id=request.folder_id,
            report_type=request.report_type,
            date_range=date_range,
        )

        # Convert to response format
        return DashboardDataResponse(
            success=True,
            summary_metrics=[
                DashboardMetric(**m) for m in dashboard_data.summary_metrics
            ],
            charts=[
                DashboardChart(
                    chart_type=c.chart_type,
                    title=c.title,
                    data=c.data,
                )
                for c in dashboard_data.charts
            ],
            tables=[
                DashboardTable(**t) for t in dashboard_data.tables
            ],
            insights=dashboard_data.insights,
            generated_at=dashboard_data.generated_at,
        )

    except Exception as e:
        logger.error(f"Dashboard data generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# REPORT MANAGEMENT
# =============================================================================


@router.delete(
    "/reports/{report_id}",
    operation_id="deleteReport",
    summary="Delete a report",
    tags=["Intelligence Reports"],
)
async def delete_report(
    report_id: str = Path(..., description="Report ID"),
    org_id: str = Depends(get_org_id),
):
    """Delete a report and its associated files."""
    report_dict = await report_repository.get_report_by_org(report_id, org_id)

    if not report_dict:
        raise HTTPException(status_code=404, detail="Report not found")

    # Delete from GCS (optional - could keep for audit)
    storage = get_storage()
    for path_key in ["pdf_path", "excel_path", "json_path"]:
        path = report_dict.get(path_key)
        if path:
            try:
                await storage.delete(path)
            except Exception as e:
                logger.warning(f"Failed to delete {path}: {e}")

    # Delete from database
    deleted = await report_repository.delete_report(report_id, org_id)

    return {
        "success": deleted,
        "message": "Report deleted" if deleted else "Report not found",
    }


@router.get(
    "/statistics",
    operation_id="getReportStatistics",
    summary="Get report statistics",
    tags=["Intelligence Analysis"],
)
async def get_report_statistics(
    org_id: str = Depends(get_org_id),
):
    """Get aggregate statistics about reports for the organization."""
    stats = await report_repository.get_report_statistics(org_id)

    return {
        "success": True,
        "statistics": {
            "total_reports": stats.get("total_reports", 0),
            "completed_reports": stats.get("completed_reports", 0),
            "failed_reports": stats.get("failed_reports", 0),
            "pending_reports": stats.get("pending_reports", 0),
            "total_documents_processed": stats.get("total_documents_processed", 0),
            "total_records_analyzed": stats.get("total_records_analyzed", 0),
            "avg_processing_time_ms": stats.get("avg_processing_time_ms"),
        },
    }
