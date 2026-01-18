"""Dashboard data builder for frontend display.

Transforms Report data into a format optimized for dashboard/UI consumption.
"""

from datetime import datetime
from typing import Any, Dict, List

from ..schemas import (
    Report,
    ReportType,
    DashboardData,
    ChartData,
    Insight,
)


def build_dashboard_data(
    report: Report,
    include_raw_data: bool = False,
) -> DashboardData:
    """Build dashboard-ready data from a Report object.

    Args:
        report: Report object
        include_raw_data: Whether to include raw data arrays

    Returns:
        DashboardData object ready for frontend consumption
    """
    currency = report.options.currency_symbol if report.options else "$"

    # Build summary metrics
    summary_metrics = _build_summary_metrics(report, currency)

    # Build charts (simplified for dashboard)
    charts = _build_dashboard_charts(report)

    # Build tables
    tables = _build_dashboard_tables(report, currency)

    return DashboardData(
        summary_metrics=summary_metrics,
        charts=charts,
        tables=tables,
        insights=report.insights or [],
        generated_at=report.completed_at or datetime.utcnow(),
    )


def _build_summary_metrics(report: Report, currency: str) -> List[Dict[str, Any]]:
    """Build summary metrics for dashboard display."""
    metrics = []

    if report.summary:
        s = report.summary
        metrics = [
            {
                "id": "total_amount",
                "label": "Total Spend",
                "value": s.total_amount,
                "format": "currency",
                "currency": currency,
                "icon": "dollar-sign",
            },
            {
                "id": "document_count",
                "label": "Documents",
                "value": s.document_count,
                "format": "number",
                "icon": "file-text",
            },
            {
                "id": "record_count",
                "label": "Records",
                "value": s.record_count,
                "format": "number",
                "icon": "database",
            },
            {
                "id": "vendor_count",
                "label": "Vendors",
                "value": s.vendor_count,
                "format": "number",
                "icon": "users",
            },
            {
                "id": "avg_transaction",
                "label": "Avg Transaction",
                "value": s.avg_transaction_amount,
                "format": "currency",
                "currency": currency,
                "icon": "trending-up",
            },
        ]

        # Add top category/vendor as highlights
        if s.top_category:
            metrics.append({
                "id": "top_category",
                "label": "Top Category",
                "value": s.top_category,
                "format": "text",
                "icon": "tag",
            })

        if s.top_vendor:
            metrics.append({
                "id": "top_vendor",
                "label": "Top Vendor",
                "value": s.top_vendor,
                "format": "text",
                "icon": "building",
            })

    return metrics


def _build_dashboard_charts(report: Report) -> List[ChartData]:
    """Build chart data for dashboard display.

    Returns simplified chart data without images (frontend will render).
    """
    charts = []

    if report.report_type == ReportType.EXPENSE_SUMMARY and report.expense_data:
        data = report.expense_data

        # Category pie chart
        if data.by_category:
            charts.append(ChartData(
                chart_type="pie",
                title="Spending by Category",
                data={
                    "labels": [c.category for c in data.by_category[:8]],
                    "values": [c.amount for c in data.by_category[:8]],
                    "percentages": [c.percentage for c in data.by_category[:8]],
                },
            ))

        # Vendor bar chart
        if data.by_vendor:
            charts.append(ChartData(
                chart_type="bar",
                title="Top Vendors",
                data={
                    "labels": [v.vendor for v in data.by_vendor[:10]],
                    "values": [v.amount for v in data.by_vendor[:10]],
                },
            ))

        # Monthly trend line chart
        if data.monthly_trends:
            charts.append(ChartData(
                chart_type="line",
                title="Monthly Spending",
                data={
                    "labels": [t.month for t in data.monthly_trends],
                    "values": [t.amount for t in data.monthly_trends],
                    "counts": [t.count for t in data.monthly_trends],
                },
            ))

    elif report.report_type == ReportType.VENDOR_ANALYSIS and report.vendor_data:
        data = report.vendor_data

        # Vendor bar chart
        if data.vendor_rankings:
            charts.append(ChartData(
                chart_type="bar",
                title="Vendor Spend Rankings",
                data={
                    "labels": [v.vendor for v in data.vendor_rankings[:15]],
                    "values": [v.amount for v in data.vendor_rankings[:15]],
                },
            ))

        # Concentration pie chart
        if data.vendor_rankings and data.concentration_metrics:
            top_3_pct = data.concentration_metrics.get("top_3_percentage", 0)
            charts.append(ChartData(
                chart_type="pie",
                title="Vendor Concentration",
                data={
                    "labels": ["Top 3 Vendors", "Other Vendors"],
                    "values": [top_3_pct, 100 - top_3_pct],
                },
            ))

    # Include any pre-generated charts (with images)
    if report.charts:
        for chart in report.charts:
            # Check if we already have this chart type
            existing_types = [c.chart_type for c in charts]
            if chart.chart_type not in existing_types or chart.data.get("image_base64"):
                charts.append(chart)

    return charts


def _build_dashboard_tables(report: Report, currency: str) -> List[Dict[str, Any]]:
    """Build table data for dashboard display."""
    tables = []

    if report.report_type == ReportType.EXPENSE_SUMMARY and report.expense_data:
        data = report.expense_data

        # Category table
        if data.by_category:
            tables.append({
                "id": "category_breakdown",
                "title": "Spending by Category",
                "columns": [
                    {"key": "category", "label": "Category", "type": "text"},
                    {"key": "amount", "label": "Amount", "type": "currency", "currency": currency},
                    {"key": "percentage", "label": "% of Total", "type": "percent"},
                    {"key": "count", "label": "Transactions", "type": "number"},
                ],
                "rows": [
                    {
                        "category": c.category,
                        "amount": c.amount,
                        "percentage": c.percentage,
                        "count": c.count,
                    }
                    for c in data.by_category[:10]
                ],
            })

        # Vendor table
        if data.by_vendor:
            tables.append({
                "id": "vendor_breakdown",
                "title": "Top Vendors",
                "columns": [
                    {"key": "vendor", "label": "Vendor", "type": "text"},
                    {"key": "amount", "label": "Total Spend", "type": "currency", "currency": currency},
                    {"key": "invoice_count", "label": "Invoices", "type": "number"},
                    {"key": "avg_amount", "label": "Avg Invoice", "type": "currency", "currency": currency},
                ],
                "rows": [
                    {
                        "vendor": v.vendor,
                        "amount": v.amount,
                        "invoice_count": v.invoice_count,
                        "avg_amount": v.avg_amount,
                    }
                    for v in data.by_vendor[:10]
                ],
            })

        # Top expenses table
        if data.top_expenses:
            # Dynamically build columns from first record
            first_record = data.top_expenses[0]
            columns = []
            for key in list(first_record.keys())[:6]:  # Limit columns
                col_type = "text"
                if key in ["amount", "total", "price", "cost"]:
                    col_type = "currency"
                elif key in ["count", "quantity"]:
                    col_type = "number"
                columns.append({
                    "key": key,
                    "label": key.replace("_", " ").title(),
                    "type": col_type,
                    "currency": currency if col_type == "currency" else None,
                })

            tables.append({
                "id": "top_expenses",
                "title": "Largest Expenses",
                "columns": columns,
                "rows": data.top_expenses[:10],
            })

    elif report.report_type == ReportType.VENDOR_ANALYSIS and report.vendor_data:
        data = report.vendor_data

        # Vendor rankings table
        if data.vendor_rankings:
            tables.append({
                "id": "vendor_rankings",
                "title": "Vendor Rankings",
                "columns": [
                    {"key": "rank", "label": "#", "type": "number"},
                    {"key": "vendor", "label": "Vendor", "type": "text"},
                    {"key": "amount", "label": "Total Spend", "type": "currency", "currency": currency},
                    {"key": "percentage", "label": "% of Total", "type": "percent"},
                    {"key": "invoice_count", "label": "Invoices", "type": "number"},
                ],
                "rows": [
                    {
                        "rank": i + 1,
                        "vendor": v.vendor,
                        "amount": v.amount,
                        "percentage": v.percentage,
                        "invoice_count": v.invoice_count,
                    }
                    for i, v in enumerate(data.vendor_rankings[:15])
                ],
            })

    elif report.report_type == ReportType.INVOICE_RECONCILIATION and report.reconciliation_data:
        data = report.reconciliation_data

        # Summary table
        tables.append({
            "id": "reconciliation_summary",
            "title": "Reconciliation Status",
            "columns": [
                {"key": "metric", "label": "Metric", "type": "text"},
                {"key": "value", "label": "Count", "type": "number"},
            ],
            "rows": [
                {"metric": "Total Records", "value": data.summary.get("total_records", 0)},
                {"metric": "Matched", "value": data.summary.get("matched_count", 0)},
                {"metric": "Missing PO", "value": data.summary.get("missing_po_count", 0)},
            ],
        })

    return tables


def format_metric_value(value: Any, format_type: str, currency: str = "$") -> str:
    """Format a metric value for display.

    Args:
        value: The value to format
        format_type: One of 'currency', 'number', 'percent', 'text'
        currency: Currency symbol for currency format

    Returns:
        Formatted string
    """
    if value is None:
        return "N/A"

    if format_type == "currency":
        return f"{currency}{value:,.2f}"
    elif format_type == "number":
        return f"{value:,}"
    elif format_type == "percent":
        return f"{value:.1f}%"
    else:
        return str(value)
