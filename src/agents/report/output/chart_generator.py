"""Chart data generation for business intelligence reports.

Generates chart data structures to be rendered by the frontend using Recharts.
No server-side rendering - just prepares the data in the format expected by charts.
"""

import logging
import math
from typing import Any, Dict, List, Optional

from ..schemas import ReportType, ChartData

logger = logging.getLogger(__name__)


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert a value to float, handling NaN, Infinity, and None.

    Args:
        value: Value to convert
        default: Default value if conversion fails or value is invalid

    Returns:
        Valid float value (never NaN or Infinity)
    """
    if value is None:
        return default
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def _safe_str(value: Any, default: str = "Unknown", max_length: Optional[int] = None) -> str:
    """Safely convert a value to string.

    Args:
        value: Value to convert
        default: Default value if conversion fails
        max_length: Optional max length to truncate to

    Returns:
        String value
    """
    if value is None:
        return default
    result = str(value) if value else default
    if max_length and len(result) > max_length:
        return result[:max_length]
    return result


async def generate_charts(
    analysis_results: Dict[str, Any],
    report_type: ReportType,
    chart_types: List[str],
    dpi: int = 150,  # Kept for API compatibility, but unused
) -> List[ChartData]:
    """Generate chart data structures from analysis results.

    Args:
        analysis_results: Results from analysis queries
        report_type: Type of report
        chart_types: List of chart types to generate ("pie", "bar", "line")
        dpi: Unused - kept for API compatibility

    Returns:
        List of ChartData objects with chart configurations for frontend rendering
    """
    charts = []

    try:
        # Generate category pie chart data
        if "pie" in chart_types and "by_category" in analysis_results:
            chart = _build_pie_chart_data(
                analysis_results["by_category"],
                "Spending by Category",
                "category",
                "total",
            )
            if chart:
                charts.append(chart)

        # Generate vendor bar chart data
        if "bar" in chart_types and "by_vendor" in analysis_results:
            chart = _build_bar_chart_data(
                analysis_results["by_vendor"][:10],  # Top 10
                "Top Vendors by Spend",
                "vendor",
                "total",
            )
            if chart:
                charts.append(chart)

        # Generate monthly trend line chart data
        if "line" in chart_types and "monthly_trends" in analysis_results:
            chart = _build_line_chart_data(
                analysis_results["monthly_trends"],
                "Monthly Spending Trend",
                "month",
                "total",
            )
            if chart:
                charts.append(chart)

        # Generate top expenses bar chart
        if "bar" in chart_types and "top_expenses" in analysis_results:
            chart = _build_top_expenses_chart(
                analysis_results["top_expenses"][:10],
                "Top Expenses",
            )
            if chart:
                charts.append(chart)

    except Exception as e:
        logger.error(f"Chart data generation failed: {e}")

    return charts


def _build_pie_chart_data(
    data: List[Dict[str, Any]],
    title: str,
    label_field: str,
    value_field: str,
) -> ChartData | None:
    """Build pie chart data structure.

    Args:
        data: List of dicts with label and value fields
        title: Chart title
        label_field: Field name for labels
        value_field: Field name for values

    Returns:
        ChartData object or None if no data
    """
    if not data:
        return None

    # Limit to top 8 for readability
    limited_data = data[:8]

    # Calculate percentages (use safe_float to handle NaN)
    total = sum(_safe_float(d.get(value_field)) for d in limited_data) or 1

    chart_items = []
    for d in limited_data:
        value = _safe_float(d.get(value_field))
        chart_items.append({
            "name": _safe_str(d.get(label_field), max_length=25),
            "value": round(value, 2),
            "percentage": round((value / total) * 100, 1),
        })

    return ChartData(
        chart_type="pie",
        title=title,
        data={
            "items": chart_items,
            "total": round(total, 2),
        },
    )


def _build_bar_chart_data(
    data: List[Dict[str, Any]],
    title: str,
    label_field: str,
    value_field: str,
) -> ChartData | None:
    """Build bar chart data structure.

    Args:
        data: List of dicts with label and value fields
        title: Chart title
        label_field: Field name for labels
        value_field: Field name for values

    Returns:
        ChartData object or None if no data
    """
    if not data:
        return None

    chart_items = []
    for d in data:
        chart_items.append({
            "name": _safe_str(d.get(label_field), max_length=30),
            "value": round(_safe_float(d.get(value_field)), 2),
        })

    return ChartData(
        chart_type="bar",
        title=title,
        data={
            "items": chart_items,
        },
    )


def _build_line_chart_data(
    data: List[Dict[str, Any]],
    title: str,
    x_field: str,
    y_field: str,
) -> ChartData | None:
    """Build line chart data structure for trends.

    Args:
        data: List of dicts with x and y fields
        title: Chart title
        x_field: Field name for x-axis
        y_field: Field name for y-axis

    Returns:
        ChartData object or None if insufficient data
    """
    if not data or len(data) < 2:
        return None

    chart_items = []
    prev_value = None

    for d in data:
        value = _safe_float(d.get(y_field))
        item = {
            "name": _safe_str(d.get(x_field), default=""),
            "value": round(value, 2),
        }

        # Calculate change percentage from previous
        if prev_value is not None and prev_value != 0:
            change = ((value - prev_value) / prev_value) * 100
            # Ensure change is also safe (could be inf if prev_value is very small)
            item["change"] = round(_safe_float(change), 1)

        chart_items.append(item)
        prev_value = value

    return ChartData(
        chart_type="line",
        title=title,
        data={
            "items": chart_items,
        },
    )


def _build_top_expenses_chart(
    data: List[Dict[str, Any]],
    title: str,
) -> ChartData | None:
    """Build chart data for top expenses.

    Args:
        data: List of expense records
        title: Chart title

    Returns:
        ChartData object or None if no data
    """
    if not data:
        return None

    chart_items = []
    for d in data:
        # Try different field names for vendor/description
        name = d.get("vendor") or d.get("merchant") or d.get("description") or "Unknown"
        amount = d.get("amount") or d.get("total") or d.get("total_amount") or 0

        # Get date value, convert to string if it's a date object
        date_val = d.get("date") or d.get("invoice_date")
        if date_val is not None:
            date_val = str(date_val) if date_val else None

        chart_items.append({
            "name": _safe_str(name, max_length=30),
            "value": round(_safe_float(amount), 2),
            "date": date_val,
            "category": d.get("category"),
        })

    return ChartData(
        chart_type="bar",
        title=title,
        data={
            "items": chart_items,
        },
    )


# Legacy function for backwards compatibility - returns None
def render_chart_to_bytes(chart_data: ChartData, dpi: int = 150) -> None:
    """Legacy function - charts are now rendered on the frontend.

    Returns None as server-side rendering is no longer used.
    """
    logger.info("Server-side chart rendering deprecated - use frontend Recharts instead")
    return None
