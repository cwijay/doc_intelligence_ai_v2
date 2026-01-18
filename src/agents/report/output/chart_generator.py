"""Chart generation for business intelligence reports.

Uses matplotlib for server-side chart rendering.
Charts are saved as PNG images and can be embedded in PDF reports.
"""

import asyncio
import io
import logging
from typing import Any, Dict, List, Optional

from ..schemas import ReportType, ChartData

logger = logging.getLogger(__name__)

# Check if matplotlib is available
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server-side rendering
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available - chart generation disabled")


async def generate_charts(
    analysis_results: Dict[str, Any],
    report_type: ReportType,
    chart_types: List[str],
    dpi: int = 150,
) -> List[ChartData]:
    """Generate charts from analysis results.

    Args:
        analysis_results: Results from analysis queries
        report_type: Type of report
        chart_types: List of chart types to generate ("pie", "bar", "line")
        dpi: DPI for chart images

    Returns:
        List of ChartData objects with rendered chart data
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Skipping chart generation - matplotlib not available")
        return []

    charts = []

    try:
        # Generate category pie chart
        if "pie" in chart_types and "by_category" in analysis_results:
            chart = await _generate_pie_chart(
                analysis_results["by_category"],
                "Spending by Category",
                "category",
                "total",
                dpi,
            )
            if chart:
                charts.append(chart)

        # Generate vendor bar chart
        if "bar" in chart_types and "by_vendor" in analysis_results:
            chart = await _generate_bar_chart(
                analysis_results["by_vendor"][:10],  # Top 10
                "Top Vendors by Spend",
                "vendor",
                "total",
                dpi,
            )
            if chart:
                charts.append(chart)

        # Generate monthly trend line chart
        if "line" in chart_types and "monthly_trends" in analysis_results:
            chart = await _generate_line_chart(
                analysis_results["monthly_trends"],
                "Monthly Spending Trend",
                "month",
                "total",
                dpi,
            )
            if chart:
                charts.append(chart)

    except Exception as e:
        logger.error(f"Chart generation failed: {e}")

    return charts


async def _generate_pie_chart(
    data: List[Dict[str, Any]],
    title: str,
    label_field: str,
    value_field: str,
    dpi: int,
) -> Optional[ChartData]:
    """Generate a pie chart.

    Args:
        data: List of dicts with label and value fields
        title: Chart title
        label_field: Field name for labels
        value_field: Field name for values
        dpi: DPI for image

    Returns:
        ChartData object or None if generation fails
    """
    if not data:
        return None

    try:
        loop = asyncio.get_event_loop()
        image_bytes = await loop.run_in_executor(
            None,
            lambda: _render_pie_chart(data, title, label_field, value_field, dpi)
        )

        return ChartData(
            chart_type="pie",
            title=title,
            data={
                "labels": [d.get(label_field, "Unknown") for d in data],
                "values": [d.get(value_field, 0) for d in data],
                "image_base64": image_bytes.decode('latin-1') if image_bytes else None,
            },
        )

    except Exception as e:
        logger.error(f"Pie chart generation failed: {e}")
        return None


def _render_pie_chart(
    data: List[Dict[str, Any]],
    title: str,
    label_field: str,
    value_field: str,
    dpi: int,
) -> bytes:
    """Render pie chart to bytes (runs in executor)."""
    fig, ax = plt.subplots(figsize=(8, 6))

    labels = [d.get(label_field, "Unknown")[:20] for d in data[:8]]  # Top 8, truncate labels
    values = [d.get(value_field, 0) for d in data[:8]]

    # Calculate percentages for labels
    total = sum(values) or 1
    percentages = [(v / total) * 100 for v in values]

    # Only show labels for slices > 5%
    def autopct_func(pct):
        return f'{pct:.1f}%' if pct > 5 else ''

    colors = plt.cm.Set3.colors[:len(labels)]

    wedges, texts, autotexts = ax.pie(
        values,
        labels=None,  # We'll add legend instead
        autopct=autopct_func,
        colors=colors,
        startangle=90,
    )

    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add legend
    ax.legend(
        wedges,
        [f"{l} (${v:,.0f})" for l, v in zip(labels, values)],
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=9,
    )

    plt.tight_layout()

    # Save to bytes
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    image_bytes = buf.getvalue()
    plt.close(fig)

    return image_bytes


async def _generate_bar_chart(
    data: List[Dict[str, Any]],
    title: str,
    label_field: str,
    value_field: str,
    dpi: int,
) -> Optional[ChartData]:
    """Generate a horizontal bar chart.

    Args:
        data: List of dicts with label and value fields
        title: Chart title
        label_field: Field name for labels
        value_field: Field name for values
        dpi: DPI for image

    Returns:
        ChartData object or None if generation fails
    """
    if not data:
        return None

    try:
        loop = asyncio.get_event_loop()
        image_bytes = await loop.run_in_executor(
            None,
            lambda: _render_bar_chart(data, title, label_field, value_field, dpi)
        )

        return ChartData(
            chart_type="bar",
            title=title,
            data={
                "labels": [d.get(label_field, "Unknown") for d in data],
                "values": [d.get(value_field, 0) for d in data],
                "image_base64": image_bytes.decode('latin-1') if image_bytes else None,
            },
        )

    except Exception as e:
        logger.error(f"Bar chart generation failed: {e}")
        return None


def _render_bar_chart(
    data: List[Dict[str, Any]],
    title: str,
    label_field: str,
    value_field: str,
    dpi: int,
) -> bytes:
    """Render horizontal bar chart to bytes (runs in executor)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = [d.get(label_field, "Unknown")[:25] for d in data]  # Truncate long labels
    values = [d.get(value_field, 0) for d in data]

    # Reverse for horizontal bar (top item at top)
    labels = labels[::-1]
    values = values[::-1]

    colors = plt.cm.Blues([0.3 + 0.5 * (i / len(values)) for i in range(len(values))])

    bars = ax.barh(labels, values, color=colors)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + max(values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f'${value:,.0f}',
            va='center',
            fontsize=9,
        )

    ax.set_xlabel('Amount ($)', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # Save to bytes
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    image_bytes = buf.getvalue()
    plt.close(fig)

    return image_bytes


async def _generate_line_chart(
    data: List[Dict[str, Any]],
    title: str,
    x_field: str,
    y_field: str,
    dpi: int,
) -> Optional[ChartData]:
    """Generate a line chart for trends.

    Args:
        data: List of dicts with x and y fields
        title: Chart title
        x_field: Field name for x-axis
        y_field: Field name for y-axis
        dpi: DPI for image

    Returns:
        ChartData object or None if generation fails
    """
    if not data or len(data) < 2:
        return None

    try:
        loop = asyncio.get_event_loop()
        image_bytes = await loop.run_in_executor(
            None,
            lambda: _render_line_chart(data, title, x_field, y_field, dpi)
        )

        return ChartData(
            chart_type="line",
            title=title,
            data={
                "x_values": [d.get(x_field, "") for d in data],
                "y_values": [d.get(y_field, 0) for d in data],
                "image_base64": image_bytes.decode('latin-1') if image_bytes else None,
            },
        )

    except Exception as e:
        logger.error(f"Line chart generation failed: {e}")
        return None


def _render_line_chart(
    data: List[Dict[str, Any]],
    title: str,
    x_field: str,
    y_field: str,
    dpi: int,
) -> bytes:
    """Render line chart to bytes (runs in executor)."""
    fig, ax = plt.subplots(figsize=(10, 5))

    x_values = [d.get(x_field, "") for d in data]
    y_values = [d.get(y_field, 0) for d in data]

    ax.plot(x_values, y_values, marker='o', linewidth=2, markersize=6, color='#2196F3')
    ax.fill_between(x_values, y_values, alpha=0.3, color='#2196F3')

    # Add data labels
    for x, y in zip(x_values, y_values):
        ax.annotate(
            f'${y:,.0f}',
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=8,
        )

    ax.set_xlabel('Month', fontsize=11)
    ax.set_ylabel('Amount ($)', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Rotate x labels if many data points
    if len(x_values) > 6:
        plt.xticks(rotation=45, ha='right')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    # Save to bytes
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    image_bytes = buf.getvalue()
    plt.close(fig)

    return image_bytes


def render_chart_to_bytes(chart_data: ChartData, dpi: int = 150) -> Optional[bytes]:
    """Re-render a chart from its data to bytes.

    Useful when you have ChartData but need the raw image bytes.

    Args:
        chart_data: ChartData object with chart configuration
        dpi: DPI for output image

    Returns:
        PNG image bytes or None
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    if chart_data.chart_type == "pie":
        data = [
            {"label": l, "value": v}
            for l, v in zip(
                chart_data.data.get("labels", []),
                chart_data.data.get("values", [])
            )
        ]
        return _render_pie_chart(data, chart_data.title, "label", "value", dpi)

    elif chart_data.chart_type == "bar":
        data = [
            {"label": l, "value": v}
            for l, v in zip(
                chart_data.data.get("labels", []),
                chart_data.data.get("values", [])
            )
        ]
        return _render_bar_chart(data, chart_data.title, "label", "value", dpi)

    elif chart_data.chart_type == "line":
        data = [
            {"x": x, "y": y}
            for x, y in zip(
                chart_data.data.get("x_values", []),
                chart_data.data.get("y_values", [])
            )
        ]
        return _render_line_chart(data, chart_data.title, "x", "y", dpi)

    return None
