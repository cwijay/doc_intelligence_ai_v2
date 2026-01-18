"""PDF generation for business intelligence reports.

Uses a simple HTML-to-PDF approach with basic styling.
For production, consider using weasyprint or reportlab for richer output.
"""

import asyncio
import base64
import io
import logging
from datetime import datetime
from typing import Optional

from ..schemas import Report, ReportType, Insight

logger = logging.getLogger(__name__)

# Check if weasyprint is available
try:
    from weasyprint import HTML, CSS
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False
    logger.warning("weasyprint not available - using basic PDF generation")


async def generate_pdf(report: Report) -> bytes:
    """Generate PDF report from Report object.

    Args:
        report: Report object with all data

    Returns:
        PDF file as bytes
    """
    # Generate HTML content
    html_content = _generate_html(report)

    if WEASYPRINT_AVAILABLE:
        return await _render_with_weasyprint(html_content)
    else:
        # Fallback: return HTML as bytes (can be opened in browser)
        logger.warning("Returning HTML instead of PDF - install weasyprint for PDF output")
        return html_content.encode('utf-8')


async def _render_with_weasyprint(html_content: str) -> bytes:
    """Render HTML to PDF using weasyprint."""
    loop = asyncio.get_event_loop()

    def render():
        html = HTML(string=html_content)
        pdf_bytes = html.write_pdf()
        return pdf_bytes

    return await loop.run_in_executor(None, render)


def _generate_html(report: Report) -> str:
    """Generate HTML content for the report."""
    # Build sections based on report type
    sections = []

    # Header
    sections.append(_generate_header(report))

    # Executive Summary
    sections.append(_generate_executive_summary(report))

    # Insights
    if report.insights:
        sections.append(_generate_insights_section(report.insights))

    # Charts (if available)
    if report.charts:
        sections.append(_generate_charts_section(report))

    # Data Tables based on report type
    if report.report_type == ReportType.EXPENSE_SUMMARY and report.expense_data:
        sections.append(_generate_expense_tables(report))
    elif report.report_type == ReportType.VENDOR_ANALYSIS and report.vendor_data:
        sections.append(_generate_vendor_tables(report))
    elif report.report_type == ReportType.INVOICE_RECONCILIATION and report.reconciliation_data:
        sections.append(_generate_reconciliation_tables(report))

    # Footer
    sections.append(_generate_footer(report))

    return _wrap_html(sections)


def _wrap_html(sections: list) -> str:
    """Wrap sections in HTML document with styling."""
    css = """
    <style>
        @page {
            margin: 1in;
            @top-right {
                content: "Page " counter(page) " of " counter(pages);
            }
        }

        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 11pt;
            line-height: 1.5;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
        }

        h1 {
            color: #1a73e8;
            font-size: 24pt;
            margin-bottom: 5px;
            border-bottom: 3px solid #1a73e8;
            padding-bottom: 10px;
        }

        h2 {
            color: #202124;
            font-size: 16pt;
            margin-top: 30px;
            margin-bottom: 15px;
            border-bottom: 1px solid #dadce0;
            padding-bottom: 5px;
        }

        h3 {
            color: #5f6368;
            font-size: 13pt;
            margin-top: 20px;
        }

        .subtitle {
            color: #5f6368;
            font-size: 12pt;
            margin-bottom: 20px;
        }

        .summary-box {
            background: #f8f9fa;
            border: 1px solid #dadce0;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }

        .metric {
            display: inline-block;
            text-align: center;
            padding: 15px 25px;
            margin: 5px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }

        .metric-value {
            font-size: 24pt;
            font-weight: bold;
            color: #1a73e8;
        }

        .metric-label {
            font-size: 10pt;
            color: #5f6368;
            margin-top: 5px;
        }

        .insight {
            background: #fff;
            border-left: 4px solid #1a73e8;
            padding: 15px;
            margin: 10px 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        }

        .insight.warning {
            border-left-color: #f9ab00;
        }

        .insight.critical {
            border-left-color: #d93025;
        }

        .insight-title {
            font-weight: bold;
            margin-bottom: 5px;
        }

        .insight-category {
            font-size: 9pt;
            color: #5f6368;
            text-transform: uppercase;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            font-size: 10pt;
        }

        th {
            background: #f1f3f4;
            color: #202124;
            font-weight: 600;
            text-align: left;
            padding: 12px 10px;
            border-bottom: 2px solid #dadce0;
        }

        td {
            padding: 10px;
            border-bottom: 1px solid #e8eaed;
        }

        tr:nth-child(even) {
            background: #fafafa;
        }

        .text-right {
            text-align: right;
        }

        .text-center {
            text-align: center;
        }

        .chart-container {
            text-align: center;
            margin: 20px 0;
            page-break-inside: avoid;
        }

        .chart-container img {
            max-width: 100%;
            height: auto;
        }

        .footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #dadce0;
            font-size: 9pt;
            color: #5f6368;
            text-align: center;
        }

        .page-break {
            page-break-before: always;
        }
    </style>
    """

    content = "\n".join(sections)

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Business Intelligence Report</title>
    {css}
</head>
<body>
{content}
</body>
</html>"""


def _generate_header(report: Report) -> str:
    """Generate report header section."""
    title_map = {
        ReportType.EXPENSE_SUMMARY: "Expense Summary Report",
        ReportType.VENDOR_ANALYSIS: "Vendor Analysis Report",
        ReportType.INVOICE_RECONCILIATION: "Invoice Reconciliation Report",
        ReportType.SPEND_TRENDS: "Spending Trends Report",
        ReportType.CASH_FLOW_PROJECTION: "Cash Flow Projection Report",
        ReportType.TAX_PREPARATION: "Tax Preparation Report",
    }

    title = title_map.get(report.report_type, "Business Intelligence Report")
    date_range = f"{report.date_range.start.strftime('%B %d, %Y')} - {report.date_range.end.strftime('%B %d, %Y')}"

    return f"""
    <h1>{title}</h1>
    <p class="subtitle">
        Period: {date_range}<br>
        Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
    </p>
    """


def _generate_executive_summary(report: Report) -> str:
    """Generate executive summary section."""
    if not report.summary:
        return ""

    s = report.summary
    currency = report.options.currency_symbol if report.options else "$"

    return f"""
    <h2>Executive Summary</h2>
    <div class="summary-box">
        <div class="metric">
            <div class="metric-value">{currency}{s.total_amount:,.0f}</div>
            <div class="metric-label">Total Amount</div>
        </div>
        <div class="metric">
            <div class="metric-value">{s.document_count}</div>
            <div class="metric-label">Documents</div>
        </div>
        <div class="metric">
            <div class="metric-value">{s.record_count}</div>
            <div class="metric-label">Records</div>
        </div>
        <div class="metric">
            <div class="metric-value">{s.vendor_count}</div>
            <div class="metric-label">Vendors</div>
        </div>
    </div>
    <p>
        <strong>Top Category:</strong> {s.top_category or 'N/A'}<br>
        <strong>Top Vendor:</strong> {s.top_vendor or 'N/A'}<br>
        <strong>Average Transaction:</strong> {currency}{s.avg_transaction_amount:,.2f}
    </p>
    """


def _generate_insights_section(insights: list) -> str:
    """Generate insights section."""
    if not insights:
        return ""

    items = []
    for insight in insights:
        severity_class = insight.severity if insight.severity in ['warning', 'critical'] else ''
        items.append(f"""
        <div class="insight {severity_class}">
            <div class="insight-category">{insight.category}</div>
            <div class="insight-title">{insight.title}</div>
            <div class="insight-description">{insight.description}</div>
        </div>
        """)

    return f"""
    <h2>Key Insights</h2>
    {"".join(items)}
    """


def _generate_charts_section(report: Report) -> str:
    """Generate charts section."""
    if not report.charts:
        return ""

    items = []
    for chart in report.charts:
        # Check if we have base64 image data
        image_data = chart.data.get("image_base64") if chart.data else None
        if image_data:
            # Convert to proper base64 if needed
            if isinstance(image_data, str):
                img_src = f"data:image/png;base64,{base64.b64encode(image_data.encode('latin-1')).decode('ascii')}"
            else:
                img_src = f"data:image/png;base64,{base64.b64encode(image_data).decode('ascii')}"

            items.append(f"""
            <div class="chart-container">
                <h3>{chart.title}</h3>
                <img src="{img_src}" alt="{chart.title}">
            </div>
            """)

    if not items:
        return ""

    return f"""
    <h2>Charts</h2>
    {"".join(items)}
    """


def _generate_expense_tables(report: Report) -> str:
    """Generate expense-specific tables."""
    if not report.expense_data:
        return ""

    data = report.expense_data
    currency = report.options.currency_symbol if report.options else "$"

    # Category breakdown table
    category_rows = []
    for item in data.by_category[:10]:
        category_rows.append(f"""
        <tr>
            <td>{item.category}</td>
            <td class="text-right">{currency}{item.amount:,.2f}</td>
            <td class="text-right">{item.percentage:.1f}%</td>
            <td class="text-right">{item.count}</td>
        </tr>
        """)

    # Vendor breakdown table
    vendor_rows = []
    for item in data.by_vendor[:10]:
        vendor_rows.append(f"""
        <tr>
            <td>{item.vendor}</td>
            <td class="text-right">{currency}{item.amount:,.2f}</td>
            <td class="text-right">{item.invoice_count}</td>
            <td class="text-right">{currency}{item.avg_amount:,.2f}</td>
        </tr>
        """)

    return f"""
    <h2>Spending by Category</h2>
    <table>
        <thead>
            <tr>
                <th>Category</th>
                <th class="text-right">Amount</th>
                <th class="text-right">% of Total</th>
                <th class="text-right">Transactions</th>
            </tr>
        </thead>
        <tbody>
            {"".join(category_rows)}
        </tbody>
    </table>

    <h2>Top Vendors</h2>
    <table>
        <thead>
            <tr>
                <th>Vendor</th>
                <th class="text-right">Total Spend</th>
                <th class="text-right">Invoices</th>
                <th class="text-right">Avg per Invoice</th>
            </tr>
        </thead>
        <tbody>
            {"".join(vendor_rows)}
        </tbody>
    </table>
    """


def _generate_vendor_tables(report: Report) -> str:
    """Generate vendor analysis tables."""
    if not report.vendor_data:
        return ""

    data = report.vendor_data
    currency = report.options.currency_symbol if report.options else "$"

    # Vendor rankings table
    rows = []
    for i, item in enumerate(data.vendor_rankings[:15], 1):
        rows.append(f"""
        <tr>
            <td>{i}</td>
            <td>{item.vendor}</td>
            <td class="text-right">{currency}{item.amount:,.2f}</td>
            <td class="text-right">{item.percentage:.1f}%</td>
            <td class="text-right">{item.invoice_count}</td>
        </tr>
        """)

    concentration = data.concentration_metrics or {}

    return f"""
    <h2>Vendor Concentration Analysis</h2>
    <div class="summary-box">
        <p><strong>Top 3 Vendors:</strong> {concentration.get('top_3_percentage', 0):.1f}% of total spend</p>
        <p><strong>Total Vendors:</strong> {concentration.get('vendor_count', 0)}</p>
        <p><strong>Average per Vendor:</strong> {currency}{concentration.get('avg_per_vendor', 0):,.2f}</p>
    </div>

    <h2>Vendor Rankings</h2>
    <table>
        <thead>
            <tr>
                <th>Rank</th>
                <th>Vendor</th>
                <th class="text-right">Total Spend</th>
                <th class="text-right">% of Total</th>
                <th class="text-right">Invoices</th>
            </tr>
        </thead>
        <tbody>
            {"".join(rows)}
        </tbody>
    </table>
    """


def _generate_reconciliation_tables(report: Report) -> str:
    """Generate reconciliation tables."""
    if not report.reconciliation_data:
        return ""

    data = report.reconciliation_data
    summary = data.summary or {}

    return f"""
    <h2>Reconciliation Summary</h2>
    <div class="summary-box">
        <div class="metric">
            <div class="metric-value">{summary.get('total_records', 0)}</div>
            <div class="metric-label">Total Records</div>
        </div>
        <div class="metric">
            <div class="metric-value">{summary.get('matched_count', 0)}</div>
            <div class="metric-label">Matched</div>
        </div>
        <div class="metric">
            <div class="metric-value">{summary.get('missing_po_count', 0)}</div>
            <div class="metric-label">Missing PO</div>
        </div>
    </div>
    """


def _generate_footer(report: Report) -> str:
    """Generate report footer."""
    return f"""
    <div class="footer">
        Report ID: {report.id}<br>
        Generated by Document Intelligence AI<br>
        Processing time: {report.processing_time_ms or 0}ms
    </div>
    """
