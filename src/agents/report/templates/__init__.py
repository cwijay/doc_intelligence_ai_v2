"""Report templates module.

Provides prompt templates and report structures for different report types.
"""

from typing import Any, Dict

from ..schemas import ReportType


def get_insights_prompt(
    report_type: ReportType,
    analysis_results: Dict[str, Any],
    max_insights: int = 5,
) -> str:
    """Get the insights generation prompt for a report type.

    Args:
        report_type: Type of report
        analysis_results: Results from analysis queries
        max_insights: Maximum number of insights to generate

    Returns:
        Formatted prompt string
    """
    if report_type == ReportType.EXPENSE_SUMMARY:
        return _get_expense_summary_prompt(analysis_results, max_insights)
    elif report_type == ReportType.VENDOR_ANALYSIS:
        return _get_vendor_analysis_prompt(analysis_results, max_insights)
    elif report_type == ReportType.INVOICE_RECONCILIATION:
        return _get_reconciliation_prompt(analysis_results, max_insights)
    elif report_type == ReportType.SPEND_TRENDS:
        return _get_spend_trends_prompt(analysis_results, max_insights)
    elif report_type == ReportType.CASH_FLOW_PROJECTION:
        return _get_cash_flow_prompt(analysis_results, max_insights)
    elif report_type == ReportType.TAX_PREPARATION:
        return _get_tax_prep_prompt(analysis_results, max_insights)
    else:
        return _get_generic_prompt(analysis_results, max_insights)


def _format_data_summary(analysis_results: Dict[str, Any]) -> str:
    """Format analysis results as a readable summary for the LLM."""
    lines = []

    if "total_amount" in analysis_results:
        lines.append(f"Total Amount: ${analysis_results['total_amount']:,.2f}")

    if "record_count" in analysis_results:
        lines.append(f"Total Records: {analysis_results['record_count']}")

    if "vendor_count" in analysis_results:
        lines.append(f"Unique Vendors: {analysis_results['vendor_count']}")

    if "by_category" in analysis_results and analysis_results["by_category"]:
        lines.append("\nTop Categories:")
        for item in analysis_results["by_category"][:5]:
            cat = item.get("category", "Unknown")
            total = item.get("total", 0)
            count = item.get("count", 0)
            lines.append(f"  - {cat}: ${total:,.2f} ({count} transactions)")

    if "by_vendor" in analysis_results and analysis_results["by_vendor"]:
        lines.append("\nTop Vendors:")
        for item in analysis_results["by_vendor"][:5]:
            vendor = item.get("vendor", "Unknown")
            total = item.get("total", 0)
            count = item.get("invoice_count", 0)
            lines.append(f"  - {vendor}: ${total:,.2f} ({count} invoices)")

    if "monthly_trends" in analysis_results and analysis_results["monthly_trends"]:
        lines.append("\nMonthly Trends:")
        for item in analysis_results["monthly_trends"][-6:]:
            month = item.get("month", "")
            total = item.get("total", 0)
            lines.append(f"  - {month}: ${total:,.2f}")

    return "\n".join(lines)


def _get_expense_summary_prompt(
    analysis_results: Dict[str, Any],
    max_insights: int,
) -> str:
    """Generate prompt for expense summary insights."""
    data_summary = _format_data_summary(analysis_results)

    return f"""You are a business intelligence analyst helping a small business owner understand their expenses.

Analyze the following expense data and provide {max_insights} actionable insights:

{data_summary}

For each insight, provide:
1. A clear, specific observation about the data
2. Why this matters for the business
3. A concrete recommendation if applicable

Focus on:
- Unusual spending patterns or anomalies
- Cost-saving opportunities
- Budget concerns (categories that seem high)
- Vendor concentration risks
- Seasonal or trending patterns

Respond in JSON format as an array of objects with these fields:
- "category": one of "spending", "trend", "anomaly", "recommendation", "risk"
- "title": brief title (max 10 words)
- "description": detailed explanation (2-3 sentences)
- "severity": one of "info", "warning", "critical"
- "data_points": optional dict with supporting numbers

Example:
```json
[
  {{
    "category": "spending",
    "title": "High Office Supplies Spending",
    "description": "Office supplies account for 34% of total spend, which is above typical benchmarks for service businesses (usually 10-15%). Consider bulk purchasing or negotiating vendor discounts.",
    "severity": "warning",
    "data_points": {{"percentage": 34, "benchmark": 15}}
  }}
]
```

Provide exactly {max_insights} insights as a JSON array:"""


def _get_vendor_analysis_prompt(
    analysis_results: Dict[str, Any],
    max_insights: int,
) -> str:
    """Generate prompt for vendor analysis insights."""
    data_summary = _format_data_summary(analysis_results)

    # Calculate concentration metrics
    by_vendor = analysis_results.get("by_vendor", [])
    total = analysis_results.get("total_amount", 1)
    top_3_pct = sum(v.get("total", 0) for v in by_vendor[:3]) / total * 100 if total > 0 else 0

    return f"""You are a business intelligence analyst helping a small business optimize vendor relationships.

Analyze the following vendor data and provide {max_insights} actionable insights:

{data_summary}

Vendor Concentration: Top 3 vendors account for {top_3_pct:.1f}% of total spend.

For each insight, provide analysis on:
- Vendor concentration risks (dependency on few vendors)
- Negotiation opportunities (high-volume vendors)
- Price comparison opportunities
- Payment terms optimization
- Vendor diversification recommendations

Respond in JSON format as an array of objects with these fields:
- "category": one of "risk", "opportunity", "trend", "recommendation"
- "title": brief title (max 10 words)
- "description": detailed explanation (2-3 sentences)
- "severity": one of "info", "warning", "critical"
- "data_points": optional dict with supporting numbers

Provide exactly {max_insights} insights as a JSON array:"""


def _get_reconciliation_prompt(
    analysis_results: Dict[str, Any],
    max_insights: int,
) -> str:
    """Generate prompt for invoice reconciliation insights."""
    data_summary = _format_data_summary(analysis_results)
    reconciliation = analysis_results.get("reconciliation", {})

    return f"""You are a business intelligence analyst helping with accounts payable reconciliation.

Analyze the following reconciliation data and provide {max_insights} actionable insights:

{data_summary}

Reconciliation Status:
- Matched invoices: {reconciliation.get('matched_count', 0)}
- Missing PO references: {reconciliation.get('missing_po_count', 0)}

For each insight, focus on:
- Process improvements for PO compliance
- Common reconciliation issues
- Discrepancy patterns
- Control recommendations

Respond in JSON format as an array of objects with these fields:
- "category": one of "process", "risk", "compliance", "recommendation"
- "title": brief title (max 10 words)
- "description": detailed explanation (2-3 sentences)
- "severity": one of "info", "warning", "critical"

Provide exactly {max_insights} insights as a JSON array:"""


def _get_spend_trends_prompt(
    analysis_results: Dict[str, Any],
    max_insights: int,
) -> str:
    """Generate prompt for spend trends insights."""
    data_summary = _format_data_summary(analysis_results)

    return f"""You are a business intelligence analyst helping a small business understand spending trends.

Analyze the following trend data and provide {max_insights} actionable insights:

{data_summary}

For each insight, focus on:
- Month-over-month changes
- Seasonal patterns
- Growth or decline trends
- Anomalies or spikes
- Budget forecasting implications

Respond in JSON format as an array of objects with these fields:
- "category": one of "trend", "seasonal", "anomaly", "forecast", "recommendation"
- "title": brief title (max 10 words)
- "description": detailed explanation (2-3 sentences)
- "severity": one of "info", "warning", "critical"
- "data_points": optional dict with supporting numbers

Provide exactly {max_insights} insights as a JSON array:"""


def _get_cash_flow_prompt(
    analysis_results: Dict[str, Any],
    max_insights: int,
) -> str:
    """Generate prompt for cash flow projection insights."""
    data_summary = _format_data_summary(analysis_results)

    return f"""You are a business intelligence analyst helping a small business with cash flow planning.

Analyze the following data and provide {max_insights} actionable insights:

{data_summary}

For each insight, focus on:
- Expected payment timing
- Cash flow gaps or surpluses
- Seasonal cash needs
- Working capital recommendations

Respond in JSON format as an array of objects with these fields:
- "category": one of "inflow", "outflow", "timing", "recommendation"
- "title": brief title (max 10 words)
- "description": detailed explanation (2-3 sentences)
- "severity": one of "info", "warning", "critical"

Provide exactly {max_insights} insights as a JSON array:"""


def _get_tax_prep_prompt(
    analysis_results: Dict[str, Any],
    max_insights: int,
) -> str:
    """Generate prompt for tax preparation insights."""
    data_summary = _format_data_summary(analysis_results)

    return f"""You are a business intelligence analyst helping a small business prepare for tax filing.

Analyze the following expense data and provide {max_insights} actionable insights:

{data_summary}

For each insight, focus on:
- Deductible expense categories
- Missing documentation
- Category organization for tax reporting
- Potential deduction opportunities

Note: This is informational only - recommend consulting a tax professional.

Respond in JSON format as an array of objects with these fields:
- "category": one of "deduction", "documentation", "organization", "recommendation"
- "title": brief title (max 10 words)
- "description": detailed explanation (2-3 sentences)
- "severity": one of "info", "warning", "critical"

Provide exactly {max_insights} insights as a JSON array:"""


def _get_generic_prompt(
    analysis_results: Dict[str, Any],
    max_insights: int,
) -> str:
    """Generate generic insights prompt."""
    data_summary = _format_data_summary(analysis_results)

    return f"""You are a business intelligence analyst helping a small business owner.

Analyze the following data and provide {max_insights} actionable insights:

{data_summary}

For each insight, provide:
1. A clear observation about the data
2. Why this matters
3. A recommendation if applicable

Respond in JSON format as an array of objects with these fields:
- "category": one of "general", "trend", "anomaly", "recommendation"
- "title": brief title (max 10 words)
- "description": detailed explanation (2-3 sentences)
- "severity": one of "info", "warning", "critical"

Provide exactly {max_insights} insights as a JSON array:"""
