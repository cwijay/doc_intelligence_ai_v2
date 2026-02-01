"""Report templates module.

Provides prompt templates and report structures for different report types.
"""

from typing import Any, Dict, Optional

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


# =============================================================================
# DOCUMENT DATA EXTRACTION PROMPTS
# =============================================================================


def get_document_data_extraction_prompt(
    document_content: str,
    filename: str,
    max_chars: int = 50000,
) -> str:
    """Generate prompt for extracting structured data from parsed document content.

    This prompt instructs the LLM to extract standardized fields from document
    text (typically Markdown from LlamaParse) for use in report analysis.

    Args:
        document_content: The parsed document content (usually Markdown)
        filename: Original filename for context
        max_chars: Maximum characters to include from document

    Returns:
        Formatted prompt string
    """
    # Truncate if needed
    if len(document_content) > max_chars:
        document_content = document_content[:max_chars] + "\n\n[... content truncated ...]"

    return f"""You are a data extraction specialist. Extract structured financial/business data from the following document.

**Document**: {filename}

**Document Content**:
```
{document_content}
```

**Instructions**:
Analyze the document and extract all relevant financial/business data. Return a JSON object with the following fields (include only fields that have actual values in the document):

Required fields (always include if identifiable):
- "document_type": string - Type of document (e.g., "invoice", "receipt", "contract", "purchase_order", "expense_report", "statement", "unknown")
- "source_file": string - The filename "{filename}"

Financial fields (include if present):
- "amount" or "total": number - The main monetary amount/total
- "subtotal": number - Subtotal before tax/fees if separate from total
- "tax": number - Tax amount if listed
- "currency": string - Currency code (e.g., "USD", "EUR") or symbol

Date fields (include if present):
- "date": string (YYYY-MM-DD format) - Primary date (invoice date, transaction date, etc.)
- "due_date": string (YYYY-MM-DD format) - Due date if applicable
- "payment_date": string (YYYY-MM-DD format) - Payment/receipt date if different

Entity fields (include if present):
- "vendor" or "merchant": string - Name of vendor/supplier/merchant
- "vendor_address": string - Vendor's address
- "customer": string - Customer/buyer name if not the document owner
- "invoice_number": string - Invoice or reference number
- "po_number": string - Purchase order number if referenced

Categorization (include if determinable):
- "category": string - Expense category (e.g., "Office Supplies", "Travel", "Software", "Utilities", "Professional Services")
- "payment_method": string - How payment was made (e.g., "Credit Card", "Check", "ACH", "Wire")
- "payment_status": string - Status (e.g., "paid", "pending", "overdue")

Line items (include if document has itemized entries):
- "line_items": array of objects, each with:
  - "description": string
  - "quantity": number
  - "unit_price": number
  - "amount": number

Additional notes:
- "notes": string - Any important notes or special terms

**Important**:
1. Only include fields that have actual values found in the document
2. Use null for truly missing values, but prefer to omit the field entirely
3. Parse dates into YYYY-MM-DD format when possible
4. Convert amounts to numbers (remove currency symbols, commas)
5. Be conservative - only extract what's clearly stated in the document

Respond with ONLY a valid JSON object, no markdown code blocks or explanation:"""


def get_batch_extraction_prompt(
    documents: list[tuple[str, str]],
    max_chars_per_doc: int = 10000,
) -> str:
    """Generate prompt for batch extracting data from multiple documents.

    Args:
        documents: List of (filename, content) tuples
        max_chars_per_doc: Maximum characters per document

    Returns:
        Formatted prompt string
    """
    doc_sections = []
    for i, (filename, content) in enumerate(documents, 1):
        truncated = content[:max_chars_per_doc] if len(content) > max_chars_per_doc else content
        doc_sections.append(f"""
--- DOCUMENT {i}: {filename} ---
{truncated}
--- END DOCUMENT {i} ---
""")

    all_docs = "\n".join(doc_sections)

    return f"""You are a data extraction specialist. Extract structured financial/business data from the following {len(documents)} documents.

{all_docs}

**Instructions**:
For EACH document, extract all relevant financial/business data. Return a JSON array with one object per document.

Each object should contain (include only fields with actual values):
- "source_file": string - The document filename (REQUIRED)
- "document_type": string - Type (invoice, receipt, contract, etc.)
- "amount" or "total": number - Main monetary amount
- "date": string (YYYY-MM-DD) - Primary date
- "vendor" or "merchant": string - Vendor/supplier name
- "category": string - Expense category if determinable
- "invoice_number": string - Reference number if present
- Other relevant fields as described in standard extraction

**Important**:
1. Return a JSON array with exactly {len(documents)} objects (one per document)
2. Each object MUST have "source_file" matching the document filename
3. Only include fields with actual values from each document
4. Parse dates to YYYY-MM-DD, amounts to numbers

Respond with ONLY a valid JSON array, no markdown or explanation:"""
