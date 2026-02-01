"""Output generators for Report Agent.

This module provides functions to generate various output formats:
- PDF reports
- Excel spreadsheets
- Charts (data only - rendered on frontend with Recharts)
- Dashboard data (JSON)

Note: Uses lazy imports to avoid loading heavy dependencies (like WeasyPrint)
until they are actually needed.
"""

# Lazy imports to avoid loading dependencies until needed


def generate_charts(*args, **kwargs):
    """Generate chart data structures for frontend rendering (no matplotlib needed)."""
    from .chart_generator import generate_charts as _generate_charts
    return _generate_charts(*args, **kwargs)


def generate_pdf(*args, **kwargs):
    """Generate PDF - lazy import."""
    from .pdf_generator import generate_pdf as _generate_pdf
    return _generate_pdf(*args, **kwargs)


def generate_excel(*args, **kwargs):
    """Generate Excel - lazy import."""
    from .excel_generator import generate_excel as _generate_excel
    return _generate_excel(*args, **kwargs)


def build_dashboard_data(*args, **kwargs):
    """Build dashboard data - lazy import."""
    from .dashboard_data import build_dashboard_data as _build_dashboard_data
    return _build_dashboard_data(*args, **kwargs)


__all__ = [
    "generate_charts",
    "generate_pdf",
    "generate_excel",
    "build_dashboard_data",
]
