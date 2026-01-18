"""Output generators for Report Agent.

This module provides functions to generate various output formats:
- PDF reports
- Excel spreadsheets
- Charts (PNG images)
- Dashboard data (JSON)
"""

from .chart_generator import generate_charts
from .pdf_generator import generate_pdf
from .excel_generator import generate_excel
from .dashboard_data import build_dashboard_data

__all__ = [
    "generate_charts",
    "generate_pdf",
    "generate_excel",
    "build_dashboard_data",
]
