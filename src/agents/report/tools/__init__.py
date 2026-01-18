"""Tools module for Report Agent.

The ReportAgent uses internal methods rather than LangGraph tools
for better control over the report generation workflow.

The main operations are:
- Data loading from GCS extracted folder
- Data aggregation into pandas DataFrame
- DuckDB analysis queries
- LLM insight generation
- Chart/PDF/Excel rendering

These are all implemented as methods in ReportAgent.core.py.
"""
