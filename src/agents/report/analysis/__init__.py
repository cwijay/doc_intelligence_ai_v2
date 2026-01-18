"""Analysis module for Report Agent.

Data aggregation and DuckDB query execution are implemented
directly in the ReportAgent core for simplicity.

This module provides additional analysis utilities if needed.
"""

# Analysis logic is implemented in ReportAgent.core.py:
# - _aggregate_data(): Converts extracted records to pandas DataFrame
# - _run_analysis(): Executes DuckDB queries on the DataFrame
# - _run_reconciliation_queries(): Invoice reconciliation specific queries
