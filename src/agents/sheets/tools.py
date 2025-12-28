"""Tools for Sheets Agent using LangChain and DuckDB."""

import os
import tempfile
import time
import json
import asyncio
import functools
import threading
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from pathlib import Path
import pandas as pd
import duckdb
import logging

from langchain_core.tools import BaseTool

from src.core.executors import get_executors

from src.utils.timer_utils import elapsed_ms
from .gcs_loader import load_dataframe, validate_file_path, get_file_extension, get_excel_sheet_names

if TYPE_CHECKING:
    from .config import SheetsAgentConfig

logger = logging.getLogger(__name__)

# Security: Dangerous SQL keywords to block
DANGEROUS_SQL_KEYWORDS = frozenset([
    'DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE',
    'TRUNCATE', 'EXEC', 'EXECUTE', 'GRANT', 'REVOKE'
])

# Performance: Default max rows for query results
DEFAULT_MAX_RESULT_ROWS = 10000


class DuckDBPool:
    """Connection pool for DuckDB to improve performance."""

    def __init__(self, max_connections: int = 5):
        self._pool: List[duckdb.DuckDBPyConnection] = []
        self._max = max_connections
        self._lock = threading.Lock()

    def get_connection(self) -> duckdb.DuckDBPyConnection:
        """Get a connection from the pool or create a new one."""
        with self._lock:
            if self._pool:
                return self._pool.pop()
        return duckdb.connect(':memory:')

    def return_connection(self, conn: duckdb.DuckDBPyConnection):
        """Return a connection to the pool."""
        with self._lock:
            if len(self._pool) < self._max:
                # Clear any registered tables before returning to pool
                try:
                    conn.execute("SELECT 1")  # Test connection is valid
                    self._pool.append(conn)
                except Exception:
                    try:
                        conn.close()
                    except Exception:
                        pass
            else:
                try:
                    conn.close()
                except Exception:
                    pass

    def close_all(self):
        """Close all connections in the pool."""
        with self._lock:
            for conn in self._pool:
                try:
                    conn.close()
                except Exception:
                    pass
            self._pool.clear()


# Global connection pool instance
_duckdb_pool: Optional[DuckDBPool] = None


def get_duckdb_pool(pool_size: int = 5) -> DuckDBPool:
    """Get or create the global DuckDB connection pool."""
    global _duckdb_pool
    if _duckdb_pool is None:
        _duckdb_pool = DuckDBPool(max_connections=pool_size)
    return _duckdb_pool


def validate_sql_query(query: str) -> tuple[bool, Optional[str]]:
    """
    Validate SQL query for dangerous operations.

    Returns:
        Tuple of (is_valid, error_message)
    """
    query_upper = query.upper()
    for keyword in DANGEROUS_SQL_KEYWORDS:
        # Check for keyword as a whole word
        if f' {keyword} ' in f' {query_upper} ' or query_upper.startswith(f'{keyword} '):
            return False, f"Dangerous SQL operation '{keyword}' is not allowed"
    return True, None


def add_query_limit(query: str, max_rows: int = DEFAULT_MAX_RESULT_ROWS) -> str:
    """Add LIMIT clause to query if not present to prevent memory exhaustion."""
    query_upper = query.upper().strip()
    if 'LIMIT' not in query_upper:
        # Remove trailing semicolon if present
        query = query.rstrip().rstrip(';')
        return f"{query} LIMIT {max_rows}"
    return query


class FilePreviewTool(BaseTool):
    """Tool for previewing file structure and metadata."""

    name: str = "load_file_preview"
    description: str = """Load and preview Excel/CSV file structure and metadata.
    Input should be a file path (local path or GCS URI like 'gs://bucket/path/file.xlsx').
    Returns file structure, column names, data types, and sample data."""

    _config: Any = None
    _file_cache: Any = None

    def __init__(self, config: "SheetsAgentConfig", file_cache: Optional[Any] = None):
        super().__init__()
        self._config = config
        self._file_cache = file_cache

    def _run(self, file_path: str) -> str:
        """Load and analyze file structure."""
        try:
            logger.info(f"Loading file preview for: {file_path}")

            # Validate file path (supports both local and GCS)
            is_valid, error = validate_file_path(file_path)
            if not is_valid:
                return f"Error: {error}"

            # Determine file type
            file_ext = get_file_extension(file_path)

            # Get sheet names for Excel files
            sheet_names = None
            if file_ext in ['.xlsx', '.xls']:
                try:
                    sheet_names = get_excel_sheet_names(file_path)
                except Exception as e:
                    logger.warning(f"Could not get sheet names: {e}")

            # Try cache first
            df = None
            if self._file_cache:
                cached_df = self._file_cache.get(file_path)
                if cached_df is not None:
                    logger.info(f"Using cached data for preview: {file_path}")
                    df = cached_df.head(100)  # Preview from cache

            # Load if not cached
            if df is None:
                df, metadata = load_dataframe(file_path, nrows=100)
            else:
                metadata = {"source": "cache"}

            # Generate file preview information
            preview_info = {
                "file_path": file_path,
                "file_type": file_ext,
                "source": metadata.get("source", "unknown"),
                "sheet_names": sheet_names,
                "shape": {"rows": len(df), "columns": len(df.columns)},
                "columns": [{"name": col, "type": str(df[col].dtype)} for col in df.columns],
                "sample_data": df.head(self._config.preview_rows).to_dict('records'),
                "null_counts": df.isnull().sum().to_dict(),
                "memory_usage": metadata.get("memory_usage_bytes", int(df.memory_usage(deep=True).sum()))
            }

            return json.dumps(preview_info, indent=2, default=str)

        except Exception as e:
            logger.error(f"Error loading file preview for {file_path}: {e}")
            return f"Error loading file preview: {str(e)}"

    async def _arun(self, file_path: str) -> str:
        """Async version of _run using dedicated query executor pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            get_executors().query_executor,
            functools.partial(self._run, file_path)
        )


class CrossFileQueryTool(BaseTool):
    """Tool for executing queries across multiple files using DuckDB."""

    name: str = "execute_cross_file_query"
    description: str = """Execute SQL queries across multiple Excel/CSV files using DuckDB.
    Input should be a JSON object with 'file_paths' (list of local paths or GCS URIs) and 'query' (SQL query).
    Files are automatically loaded as tables named 'file_0', 'file_1', etc.
    Use this for multi-file analysis (2+ files).
    Returns query results as JSON."""

    _config: Any = None
    _file_cache: Any = None

    def __init__(self, config: "SheetsAgentConfig", file_cache: Optional[Any] = None):
        super().__init__()
        self._config = config
        self._file_cache = file_cache

    def _load_file_cached(self, file_path: str) -> tuple:
        """Load file with cache support."""
        if self._file_cache:
            cached_df = self._file_cache.get(file_path)
            if cached_df is not None:
                logger.info(f"Using cached data for: {file_path}")
                return cached_df, {"source": "cache"}

        df, metadata = load_dataframe(file_path)

        # Cache the loaded DataFrame
        if self._file_cache:
            self._file_cache.put(file_path, df)

        return df, metadata

    def _run(self, input_json: str) -> str:
        """Execute cross-file query."""
        try:
            # Parse input
            input_data = json.loads(input_json)
            file_paths = input_data.get('file_paths', [])
            query = input_data.get('query', '')

            if not file_paths or not query:
                return "Error: Both 'file_paths' and 'query' are required"

            # Security: Validate SQL query
            is_valid, error_msg = validate_sql_query(query)
            if not is_valid:
                return f"Error: {error_msg}"

            # Performance: Add LIMIT if not present
            query = add_query_limit(query)

            logger.info(f"Executing cross-file query across {len(file_paths)} files")

            # For single files, use pandas directly (more efficient)
            if len(file_paths) == 1:
                return self._handle_single_file(file_paths[0], query)

            # Use connection pool for better performance
            pool = get_duckdb_pool()
            conn = pool.get_connection()
            table_names = []
            try:

                for i, file_path in enumerate(file_paths):
                    try:
                        # Validate file path (supports both local and GCS)
                        is_valid, error = validate_file_path(file_path)
                        if not is_valid:
                            logger.warning(f"Invalid file path: {file_path} - {error}")
                            continue

                        # Determine table name and load data using cached loader
                        table_name = f"file_{i}"
                        df, metadata = self._load_file_cached(file_path)

                        # Register DataFrame with DuckDB
                        conn.register(table_name, df)
                        table_names.append({
                            "table_name": table_name,
                            "file_path": file_path,
                            "source": metadata.get("source", "unknown"),
                            "shape": list(df.shape),
                            "columns": list(df.columns)
                        })

                        logger.info(f"Loaded {file_path} as {table_name}")

                    except Exception as e:
                        logger.error(f"Error loading {file_path}: {e}")
                        continue

                if not table_names:
                    return "Error: No files could be loaded successfully"

                # Execute query
                start_time = time.time()
                result = conn.execute(query).fetchall()
                execution_time = elapsed_ms(start_time)

                # Get column names
                columns = [desc[0] for desc in conn.description] if conn.description else []

                # Format results
                result_data = [dict(zip(columns, row)) for row in result]

                response = {
                    "success": True,
                    "query": query,
                    "tables_loaded": table_names,
                    "results": result_data,
                    "row_count": len(result_data),
                    "execution_time_ms": execution_time,
                    "columns": columns
                }

                return json.dumps(response, indent=2, default=str)

            finally:
                # Return connection to pool instead of closing
                if conn is not None:
                    pool.return_connection(conn)

        except json.JSONDecodeError:
            return "Error: Input must be valid JSON with 'file_paths' and 'query' fields"
        except Exception as e:
            logger.error(f"Error executing cross-file query: {e}")
            return f"Error executing query: {str(e)}"

    def _handle_single_file(self, file_path: str, query: str) -> str:
        """Handle single file queries with pandas."""
        pool = get_duckdb_pool()
        conn = None
        try:
            logger.info(f"Processing single file with pandas: {file_path}")

            # Validate file path (supports both local and GCS)
            is_valid, error = validate_file_path(file_path)
            if not is_valid:
                return f"Error: {error}"

            # Load file into DataFrame using cached loader
            df, metadata = self._load_file_cached(file_path)

            logger.info(f"Loaded single file from {metadata.get('source', 'unknown')} with shape: {df.shape}")

            # Use connection pool for DuckDB
            conn = pool.get_connection()
            conn.register('file_0', df)

            start_time = time.time()
            result = conn.execute(query).fetchall()
            execution_time = elapsed_ms(start_time)

            columns = [desc[0] for desc in conn.description] if conn.description else []
            result_data = [dict(zip(columns, row)) for row in result]

            response = {
                "success": True,
                "query": query,
                "tables_loaded": [{
                    "table_name": "file_0",
                    "file_path": file_path,
                    "source": metadata.get("source", "unknown"),
                    "shape": list(df.shape),
                    "columns": list(df.columns)
                }],
                "results": result_data,
                "row_count": len(result_data),
                "execution_time_ms": execution_time,
                "columns": columns
            }

            return json.dumps(response, indent=2, default=str)

        except Exception as e:
            logger.error(f"Error processing single file: {e}")
            return f"Error processing single file: {str(e)}"
        finally:
            if conn is not None:
                pool.return_connection(conn)

    async def _arun(self, input_json: str) -> str:
        """Async version of _run using dedicated query executor pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            get_executors().query_executor,
            functools.partial(self._run, input_json)
        )


class SmartAnalysisTool(BaseTool):
    """Smart tool that combines file preview and analysis in one efficient call."""

    name: str = "smart_analysis"
    description: str = """Analyze Excel/CSV files efficiently with preview and query in one call.
    Input should be a JSON object with 'file_path' (local path or GCS URI) and 'query' (what you want to find).
    This tool loads the file once and provides both preview and analysis results.
    Use this as the primary tool for single file analysis - much faster than separate preview + query calls."""

    _config: Any = None
    _file_cache: Any = None

    def __init__(self, config: "SheetsAgentConfig", file_cache: Optional[Any] = None):
        super().__init__()
        self._config = config
        self._file_cache = file_cache

    def _load_file(self, file_path: str) -> pd.DataFrame:
        """Load file using cache if available. Supports both local and GCS paths."""
        # Try cache first
        if self._file_cache:
            cached_df = self._file_cache.get(file_path)
            if cached_df is not None:
                logger.info(f"Using cached data for {file_path}")
                return cached_df

        # Validate file path (supports both local and GCS)
        is_valid, error = validate_file_path(file_path)
        if not is_valid:
            raise ValueError(error)

        # Load file using GCS-aware loader
        df, metadata = load_dataframe(file_path)

        # Cache the loaded DataFrame
        if self._file_cache:
            self._file_cache.put(file_path, df)

        logger.info(f"Loaded and cached {file_path} from {metadata.get('source', 'unknown')} with shape: {df.shape}")
        return df

    def _run(self, input_json: str) -> str:
        """Execute smart analysis with preview and query."""
        try:
            # Parse input
            input_data = json.loads(input_json)
            file_path = input_data.get('file_path', '')
            query = input_data.get('query', '')

            if not file_path:
                return "Error: 'file_path' is required"

            logger.info(f"Smart analysis for: {file_path}")

            # Load file (with caching)
            df = self._load_file(file_path)

            # Generate comprehensive analysis
            analysis_result = {
                "success": True,
                "file_path": file_path,
                "query": query,

                # File preview information
                "preview": {
                    "shape": {"rows": len(df), "columns": len(df.columns)},
                    "columns": list(df.columns),
                    "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
                    "sample_data": df.head(self._config.sample_rows).to_dict('records'),
                    "null_counts": df.isnull().sum().to_dict()
                },

                # Query-specific analysis
                "analysis": self._analyze_query(df, query, file_path),

                # Performance info
                "cached": self._file_cache is not None and self._file_cache.get(file_path) is not None
            }

            return json.dumps(analysis_result, indent=2, default=str)

        except json.JSONDecodeError:
            return "Error: Input must be valid JSON with 'file_path' field"
        except Exception as e:
            logger.error(f"Error in smart analysis: {e}")
            return f"Error in smart analysis: {str(e)}"

    def _analyze_query(self, df: pd.DataFrame, query: str, file_path: str) -> dict:
        """Intelligent query analysis based on natural language."""
        if not query:
            return {"message": "No specific query provided, showing basic summary"}

        query_lower = query.lower()
        result = {
            "query_interpretation": query,
            "findings": [],
            "data": {},
            "insights": []
        }

        try:
            import re

            # OPTIMIZATION: Pre-compute lowercase column names once (avoids repeated .lower() calls)
            col_lower_map = {col: col.lower() for col in df.columns}

            # 1. FILTERING QUERIES: Look for specific values to filter by
            # Extract numbers/IDs from query (e.g., "invoice 65250", "order #123")
            numbers_in_query = re.findall(r'\b\d+\b', query)

            if numbers_in_query:
                search_terms = numbers_in_query

                # OPTIMIZATION: Pre-convert DataFrame to string once (instead of per column)
                # and collect all masks before combining (avoids 1000s of intermediate DataFrames)
                str_df = df.astype(str)
                all_masks = []

                for term in search_terms:
                    for col in str_df.columns:
                        try:
                            mask = str_df[col].str.contains(str(term), case=False, na=False)
                            if mask.any():
                                all_masks.append(mask)
                        except Exception:
                            continue

                # Combine all masks with OR and deduplicate once at the end
                if all_masks:
                    combined_mask = all_masks[0]
                    for mask in all_masks[1:]:
                        combined_mask = combined_mask | mask
                    filtered_rows = df[combined_mask].drop_duplicates()
                else:
                    filtered_rows = pd.DataFrame()

                if not filtered_rows.empty:
                    result["data"]["filtered_rows"] = filtered_rows.to_dict('records')
                    result["data"]["row_count"] = len(filtered_rows)
                    result["data"]["columns"] = list(filtered_rows.columns)
                    result["findings"].append(f"Found {len(filtered_rows)} rows matching: {search_terms}")
                    result["insights"].append(f"Filtered data for values: {search_terms}")
                    return result  # Return early with filtered data

            # 2. LIST/SHOW QUERIES: Return actual data rows
            display_keywords = self._config.display_keywords if self._config else ['list', 'show', 'display', 'all items', 'all rows', 'line items']
            max_display = self._config.max_display_rows if self._config else 100
            if any(kw in query_lower for kw in display_keywords):
                # Check if asking for specific columns
                words = query_lower.replace(',', ' ').split()
                matching_cols = [col for col in df.columns if any(word in col.lower() for word in words if len(word) > 2)]

                if matching_cols and len(matching_cols) < len(df.columns):
                    result["data"]["requested_data"] = df[matching_cols].to_dict('records')
                    result["data"]["columns"] = matching_cols
                else:
                    # Return all data (limited to prevent huge responses)
                    result["data"]["all_rows"] = df.head(max_display).to_dict('records')
                    result["data"]["total_rows"] = len(df)
                    result["data"]["columns"] = list(df.columns)

                result["findings"].append(f"Listing {min(max_display, len(df))} of {len(df)} rows")
                result["insights"].append("Showing actual data rows as requested")
                return result

            # Revenue/Financial analysis
            financial_terms = set(self._config.financial_terms) if self._config else {'revenue', 'sales', 'income', 'profit', 'earnings', 'cost', 'expense'}
            if any(term in query_lower for term in financial_terms):
                revenue_cols = [col for col, lower in col_lower_map.items()
                                if any(term in lower for term in financial_terms)]
                if revenue_cols:
                    revenue_data = {}
                    for col in revenue_cols:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            revenue_data[col] = {
                                "total": float(df[col].sum()),
                                "average": float(df[col].mean()),
                                "max": float(df[col].max()),
                                "min": float(df[col].min())
                            }
                    result["data"]["revenue_analysis"] = revenue_data
                    result["findings"].append(f"Found {len(revenue_cols)} revenue-related columns")

            # Quarterly analysis (Q1, Q2, etc.)
            quarterly_terms = set(self._config.quarterly_terms) if self._config else {'q1', 'q2', 'q3', 'q4', 'quarter'}
            if any(q in query_lower for q in quarterly_terms):
                quarterly_cols = [col for col, lower in col_lower_map.items()
                                  if any(q in lower for q in quarterly_terms)]
                if quarterly_cols:
                    quarterly_data = {}
                    for col in quarterly_cols:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            quarterly_data[col] = {
                                "total": float(df[col].sum()),
                                "average": float(df[col].mean())
                            }
                    result["data"]["quarterly_analysis"] = quarterly_data
                    result["findings"].append(f"Found {len(quarterly_cols)} quarterly columns")

            # Total/sum analysis
            if 'total' in query_lower or 'sum' in query_lower:
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                if numeric_cols:
                    totals = {}
                    for col in numeric_cols:
                        totals[col] = float(df[col].sum())
                    result["data"]["totals"] = totals
                    result["findings"].append(f"Calculated totals for {len(numeric_cols)} numeric columns")

            # Fiscal Year analysis (dynamic based on config)
            fy = self._config.current_fiscal_year if self._config else "25"
            fy_patterns = [f'fy{fy}', f'fy {fy}', f'fy20{fy}', f'fiscal{fy}', f'fiscal {fy}']
            if any(p in query_lower for p in fy_patterns):
                fy_cols = [col for col, lower in col_lower_map.items()
                           if any(p in lower for p in fy_patterns)]
                if fy_cols:
                    fy_data = {}
                    for col in fy_cols:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            fy_data[col] = float(df[col].sum())
                    result["data"]["fiscal_year_analysis"] = fy_data
                    result["findings"].append(f"Found {len(fy_cols)} FY{fy} columns")

            # Generate insights
            if result["data"]:
                result["insights"].append("Analysis completed successfully with specific query matching")

                # Smart insights based on findings
                currency = self._config.currency_symbol if self._config else "$"
                if "revenue_analysis" in result["data"]:
                    revenue_totals = [v["total"] for v in result["data"]["revenue_analysis"].values()]
                    if revenue_totals:
                        max_revenue = max(revenue_totals)
                        result["insights"].append(f"Highest revenue stream: {currency}{max_revenue:,.2f}")

                if "quarterly_analysis" in result["data"]:
                    quarterly_totals = result["data"]["quarterly_analysis"]
                    if quarterly_totals:
                        for col, data in quarterly_totals.items():
                            result["insights"].append(f"{col}: {currency}{data['total']:,.2f}")
            else:
                result["insights"].append("No specific matches found for query, showing general data summary")
                # Fallback: basic numeric summary
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                if numeric_cols:
                    result["data"]["summary_stats"] = df[numeric_cols].describe().to_dict()

        except Exception as e:
            result["error"] = f"Analysis error: {str(e)}"
            result["insights"].append("Encountered error during analysis")

        return result

    async def _arun(self, input_json: str) -> str:
        """Async version of _run using dedicated query executor pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            get_executors().query_executor,
            functools.partial(self._run, input_json)
        )
