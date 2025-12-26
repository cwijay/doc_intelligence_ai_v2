"""Tools for Sheets Agent using LangChain and DuckDB."""

import os
import tempfile
import time
import json
import asyncio
import threading
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from pathlib import Path
import pandas as pd
import duckdb
import logging

from langchain_core.tools import BaseTool

from src.utils.timer_utils import elapsed_ms

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
    Input should be a local file path (e.g., '/path/to/file.xlsx').
    Returns file structure, column names, data types, and sample data."""

    _config: Any = None

    def __init__(self, config: "SheetsAgentConfig"):
        super().__init__()
        self._config = config

    def _run(self, file_path: str) -> str:
        """Load and analyze file structure."""
        try:
            logger.info(f"Loading file preview for: {file_path}")

            # Validate file exists
            if not os.path.exists(file_path):
                return f"Error: File not found: {file_path}"

            # Determine file type and load
            file_ext = Path(file_path).suffix.lower()

            if file_ext == '.csv':
                df = pd.read_csv(file_path, nrows=100)
                sheet_names = None
            elif file_ext == '.tsv':
                df = pd.read_csv(file_path, sep='\t', nrows=100)
                sheet_names = None
            elif file_ext in ['.xlsx', '.xls']:
                excel_file = pd.ExcelFile(file_path)
                sheet_names = excel_file.sheet_names
                df = pd.read_excel(file_path, sheet_name=sheet_names[0], nrows=100)
            else:
                return f"Error: Unsupported file type: {file_ext}"

            # Generate file preview information
            preview_info = {
                "file_path": file_path,
                "file_type": file_ext,
                "sheet_names": sheet_names,
                "shape": {"rows": len(df), "columns": len(df.columns)},
                "columns": [{"name": col, "type": str(df[col].dtype)} for col in df.columns],
                "sample_data": df.head(5).to_dict('records'),
                "null_counts": df.isnull().sum().to_dict(),
                "memory_usage": int(df.memory_usage(deep=True).sum())
            }

            return json.dumps(preview_info, indent=2, default=str)

        except Exception as e:
            logger.error(f"Error loading file preview for {file_path}: {e}")
            return f"Error loading file preview: {str(e)}"

    async def _arun(self, file_path: str) -> str:
        """Async version of _run using thread pool for blocking I/O."""
        return await asyncio.to_thread(self._run, file_path)


class CrossFileQueryTool(BaseTool):
    """Tool for executing queries across multiple files using DuckDB."""

    name: str = "execute_cross_file_query"
    description: str = """Execute SQL queries across multiple Excel/CSV files using DuckDB.
    Input should be a JSON object with 'file_paths' (list of local paths) and 'query' (SQL query).
    Files are automatically loaded as tables named 'file_0', 'file_1', etc.
    Use this ONLY for multiple files (2+). For single files, use query_single_file instead.
    Returns query results as JSON."""

    _config: Any = None

    def __init__(self, config: "SheetsAgentConfig"):
        super().__init__()
        self._config = config

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
                        if not os.path.exists(file_path):
                            logger.warning(f"File not found: {file_path}")
                            continue

                        # Determine table name and load data
                        table_name = f"file_{i}"
                        file_ext = Path(file_path).suffix.lower()

                        if file_ext == '.csv':
                            df = pd.read_csv(file_path)
                        elif file_ext == '.tsv':
                            df = pd.read_csv(file_path, sep='\t')
                        elif file_ext in ['.xlsx', '.xls']:
                            df = pd.read_excel(file_path)
                        else:
                            continue

                        # Register DataFrame with DuckDB
                        conn.register(table_name, df)
                        table_names.append({
                            "table_name": table_name,
                            "file_path": file_path,
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

            if not os.path.exists(file_path):
                return f"Error: File not found: {file_path}"

            # Load file into DataFrame
            file_ext = Path(file_path).suffix.lower()

            if file_ext == '.csv':
                df = pd.read_csv(file_path)
            elif file_ext == '.tsv':
                df = pd.read_csv(file_path, sep='\t')
            elif file_ext in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                return f"Error: Unsupported file type: {file_ext}"

            logger.info(f"Loaded single file with shape: {df.shape}")

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
        """Async version of _run using thread pool for blocking I/O."""
        return await asyncio.to_thread(self._run, input_json)


class SingleFileQueryTool(BaseTool):
    """Tool for querying a single Excel/CSV file using pandas."""

    name: str = "query_single_file"
    description: str = """Query a single Excel/CSV file using natural language description.
    Input should be a JSON object with 'file_path' (local path) and 'query_description' (what you want to find/calculate).
    Use this for single file analysis - more efficient than cross-file queries.
    Examples: calculate totals, find specific rows, aggregate by columns, etc."""

    _config: Any = None

    def __init__(self, config: "SheetsAgentConfig"):
        super().__init__()
        self._config = config

    def _run(self, input_json: str) -> str:
        """Execute single file query using pandas."""
        try:
            # Parse input
            input_data = json.loads(input_json)
            file_path = input_data.get('file_path', '')
            query_description = input_data.get('query_description', '')

            if not file_path or not query_description:
                return "Error: Both 'file_path' and 'query_description' are required"

            logger.info(f"Executing single file query on: {file_path}")

            if not os.path.exists(file_path):
                return f"Error: File not found: {file_path}"

            # Load file into DataFrame
            file_ext = Path(file_path).suffix.lower()

            if file_ext == '.csv':
                df = pd.read_csv(file_path)
            elif file_ext == '.tsv':
                df = pd.read_csv(file_path, sep='\t')
            elif file_ext in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                return f"Error: Unsupported file type: {file_ext}"

            logger.info(f"Loaded file with shape: {df.shape}")

            # Perform analysis based on query description
            result = self._analyze_dataframe(df, query_description, file_path)

            return json.dumps({
                "success": True,
                "file_path": file_path,
                "query": query_description,
                "shape": {"rows": len(df), "columns": len(df.columns)},
                "columns": list(df.columns),
                "result": result
            }, indent=2, default=str)

        except json.JSONDecodeError:
            return "Error: Input must be valid JSON with 'file_path' and 'query_description' fields"
        except Exception as e:
            logger.error(f"Error executing single file query: {e}")
            return f"Error executing query: {str(e)}"

    def _analyze_dataframe(self, df: pd.DataFrame, query_description: str, file_path: str) -> dict:
        """Analyze DataFrame based on query description."""
        result = {
            "analysis": f"Analysis of {file_path}",
            "summary_stats": {},
            "sample_data": [],
            "insights": []
        }

        try:
            # Basic summary statistics for numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                result["summary_stats"] = df[numeric_cols].describe().to_dict()

            # Sample data (first 5 rows)
            result["sample_data"] = df.head(5).to_dict('records')

            # Query-specific analysis
            query_lower = query_description.lower()

            if 'total' in query_lower or 'sum' in query_lower:
                if numeric_cols:
                    totals = df[numeric_cols].sum().to_dict()
                    result["totals"] = totals
                    result["insights"].append(f"Calculated totals for numeric columns: {list(totals.keys())}")

            if 'revenue' in query_lower:
                revenue_cols = [col for col in df.columns if 'revenue' in col.lower() or 'sales' in col.lower()]
                if revenue_cols:
                    revenue_data = df[revenue_cols].sum().to_dict()
                    result["revenue_analysis"] = revenue_data
                    result["insights"].append(f"Found revenue columns: {revenue_cols}")

            if 'q1' in query_lower or 'quarter' in query_lower:
                q1_cols = [col for col in df.columns if 'q1' in col.lower() or 'quarter' in col.lower()]
                if q1_cols:
                    q1_data = df[q1_cols].sum().to_dict()
                    result["quarterly_analysis"] = q1_data
                    result["insights"].append(f"Found Q1/quarterly columns: {q1_cols}")

            if not result["insights"]:
                result["insights"].append("Performed basic data analysis. Use more specific queries for detailed insights.")

        except Exception as e:
            result["error"] = f"Analysis error: {str(e)}"

        return result

    async def _arun(self, input_json: str) -> str:
        """Async version of _run using thread pool for blocking I/O."""
        return await asyncio.to_thread(self._run, input_json)


class SmartAnalysisTool(BaseTool):
    """Smart tool that combines file preview and analysis in one efficient call."""

    name: str = "smart_analysis"
    description: str = """Analyze Excel/CSV files efficiently with preview and query in one call.
    Input should be a JSON object with 'file_path' (local path) and 'query' (what you want to find).
    This tool loads the file once and provides both preview and analysis results.
    Use this as the primary tool for single file analysis - much faster than separate preview + query calls."""

    _config: Any = None
    _file_cache: Any = None

    def __init__(self, config: "SheetsAgentConfig", file_cache: Optional[Any] = None):
        super().__init__()
        self._config = config
        self._file_cache = file_cache

    def _load_file(self, file_path: str) -> pd.DataFrame:
        """Load file using cache if available."""
        # Try cache first
        if self._file_cache:
            cached_df = self._file_cache.get(file_path)
            if cached_df is not None:
                logger.info(f"Using cached data for {file_path}")
                return cached_df

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_ext = Path(file_path).suffix.lower()

        if file_ext == '.csv':
            df = pd.read_csv(file_path)
        elif file_ext == '.tsv':
            df = pd.read_csv(file_path, sep='\t')
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

        # Cache the loaded DataFrame
        if self._file_cache:
            self._file_cache.put(file_path, df)

        logger.info(f"Loaded and cached {file_path} with shape: {df.shape}")
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
                    "sample_data": df.head(3).to_dict('records'),
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
            # Revenue analysis
            if 'revenue' in query_lower or 'sales' in query_lower:
                revenue_cols = [col for col in df.columns if any(term in col.lower() for term in ['revenue', 'sales', 'income'])]
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
            if any(q in query_lower for q in ['q1', 'q2', 'q3', 'q4', 'quarter']):
                quarterly_cols = [col for col in df.columns if any(q in col.lower() for q in ['q1', 'q2', 'q3', 'q4', 'quarter'])]
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

            # FY25 specific
            if 'fy25' in query_lower or 'fy 25' in query_lower:
                fy25_cols = [col for col in df.columns if 'fy25' in col.lower() or 'fy 25' in col.lower()]
                if fy25_cols:
                    fy25_data = {}
                    for col in fy25_cols:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            fy25_data[col] = float(df[col].sum())
                    result["data"]["fy25_analysis"] = fy25_data
                    result["findings"].append(f"Found {len(fy25_cols)} FY25 columns")

            # Generate insights
            if result["data"]:
                result["insights"].append("Analysis completed successfully with specific query matching")

                # Smart insights based on findings
                if "revenue_analysis" in result["data"]:
                    revenue_totals = [v["total"] for v in result["data"]["revenue_analysis"].values()]
                    if revenue_totals:
                        max_revenue = max(revenue_totals)
                        result["insights"].append(f"Highest revenue stream: ${max_revenue:,.2f}")

                if "quarterly_analysis" in result["data"]:
                    quarterly_totals = result["data"]["quarterly_analysis"]
                    if quarterly_totals:
                        for col, data in quarterly_totals.items():
                            result["insights"].append(f"{col}: ${data['total']:,.2f}")
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
        """Async version of _run using thread pool for blocking I/O."""
        return await asyncio.to_thread(self._run, input_json)


class DataAnalysisTool(BaseTool):
    """Tool for performing statistical analysis and data insights."""

    name: str = "analyze_data_patterns"
    description: str = """Analyze data patterns and generate insights from loaded files.
    Input should be a JSON object with 'file_paths' and 'analysis_type'
    (options: 'summary', 'correlation', 'trends', 'outliers', 'quality').
    Returns analytical insights as JSON."""

    _config: Any = None

    def __init__(self, config: "SheetsAgentConfig"):
        super().__init__()
        self._config = config

    def _run(self, input_json: str) -> str:
        """Perform data analysis."""
        try:
            # Parse input
            input_data = json.loads(input_json)
            file_paths = input_data.get('file_paths', [])
            analysis_type = input_data.get('analysis_type', 'summary')

            if not file_paths:
                return "Error: 'file_paths' is required"

            logger.info(f"Performing {analysis_type} analysis on {len(file_paths)} files")

            results = []

            for file_path in file_paths:
                if not os.path.exists(file_path):
                    results.append({"file_path": file_path, "error": "File not found"})
                    continue

                file_ext = Path(file_path).suffix.lower()

                try:
                    if file_ext == '.csv':
                        df = pd.read_csv(file_path)
                    elif file_ext == '.tsv':
                        df = pd.read_csv(file_path, sep='\t')
                    elif file_ext in ['.xlsx', '.xls']:
                        df = pd.read_excel(file_path)
                    else:
                        results.append({"file_path": file_path, "error": f"Unsupported file type: {file_ext}"})
                        continue

                    # Perform analysis based on type
                    if analysis_type == 'summary':
                        analysis_result = self._generate_summary(df, file_path)
                    elif analysis_type == 'correlation':
                        analysis_result = self._generate_correlation(df, file_path)
                    elif analysis_type == 'trends':
                        analysis_result = self._generate_trends(df, file_path)
                    elif analysis_type == 'outliers':
                        analysis_result = self._generate_outliers(df, file_path)
                    elif analysis_type == 'quality':
                        analysis_result = self._generate_quality_report(df, file_path)
                    else:
                        analysis_result = {"error": f"Unknown analysis type: {analysis_type}"}

                    results.append(analysis_result)

                except Exception as e:
                    results.append({"file_path": file_path, "error": str(e)})

            return json.dumps({
                "analysis_type": analysis_type,
                "files_analyzed": len(results),
                "results": results
            }, indent=2, default=str)

        except json.JSONDecodeError:
            return "Error: Input must be valid JSON"
        except Exception as e:
            logger.error(f"Error in data analysis: {e}")
            return f"Error performing analysis: {str(e)}"

    def _generate_summary(self, df: pd.DataFrame, file_path: str) -> Dict:
        """Generate summary statistics."""
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

        return {
            "file_path": file_path,
            "shape": list(df.shape),
            "columns": list(df.columns),
            "numeric_columns": numeric_cols,
            "summary_stats": df.describe().to_dict() if numeric_cols else {},
            "null_counts": df.isnull().sum().to_dict(),
            "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()}
        }

    def _generate_correlation(self, df: pd.DataFrame, file_path: str) -> Dict:
        """Generate correlation analysis."""
        numeric_df = df.select_dtypes(include=['number'])

        if numeric_df.empty:
            return {
                "file_path": file_path,
                "error": "No numeric columns found for correlation analysis"
            }

        correlation_matrix = numeric_df.corr()

        return {
            "file_path": file_path,
            "correlation_matrix": correlation_matrix.to_dict(),
            "high_correlations": self._find_high_correlations(correlation_matrix)
        }

    def _generate_trends(self, df: pd.DataFrame, file_path: str) -> Dict:
        """Generate trend analysis."""
        date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

        return {
            "file_path": file_path,
            "date_columns": date_cols,
            "numeric_columns": numeric_cols,
            "trends": "Trend analysis requires time-series data with date columns" if not date_cols else "Date columns detected"
        }

    def _generate_outliers(self, df: pd.DataFrame, file_path: str) -> Dict:
        """Generate outlier analysis."""
        numeric_df = df.select_dtypes(include=['number'])

        if numeric_df.empty:
            return {
                "file_path": file_path,
                "error": "No numeric columns found for outlier analysis"
            }

        outliers = {}
        for col in numeric_df.columns:
            Q1 = numeric_df[col].quantile(0.25)
            Q3 = numeric_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outlier_count = ((numeric_df[col] < lower_bound) | (numeric_df[col] > upper_bound)).sum()
            outliers[col] = {
                "count": int(outlier_count),
                "percentage": float(outlier_count / len(numeric_df) * 100),
                "bounds": {"lower": float(lower_bound), "upper": float(upper_bound)}
            }

        return {
            "file_path": file_path,
            "outliers_by_column": outliers
        }

    def _generate_quality_report(self, df: pd.DataFrame, file_path: str) -> Dict:
        """Generate data quality report."""
        total_rows = len(df)

        quality_metrics = {
            "file_path": file_path,
            "total_rows": total_rows,
            "total_columns": len(df.columns),
            "completeness": {},
            "duplicates": {
                "duplicate_rows": int(df.duplicated().sum()),
                "duplicate_percentage": float(df.duplicated().sum() / total_rows * 100) if total_rows > 0 else 0
            },
            "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()}
        }

        # Calculate completeness for each column
        for col in df.columns:
            null_count = df[col].isnull().sum()
            quality_metrics["completeness"][col] = {
                "null_count": int(null_count),
                "completeness_percentage": float((total_rows - null_count) / total_rows * 100) if total_rows > 0 else 0
            }

        return quality_metrics

    def _find_high_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.7) -> List[Dict]:
        """Find highly correlated pairs."""
        high_corr = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                correlation = corr_matrix.iloc[i, j]
                if abs(correlation) >= threshold:
                    high_corr.append({
                        "column_1": corr_matrix.columns[i],
                        "column_2": corr_matrix.columns[j],
                        "correlation": float(correlation)
                    })

        return high_corr

    async def _arun(self, input_json: str) -> str:
        """Async version of _run using thread pool for blocking I/O."""
        return await asyncio.to_thread(self._run, input_json)
