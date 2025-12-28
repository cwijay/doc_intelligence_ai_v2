"""GCS and local file loading utilities for SheetsAgent.

This module provides utilities to load DataFrames from either GCS paths
(gs://bucket/path) or local file paths. Uses gcsfs for GCS support with pandas.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import pandas as pd

from src.utils.gcs_utils import is_gcs_path, parse_gcs_uri

logger = logging.getLogger(__name__)

# Supported file extensions
SUPPORTED_EXTENSIONS = frozenset(['.xlsx', '.xls', '.csv', '.tsv'])


def validate_file_path(file_path: str) -> Tuple[bool, Optional[str]]:
    """
    Validate that a file path is valid and accessible.

    For GCS paths: validates the URI format
    For local paths: checks that the file exists

    Args:
        file_path: Local path or GCS URI (gs://bucket/path)

    Returns:
        Tuple of (is_valid, error_message)
        - (True, None) if valid
        - (False, error_message) if invalid
    """
    if is_gcs_path(file_path):
        try:
            bucket, blob = parse_gcs_uri(file_path)
            if not bucket:
                return False, f"Invalid GCS URI - missing bucket: {file_path}"
            if not blob:
                return False, f"Invalid GCS URI - missing blob path: {file_path}"

            # Check file extension
            ext = Path(blob).suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                return False, f"Unsupported file type '{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"

            return True, None
        except ValueError as e:
            return False, str(e)
    else:
        # Local path validation
        if not os.path.exists(file_path):
            return False, f"File not found: {file_path}"

        if not os.path.isfile(file_path):
            return False, f"Path is not a file: {file_path}"

        ext = Path(file_path).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            return False, f"Unsupported file type '{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"

        return True, None


def get_file_extension(file_path: str) -> str:
    """
    Get the file extension from a path (works for both GCS and local paths).

    Args:
        file_path: Local path or GCS URI

    Returns:
        File extension in lowercase (e.g., '.xlsx', '.csv')
    """
    if is_gcs_path(file_path):
        _, blob = parse_gcs_uri(file_path)
        return Path(blob).suffix.lower()
    return Path(file_path).suffix.lower()


def load_dataframe(
    file_path: str,
    nrows: Optional[int] = None,
    sheet_name: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load a DataFrame from either GCS or local path.

    Uses gcsfs to enable pandas to read directly from GCS paths.

    Args:
        file_path: Local path or GCS URI (gs://bucket/path)
        nrows: Optional limit on rows to read (for preview)
        sheet_name: Optional sheet name for Excel files (default: first sheet)

    Returns:
        Tuple of (DataFrame, metadata_dict)
        metadata_dict contains: source, path, shape, columns, dtypes

    Raises:
        ValueError: If file type is not supported
        FileNotFoundError: If file does not exist
        Exception: For other loading errors
    """
    # Validate path first
    is_valid, error = validate_file_path(file_path)
    if not is_valid:
        raise ValueError(error)

    metadata: Dict[str, Any] = {
        "source": "gcs" if is_gcs_path(file_path) else "local",
        "path": file_path,
    }

    file_ext = get_file_extension(file_path)
    read_kwargs: Dict[str, Any] = {}

    if nrows is not None:
        read_kwargs["nrows"] = nrows

    try:
        if file_ext == '.csv':
            df = pd.read_csv(file_path, **read_kwargs)
        elif file_ext == '.tsv':
            df = pd.read_csv(file_path, sep='\t', **read_kwargs)
        elif file_ext in ['.xlsx', '.xls']:
            # For Excel, handle sheet_name
            excel_kwargs = {**read_kwargs}
            if sheet_name:
                excel_kwargs["sheet_name"] = sheet_name
            df = pd.read_excel(file_path, **excel_kwargs)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

        # Populate metadata
        metadata["shape"] = list(df.shape)
        metadata["columns"] = list(df.columns)
        metadata["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}
        metadata["memory_usage_bytes"] = int(df.memory_usage(deep=True).sum())

        logger.info(
            f"Loaded DataFrame from {metadata['source']}: {file_path}, "
            f"shape={df.shape}, memory={metadata['memory_usage_bytes']} bytes"
        )

        return df, metadata

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {e}")
        raise


def load_multiple_dataframes(
    file_paths: List[str],
    nrows: Optional[int] = None
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, Any]]]:
    """
    Load multiple DataFrames from a list of paths.

    Args:
        file_paths: List of local paths or GCS URIs
        nrows: Optional limit on rows per file

    Returns:
        Tuple of (dataframes_dict, metadata_dict)
        - dataframes_dict: Maps file_path to DataFrame
        - metadata_dict: Maps file_path to metadata
    """
    dataframes: Dict[str, pd.DataFrame] = {}
    all_metadata: Dict[str, Dict[str, Any]] = {}

    for file_path in file_paths:
        df, metadata = load_dataframe(file_path, nrows=nrows)
        dataframes[file_path] = df
        all_metadata[file_path] = metadata

    return dataframes, all_metadata


def get_excel_sheet_names(file_path: str) -> List[str]:
    """
    Get the list of sheet names from an Excel file.

    Args:
        file_path: Local path or GCS URI to an Excel file

    Returns:
        List of sheet names

    Raises:
        ValueError: If not an Excel file
    """
    file_ext = get_file_extension(file_path)
    if file_ext not in ['.xlsx', '.xls']:
        raise ValueError(f"Not an Excel file: {file_path}")

    try:
        excel_file = pd.ExcelFile(file_path)
        return excel_file.sheet_names
    except Exception as e:
        logger.error(f"Error reading Excel sheet names from {file_path}: {e}")
        raise


def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    Get basic file information without loading the full DataFrame.

    For Excel files, includes sheet names.
    For CSV/TSV, includes row count estimate.

    Args:
        file_path: Local path or GCS URI

    Returns:
        Dictionary with file information
    """
    is_valid, error = validate_file_path(file_path)
    if not is_valid:
        return {"error": error, "valid": False}

    file_ext = get_file_extension(file_path)

    info: Dict[str, Any] = {
        "valid": True,
        "path": file_path,
        "source": "gcs" if is_gcs_path(file_path) else "local",
        "file_type": file_ext.lstrip('.'),
    }

    try:
        if file_ext in ['.xlsx', '.xls']:
            info["sheet_names"] = get_excel_sheet_names(file_path)
            info["sheet_count"] = len(info["sheet_names"])

        # Load preview to get column info
        df, metadata = load_dataframe(file_path, nrows=5)
        info["columns"] = metadata["columns"]
        info["column_count"] = len(metadata["columns"])
        info["dtypes"] = metadata["dtypes"]
        info["sample_shape"] = metadata["shape"]

    except Exception as e:
        info["error"] = str(e)
        logger.error(f"Error getting file info for {file_path}: {e}")

    return info
