"""GCS path parsing and building utilities.

This module provides consistent utilities for working with Google Cloud Storage
URIs (gs://bucket/path/to/object) to avoid duplicated path parsing logic
throughout the codebase.
"""

from typing import Tuple, Optional

from ..constants import GCS_URI_PREFIX, GCS_URI_PREFIX_LEN


def is_gcs_path(path: str) -> bool:
    """
    Check if a path is a GCS URI.

    Args:
        path: Path string to check

    Returns:
        True if path starts with 'gs://', False otherwise

    Example:
        >>> is_gcs_path("gs://bucket/path/file.txt")
        True
        >>> is_gcs_path("/local/path/file.txt")
        False
    """
    return path.startswith(GCS_URI_PREFIX)


def parse_gcs_uri(gcs_uri: str) -> Tuple[str, str]:
    """
    Parse a GCS URI into bucket name and blob path.

    Args:
        gcs_uri: Full GCS URI (e.g., "gs://bucket/path/to/file.txt")

    Returns:
        Tuple of (bucket_name, blob_path)

    Raises:
        ValueError: If the path is not a valid GCS URI

    Example:
        >>> parse_gcs_uri("gs://my-bucket/org/parsed/doc.md")
        ('my-bucket', 'org/parsed/doc.md')
        >>> parse_gcs_uri("gs://bucket")
        ('bucket', '')
    """
    if not is_gcs_path(gcs_uri):
        raise ValueError(f"Not a valid GCS URI (must start with 'gs://'): {gcs_uri}")

    path_without_prefix = gcs_uri[GCS_URI_PREFIX_LEN:]
    parts = path_without_prefix.split("/", 1)
    bucket = parts[0]
    blob_path = parts[1] if len(parts) > 1 else ""
    return bucket, blob_path


def extract_org_from_gcs_path(gcs_uri: str, fallback: Optional[str] = None) -> Optional[str]:
    """
    Extract organization name from a GCS path.

    Assumes path structure: gs://bucket/org_name/...

    Args:
        gcs_uri: Full GCS URI
        fallback: Value to return if org cannot be extracted

    Returns:
        Organization name or fallback value

    Example:
        >>> extract_org_from_gcs_path("gs://bucket/Acme Corp/parsed/doc.md")
        'Acme Corp'
        >>> extract_org_from_gcs_path("/local/path", fallback="default-org")
        'default-org'
    """
    if not is_gcs_path(gcs_uri):
        return fallback

    try:
        _, blob_path = parse_gcs_uri(gcs_uri)
        parts = blob_path.split("/")
        return parts[0] if parts and parts[0] else fallback
    except (ValueError, IndexError):
        return fallback


def build_gcs_uri(bucket: str, *path_parts: str) -> str:
    """
    Build a GCS URI from bucket name and path parts.

    Args:
        bucket: GCS bucket name
        *path_parts: Path segments to join

    Returns:
        Complete GCS URI

    Example:
        >>> build_gcs_uri("my-bucket", "org", "parsed", "doc.md")
        'gs://my-bucket/org/parsed/doc.md'
        >>> build_gcs_uri("bucket", "path/to/file.txt")
        'gs://bucket/path/to/file.txt'
    """
    # Filter out empty parts and join
    path = "/".join(part for part in path_parts if part)
    if path:
        return f"{GCS_URI_PREFIX}{bucket}/{path}"
    return f"{GCS_URI_PREFIX}{bucket}"


def strip_gcs_prefix(gcs_uri: str) -> str:
    """
    Remove the 'gs://' prefix from a GCS URI.

    Args:
        gcs_uri: Full GCS URI

    Returns:
        Path without the 'gs://' prefix

    Example:
        >>> strip_gcs_prefix("gs://bucket/path/file.txt")
        'bucket/path/file.txt'
    """
    if is_gcs_path(gcs_uri):
        return gcs_uri[GCS_URI_PREFIX_LEN:]
    return gcs_uri


def extract_gcs_path_parts(gcs_uri: str, max_splits: int = -1) -> list:
    """
    Extract path parts from a GCS URI.

    Splits the path after removing 'gs://' prefix.
    First element is always the bucket name.

    Args:
        gcs_uri: Full GCS URI
        max_splits: Maximum number of splits (-1 for unlimited)

    Returns:
        List of path parts: [bucket, part1, part2, ...]

    Example:
        >>> extract_gcs_path_parts("gs://bucket/org/parsed/doc.md")
        ['bucket', 'org', 'parsed', 'doc.md']
        >>> extract_gcs_path_parts("gs://bucket/org/parsed/doc.md", max_splits=2)
        ['bucket', 'org', 'parsed/doc.md']
    """
    if not is_gcs_path(gcs_uri):
        return []

    path_without_prefix = gcs_uri[GCS_URI_PREFIX_LEN:]
    if max_splits == -1:
        return path_without_prefix.split("/")
    return path_without_prefix.split("/", max_splits)
