"""Formatting utilities for frontend-friendly display values."""

from datetime import datetime, timedelta, timezone
from typing import Optional


def format_duration_ms(duration_ms: Optional[int]) -> Optional[str]:
    """
    Format duration in milliseconds to human-readable string.

    Examples:
        500 -> "500ms"
        2345 -> "2.3s"
        65000 -> "1m 5s"
        3665000 -> "1h 1m"
    """
    if duration_ms is None:
        return None

    if duration_ms < 1000:
        return f"{duration_ms}ms"
    elif duration_ms < 60000:
        return f"{duration_ms / 1000:.1f}s"
    elif duration_ms < 3600000:
        minutes = duration_ms // 60000
        seconds = (duration_ms % 60000) // 1000
        if seconds > 0:
            return f"{minutes}m {seconds}s"
        return f"{minutes}m"
    else:
        hours = duration_ms // 3600000
        minutes = (duration_ms % 3600000) // 60000
        if minutes > 0:
            return f"{hours}h {minutes}m"
        return f"{hours}h"


def format_time_ago(dt: Optional[datetime]) -> Optional[str]:
    """
    Format datetime to relative time string.

    Examples:
        "just now", "5 minutes ago", "2 hours ago", "3 days ago"
    """
    if dt is None:
        return None

    # Ensure we're comparing UTC times
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    diff = now - dt

    if diff < timedelta(seconds=60):
        return "just now"
    elif diff < timedelta(hours=1):
        minutes = int(diff.total_seconds() // 60)
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    elif diff < timedelta(days=1):
        hours = int(diff.total_seconds() // 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    elif diff < timedelta(days=30):
        days = diff.days
        return f"{days} day{'s' if days != 1 else ''} ago"
    elif diff < timedelta(days=365):
        months = diff.days // 30
        return f"{months} month{'s' if months != 1 else ''} ago"
    else:
        return dt.strftime("%b %d, %Y")


def format_file_size(size_bytes: Optional[int]) -> Optional[str]:
    """
    Format file size in bytes to human-readable string.

    Examples:
        500 -> "500 B"
        1536 -> "1.5 KB"
        1572864 -> "1.5 MB"
        1610612736 -> "1.5 GB"
    """
    if size_bytes is None:
        return None

    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def get_status_color(status: Optional[str]) -> str:
    """
    Get suggested UI color for job status.

    Returns CSS-friendly color names that map to common UI frameworks.
    """
    if status is None:
        return "gray"

    colors = {
        "completed": "green",
        "processing": "blue",
        "pending": "gray",
        "failed": "red",
        "cancelled": "orange",
        "running": "blue",
        "queued": "yellow",
    }
    return colors.get(status.lower(), "gray")


def get_activity_title(event_type: Optional[str]) -> str:
    """
    Get human-readable title for event type.

    Transforms snake_case event types to display-friendly titles.
    """
    if event_type is None:
        return "Activity"

    titles = {
        "generation_started": "Generation Started",
        "generation_completed": "Generation Completed",
        "generation_cache_hit": "Cache Hit",
        "parse_started": "Parsing Started",
        "parse_completed": "Parsing Completed",
        "document_loaded": "Document Loaded",
        "document_agent_query": "Document Query",
        "summary_generated": "Summary Generated",
        "faqs_generated": "FAQs Generated",
        "questions_generated": "Questions Generated",
        "content_generated": "Content Generated",
        "error": "Error Occurred",
        "cache_hit": "Cache Hit",
        "document_uploaded": "Document Uploaded",
        "document_parsed": "Document Parsed",
        "extraction_started": "Extraction Started",
        "extraction_completed": "Extraction Completed",
    }
    return titles.get(event_type, event_type.replace("_", " ").title())


def get_activity_icon(event_type: Optional[str]) -> str:
    """
    Get suggested icon name for event type.

    Returns icon names compatible with common icon libraries (Lucide, Heroicons).
    """
    if event_type is None:
        return "activity"

    icons = {
        "generation_started": "play",
        "generation_completed": "check-circle",
        "generation_cache_hit": "zap",
        "parse_started": "upload",
        "parse_completed": "file-check",
        "document_loaded": "file",
        "document_agent_query": "message-square",
        "summary_generated": "file-text",
        "faqs_generated": "help-circle",
        "questions_generated": "list",
        "content_generated": "file-plus",
        "error": "alert-triangle",
        "cache_hit": "zap",
        "document_uploaded": "upload-cloud",
        "document_parsed": "file-search",
        "extraction_started": "table",
        "extraction_completed": "table-2",
    }
    return icons.get(event_type, "activity")


def build_activity_description(
    event_type: Optional[str],
    file_name: Optional[str] = None,
    details: Optional[dict] = None,
) -> str:
    """
    Build human-readable description from event data.

    Constructs a meaningful sentence describing the activity.
    """
    if event_type is None:
        return "Unknown activity"

    file_ref = file_name or "document"
    details = details or {}

    descriptions = {
        "generation_completed": lambda: f"Generated {details.get('generation_type', 'content')} for {file_ref}",
        "generation_cache_hit": lambda: f"Retrieved cached {details.get('generation_type', 'content')} for {file_ref}",
        "generation_started": lambda: f"Started generating {details.get('generation_type', 'content')} for {file_ref}",
        "parse_completed": lambda: f"Parsed {file_ref}",
        "parse_started": lambda: f"Started parsing {file_ref}",
        "document_loaded": lambda: f"Loaded {file_ref}",
        "document_agent_query": lambda: f"Processed query on {file_ref}",
        "summary_generated": lambda: f"Generated summary for {file_ref}",
        "faqs_generated": lambda: f"Generated FAQs for {file_ref}",
        "questions_generated": lambda: f"Generated questions for {file_ref}",
        "content_generated": lambda: f"Generated all content for {file_ref}",
        "error": lambda: f"Error processing {file_ref}: {details.get('error', 'Unknown error')[:50]}",
        "cache_hit": lambda: f"Cache hit for {file_ref}",
        "document_uploaded": lambda: f"Uploaded {file_ref}",
        "extraction_completed": lambda: f"Extracted data from {file_ref}",
    }

    builder = descriptions.get(event_type)
    if builder:
        return builder()

    return f"Activity on {file_ref}"
