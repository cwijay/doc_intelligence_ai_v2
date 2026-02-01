"""Repository for intelligence report database operations.

Handles CRUD operations for intelligence reports in PostgreSQL.
"""

import json
import logging
import math
from datetime import date, datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from sqlalchemy import text

from src.db.connection import db

logger = logging.getLogger(__name__)


class SafeJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles date, datetime, NaN, and Infinity values.

    PostgreSQL JSONB doesn't support NaN or Infinity, so we convert them to null.
    """

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, date):
            return obj.isoformat()
        return super().default(obj)

    def encode(self, obj):
        """Override encode to handle NaN/Infinity in the final JSON string."""
        return super().encode(self._sanitize(obj))

    def _sanitize(self, obj):
        """Recursively sanitize object, replacing NaN/Infinity with None."""
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        elif isinstance(obj, dict):
            return {k: self._sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._sanitize(item) for item in obj)
        return obj


# Alias for backwards compatibility
DateTimeEncoder = SafeJSONEncoder


def _parse_date(date_str: Optional[str]) -> Optional[date]:
    """Parse a date string to a date object."""
    if not date_str:
        return None
    if isinstance(date_str, date):
        return date_str
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        logger.warning(f"Invalid date format: {date_str}")
        return None


# IntelligenceReportModel is defined in biz2bricks_core.models.intelligence
# Raw SQL is used here for more control over query execution

# SQL for creating the table (run via migration)
# Note: organization_id is VARCHAR to match organizations.id (not UUID)
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS intelligence_reports (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid()::text,
    organization_id VARCHAR(36) NOT NULL REFERENCES organizations(id),
    folder_id VARCHAR(36) NOT NULL,
    report_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    date_range_start DATE,
    date_range_end DATE,
    options JSONB,
    document_count INTEGER DEFAULT 0,
    extracted_record_count INTEGER DEFAULT 0,
    summary JSONB,
    insights JSONB,
    charts JSONB,
    pdf_path VARCHAR(500),
    excel_path VARCHAR(500),
    json_path VARCHAR(500),
    error_message TEXT,
    processing_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    created_by VARCHAR(36) REFERENCES users(id)
);

CREATE INDEX IF NOT EXISTS idx_intelligence_reports_org_id ON intelligence_reports(organization_id);
CREATE INDEX IF NOT EXISTS idx_intelligence_reports_status ON intelligence_reports(status);
CREATE INDEX IF NOT EXISTS idx_intelligence_reports_created_at ON intelligence_reports(created_at DESC);
"""


async def create_report(
    organization_id: str,
    folder_id: str,
    report_type: str,
    date_range_start: Optional[str] = None,
    date_range_end: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
    created_by: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a new intelligence report record.

    Args:
        organization_id: Organization ID
        folder_id: Folder ID containing documents
        report_type: Type of report (expense_summary, vendor_analysis, etc.)
        date_range_start: Start date for filtering
        date_range_end: End date for filtering
        options: Report generation options
        created_by: User ID who created the report

    Returns:
        Created report record as dict
    """
    report_id = str(uuid4())

    async with db.session() as session:
        if not session:
            logger.warning("Database disabled, returning mock report")
            return {"id": report_id, "status": "pending"}

        result = await session.execute(
            text("""
            INSERT INTO intelligence_reports
            (id, organization_id, folder_id, report_type, status,
             date_range_start, date_range_end, options, created_by)
            VALUES
            (:id, :org_id, :folder_id, :report_type, 'pending',
             :date_start, :date_end, CAST(:options AS jsonb), :created_by)
            RETURNING *
            """),
            {
                "id": report_id,
                "org_id": organization_id,
                "folder_id": folder_id,
                "report_type": report_type,
                "date_start": _parse_date(date_range_start),
                "date_end": _parse_date(date_range_end),
                "options": json.dumps(options, cls=DateTimeEncoder) if options else None,
                "created_by": created_by,
            }
        )
        row = result.mappings().first()
        return dict(row) if row else {"id": report_id}


async def get_report(report_id: str) -> Optional[Dict[str, Any]]:
    """Get a report by ID.

    Args:
        report_id: Report ID

    Returns:
        Report record as dict or None
    """
    async with db.session() as session:
        if not session:
            return None

        result = await session.execute(
            text("SELECT * FROM intelligence_reports WHERE id = :id"),
            {"id": report_id}
        )
        row = result.mappings().first()
        return dict(row) if row else None


async def get_report_by_org(report_id: str, organization_id: str) -> Optional[Dict[str, Any]]:
    """Get a report by ID, verifying organization ownership.

    Args:
        report_id: Report ID
        organization_id: Organization ID for verification

    Returns:
        Report record as dict or None
    """
    async with db.session() as session:
        if not session:
            return None

        result = await session.execute(
            text("""
            SELECT * FROM intelligence_reports
            WHERE id = :id AND organization_id = :org_id
            """),
            {"id": report_id, "org_id": organization_id}
        )
        row = result.mappings().first()
        return dict(row) if row else None


async def update_report_status(
    report_id: str,
    status: str,
    error_message: Optional[str] = None,
    processing_time_ms: Optional[int] = None,
) -> bool:
    """Update report status.

    Args:
        report_id: Report ID
        status: New status
        error_message: Error message if failed
        processing_time_ms: Processing time

    Returns:
        True if updated successfully
    """
    async with db.session() as session:
        if not session:
            return False

        # Build dynamic update
        set_parts = ["status = :status"]
        params = {"id": report_id, "status": status}

        if error_message is not None:
            set_parts.append("error_message = :error_message")
            params["error_message"] = error_message

        if processing_time_ms is not None:
            set_parts.append("processing_time_ms = :processing_time_ms")
            params["processing_time_ms"] = processing_time_ms

        if status == "completed":
            set_parts.append("completed_at = NOW()")

        set_clause = ", ".join(set_parts)
        query = f"UPDATE intelligence_reports SET {set_clause} WHERE id = :id"

        result = await session.execute(text(query), params)
        return result.rowcount > 0


async def update_report_data(
    report_id: str,
    document_count: Optional[int] = None,
    extracted_record_count: Optional[int] = None,
    summary: Optional[Dict[str, Any]] = None,
    insights: Optional[List[Dict[str, Any]]] = None,
    charts: Optional[List[Dict[str, Any]]] = None,
    pdf_path: Optional[str] = None,
    excel_path: Optional[str] = None,
    json_path: Optional[str] = None,
) -> bool:
    """Update report data fields.

    Args:
        report_id: Report ID
        document_count: Number of documents processed
        extracted_record_count: Number of records extracted
        summary: Report summary data
        insights: Generated insights
        charts: Chart data
        pdf_path: GCS path to PDF
        excel_path: GCS path to Excel
        json_path: GCS path to JSON

    Returns:
        True if updated successfully
    """
    async with db.session() as session:
        if not session:
            return False

        set_parts = []
        params = {"id": report_id}

        if document_count is not None:
            set_parts.append("document_count = :document_count")
            params["document_count"] = document_count

        if extracted_record_count is not None:
            set_parts.append("extracted_record_count = :extracted_record_count")
            params["extracted_record_count"] = extracted_record_count

        if summary is not None:
            set_parts.append("summary = CAST(:summary AS jsonb)")
            params["summary"] = json.dumps(summary, cls=DateTimeEncoder)

        if insights is not None:
            set_parts.append("insights = CAST(:insights AS jsonb)")
            params["insights"] = json.dumps(insights, cls=DateTimeEncoder)

        if charts is not None:
            set_parts.append("charts = CAST(:charts AS jsonb)")
            params["charts"] = json.dumps(charts, cls=DateTimeEncoder)

        if pdf_path is not None:
            set_parts.append("pdf_path = :pdf_path")
            params["pdf_path"] = pdf_path

        if excel_path is not None:
            set_parts.append("excel_path = :excel_path")
            params["excel_path"] = excel_path

        if json_path is not None:
            set_parts.append("json_path = :json_path")
            params["json_path"] = json_path

        if not set_parts:
            return True

        set_clause = ", ".join(set_parts)
        query = f"UPDATE intelligence_reports SET {set_clause} WHERE id = :id"

        result = await session.execute(text(query), params)
        return result.rowcount > 0


async def list_reports(
    organization_id: str,
    status: Optional[str] = None,
    report_type: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """List reports for an organization.

    Args:
        organization_id: Organization ID
        status: Optional status filter
        report_type: Optional report type filter
        limit: Maximum number of results
        offset: Number of results to skip

    Returns:
        List of report records
    """
    async with db.session() as session:
        if not session:
            return []

        where_parts = ["organization_id = :org_id"]
        params = {"org_id": organization_id, "limit": limit, "offset": offset}

        if status:
            where_parts.append("status = :status")
            params["status"] = status

        if report_type:
            where_parts.append("report_type = :report_type")
            params["report_type"] = report_type

        where_clause = " AND ".join(where_parts)
        query = f"""
            SELECT * FROM intelligence_reports
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT :limit OFFSET :offset
        """

        result = await session.execute(text(query), params)
        rows = result.mappings().all()
        return [dict(row) for row in rows]


async def count_reports(
    organization_id: str,
    status: Optional[str] = None,
    report_type: Optional[str] = None,
) -> int:
    """Count reports for an organization.

    Args:
        organization_id: Organization ID
        status: Optional status filter
        report_type: Optional report type filter

    Returns:
        Count of matching reports
    """
    async with db.session() as session:
        if not session:
            return 0

        where_parts = ["organization_id = :org_id"]
        params = {"org_id": organization_id}

        if status:
            where_parts.append("status = :status")
            params["status"] = status

        if report_type:
            where_parts.append("report_type = :report_type")
            params["report_type"] = report_type

        where_clause = " AND ".join(where_parts)
        query = f"SELECT COUNT(*) FROM intelligence_reports WHERE {where_clause}"

        result = await session.execute(text(query), params)
        return result.scalar() or 0


async def delete_report(report_id: str, organization_id: str) -> bool:
    """Delete a report.

    Args:
        report_id: Report ID
        organization_id: Organization ID for verification

    Returns:
        True if deleted successfully
    """
    async with db.session() as session:
        if not session:
            return False

        result = await session.execute(
            text("""
            DELETE FROM intelligence_reports
            WHERE id = :id AND organization_id = :org_id
            """),
            {"id": report_id, "org_id": organization_id}
        )
        return result.rowcount > 0


async def get_report_statistics(organization_id: str) -> Dict[str, Any]:
    """Get report statistics for an organization.

    Args:
        organization_id: Organization ID

    Returns:
        Statistics dict with counts and totals
    """
    async with db.session() as session:
        if not session:
            return {
                "total_reports": 0,
                "completed_reports": 0,
                "failed_reports": 0,
                "pending_reports": 0,
                "total_documents_processed": 0,
                "total_records_analyzed": 0,
                "avg_processing_time_ms": None,
            }

        result = await session.execute(
            text("""
            SELECT
                COUNT(*) as total_reports,
                COUNT(*) FILTER (WHERE status = 'completed') as completed_reports,
                COUNT(*) FILTER (WHERE status = 'failed') as failed_reports,
                COUNT(*) FILTER (WHERE status IN ('pending', 'extracting', 'aggregating', 'analyzing', 'generating')) as pending_reports,
                COALESCE(SUM(document_count), 0) as total_documents_processed,
                COALESCE(SUM(extracted_record_count), 0) as total_records_analyzed,
                AVG(processing_time_ms) as avg_processing_time_ms
            FROM intelligence_reports
            WHERE organization_id = :org_id
            """),
            {"org_id": organization_id}
        )
        row = result.mappings().first()
        if row:
            return {
                "total_reports": row["total_reports"] or 0,
                "completed_reports": row["completed_reports"] or 0,
                "failed_reports": row["failed_reports"] or 0,
                "pending_reports": row["pending_reports"] or 0,
                "total_documents_processed": row["total_documents_processed"] or 0,
                "total_records_analyzed": row["total_records_analyzed"] or 0,
                "avg_processing_time_ms": row["avg_processing_time_ms"],
            }
        return {
            "total_reports": 0,
            "completed_reports": 0,
            "failed_reports": 0,
            "pending_reports": 0,
            "total_documents_processed": 0,
            "total_records_analyzed": 0,
            "avg_processing_time_ms": None,
        }


async def find_cached_report(
    organization_id: str,
    folder_id: str,
    report_type: str,
    date_range_start: Optional[str] = None,
    date_range_end: Optional[str] = None,
    max_age_hours: int = 24,
) -> Optional[Dict[str, Any]]:
    """Find a recently generated report that matches the criteria.

    Useful for returning cached results instead of regenerating.

    Args:
        organization_id: Organization ID
        folder_id: Folder ID
        report_type: Report type
        date_range_start: Start date
        date_range_end: End date
        max_age_hours: Maximum age of cached report in hours

    Returns:
        Cached report or None
    """
    async with db.session() as session:
        if not session:
            return None

        # Build query with optional date filters
        where_parts = [
            "organization_id = :org_id",
            "folder_id = :folder_id",
            "report_type = :report_type",
            "status = 'completed'",
            f"created_at > NOW() - INTERVAL '{max_age_hours} hours'",
        ]
        params = {
            "org_id": organization_id,
            "folder_id": folder_id,
            "report_type": report_type,
        }

        if date_range_start:
            where_parts.append("date_range_start = :date_start")
            params["date_start"] = _parse_date(date_range_start)
        else:
            where_parts.append("date_range_start IS NULL")

        if date_range_end:
            where_parts.append("date_range_end = :date_end")
            params["date_end"] = _parse_date(date_range_end)
        else:
            where_parts.append("date_range_end IS NULL")

        where_clause = " AND ".join(where_parts)
        query = f"""
            SELECT * FROM intelligence_reports
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT 1
        """

        result = await session.execute(text(query), params)
        row = result.mappings().first()
        return dict(row) if row else None
