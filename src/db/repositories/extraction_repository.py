"""
Extraction repository - Dynamic table creation and data persistence for extracted data.

Multi-tenancy: Creates org-specific tables and scopes all operations by organization_id.

Features:
- Dynamic table creation from extraction template schemas
- Type mapping from JSON schema to PostgreSQL types
- Parent table for header fields, child table for line items
- Safe table/column naming with SQL injection prevention
"""

import logging
import re
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any

from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError

from ..connection import db
from ..utils import with_db_retry

logger = logging.getLogger(__name__)


# =============================================================================
# TYPE MAPPING
# =============================================================================

# Map JSON schema types to PostgreSQL types
SCHEMA_TYPE_TO_PG = {
    "string": "VARCHAR",
    "number": "DECIMAL(12,2)",
    "integer": "INTEGER",
    "currency": "DECIMAL(12,2)",
    "date": "DATE",
    "boolean": "BOOLEAN",
    "text": "TEXT",
}

# Common entity ID field names for duplicate detection
ENTITY_ID_FIELDS = [
    "invoice_id", "invoice_number", "invoice_no",
    "order_id", "order_number", "order_no",
    "po_number", "po_id", "purchase_order_number",
    "contract_id", "contract_number", "contract_no",
    "receipt_id", "receipt_number", "receipt_no",
    "quote_id", "quote_number",
    "bill_id", "bill_number",
    "reference_number", "reference_id", "ref_number",
]


def _find_entity_id_field(schema: Dict[str, Any]) -> Optional[str]:
    """Find entity ID field in schema for duplicate detection.

    Looks for common ID field names like invoice_id, invoice_number, etc.

    Args:
        schema: JSON schema with properties

    Returns:
        Field name if found, None otherwise
    """
    properties = schema.get("properties", {})
    field_names = [name.lower() for name in properties.keys()]

    for entity_field in ENTITY_ID_FIELDS:
        if entity_field in field_names:
            # Return the original case field name
            for name in properties.keys():
                if name.lower() == entity_field:
                    return name
    return None


def _map_schema_type_to_pg(schema_type: str) -> str:
    """Map JSON schema type to PostgreSQL type."""
    return SCHEMA_TYPE_TO_PG.get(schema_type.lower(), "VARCHAR")


# =============================================================================
# ORGANIZATION LOOKUP
# =============================================================================

async def get_organization_name(org_id: str) -> Optional[str]:
    """Lookup organization name from org_id.

    Args:
        org_id: Organization UUID

    Returns:
        Organization name if found, None otherwise
    """
    try:
        async with db.session() as session:
            result = await session.execute(
                text("SELECT name FROM organizations WHERE id = :org_id"),
                {"org_id": org_id}
            )
            row = result.fetchone()
            return row[0] if row else None
    except Exception as e:
        logger.warning(f"Failed to lookup org name for {org_id}: {e}")
        return None


# =============================================================================
# TABLE NAMING
# =============================================================================

def _sanitize_identifier(name: str, max_length: int = 63) -> str:
    """
    Sanitize a string for use as a PostgreSQL identifier.

    - Converts to lowercase
    - Replaces spaces and hyphens with underscores
    - Removes non-alphanumeric characters (except underscores)
    - Truncates to max_length (Postgres limit is 63)
    """
    # Lowercase and replace common separators
    name = name.lower().strip()
    name = re.sub(r'[\s\-]+', '_', name)
    # Remove any character that's not alphanumeric or underscore
    name = re.sub(r'[^a-z0-9_]', '', name)
    # Ensure doesn't start with a number
    if name and name[0].isdigit():
        name = '_' + name
    # Truncate
    return name[:max_length]


def _get_table_name(org_name: str, template_name: str) -> str:
    """Generate parent table name: {org_name}_{entity_name}.

    Example: 'Acme Corp' + 'invoice_template' -> 'acme_corp_invoice'
    """
    safe_org = _sanitize_identifier(org_name, 25)
    # Strip '_template' or 'template' suffix to get entity name
    entity = template_name.lower()
    entity = entity.replace('_template', '').replace('template', '').strip('_')
    safe_entity = _sanitize_identifier(entity, 30)
    return f"{safe_org}_{safe_entity}"


def _get_line_items_table_name(org_name: str, template_name: str) -> str:
    """Generate child table name for line items."""
    base = _get_table_name(org_name, template_name)
    return f"{base}_line_items"[:63]  # Respect Postgres limit


# =============================================================================
# SCHEMA PARSING
# =============================================================================

def _extract_header_fields(schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract header fields (non-line-item) from schema."""
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    fields = []
    for field_name, field_def in properties.items():
        if field_name == "line_items":
            continue  # Skip line items

        field_type = field_def.get("type", "string")
        if field_type == "array" or field_type == "object":
            continue  # Skip complex types

        fields.append({
            "name": field_name,
            "type": field_type,
            "required": field_name in required,
            "description": field_def.get("description", ""),
        })

    return fields


def _extract_line_item_fields(schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract line item fields from schema."""
    properties = schema.get("properties", {})
    line_items_def = properties.get("line_items", {})

    if line_items_def.get("type") != "array":
        return []

    items_def = line_items_def.get("items", {})
    item_properties = items_def.get("properties", {})
    required = set(items_def.get("required", []))

    fields = []
    for field_name, field_def in item_properties.items():
        field_type = field_def.get("type", "string")
        if field_type in ("array", "object"):
            continue

        fields.append({
            "name": field_name,
            "type": field_type,
            "required": field_name in required,
            "description": field_def.get("description", ""),
        })

    return fields


# =============================================================================
# DDL GENERATION
# =============================================================================

def _generate_parent_table_ddl(
    table_name: str,
    header_fields: List[Dict[str, Any]],
    entity_id_field: Optional[str] = None
) -> List[str]:
    """Generate CREATE TABLE DDL statements for parent (header) table.

    Args:
        table_name: Name of the table
        header_fields: List of header field definitions
        entity_id_field: Optional entity ID field for unique constraint

    Returns a list of separate statements for asyncpg compatibility.
    """
    columns = [
        "id UUID PRIMARY KEY DEFAULT gen_random_uuid()",
        "organization_id VARCHAR NOT NULL",
        "document_id VARCHAR NOT NULL",
        "extraction_job_id VARCHAR NOT NULL",
        "source_file_path VARCHAR",
        "template_name VARCHAR NOT NULL",
        "extracted_at TIMESTAMP DEFAULT NOW()",
        "updated_at TIMESTAMP DEFAULT NOW()",
    ]

    for field in header_fields:
        col_name = _sanitize_identifier(field["name"])
        col_type = _map_schema_type_to_pg(field["type"])
        columns.append(f"{col_name} {col_type}")

    # Add unique constraint on entity ID if found
    if entity_id_field:
        entity_col = _sanitize_identifier(entity_id_field)
        columns.append(f"CONSTRAINT uq_{table_name}_entity UNIQUE (organization_id, {entity_col})")

    columns_sql = ",\n    ".join(columns)

    # Return separate statements - asyncpg can't execute multiple statements at once
    statements = [
        f"CREATE TABLE IF NOT EXISTS {table_name} (\n    {columns_sql}\n)",
        f"CREATE INDEX IF NOT EXISTS idx_{table_name}_org ON {table_name}(organization_id)",
        f"CREATE INDEX IF NOT EXISTS idx_{table_name}_doc ON {table_name}(document_id)",
    ]

    return statements


def _generate_line_items_table_ddl(
    table_name: str,
    parent_table_name: str,
    line_item_fields: List[Dict[str, Any]]
) -> List[str]:
    """Generate CREATE TABLE DDL statements for line items child table.

    Returns a list of separate statements for asyncpg compatibility.
    """
    columns = [
        "id UUID PRIMARY KEY DEFAULT gen_random_uuid()",
        f"parent_id UUID NOT NULL REFERENCES {parent_table_name}(id) ON DELETE CASCADE",
        "line_number INTEGER NOT NULL",
    ]

    for field in line_item_fields:
        col_name = _sanitize_identifier(field["name"])
        col_type = _map_schema_type_to_pg(field["type"])
        columns.append(f"{col_name} {col_type}")

    columns.append("created_at TIMESTAMP DEFAULT NOW()")
    columns_sql = ",\n    ".join(columns)

    # Return separate statements - asyncpg can't execute multiple statements at once
    return [
        f"CREATE TABLE IF NOT EXISTS {table_name} (\n    {columns_sql}\n)",
        f"CREATE INDEX IF NOT EXISTS idx_{table_name}_parent ON {table_name}(parent_id)",
    ]


# =============================================================================
# TABLE MANAGEMENT
# =============================================================================

async def _get_existing_columns(session, table_name: str) -> set:
    """Get set of existing column names for a table."""
    result = await session.execute(text("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = :table_name
    """), {"table_name": table_name})
    return {row[0] for row in result.fetchall()}


# System columns that should exist on parent tables
PARENT_SYSTEM_COLUMNS = [
    ("extracted_at", "TIMESTAMP DEFAULT NOW()"),
    ("updated_at", "TIMESTAMP DEFAULT NOW()"),
]


async def _add_missing_columns(
    session,
    table_name: str,
    fields: List[Dict[str, Any]],
    existing_columns: set
) -> int:
    """Add missing columns to an existing table.

    Args:
        session: Database session
        table_name: Name of the table
        fields: List of field definitions from schema
        existing_columns: Set of existing column names

    Returns:
        Number of columns added
    """
    added = 0
    for field in fields:
        col_name = _sanitize_identifier(field["name"])
        if col_name not in existing_columns:
            col_type = _map_schema_type_to_pg(field["type"])
            alter_sql = f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type}"
            try:
                await session.execute(text(alter_sql))
                logger.info(f"Added column {col_name} to {table_name}")
                added += 1
            except ProgrammingError as e:
                # Column might already exist due to race condition
                if "already exists" not in str(e):
                    raise
    return added


async def _add_missing_system_columns(
    session,
    table_name: str,
    existing_columns: set
) -> int:
    """Add missing system columns (extracted_at, updated_at) to parent table.

    Args:
        session: Database session
        table_name: Name of the table
        existing_columns: Set of existing column names

    Returns:
        Number of columns added
    """
    added = 0
    for col_name, col_def in PARENT_SYSTEM_COLUMNS:
        if col_name not in existing_columns:
            alter_sql = f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_def}"
            try:
                await session.execute(text(alter_sql))
                logger.info(f"Added system column {col_name} to {table_name}")
                added += 1
            except ProgrammingError as e:
                if "already exists" not in str(e):
                    raise
    return added


@with_db_retry
async def ensure_extraction_tables_exist(
    org_name: str,
    template_name: str,
    schema: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Ensure extraction tables exist for the given template.
    Creates parent and line items tables if they don't exist.
    Adds missing columns if tables exist but schema has evolved.

    Args:
        org_name: Organization name (from X-Organization-ID header)
        template_name: Template name
        schema: JSON schema from template

    Returns:
        Dict with table names and entity ID field:
        {"parent": "...", "line_items": "...", "entity_id_field": "..."}
    """
    parent_table = _get_table_name(org_name, template_name)
    line_items_table = _get_line_items_table_name(org_name, template_name)

    header_fields = _extract_header_fields(schema)
    line_item_fields = _extract_line_item_fields(schema)

    # Detect entity ID field for UPSERT support
    entity_id_field = _find_entity_id_field(schema)
    if entity_id_field:
        logger.info(f"Detected entity ID field: {entity_id_field} (enables UPSERT)")

    async with db.session() as session:
        # Create parent table with entity ID unique constraint
        parent_statements = _generate_parent_table_ddl(
            parent_table, header_fields, entity_id_field
        )
        for statement in parent_statements:
            await session.execute(text(statement))
        logger.info(f"Ensured parent table exists: {parent_table}")

        # Add any missing columns to parent table (schema evolution)
        existing_parent_cols = await _get_existing_columns(session, parent_table)

        # Add missing system columns (extracted_at, updated_at)
        added_system = await _add_missing_system_columns(
            session, parent_table, existing_parent_cols
        )
        if added_system > 0:
            logger.info(f"Added {added_system} system columns to {parent_table}")
            # Refresh existing columns after adding system columns
            existing_parent_cols = await _get_existing_columns(session, parent_table)

        # Add missing user-defined columns
        added_parent = await _add_missing_columns(
            session, parent_table, header_fields, existing_parent_cols
        )
        if added_parent > 0:
            logger.info(f"Added {added_parent} new columns to {parent_table}")

        # Create line items table if schema has line items
        if line_item_fields:
            line_items_statements = _generate_line_items_table_ddl(
                line_items_table, parent_table, line_item_fields
            )
            for statement in line_items_statements:
                await session.execute(text(statement))
            logger.info(f"Ensured line items table exists: {line_items_table}")

            # Add any missing columns to line items table
            existing_li_cols = await _get_existing_columns(session, line_items_table)
            added_li = await _add_missing_columns(
                session, line_items_table, line_item_fields, existing_li_cols
            )
            if added_li > 0:
                logger.info(f"Added {added_li} new columns to {line_items_table}")

        await session.commit()

    return {
        "parent": parent_table,
        "line_items": line_items_table if line_item_fields else None,
        "entity_id_field": entity_id_field
    }


@with_db_retry
async def check_table_exists(table_name: str) -> bool:
    """Check if a table exists in the database."""
    async with db.session() as session:
        result = await session.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = :table_name
            )
        """), {"table_name": table_name})
        return result.scalar()


# =============================================================================
# DATA PERSISTENCE
# =============================================================================

@with_db_retry
async def save_extracted_record(
    org_id: str,
    template_name: str,
    schema: Dict[str, Any],
    extraction_job_id: str,
    document_id: str,
    extracted_data: Dict[str, Any],
    source_file_path: Optional[str] = None,
    org_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Save extracted data to the database using UPSERT when entity ID is available.

    If an entity ID field (like invoice_number) is found in the schema and data,
    uses INSERT ... ON CONFLICT to update existing records instead of creating duplicates.

    Args:
        org_id: Organization ID (stored in records for querying)
        template_name: Template name
        schema: JSON schema from template
        extraction_job_id: Unique job ID
        document_id: Source document ID
        extracted_data: Extracted data dict
        source_file_path: Optional source file path
        org_name: Organization name for table naming (if None, uses org_id)

    Returns:
        Dict with record_id and whether it was an update:
        {"record_id": "...", "updated": True/False}
    """
    # Use org_name for table naming, fallback to org_id
    table_org_name = org_name if org_name else org_id

    # Ensure tables exist (also returns entity_id_field)
    tables = await ensure_extraction_tables_exist(table_org_name, template_name, schema)
    parent_table = tables["parent"]
    line_items_table = tables["line_items"]
    entity_id_field = tables.get("entity_id_field")

    # Extract fields from schema
    header_fields = _extract_header_fields(schema)
    line_item_fields = _extract_line_item_fields(schema)

    # Check if we can use UPSERT (entity ID field exists and has a value)
    entity_id_value = None
    entity_id_col = None
    if entity_id_field:
        entity_id_value = extracted_data.get(entity_id_field)
        entity_id_col = _sanitize_identifier(entity_id_field)

    use_upsert = entity_id_field and entity_id_value is not None

    # Build parent row data
    record_id = str(uuid.uuid4())
    parent_data = {
        "id": record_id,
        "organization_id": org_id,
        "document_id": document_id,
        "extraction_job_id": extraction_job_id,
        "source_file_path": source_file_path,
        "template_name": template_name,
    }

    # Add header field values
    for field in header_fields:
        field_name = field["name"]
        col_name = _sanitize_identifier(field_name)
        value = extracted_data.get(field_name)
        parent_data[col_name] = value

    was_updated = False

    async with db.session() as session:
        if use_upsert:
            # UPSERT: Insert or update on conflict with entity ID
            columns = ", ".join(parent_data.keys())
            placeholders = ", ".join(f":{k}" for k in parent_data.keys())

            # Build SET clause for UPDATE (exclude id, organization_id, and entity_id_col)
            update_cols = [k for k in parent_data.keys()
                          if k not in ("id", "organization_id", entity_id_col)]
            update_set = ", ".join(f"{k} = EXCLUDED.{k}" for k in update_cols)
            update_set += ", updated_at = NOW()"

            upsert_sql = f"""
                INSERT INTO {parent_table} ({columns})
                VALUES ({placeholders})
                ON CONFLICT (organization_id, {entity_id_col}) DO UPDATE SET {update_set}
                RETURNING id, (xmax = 0) AS inserted
            """

            result = await session.execute(text(upsert_sql), parent_data)
            row = result.fetchone()
            record_id = str(row[0])
            was_inserted = row[1]
            was_updated = not was_inserted

            if was_updated:
                logger.info(f"Updated existing record: {record_id} (entity: {entity_id_value})")
                # Delete old line items before inserting new ones
                if line_items_table:
                    await session.execute(
                        text(f"DELETE FROM {line_items_table} WHERE parent_id = :parent_id"),
                        {"parent_id": record_id}
                    )
            else:
                logger.info(f"Inserted new record: {record_id} (entity: {entity_id_value})")
        else:
            # Regular INSERT (no entity ID available)
            columns = ", ".join(parent_data.keys())
            placeholders = ", ".join(f":{k}" for k in parent_data.keys())
            insert_sql = f"INSERT INTO {parent_table} ({columns}) VALUES ({placeholders})"
            await session.execute(text(insert_sql), parent_data)
            logger.info(f"Inserted parent record: {record_id}")

        # Insert line items if present
        line_items = extracted_data.get("line_items", [])
        if line_items and line_items_table and line_item_fields:
            for idx, item in enumerate(line_items, start=1):
                item_data = {
                    "id": str(uuid.uuid4()),
                    "parent_id": record_id,
                    "line_number": idx,
                }

                for field in line_item_fields:
                    field_name = field["name"]
                    col_name = _sanitize_identifier(field_name)
                    value = item.get(field_name)
                    item_data[col_name] = value

                columns = ", ".join(item_data.keys())
                placeholders = ", ".join(f":{k}" for k in item_data.keys())
                item_sql = f"INSERT INTO {line_items_table} ({columns}) VALUES ({placeholders})"
                await session.execute(text(item_sql), item_data)

            logger.info(f"Inserted {len(line_items)} line items for record: {record_id}")

        await session.commit()

    return {"record_id": record_id, "updated": was_updated}


# =============================================================================
# DATA RETRIEVAL
# =============================================================================

@with_db_retry
async def get_extracted_records(
    org_id: str,
    template_name: str,
    limit: int = 100,
    offset: int = 0,
    org_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Get extracted records for a template.

    Args:
        org_id: Organization ID (used for WHERE clause)
        template_name: Template name
        limit: Max records to return
        offset: Pagination offset
        org_name: Organization name for table naming (if None, uses org_id)

    Returns:
        List of extracted records (without line items)
    """
    table_org_name = org_name if org_name else org_id
    parent_table = _get_table_name(table_org_name, template_name)

    # Check if table exists
    if not await check_table_exists(parent_table):
        return []

    async with db.session() as session:
        result = await session.execute(text(f"""
            SELECT * FROM {parent_table}
            WHERE organization_id = :org_id
            ORDER BY extracted_at DESC
            LIMIT :limit OFFSET :offset
        """), {"org_id": org_id, "limit": limit, "offset": offset})

        rows = result.fetchall()
        columns = result.keys()

        return [dict(zip(columns, row)) for row in rows]


@with_db_retry
async def get_extracted_record_with_line_items(
    org_id: str,
    template_name: str,
    record_id: str,
    org_name: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Get a single extracted record with its line items.

    Args:
        org_id: Organization ID (used for WHERE clause)
        template_name: Template name
        record_id: Record UUID
        org_name: Organization name for table naming (if None, uses org_id)

    Returns:
        Record dict with line_items array, or None if not found
    """
    table_org_name = org_name if org_name else org_id
    parent_table = _get_table_name(table_org_name, template_name)
    line_items_table = _get_line_items_table_name(table_org_name, template_name)

    async with db.session() as session:
        # Get parent record
        result = await session.execute(text(f"""
            SELECT * FROM {parent_table}
            WHERE id = :record_id AND organization_id = :org_id
        """), {"record_id": record_id, "org_id": org_id})

        row = result.fetchone()
        if not row:
            return None

        record = dict(zip(result.keys(), row))

        # Get line items if table exists
        if await check_table_exists(line_items_table):
            items_result = await session.execute(text(f"""
                SELECT * FROM {line_items_table}
                WHERE parent_id = :parent_id
                ORDER BY line_number
            """), {"parent_id": record_id})

            items_rows = items_result.fetchall()
            items_columns = items_result.keys()
            record["line_items"] = [dict(zip(items_columns, r)) for r in items_rows]
        else:
            record["line_items"] = []

        return record


@with_db_retry
async def get_record_count(
    org_id: str,
    template_name: str,
    org_name: Optional[str] = None
) -> int:
    """Get total count of extracted records for a template."""
    table_org_name = org_name if org_name else org_id
    parent_table = _get_table_name(table_org_name, template_name)

    if not await check_table_exists(parent_table):
        return 0

    async with db.session() as session:
        result = await session.execute(text(f"""
            SELECT COUNT(*) FROM {parent_table}
            WHERE organization_id = :org_id
        """), {"org_id": org_id})

        return result.scalar() or 0


@with_db_retry
async def delete_extracted_record(
    org_id: str,
    template_name: str,
    record_id: str,
    org_name: Optional[str] = None
) -> bool:
    """
    Delete an extracted record and its line items.

    Args:
        org_id: Organization ID (used for WHERE clause)
        template_name: Template name
        record_id: Record UUID
        org_name: Organization name for table naming (if None, uses org_id)

    Returns:
        True if deleted, False if not found
    """
    table_org_name = org_name if org_name else org_id
    parent_table = _get_table_name(table_org_name, template_name)

    async with db.session() as session:
        result = await session.execute(text(f"""
            DELETE FROM {parent_table}
            WHERE id = :record_id AND organization_id = :org_id
            RETURNING id
        """), {"record_id": record_id, "org_id": org_id})

        deleted = result.fetchone()
        await session.commit()

        if deleted:
            logger.info(f"Deleted extraction record: {record_id}")
            return True
        return False
