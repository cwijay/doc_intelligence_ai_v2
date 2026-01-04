"""Sheets Agent API endpoints.

Multi-tenancy: All endpoints are scoped by organization_id from request headers.
"""

import asyncio
import functools
import logging
import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from src.core.executors import get_executors

from ..dependencies import get_sheets_agent, get_org_id
from ..schemas.common import TokenUsage
from ..schemas.errors import FILE_ERROR_RESPONSES, BASE_ERROR_RESPONSES
from src.utils.timer_utils import elapsed_ms
from src.core.usage import check_quota
from ..schemas.sheets import (
    SheetsAnalyzeRequest,
    SheetsAnalyzeResponse,
    SheetsPreviewRequest,
    SheetsPreviewResponse,
    SheetsHealthResponse,
    FileMetadata,
    ToolUsage,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/analyze",
    response_model=SheetsAnalyzeResponse,
    responses=FILE_ERROR_RESPONSES,
    operation_id="analyzeSheets",
    summary="Analyze Excel/CSV files with natural language",
)
@check_quota(usage_type="tokens", estimated_usage=1500)
async def analyze_sheets(
    request: SheetsAnalyzeRequest,
    agent=Depends(get_sheets_agent),
    org_id: str = Depends(get_org_id),
):
    """
    Analyze Excel/CSV files using natural language queries.

    The Sheets Agent uses OpenAI GPT and DuckDB for SQL-based analysis:

    - **Single file analysis**: Query data from one file
    - **Cross-file queries**: Join and analyze multiple files
    - **Statistical analysis**: Summary statistics, correlations, trends
    - **Data visualization recommendations**: Suggested chart types

    **Multi-tenancy**: Scoped by X-Organization-ID header.

    **Rate Limit**: 10 requests per 60 seconds per session.
    """
    start_time = time.time()

    try:
        from src.agents.sheets.schemas import ChatRequest

        # Build agent request with org_id for multi-tenancy
        agent_request = ChatRequest(
            file_paths=request.file_paths,
            query=request.query,
            session_id=request.session_id,
            user_id=request.user_id,
            organization_id=org_id,  # Multi-tenancy
            options=request.options or {},
        )

        # Process with agent
        response = await agent.process_chat(agent_request)

        processing_time = elapsed_ms(start_time)

        # Map response
        files_processed = []
        if response.files_processed:
            for f in response.files_processed:
                files_processed.append(FileMetadata(
                    file_path=f.file_path,
                    file_type=f.file_type,
                    size_bytes=f.size_bytes,
                    rows=f.shape[0] if f.shape else None,
                    columns=f.shape[1] if f.shape else None,
                    column_names=f.columns,
                    processing_time_ms=f.processing_time_ms,
                ))

        tools_used = []
        if response.tools_used:
            for t in response.tools_used:
                tools_used.append(ToolUsage(
                    tool_name=t.tool_name,
                    input_summary=str(t.input_data)[:200] if t.input_data else None,
                    output_summary=str(t.output_data)[:200] if t.output_data else None,
                    execution_time_ms=t.execution_time_ms,
                    success=t.success,
                    error_message=t.error_message,
                ))

        token_usage = None
        if response.token_usage:
            token_usage = TokenUsage(
                prompt_tokens=response.token_usage.prompt_tokens,
                completion_tokens=response.token_usage.completion_tokens,
                total_tokens=response.token_usage.total_tokens,
                estimated_cost_usd=response.token_usage.estimated_cost_usd,
            )
            # Token usage is now tracked via callback handlers in the agent
            # (see TokenTrackingCallbackHandler with use_context=True)

        return SheetsAnalyzeResponse(
            success=response.success,
            message=response.message,
            response=response.response,
            files_processed=files_processed,
            tools_used=tools_used,
            token_usage=token_usage,
            session_id=response.session_id,
            processing_time_ms=processing_time,
        )

    except Exception as e:
        logger.exception(f"Sheets analysis failed: {e}")
        return SheetsAnalyzeResponse(
            success=False,
            error=str(e),
            processing_time_ms=elapsed_ms(start_time),
        )


@router.post(
    "/preview",
    response_model=SheetsPreviewResponse,
    responses=FILE_ERROR_RESPONSES,
    operation_id="previewFile",
    summary="Preview Excel/CSV file contents",
)
async def preview_file(
    request: SheetsPreviewRequest,
    agent=Depends(get_sheets_agent),
    org_id: str = Depends(get_org_id),
):
    """
    Preview the contents of an Excel/CSV file.

    Returns file metadata (rows, columns, data types) and a preview of the first N rows.
    Useful for understanding file structure before running analysis queries.

    **Multi-tenancy**: Scoped by X-Organization-ID header.
    """
    try:
        import pandas as pd
        from pathlib import Path

        file_path = Path(request.file_path)
        if not file_path.exists():
            return SheetsPreviewResponse(
                success=False,
                error=f"File not found: {request.file_path}"
            )

        # Load file based on extension (use executor to avoid blocking event loop)
        ext = file_path.suffix.lower()
        loop = asyncio.get_running_loop()
        io_executor = get_executors().io_executor

        if ext in ['.xlsx', '.xls']:
            # Run blocking pandas I/O in executor
            df = await loop.run_in_executor(
                io_executor,
                functools.partial(pd.read_excel, file_path, sheet_name=request.sheet_name or 0, nrows=request.rows)
            )
            # Get sheet names
            xl = await loop.run_in_executor(io_executor, pd.ExcelFile, file_path)
            sheet_names = xl.sheet_names
        elif ext == '.csv':
            df = await loop.run_in_executor(
                io_executor,
                functools.partial(pd.read_csv, file_path, nrows=request.rows)
            )
            sheet_names = None
        else:
            return SheetsPreviewResponse(
                success=False,
                error=f"Unsupported file type: {ext}"
            )

        # Get full row count (in executor)
        if ext in ['.xlsx', '.xls']:
            full_df = await loop.run_in_executor(
                io_executor,
                functools.partial(pd.read_excel, file_path, sheet_name=request.sheet_name or 0)
            )
        else:
            full_df = await loop.run_in_executor(io_executor, pd.read_csv, file_path)

        file_info = FileMetadata(
            file_path=request.file_path,
            file_type=ext.lstrip('.'),
            size_bytes=file_path.stat().st_size,
            rows=len(full_df),
            columns=len(df.columns),
            column_names=list(df.columns),
            sheet_names=sheet_names,
        )

        # Convert to preview data
        preview_data = df.to_dict(orient='records')
        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

        return SheetsPreviewResponse(
            success=True,
            file_info=file_info,
            preview_data=preview_data,
            dtypes=dtypes,
        )

    except Exception as e:
        logger.exception(f"File preview failed: {e}")
        return SheetsPreviewResponse(
            success=False,
            error=str(e)
        )


@router.get(
    "/health",
    response_model=SheetsHealthResponse,
    responses=BASE_ERROR_RESPONSES,
    operation_id="getSheetsHealth",
    summary="Get Sheets Agent health status",
)
async def sheets_health(
    agent=Depends(get_sheets_agent),
):
    """
    Get the health status of the Sheets Agent.

    Returns agent readiness, DuckDB availability, list of available tools, and cache statistics.
    Does not require X-Organization-ID header.
    """
    try:
        health = agent.get_health_status()
        return SheetsHealthResponse(
            status="healthy" if health.get("status") == "healthy" else "unhealthy",
            agent_ready=True,
            duckdb_ready=health.get("duckdb_ready", False),
            tools_available=list(health.get("tools", {}).keys()),
            cache_stats=health.get("cache_stats"),
        )
    except Exception as e:
        logger.exception(f"Health check failed: {e}")
        return SheetsHealthResponse(
            status="unhealthy",
            agent_ready=False,
            duckdb_ready=False,
            tools_available=[],
        )
