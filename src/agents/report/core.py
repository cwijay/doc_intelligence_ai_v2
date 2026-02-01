"""Report Agent for Business Intelligence Report Generation.

This agent generates business intelligence reports from extracted document data:
- Expense summaries
- Vendor analysis
- Invoice reconciliation
- Cash flow projections
- Spend trends
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from langchain.chat_models import init_chat_model

from src.agents.core.base_agent import BaseAgent
from src.core.executors import get_executors

from .config import ReportAgentConfig
from .schemas import (
    Report,
    ReportType,
    ReportStatus,
    ReportOptions,
    ReportRequest,
    DateRange,
    ReportSummary,
    Insight,
    ChartData,
    CategoryBreakdown,
    VendorBreakdown,
    MonthlyTrend,
    ExpenseSummaryData,
    VendorAnalysisData,
    InvoiceReconciliationData,
    ReconciliationMatch,
    DashboardData,
)

logger = logging.getLogger(__name__)


class ReportAgent(BaseAgent):
    """Agent for generating business intelligence reports from document data.

    Workflow:
    1. Load extracted records from folder (via ExtractorAgent outputs)
    2. Aggregate data into unified dataset
    3. Run analysis queries (DuckDB)
    4. Generate LLM insights
    5. Create charts
    6. Render final report (PDF/Excel/JSON)
    """

    def __init__(self, config: Optional[ReportAgentConfig] = None):
        """Initialize the Report Agent.

        Args:
            config: Agent configuration. Uses defaults if not provided.
        """
        self.config = config or ReportAgentConfig()
        super().__init__(self.config)

        # Initialize LLM for insights generation
        self.llm = self._init_llm()

        # DuckDB connection for analysis queries
        self._duckdb_conn = None

        # Get executors for async operations
        self._executors = get_executors()

        logger.info(
            f"ReportAgent initialized with model={self.config.openai_model}, "
            f"max_documents={self.config.max_documents_per_report}"
        )

    def _init_llm(self) -> Any:
        """Initialize the language model for insights generation."""
        callbacks = self._create_token_tracking_callback("report_agent")

        llm = init_chat_model(
            model=self.config.openai_model,
            model_provider="openai",
            temperature=self.config.temperature,
            callbacks=callbacks,
        )
        logger.info(f"Initialized LLM: {self.config.openai_model}")
        return llm

    def _init_tools(self) -> List:
        """Initialize agent tools (not used directly - ReportAgent uses internal methods)."""
        return []

    def _create_agent(self) -> Any:
        """Create agent instance (ReportAgent doesn't use LangGraph agent)."""
        return None

    def _get_agent_type(self) -> str:
        """Get agent type for audit logging."""
        return "report"

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the Report Agent."""
        base_status = self._get_base_health_status()

        return {
            **base_status,
            "agent_type": "report",
            "model": self.config.openai_model,
            "max_documents_per_report": self.config.max_documents_per_report,
            "charts_enabled": self.config.enable_charts,
        }

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown agent resources gracefully."""
        logger.info("Shutting down ReportAgent...")
        self._cleanup_resources()

        # Close DuckDB connection if open
        if self._duckdb_conn:
            try:
                self._duckdb_conn.close()
            except Exception:
                pass

        logger.info("ReportAgent shutdown complete")

    async def generate_report(
        self,
        org_id: str,
        folder_id: str,
        report_type: ReportType,
        date_range: Optional[DateRange] = None,
        options: Optional[ReportOptions] = None,
        session_id: Optional[str] = None,
        folder_name: Optional[str] = None,
    ) -> Report:
        """Generate a business intelligence report.

        Args:
            org_id: Organization ID
            folder_id: Folder containing documents
            report_type: Type of report to generate
            date_range: Optional date range filter
            options: Report generation options
            session_id: Optional session ID for tracking
            folder_name: Optional folder name for direct path lookup (avoids UUID resolution)

        Returns:
            Generated Report object
        """
        start_time = time.time()
        options = options or ReportOptions()
        session_id = session_id or str(uuid.uuid4())

        # Create report record
        report_id = str(uuid.uuid4())
        report = Report(
            id=report_id,
            organization_id=org_id,
            folder_id=folder_id,
            report_type=report_type,
            status=ReportStatus.PENDING,
            date_range=date_range or DateRange(
                start=datetime.now().replace(day=1).date(),
                end=datetime.now().date()
            ),
            options=options,
            document_count=0,
            extracted_record_count=0,
            created_at=datetime.utcnow(),
        )

        try:
            # Step 1: Extract/load data from folder
            report.status = ReportStatus.EXTRACTING
            extracted_data = await self._load_extracted_data(org_id, folder_id, date_range, folder_name)
            report.document_count = extracted_data.get("document_count", 0)
            report.extracted_record_count = len(extracted_data.get("records", []))

            if report.extracted_record_count == 0:
                report.status = ReportStatus.FAILED
                report.error_message = "No extracted records found in folder"
                return report

            # Step 2: Aggregate data
            report.status = ReportStatus.AGGREGATING
            dataset = await self._aggregate_data(extracted_data, report_type)

            # Step 3: Run analysis
            report.status = ReportStatus.ANALYZING
            analysis_results = await self._run_analysis(dataset, report_type, options)

            # Step 4: Generate report content
            report.status = ReportStatus.GENERATING

            # Generate insights via LLM
            insights = await self._generate_insights(analysis_results, report_type, options)
            report.insights = insights

            # Generate charts if enabled (non-fatal if it fails)
            charts = []
            if options.include_charts:
                try:
                    charts = await self._generate_charts(analysis_results, report_type, options)
                    report.charts = charts
                except Exception as e:
                    logger.warning(f"Chart generation failed (non-fatal): {e}")
                    # Continue without charts

            # Build summary
            report.summary = self._build_summary(analysis_results, date_range)

            # Build type-specific data
            if report_type == ReportType.EXPENSE_SUMMARY:
                report.expense_data = self._build_expense_data(analysis_results, insights, charts)
            elif report_type == ReportType.VENDOR_ANALYSIS:
                report.vendor_data = self._build_vendor_data(analysis_results, insights, charts)
            elif report_type == ReportType.INVOICE_RECONCILIATION:
                report.reconciliation_data = self._build_reconciliation_data(
                    analysis_results, insights
                )

            # Step 5: Render outputs (non-fatal if individual formats fail)
            if options.output_format.value == "pdf" or "pdf" in [f.value for f in options.additional_formats]:
                try:
                    report.pdf_path = await self._render_pdf(report, org_id)
                except Exception as e:
                    logger.warning(f"PDF generation failed (non-fatal): {e}")

            if options.output_format.value == "excel" or "excel" in [f.value for f in options.additional_formats]:
                try:
                    report.excel_path = await self._render_excel(report, org_id)
                except Exception as e:
                    logger.warning(f"Excel generation failed (non-fatal): {e}")

            if options.output_format.value == "json" or "json" in [f.value for f in options.additional_formats]:
                try:
                    report.json_path = await self._save_json(report, org_id)
                except Exception as e:
                    logger.warning(f"JSON save failed (non-fatal): {e}")

            # Mark complete
            report.status = ReportStatus.COMPLETED
            report.completed_at = datetime.utcnow()
            report.processing_time_ms = int((time.time() - start_time) * 1000)

            logger.info(
                f"Report {report_id} generated: type={report_type.value}, "
                f"documents={report.document_count}, records={report.extracted_record_count}, "
                f"time={report.processing_time_ms}ms"
            )

            return report

        except Exception as e:
            logger.error(f"Report generation failed: {e}", exc_info=True)
            report.status = ReportStatus.FAILED
            report.error_message = str(e)
            report.processing_time_ms = int((time.time() - start_time) * 1000)
            return report

    async def _load_extracted_data(
        self,
        org_id: str,
        folder_id: str,
        date_range: Optional[DateRange] = None,
        folder_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load extracted records from folder.

        First tries to load from extracted data directory. If no extracted data
        is found, falls back to loading and extracting data from parsed documents.

        Args:
            org_id: Organization ID
            folder_id: Folder ID containing documents
            date_range: Optional date filter
            folder_name: Optional folder name (if provided, skips UUID resolution)

        Returns:
            Dict with records, metadata, and data_source indicator
        """
        from src.storage import get_storage
        from src.db.repositories import rag_repository, bulk_repository

        storage = get_storage()
        records = []
        document_count = 0
        data_source = "extracted"  # Track where data came from

        try:
            # If folder_name is provided, use it directly (skip UUID resolution)
            if folder_name:
                logger.info(f"Using provided folder_name: {folder_name}")
            else:
                # Get folder info - try multiple lookup strategies
                # Strategy 1: Try rag document folders by ID
                folder_info = await rag_repository.get_folder_by_id(folder_id)
                if folder_info:
                    folder_name = folder_info.get("folder_name")
                    logger.info(f"Found folder in document_folders: {folder_name}")

                # Strategy 2: Try bulk jobs by ID (folder_id might be a bulk job ID)
                if not folder_name:
                    bulk_job = await bulk_repository.get_bulk_job(folder_id)
                    if bulk_job:
                        folder_name = bulk_job.get("folder_name")
                        logger.info(f"Found folder in bulk_jobs by ID: {folder_name}")

                # Strategy 3: Try bulk jobs by folder name (folder_id might be the folder name)
                if not folder_name:
                    bulk_job = await bulk_repository.get_bulk_job_by_folder(org_id, folder_id)
                    if bulk_job:
                        folder_name = bulk_job.get("folder_name")
                        logger.info(f"Found folder in bulk_jobs by name: {folder_name}")

                # Strategy 4: Use folder_id as folder_name directly
                if not folder_name:
                    folder_name = folder_id
                    logger.info(f"Using folder_id as folder_name: {folder_name}")

            # Try to load from extracted directory first
            extracted_path = f"{org_id}/extracted/{folder_name}"
            files = await storage.list_files(directory=extracted_path, use_prefix=False)
            json_files = [f for f in files if isinstance(f, str) and f.endswith(".json")]

            if json_files:
                logger.info(f"Found {len(json_files)} JSON files in extracted directory")
                for file_path in json_files:
                    document_count += 1
                    try:
                        content = await storage.read(file_path, use_prefix=False)
                        if content:
                            import json
                            data = json.loads(content)
                            if isinstance(data, list):
                                records.extend(data)
                            elif isinstance(data, dict) and "records" in data:
                                records.extend(data["records"])
                            elif isinstance(data, dict):
                                records.append(data)
                    except Exception as e:
                        logger.warning(f"Failed to load {file_path}: {e}")
            else:
                # Fallback: Load from parsed documents and extract data using LLM
                logger.info(
                    f"No extracted data found at {extracted_path}, "
                    f"falling back to parsed documents"
                )
                data_source = "parsed"
                # Pass folder_name directly (already resolved above)
                records, document_count = await self._load_from_parsed_documents(
                    org_id, folder_name, date_range
                )

            # Filter by date range if provided (only for extracted data, parsed already filtered)
            if date_range and records and data_source == "extracted":
                records = self._filter_by_date(records, date_range)

            logger.info(
                f"Loaded {len(records)} records from {document_count} documents "
                f"(source: {data_source})"
            )
            return {
                "records": records,
                "document_count": document_count,
                "folder_name": folder_name,
                "data_source": data_source,
            }

        except Exception as e:
            logger.error(f"Failed to load extracted data: {e}")
            return {"records": [], "document_count": 0, "data_source": "none"}

    def _filter_by_date(
        self,
        records: List[Dict[str, Any]],
        date_range: DateRange,
    ) -> List[Dict[str, Any]]:
        """Filter records by date range.

        Looks for common date fields: date, invoice_date, transaction_date, created_at.
        """
        from datetime import datetime as dt

        date_fields = ["date", "invoice_date", "transaction_date", "created_at", "receipt_date"]
        filtered = []

        for record in records:
            record_date = None
            for field in date_fields:
                if field in record and record[field]:
                    try:
                        if isinstance(record[field], str):
                            # Try common date formats
                            for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%dT%H:%M:%S"]:
                                try:
                                    record_date = dt.strptime(record[field][:10], fmt[:len(record[field][:10])]).date()
                                    break
                                except ValueError:
                                    continue
                        break
                    except Exception:
                        continue

            # Include record if no date found or date is in range
            if record_date is None:
                filtered.append(record)
            elif date_range.start <= record_date <= date_range.end:
                filtered.append(record)

        return filtered

    async def _load_from_parsed_documents(
        self,
        org_id: str,
        folder_name: str,
        date_range: Optional[DateRange] = None,
    ) -> tuple[List[Dict[str, Any]], int]:
        """Load parsed Markdown documents and extract structured data using LLM.

        Falls back to this method when no extracted data is found.
        Looks for files at: {org_name}/parsed/{folder_name}/*.md

        Args:
            org_id: Organization ID
            folder_name: Folder name (already resolved from folder_id)
            date_range: Optional date filter

        Returns:
            Tuple of (list of extracted records, document count)
        """
        from src.storage import get_storage
        from src.db.repositories import get_organization_name

        storage = get_storage()
        records = []
        document_count = 0

        try:
            # Get org name from org_id
            org_name = await get_organization_name(org_id)
            if not org_name:
                logger.warning(f"Could not find organization name for {org_id}, using org_id as name")
                org_name = org_id

            # List .md files from parsed directory
            # Path: {org_name}/parsed/{folder_name}/*.md
            parsed_path = f"{org_name}/{self.config.parsed_directory}/{folder_name}"
            logger.info(f"Looking for parsed documents at: {parsed_path}")

            files = await storage.list_files(directory=parsed_path, use_prefix=False)
            md_files = [f for f in files if f.endswith(".md")]

            if not md_files:
                logger.info(f"No .md files found at {parsed_path}")
                return [], 0

            logger.info(f"Found {len(md_files)} .md files in parsed directory")

            # Process documents in batches for rate limiting
            batch_size = self.config.extraction_batch_size
            for i in range(0, len(md_files), batch_size):
                batch = md_files[i:i + batch_size]
                batch_records = await self._process_document_batch(batch, storage)
                records.extend(batch_records)
                document_count += len(batch)

                # Log progress
                logger.info(
                    f"Processed batch {i // batch_size + 1}/{(len(md_files) + batch_size - 1) // batch_size}: "
                    f"{len(batch_records)} records extracted"
                )

            # Filter by date range if provided
            if date_range and records:
                records = self._filter_by_date(records, date_range)

            logger.info(f"Extracted {len(records)} records from {document_count} parsed documents")
            return records, document_count

        except Exception as e:
            logger.error(f"Failed to load from parsed documents: {e}", exc_info=True)
            return [], 0

    async def _process_document_batch(
        self,
        file_paths: List[str],
        storage: Any,
    ) -> List[Dict[str, Any]]:
        """Process a batch of documents concurrently.

        Args:
            file_paths: List of GCS file paths
            storage: Storage backend

        Returns:
            List of extracted records
        """
        import asyncio
        from pathlib import Path

        async def process_single(file_path: str) -> Optional[Dict[str, Any]]:
            try:
                # Read document content
                content = await storage.read(file_path, use_prefix=False)
                if not content:
                    logger.warning(f"Empty content for {file_path}")
                    return None

                # Extract filename from path
                filename = Path(file_path.split("/")[-1]).stem + ".md"

                # Extract data using LLM
                return await self._extract_data_from_document(content, filename)
            except Exception as e:
                logger.warning(f"Failed to process {file_path}: {e}")
                return None

        # Process all files in batch concurrently
        tasks = [process_single(fp) for fp in file_paths]
        results = await asyncio.gather(*tasks)

        # Filter out None results
        return [r for r in results if r is not None]

    async def _extract_data_from_document(
        self,
        document_content: str,
        filename: str,
    ) -> Optional[Dict[str, Any]]:
        """Use LLM to extract structured data from a parsed document.

        Args:
            document_content: The parsed document content (Markdown)
            filename: Original filename for context

        Returns:
            Extracted data dict or None if extraction fails
        """
        import json
        import re
        from langchain.chat_models import init_chat_model
        from .templates import get_document_data_extraction_prompt

        try:
            # Truncate content if too long
            max_chars = self.config.max_document_chars
            if len(document_content) > max_chars:
                document_content = document_content[:max_chars]
                logger.debug(f"Truncated {filename} to {max_chars} chars")

            # Generate extraction prompt
            prompt = get_document_data_extraction_prompt(
                document_content=document_content,
                filename=filename,
                max_chars=max_chars,
            )

            # Use cheaper extraction model
            extraction_llm = init_chat_model(
                model=self.config.extraction_model,
                model_provider="openai",
                temperature=0.1,  # Low temperature for deterministic extraction
            )

            # Run extraction in executor to avoid blocking
            response = await asyncio.get_event_loop().run_in_executor(
                self._executors.agent_executor,
                lambda: extraction_llm.invoke(prompt),
            )

            # Parse JSON response
            content = response.content.strip()

            # Try to extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', content)
            if json_match:
                content = json_match.group(1)

            data = json.loads(content)

            # Ensure source_file is set
            if "source_file" not in data:
                data["source_file"] = filename

            # Normalize amount fields
            data = self._normalize_extracted_data(data)

            logger.debug(f"Extracted data from {filename}: {list(data.keys())}")
            return data

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from LLM response for {filename}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Failed to extract data from {filename}: {e}")
            return None

    def _normalize_extracted_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize extracted data for consistent analysis.

        Ensures consistent field names and data types.

        Args:
            data: Raw extracted data

        Returns:
            Normalized data dict
        """
        # Normalize amount field (use 'amount' as standard)
        if "total" in data and "amount" not in data:
            data["amount"] = data["total"]
        elif "total_amount" in data and "amount" not in data:
            data["amount"] = data["total_amount"]

        # Normalize vendor field
        if "merchant" in data and "vendor" not in data:
            data["vendor"] = data["merchant"]

        # Ensure amount is numeric
        if "amount" in data:
            try:
                if isinstance(data["amount"], str):
                    # Remove currency symbols and commas
                    clean = data["amount"].replace("$", "").replace(",", "").replace("€", "").replace("£", "")
                    data["amount"] = float(clean)
            except (ValueError, TypeError):
                pass

        return data

    async def _aggregate_data(
        self,
        extracted_data: Dict[str, Any],
        report_type: ReportType,
    ) -> pd.DataFrame:
        """Aggregate extracted records into a DataFrame for analysis.

        Args:
            extracted_data: Dict with records list
            report_type: Report type for schema hints

        Returns:
            Pandas DataFrame with aggregated data
        """
        records = extracted_data.get("records", [])
        if not records:
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(records)

        # Normalize column names (lowercase, replace spaces with underscores)
        df.columns = [col.lower().replace(" ", "_").replace("-", "_") for col in df.columns]

        # Try to identify and convert amount columns
        amount_columns = ["amount", "total", "total_amount", "subtotal", "price", "cost"]
        for col in amount_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[,$]', '', regex=True), errors='coerce')

        # Try to parse date columns
        date_columns = ["date", "invoice_date", "transaction_date", "receipt_date"]
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        logger.info(f"Aggregated {len(df)} records with columns: {list(df.columns)}")
        return df

    async def _run_analysis(
        self,
        df: pd.DataFrame,
        report_type: ReportType,
        options: ReportOptions,
    ) -> Dict[str, Any]:
        """Run analysis queries on the aggregated data.

        Args:
            df: Pandas DataFrame with aggregated data
            report_type: Report type
            options: Report options

        Returns:
            Dict with analysis results
        """
        import duckdb

        if df.empty:
            return {"error": "No data to analyze"}

        # Create DuckDB connection and register DataFrame
        conn = duckdb.connect()
        conn.register("data", df)

        results = {}

        try:
            # Get amount column (try common names)
            amount_col = None
            for col in ["amount", "total", "total_amount", "subtotal", "price"]:
                if col in df.columns:
                    amount_col = col
                    break

            if not amount_col:
                # Create a dummy amount column if none exists
                logger.warning("No amount column found, using 1 for all records")
                df["amount"] = 1
                amount_col = "amount"
                conn.register("data", df)

            # Common queries
            results["total_amount"] = conn.execute(
                f"SELECT COALESCE(SUM({amount_col}), 0) FROM data"
            ).fetchone()[0]

            results["record_count"] = len(df)

            # Category breakdown (try common category columns)
            category_col = None
            for col in ["category", "expense_category", "type", "expense_type"]:
                if col in df.columns:
                    category_col = col
                    break

            if category_col:
                results["by_category"] = conn.execute(f"""
                    SELECT {category_col} as category,
                           SUM({amount_col}) as total,
                           COUNT(*) as count
                    FROM data
                    WHERE {category_col} IS NOT NULL
                    GROUP BY {category_col}
                    ORDER BY total DESC
                """).fetchdf().to_dict('records')

            # Vendor breakdown
            vendor_col = None
            for col in ["vendor", "vendor_name", "supplier", "merchant", "payee"]:
                if col in df.columns:
                    vendor_col = col
                    break

            if vendor_col:
                results["by_vendor"] = conn.execute(f"""
                    SELECT {vendor_col} as vendor,
                           SUM({amount_col}) as total,
                           COUNT(*) as invoice_count,
                           AVG({amount_col}) as avg_amount
                    FROM data
                    WHERE {vendor_col} IS NOT NULL
                    GROUP BY {vendor_col}
                    ORDER BY total DESC
                    LIMIT 20
                """).fetchdf().to_dict('records')
                results["vendor_count"] = conn.execute(
                    f"SELECT COUNT(DISTINCT {vendor_col}) FROM data WHERE {vendor_col} IS NOT NULL"
                ).fetchone()[0]

            # Monthly trends
            date_col = None
            for col in ["date", "invoice_date", "transaction_date", "receipt_date"]:
                if col in df.columns:
                    date_col = col
                    break

            if date_col:
                results["monthly_trends"] = conn.execute(f"""
                    SELECT strftime(CAST({date_col} AS DATE), '%Y-%m') as month,
                           SUM({amount_col}) as total,
                           COUNT(*) as count
                    FROM data
                    WHERE {date_col} IS NOT NULL
                    GROUP BY month
                    ORDER BY month
                """).fetchdf().to_dict('records')

            # Top expenses
            results["top_expenses"] = conn.execute(f"""
                SELECT * FROM data
                ORDER BY {amount_col} DESC
                LIMIT 10
            """).fetchdf().to_dict('records')

            # Report-specific queries
            if report_type == ReportType.INVOICE_RECONCILIATION:
                results["reconciliation"] = self._run_reconciliation_queries(conn, df)

            logger.info(f"Analysis complete: {list(results.keys())}")

        finally:
            conn.close()

        return results

    def _run_reconciliation_queries(
        self,
        conn: Any,
        df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Run invoice reconciliation specific queries."""
        reconciliation = {}

        # Check for PO number column
        po_col = None
        for col in ["po_number", "purchase_order", "po"]:
            if col in df.columns:
                po_col = col
                break

        invoice_col = None
        for col in ["invoice_number", "invoice_no", "invoice_id"]:
            if col in df.columns:
                invoice_col = col
                break

        if po_col and invoice_col:
            # Matched (both PO and Invoice present)
            reconciliation["matched_count"] = conn.execute(f"""
                SELECT COUNT(*) FROM data
                WHERE {po_col} IS NOT NULL AND {invoice_col} IS NOT NULL
            """).fetchone()[0]

            # Missing PO
            reconciliation["missing_po_count"] = conn.execute(f"""
                SELECT COUNT(*) FROM data
                WHERE {po_col} IS NULL AND {invoice_col} IS NOT NULL
            """).fetchone()[0]

        return reconciliation

    async def _generate_insights(
        self,
        analysis_results: Dict[str, Any],
        report_type: ReportType,
        options: ReportOptions,
    ) -> List[Insight]:
        """Generate AI insights from analysis results.

        Args:
            analysis_results: Results from analysis queries
            report_type: Report type
            options: Report options

        Returns:
            List of Insight objects
        """
        from .templates import get_insights_prompt

        prompt = get_insights_prompt(report_type, analysis_results, options.max_insights)

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                self._executors.agent_executor,
                lambda: self.llm.invoke(prompt),
            )

            # Parse LLM response into Insight objects
            insights = self._parse_insights_response(response.content)
            return insights[:options.max_insights]

        except Exception as e:
            logger.error(f"Failed to generate insights: {e}")
            return [
                Insight(
                    category="error",
                    title="Insight Generation Failed",
                    description=f"Unable to generate AI insights: {str(e)}",
                    severity="warning",
                )
            ]

    def _parse_insights_response(self, content: str) -> List[Insight]:
        """Parse LLM response into Insight objects."""
        import json
        import re

        insights = []

        # Try to parse as JSON first
        try:
            # Extract JSON array if wrapped in markdown code block
            json_match = re.search(r'```(?:json)?\s*(\[[\s\S]*?\])\s*```', content)
            if json_match:
                data = json.loads(json_match.group(1))
            else:
                data = json.loads(content)

            for item in data:
                insights.append(Insight(
                    category=item.get("category", "general"),
                    title=item.get("title", "Insight"),
                    description=item.get("description", ""),
                    severity=item.get("severity", "info"),
                    data_points=item.get("data_points"),
                ))
            return insights
        except json.JSONDecodeError:
            pass

        # Fallback: parse as text with numbered items
        lines = content.strip().split("\n")
        current_insight = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for numbered item
            if re.match(r'^\d+[\.\)]\s*', line):
                if current_insight:
                    insights.append(current_insight)
                title = re.sub(r'^\d+[\.\)]\s*', '', line)
                current_insight = Insight(
                    category="general",
                    title=title[:100],
                    description="",
                    severity="info",
                )
            elif current_insight:
                current_insight.description += " " + line

        if current_insight:
            insights.append(current_insight)

        return insights

    async def _generate_charts(
        self,
        analysis_results: Dict[str, Any],
        report_type: ReportType,
        options: ReportOptions,
    ) -> List[ChartData]:
        """Generate charts from analysis results.

        Args:
            analysis_results: Results from analysis queries
            report_type: Report type
            options: Report options

        Returns:
            List of ChartData objects
        """
        from .output.chart_generator import generate_charts

        try:
            charts = await generate_charts(
                analysis_results,
                report_type,
                options.chart_types,
                self.config.chart_dpi,
            )
            return charts
        except Exception as e:
            logger.error(f"Chart generation failed: {e}")
            return []

    def _build_summary(
        self,
        analysis_results: Dict[str, Any],
        date_range: Optional[DateRange],
    ) -> ReportSummary:
        """Build report summary from analysis results."""
        by_category = analysis_results.get("by_category", [])
        by_vendor = analysis_results.get("by_vendor", [])

        return ReportSummary(
            total_amount=analysis_results.get("total_amount", 0),
            document_count=analysis_results.get("document_count", 0),
            record_count=analysis_results.get("record_count", 0),
            vendor_count=analysis_results.get("vendor_count", 0),
            category_count=len(by_category),
            date_range=date_range or DateRange(
                start=datetime.now().replace(day=1).date(),
                end=datetime.now().date(),
            ),
            top_category=by_category[0]["category"] if by_category else None,
            top_vendor=by_vendor[0]["vendor"] if by_vendor else None,
            avg_transaction_amount=(
                analysis_results.get("total_amount", 0) / analysis_results.get("record_count", 1)
                if analysis_results.get("record_count", 0) > 0 else 0
            ),
        )

    def _build_expense_data(
        self,
        analysis_results: Dict[str, Any],
        insights: List[Insight],
        charts: List[ChartData],
    ) -> ExpenseSummaryData:
        """Build expense summary specific data."""
        total = analysis_results.get("total_amount", 0) or 1

        return ExpenseSummaryData(
            summary=self._build_summary(analysis_results, None),
            by_category=[
                CategoryBreakdown(
                    category=item.get("category", "Unknown"),
                    amount=item.get("total", 0),
                    percentage=(item.get("total", 0) / total) * 100,
                    count=item.get("count", 0),
                )
                for item in analysis_results.get("by_category", [])
            ],
            by_vendor=[
                VendorBreakdown(
                    vendor=item.get("vendor", "Unknown"),
                    amount=item.get("total", 0),
                    percentage=(item.get("total", 0) / total) * 100,
                    invoice_count=item.get("invoice_count", 0),
                    avg_amount=item.get("avg_amount", 0),
                )
                for item in analysis_results.get("by_vendor", [])
            ],
            monthly_trends=[
                MonthlyTrend(
                    month=item.get("month", ""),
                    amount=item.get("total", 0),
                    count=item.get("count", 0),
                )
                for item in analysis_results.get("monthly_trends", [])
            ],
            top_expenses=analysis_results.get("top_expenses", []),
            insights=insights,
            charts=charts,
        )

    def _build_vendor_data(
        self,
        analysis_results: Dict[str, Any],
        insights: List[Insight],
        charts: List[ChartData],
    ) -> VendorAnalysisData:
        """Build vendor analysis specific data."""
        total = analysis_results.get("total_amount", 0) or 1

        # Calculate concentration metrics
        by_vendor = analysis_results.get("by_vendor", [])
        top_3_total = sum(v.get("total", 0) for v in by_vendor[:3])

        return VendorAnalysisData(
            summary=self._build_summary(analysis_results, None),
            vendor_rankings=[
                VendorBreakdown(
                    vendor=item.get("vendor", "Unknown"),
                    amount=item.get("total", 0),
                    percentage=(item.get("total", 0) / total) * 100,
                    invoice_count=item.get("invoice_count", 0),
                    avg_amount=item.get("avg_amount", 0),
                )
                for item in by_vendor
            ],
            vendor_trends={},  # Would need per-vendor monthly breakdown
            concentration_metrics={
                "top_3_percentage": (top_3_total / total) * 100 if total > 0 else 0,
                "vendor_count": len(by_vendor),
                "avg_per_vendor": total / len(by_vendor) if by_vendor else 0,
            },
            insights=insights,
            charts=charts,
        )

    def _build_reconciliation_data(
        self,
        analysis_results: Dict[str, Any],
        insights: List[Insight],
    ) -> InvoiceReconciliationData:
        """Build invoice reconciliation specific data."""
        reconciliation = analysis_results.get("reconciliation", {})

        return InvoiceReconciliationData(
            summary={
                "total_records": analysis_results.get("record_count", 0),
                "matched_count": reconciliation.get("matched_count", 0),
                "missing_po_count": reconciliation.get("missing_po_count", 0),
            },
            matched_pairs=[],  # Would need actual matching logic
            unmatched_invoices=[],
            unmatched_pos=[],
            discrepancies=[],
            insights=insights,
        )

    async def _render_pdf(self, report: Report, org_id: str) -> Optional[str]:
        """Render report as PDF and save to GCS.

        Args:
            report: Report object
            org_id: Organization ID

        Returns:
            GCS path to PDF file
        """
        from .output.pdf_generator import generate_pdf
        from src.storage import get_storage

        try:
            pdf_bytes = await generate_pdf(report)
            storage = get_storage()

            filename = f"{report.report_type.value}_{report.id[:8]}.pdf"
            directory = f"{org_id}/{self.config.reports_directory}"

            path = await storage.save_bytes(
                content=pdf_bytes,
                filename=filename,
                directory=directory,
                use_prefix=False,
            )
            logger.info(f"PDF saved to {path}")
            return path

        except Exception as e:
            logger.error(f"PDF generation failed: {e}")
            return None

    async def _render_excel(self, report: Report, org_id: str) -> Optional[str]:
        """Render report as Excel and save to GCS.

        Args:
            report: Report object
            org_id: Organization ID

        Returns:
            GCS path to Excel file
        """
        from .output.excel_generator import generate_excel
        from src.storage import get_storage

        try:
            excel_bytes = await generate_excel(report)
            storage = get_storage()

            filename = f"{report.report_type.value}_{report.id[:8]}.xlsx"
            directory = f"{org_id}/{self.config.reports_directory}"

            path = await storage.save_bytes(
                content=excel_bytes,
                filename=filename,
                directory=directory,
                use_prefix=False,
            )
            logger.info(f"Excel saved to {path}")
            return path

        except Exception as e:
            logger.error(f"Excel generation failed: {e}")
            return None

    async def _save_json(self, report: Report, org_id: str) -> Optional[str]:
        """Save report data as JSON to GCS.

        Args:
            report: Report object
            org_id: Organization ID

        Returns:
            GCS path to JSON file
        """
        from src.storage import get_storage
        import json

        try:
            storage = get_storage()

            # Convert report to JSON-serializable dict
            report_dict = report.model_dump(mode='json')

            filename = f"{report.report_type.value}_{report.id[:8]}.json"
            directory = f"{org_id}/{self.config.reports_directory}"

            path = await storage.save(
                content=json.dumps(report_dict, indent=2, default=str),
                filename=filename,
                directory=directory,
                use_prefix=False,
            )
            logger.info(f"JSON saved to {path}")
            return path

        except Exception as e:
            logger.error(f"JSON save failed: {e}")
            return None

    async def get_dashboard_data(
        self,
        org_id: str,
        folder_id: str,
        report_type: ReportType,
        date_range: Optional[DateRange] = None,
    ) -> DashboardData:
        """Get data formatted for dashboard display.

        Quick analysis without full report generation.
        """
        # Load and analyze data
        extracted_data = await self._load_extracted_data(org_id, folder_id, date_range)
        dataset = await self._aggregate_data(extracted_data, report_type)
        analysis = await self._run_analysis(dataset, report_type, ReportOptions())

        # Generate quick insights
        insights = await self._generate_insights(analysis, report_type, ReportOptions(max_insights=3))

        # Build dashboard data
        total = analysis.get("total_amount", 0)

        return DashboardData(
            summary_metrics=[
                {"label": "Total Spend", "value": total, "format": "currency"},
                {"label": "Documents", "value": analysis.get("document_count", 0), "format": "number"},
                {"label": "Records", "value": analysis.get("record_count", 0), "format": "number"},
                {"label": "Vendors", "value": analysis.get("vendor_count", 0), "format": "number"},
            ],
            charts=[
                ChartData(
                    chart_type="pie",
                    title="Spending by Category",
                    data={"items": analysis.get("by_category", [])},
                ),
                ChartData(
                    chart_type="bar",
                    title="Top Vendors",
                    data={"items": analysis.get("by_vendor", [])[:10]},
                ),
                ChartData(
                    chart_type="line",
                    title="Monthly Trend",
                    data={"items": analysis.get("monthly_trends", [])},
                ),
            ],
            tables=[
                {
                    "title": "Top Expenses",
                    "rows": analysis.get("top_expenses", [])[:5],
                }
            ],
            insights=insights,
            generated_at=datetime.utcnow(),
        )


# Singleton instance
_report_agent: Optional[ReportAgent] = None


def get_report_agent() -> ReportAgent:
    """Get or create the ReportAgent singleton."""
    global _report_agent
    if _report_agent is None:
        _report_agent = ReportAgent()
    return _report_agent


async def shutdown_report_agent():
    """Shutdown the ReportAgent singleton."""
    global _report_agent
    if _report_agent:
        _report_agent.shutdown()
        _report_agent = None
