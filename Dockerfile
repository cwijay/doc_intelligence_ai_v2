# =============================================================================
# Document Intelligence AI v3.0 - Production Dockerfile
# =============================================================================
# Build: docker build -t doc-intelligence-ai .
# Run:   docker run -p 8080:8080 --env-file .env doc-intelligence-ai
# =============================================================================

# -----------------------------------------------------------------------------
# Builder Stage
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS builder

# Install build dependencies
# - gcc: Required for some Python packages with C extensions
# - libpq-dev: Required for psycopg2/asyncpg PostgreSQL drivers
# - git: Required for biz2bricks-core from GitHub
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Production Stage
# -----------------------------------------------------------------------------
FROM python:3.12-slim

# Install runtime dependencies only
# Note: WeasyPrint dependencies (pango, gdk-pixbuf, etc.) for PDF report generation
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    # WeasyPrint dependencies for PDF generation
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf-2.0-0 \
    libffi8 \
    libcairo2 \
    libharfbuzz0b \
    fontconfig \
    fonts-liberation \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid 1000 --shell /bin/bash --create-home appuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser scripts/ ./scripts/

# Switch to non-root user
USER appuser

# Cloud Run uses PORT env var (default 8080)
ENV PORT=8080
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check for container orchestration
# Cloud Run has its own health checks, but this is useful for local testing
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Start the application
# Using shell form to expand PORT environment variable
CMD uvicorn src.main:app --host 0.0.0.0 --port ${PORT}
