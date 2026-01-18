#!/usr/bin/env bash
# =============================================================================
# Shared configuration for GCP deployment scripts
# Document Intelligence AI v3.0
# =============================================================================

# Exit on error
set -euo pipefail

# =============================================================================
# GCP Project Configuration
# Override these via environment variables or modify defaults below
# =============================================================================
export GCP_PROJECT="${GCP_PROJECT:-biz2bricks-dev-v1}"
export GCP_REGION="${GCP_REGION:-us-central1}"
export GCP_ZONE="${GCP_ZONE:-us-central1-a}"

# =============================================================================
# Cloud Run Configuration
# =============================================================================
export SERVICE_NAME="${SERVICE_NAME:-doc-intelligence-ai}"
export CLOUD_RUN_MEMORY="${CLOUD_RUN_MEMORY:-2Gi}"
export CLOUD_RUN_CPU="${CLOUD_RUN_CPU:-2}"
export CLOUD_RUN_MIN_INSTANCES="${CLOUD_RUN_MIN_INSTANCES:-0}"
export CLOUD_RUN_MAX_INSTANCES="${CLOUD_RUN_MAX_INSTANCES:-10}"
export CLOUD_RUN_CONCURRENCY="${CLOUD_RUN_CONCURRENCY:-80}"
export CLOUD_RUN_TIMEOUT="${CLOUD_RUN_TIMEOUT:-300}"

# =============================================================================
# Cloud SQL Configuration
# =============================================================================
export CLOUD_SQL_INSTANCE="${CLOUD_SQL_INSTANCE:-biz2bricks-dev-v1:us-central1:biz-2-bricks-intelli-doc-dev}"
export DATABASE_NAME="${DATABASE_NAME:-doc_intelligence}"
export DATABASE_USER="${DATABASE_USER:-postgres}"

# =============================================================================
# GCS Configuration
# =============================================================================
export GCS_BUCKET="${GCS_BUCKET:-biz2bricks-dev-v1-document-store}"
export GCS_PREFIX="${GCS_PREFIX:-}"
export PARSED_DIRECTORY="${PARSED_DIRECTORY:-parsed}"
export GENERATED_DIRECTORY="${GENERATED_DIRECTORY:-generated}"

# =============================================================================
# Secret Manager Secret Names
# =============================================================================
export SECRET_OPENAI_API_KEY="doc-intelligence-openai-api-key"
export SECRET_GOOGLE_API_KEY="doc-intelligence-google-api-key"
export SECRET_LLAMA_API_KEY="doc-intelligence-llama-api-key"
export SECRET_DATABASE_PASSWORD="doc-intelligence-db-password"
export SECRET_WEBHOOK_SECRET="doc-intelligence-webhook-secret"

# =============================================================================
# Container Registry
# =============================================================================
export ARTIFACT_REGISTRY="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/doc-intelligence"
export IMAGE_NAME="${ARTIFACT_REGISTRY}/${SERVICE_NAME}"

# =============================================================================
# Optional Configuration (with defaults matching .env_template)
# =============================================================================
export LOG_LEVEL="${LOG_LEVEL:-INFO}"
export DEBUG="${DEBUG:-false}"
export DATABASE_ENABLED="${DATABASE_ENABLED:-true}"
export USE_CLOUD_SQL_CONNECTOR="${USE_CLOUD_SQL_CONNECTOR:-true}"
export CLOUD_SQL_IP_TYPE="${CLOUD_SQL_IP_TYPE:-PRIVATE}"

# Agent configuration
export OPENAI_SHEET_MODEL="${OPENAI_SHEET_MODEL:-gpt-5.1-codex-mini}"
export DOCUMENT_AGENT_MODEL="${DOCUMENT_AGENT_MODEL:-gpt-5-mini}"
export OPENAI_TEMPERATURE="${OPENAI_TEMPERATURE:-0.1}"
export DOCUMENT_AGENT_TEMPERATURE="${DOCUMENT_AGENT_TEMPERATURE:-0.3}"

# Database pool configuration
export DB_POOL_SIZE="${DB_POOL_SIZE:-3}"
export DB_BACKGROUND_POOL_SIZE="${DB_BACKGROUND_POOL_SIZE:-2}"
export DB_MAX_OVERFLOW="${DB_MAX_OVERFLOW:-5}"

# =============================================================================
# Utility Functions
# =============================================================================

log_info() {
    echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') - $*"
}

log_error() {
    echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') - $*" >&2
}

log_warn() {
    echo "[WARN] $(date '+%Y-%m-%d %H:%M:%S') - $*"
}

log_success() {
    echo "[SUCCESS] $(date '+%Y-%m-%d %H:%M:%S') - $*"
}

check_gcloud_auth() {
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null | head -1 > /dev/null; then
        log_error "Not authenticated with gcloud. Run: gcloud auth login"
        exit 1
    fi
    local account
    account=$(gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null | head -1)
    log_info "Authenticated as: ${account}"
}

set_project() {
    gcloud config set project "${GCP_PROJECT}" --quiet
    log_info "Project set to: ${GCP_PROJECT}"
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running. Please start Docker."
        exit 1
    fi
    log_info "Docker is available and running"
}

get_project_number() {
    gcloud projects describe "${GCP_PROJECT}" --format="value(projectNumber)"
}

get_service_url() {
    gcloud run services describe "${SERVICE_NAME}" \
        --project="${GCP_PROJECT}" \
        --region="${GCP_REGION}" \
        --format="value(status.url)" 2>/dev/null || echo ""
}
