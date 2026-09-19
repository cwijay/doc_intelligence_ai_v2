#!/usr/bin/env bash
# =============================================================================
# Run the Document Intelligence API locally, pointing at GCP resources.
# =============================================================================
# What this script does:
#   1. Loads variables from .env.local-gcp (GCP Cloud SQL + GCS)
#   2. Verifies gcloud ADC credentials exist (needed by Cloud SQL Connector,
#      GCS client, and Vertex/Gemini clients)
#   3. Optionally applies pending schema migrations (--migrate)
#   4. Starts uvicorn with --reload on port 8001
#
# Usage:
#   ./scripts/start_local_gcp.sh                # just start the server
#   ./scripts/start_local_gcp.sh --migrate      # apply agent_builder migration first
#   ./scripts/start_local_gcp.sh --db-status    # show DB status and exit
#   PORT=8080 ./scripts/start_local_gcp.sh      # override port
#
# Prerequisites (one-time):
#   gcloud auth login
#   gcloud auth application-default login
#   gcloud config set project biz2bricks-dev-v1
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-$REPO_ROOT/.env.local-gcp}"
PORT="${PORT:-8001}"
HOST="${HOST:-0.0.0.0}"
VENV_ACTIVATE="$REPO_ROOT/.venv/bin/activate"

# ---- arg parsing -----------------------------------------------------------
RUN_MIGRATE=0
DB_STATUS=0
EXTRA_UVICORN_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --migrate)    RUN_MIGRATE=1; shift ;;
    --db-status)  DB_STATUS=1; shift ;;
    --no-reload)  EXTRA_UVICORN_ARGS+=("--no-reload"); shift ;;
    *)            EXTRA_UVICORN_ARGS+=("$1"); shift ;;
  esac
done

# ---- helpers ---------------------------------------------------------------
log()  { printf "\033[1;34m[local-gcp]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[local-gcp]\033[0m %s\n" "$*" >&2; }
fail() { printf "\033[1;31m[local-gcp]\033[0m %s\n" "$*" >&2; exit 1; }

# ---- sanity checks ---------------------------------------------------------
[[ -f "$ENV_FILE" ]] || fail "Env file not found: $ENV_FILE"

command -v gcloud >/dev/null 2>&1 || fail "gcloud CLI not found on PATH"

# Application Default Credentials (used by Cloud SQL Connector + GCS client)
ADC_FILE="${HOME}/.config/gcloud/application_default_credentials.json"
if [[ ! -f "$ADC_FILE" ]]; then
  fail "No ADC credentials at $ADC_FILE. Run: gcloud auth application-default login"
fi

# Active gcloud account (sanity, not hard failure)
if ! gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null | grep -q '.'; then
  warn "No active gcloud account. Run: gcloud auth login"
fi

# ---- load env --------------------------------------------------------------
log "Loading env from $ENV_FILE"
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

: "${GCP_PROJECT:=biz2bricks-dev-v1}"
export GOOGLE_CLOUD_PROJECT="$GCP_PROJECT"

# ---- activate venv ---------------------------------------------------------
if [[ -f "$VENV_ACTIVATE" ]]; then
  log "Activating virtualenv at .venv"
  # shellcheck disable=SC1090
  source "$VENV_ACTIVATE"
else
  warn "No virtualenv at $VENV_ACTIVATE - using system python"
fi

# ---- optional DB ops -------------------------------------------------------
if [[ "$DB_STATUS" -eq 1 ]]; then
  log "Running db_setup.py status against Cloud SQL"
  cd "$REPO_ROOT"
  exec python scripts/db_setup.py status
fi

if [[ "$RUN_MIGRATE" -eq 1 ]]; then
  log "Applying agent_builder schema migration to Cloud SQL"
  cd "$REPO_ROOT"
  python scripts/apply_agent_builder_schema.py
fi

# ---- launch server ---------------------------------------------------------
log "Target: CloudSQL=$CLOUD_SQL_INSTANCE  DB=$DATABASE_NAME  GCS=$GCS_BUCKET"
log "Starting uvicorn on $HOST:$PORT"

cd "$REPO_ROOT"
# macOS bash 3.2 + set -u chokes on "${empty_array[@]}"; the ${arr[@]+...}
# pattern expands to nothing when unset and to the quoted elements otherwise.
exec uvicorn src.main:app --host "$HOST" --port "$PORT" --reload ${EXTRA_UVICORN_ARGS[@]+"${EXTRA_UVICORN_ARGS[@]}"}
