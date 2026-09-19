#!/usr/bin/env bash
# =============================================================================
# Run the Document Intelligence AI service on this machine, against the local
# Postgres container the backend API repo owns (b2blocal-postgres).
#
# Cloud SQL is off. GCS, Gemini File Search, LlamaParse and OpenAI stay remote —
# those are billed per use, not per hour, so there is nothing to save by
# emulating them.
#
# Usage:
#   ./start_local.sh                  # 127.0.0.1:8001, reload on
#   ./start_local.sh --port 8002
#   ./start_local.sh --no-reload
#   ./start_local.sh --host 0.0.0.0   # refuses unless --i-know, see below
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="$SCRIPT_DIR/.env.local"
BACKEND_REPO="${B2B_BACKEND_REPO:-$REPO_ROOT/../doc_intelligence_backend_api_v2.0}"
BACKEND_ENV="$BACKEND_REPO/scripts/local_exec/.env.local"
PG_CONTAINER="b2blocal-postgres"
VENV_DIR="$REPO_ROOT/.venv"

RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'; BLUE=$'\033[0;34m'; NC=$'\033[0m'
info() { echo "${BLUE}[INFO]${NC} $*"; }
ok()   { echo "${GREEN}[OK]${NC} $*"; }
warn() { echo "${YELLOW}[WARN]${NC} $*"; }
die()  { echo "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# Captured separately from HOST/PORT because .env.local is sourced further down
# and would otherwise clobber them — silently ignoring the flag the user passed.
CLI_HOST=""
CLI_PORT=""
RELOAD=true
FORCE_BIND=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --host) CLI_HOST="$2"; shift 2 ;;
        --port) CLI_PORT="$2"; shift 2 ;;
        --no-reload) RELOAD=false; shift ;;
        --i-know) FORCE_BIND=true; shift ;;
        -h|--help) sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "Unknown option: $1" ;;
    esac
done

# This service has no authentication. get_org_id in src/api/dependencies.py
# checks that the user in X-User-ID belongs to the org in X-Organization-ID,
# but both headers are self-asserted — anyone who can reach the port can claim
# any identity. Loopback is the only thing keeping that honest locally.
HOST="${CLI_HOST:-127.0.0.1}"
if [[ "$HOST" != "127.0.0.1" && "$HOST" != "localhost" && "$FORCE_BIND" != true ]]; then
    die "Refusing to bind to $HOST.
       Identity here comes from unverified request headers, so exposing this
       port hands any caller every organization's documents.
       Pass --i-know only if you genuinely intend that."
fi

[[ -f "$ENV_FILE" ]] || die "$ENV_FILE not found. Run ./setup_local.sh first."
[[ -d "$VENV_DIR" ]]  || die "No virtualenv at $VENV_DIR."

# ----------------------------------------------------------------- app env ---
# Loaded first so the database block below always wins: src/db/connection.py
# calls load_dotenv(), which does not override variables already exported, so
# whatever we export here beats the repo's Cloud SQL .env.
set -a; source "$ENV_FILE"; set +a

# Re-assert the command line over the env file: sourcing above may have set
# PORT/HOST, and an explicit flag must win over a file default.
PORT="${CLI_PORT:-${PORT:-8001}}"
HOST="${CLI_HOST:-$HOST}"

# ------------------------------------------------------- infra preflight ------
docker info >/dev/null 2>&1 || die "Docker daemon not reachable. Start Docker, then ./setup_local.sh"
state=$(docker inspect "$PG_CONTAINER" --format '{{.State.Status}}' 2>/dev/null || echo missing)
[[ "$state" == running ]] || die "Container $PG_CONTAINER is '$state'. Run ./setup_local.sh first."
ok "Postgres container running"

# ------------------------------------------------------ database credentials -
# Read live from the backend repo rather than duplicated into our .env.local,
# so a password rotated there is picked up on the next start.
[[ -f "$BACKEND_ENV" ]] || die "Backend env file not found: $BACKEND_ENV
       It holds the Postgres password. Run ./setup_local.sh for the full diagnosis."
backend_var() { sed -n "s/^$1=//p" "$BACKEND_ENV" | tail -1 | tr -d '"'; }

DATABASE_USER="$(backend_var DATABASE_USER)";            DATABASE_USER="${DATABASE_USER:-postgres}"
DATABASE_NAME="$(backend_var DATABASE_NAME)";            DATABASE_NAME="${DATABASE_NAME:-doc_intelligence}"
DATABASE_PASSWORD="$(backend_var POSTGRES_PASSWORD)"
POSTGRES_HOST_PORT="$(backend_var POSTGRES_HOST_PORT)";  POSTGRES_HOST_PORT="${POSTGRES_HOST_PORT:-15432}"
BOOTSTRAP_ORG_ID="$(backend_var BOOTSTRAP_ORG_ID)"
[[ -n "$DATABASE_PASSWORD" ]] || die "POSTGRES_PASSWORD missing from $BACKEND_ENV"

export DATABASE_ENABLED=true
export USE_CLOUD_SQL_CONNECTOR=false
# Blanked, not merely unused: leaving a Cloud SQL instance name in the
# environment makes the failure modes of a misconfigured run much harder to read.
export CLOUD_SQL_INSTANCE=""
export DATABASE_NAME DATABASE_USER DATABASE_PASSWORD
export DATABASE_URL="postgresql+asyncpg://${DATABASE_USER}:${DATABASE_PASSWORD}@127.0.0.1:${POSTGRES_HOST_PORT}/${DATABASE_NAME}"
export DB_POOL_SIZE="${DB_POOL_SIZE:-3}"
export DB_BACKGROUND_POOL_SIZE="${DB_BACKGROUND_POOL_SIZE:-2}"
export DB_MAX_OVERFLOW="${DB_MAX_OVERFLOW:-5}"

# Fail here, with the password in hand, rather than on the first request.
if command -v docker >/dev/null; then
    docker exec -e PGPASSWORD="$DATABASE_PASSWORD" "$PG_CONTAINER" \
        psql -U "$DATABASE_USER" -d "$DATABASE_NAME" -tAc 'SELECT 1' >/dev/null 2>&1 \
        || die "Cannot authenticate to $DATABASE_NAME with the password from $BACKEND_ENV.
       If the backend rotated it, re-run its setup_infra.sh."
fi
ok "Database reachable (${DATABASE_NAME} on 127.0.0.1:${POSTGRES_HOST_PORT})"

# ---------------------------------------------------------- object storage ---
# A service-account key rather than personal ADC: ADC cannot sign URLs without
# an iam.serviceAccounts.signBlob grant, so the signed upload URLs that bulk
# processing hands out would fail even while plain reads worked.
SA_KEY="${GCP_SA_KEY_FILE:-$REPO_ROOT/../biz2bricks_stack/secrets/gcp-sa-key.json}"
[[ "$SA_KEY" != /* ]] && SA_KEY="$REPO_ROOT/$SA_KEY"
if [[ -f "$SA_KEY" ]]; then
    export GOOGLE_APPLICATION_CREDENTIALS="$SA_KEY"
    STORAGE_DESC="gs://${GCS_BUCKET:-unset}${GCS_PREFIX:+/$GCS_PREFIX}  (service account)"
    ok "GCS via service-account key"
elif [[ -f "$HOME/.config/gcloud/application_default_credentials.json" ]]; then
    STORAGE_DESC="gs://${GCS_BUCKET:-unset}${GCS_PREFIX:+/$GCS_PREFIX}  (ADC — signed URLs will fail)"
    warn "No key at $SA_KEY; falling back to ADC."
    warn "Reads and writes work, but signed upload URLs (bulk processing) will not."
else
    die "No GCS credentials. Either place a service-account key at $SA_KEY
       (or set GCP_SA_KEY_FILE in $ENV_FILE), or run:
         gcloud auth application-default login"
fi

# ------------------------------------------------------------------- launch --
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
command -v uvicorn >/dev/null || die "uvicorn not in the venv. pip install -r requirements.txt"

cd "$REPO_ROOT"
RELOAD_ARGS=()
# --reload-dir src matters: without it, edits under .venv retrigger the reload
# loop and the server never settles.
[[ "$RELOAD" == true ]] && RELOAD_ARGS=(--reload --reload-dir src)

cat <<EOF

${GREEN}Starting Document Intelligence AI service${NC}
  URL        http://${HOST}:${PORT}
  Docs       http://${HOST}:${PORT}/docs
  Database   127.0.0.1:${POSTGRES_HOST_PORT}/${DATABASE_NAME}  (Cloud SQL off)
  Storage    ${STORAGE_DESC}
  Org ID     ${BOOTSTRAP_ORG_ID:-<run ./setup_local.sh>}
  Reload     ${RELOAD}

EOF

# macOS ships bash 3.2, where "\${arr[@]}" on an empty array trips `set -u`.
exec uvicorn src.main:app --host "$HOST" --port "$PORT" ${RELOAD_ARGS[@]+"${RELOAD_ARGS[@]}"}
