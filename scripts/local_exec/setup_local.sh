#!/usr/bin/env bash
# =============================================================================
# Prepare this machine to run the Document Intelligence AI service locally.
#
# This service does NOT own any infrastructure. The backend API repo
# (doc_intelligence_backend_api_v2.0) already runs a Postgres container —
# b2blocal-postgres — holding the shared biz2bricks_core schema, and both
# services are meant to talk to the same database. This script joins that
# stack rather than standing up a second Postgres.
#
# What it does:
#   1. Verifies b2blocal-postgres is up (delegates to the backend's
#      setup_infra.sh if it is not)
#   2. Verifies pgvector, which the RAG semantic cache needs
#   3. Applies any tables this repo declares that are not there yet
#   4. Seeds subscription tiers if the table is empty
#   5. Generates .env.local from .env.local-gcp, minus the Cloud SQL settings
#
# Idempotent: safe to re-run. It never destroys data — teardown belongs to the
# backend repo, which owns the containers.
#
# Usage:
#   ./setup_local.sh                        # prepare everything
#   ./setup_local.sh --status               # show infra and database state
#   ./setup_local.sh --migrate-agent-builder  # also apply the agent_builder tables
#
# Afterwards: ./start_local.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="$SCRIPT_DIR/.env.local"
SEED_ENV="${SEED_ENV:-$REPO_ROOT/.env.local-gcp}"
BACKEND_REPO="${B2B_BACKEND_REPO:-$REPO_ROOT/../doc_intelligence_backend_api_v2.0}"
BACKEND_ENV="$BACKEND_REPO/scripts/local_exec/.env.local"
PG_CONTAINER="b2blocal-postgres"
VENV_PYTHON="$REPO_ROOT/.venv/bin/python"

RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'; BLUE=$'\033[0;34m'; NC=$'\033[0m'
info() { echo "${BLUE}[INFO]${NC} $*"; }
ok()   { echo "${GREEN}[OK]${NC} $*"; }
warn() { echo "${YELLOW}[WARN]${NC} $*"; }
die()  { echo "${RED}[ERROR]${NC} $*" >&2; exit 1; }

STATUS_ONLY=false
MIGRATE_AGENT_BUILDER=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --status)                 STATUS_ONLY=true; shift ;;
        --migrate-agent-builder)  MIGRATE_AGENT_BUILDER=true; shift ;;
        -h|--help) sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "Unknown option: $1" ;;
    esac
done

command -v docker >/dev/null || die "docker not found"
docker info >/dev/null 2>&1 || die "Docker daemon not reachable. Start Docker Desktop first."
[[ -x "$VENV_PYTHON" ]] || die "No virtualenv at $REPO_ROOT/.venv. Create it and pip install -r requirements.txt."

# ------------------------------------------------------- backend credentials -
# The backend generates a random Postgres password into its own .env.local. We
# read it rather than copy it, so rotating it there does not silently leave
# this service pointing at a password that no longer works.
[[ -f "$BACKEND_ENV" ]] || die "Backend env file not found: $BACKEND_ENV
       The backend API repo owns the local Postgres container. Run its setup first:
         $BACKEND_REPO/scripts/local_exec/setup_infra.sh
       Or point B2B_BACKEND_REPO at the repo if it lives elsewhere."

backend_var() { sed -n "s/^$1=//p" "$BACKEND_ENV" | tail -1 | tr -d '"'; }

DATABASE_USER="$(backend_var DATABASE_USER)";      DATABASE_USER="${DATABASE_USER:-postgres}"
DATABASE_NAME="$(backend_var DATABASE_NAME)";      DATABASE_NAME="${DATABASE_NAME:-doc_intelligence}"
POSTGRES_PASSWORD="$(backend_var POSTGRES_PASSWORD)"
POSTGRES_HOST_PORT="$(backend_var POSTGRES_HOST_PORT)"; POSTGRES_HOST_PORT="${POSTGRES_HOST_PORT:-15432}"
BOOTSTRAP_ORG_ID="$(backend_var BOOTSTRAP_ORG_ID)"

[[ -n "$POSTGRES_PASSWORD" ]] || die "POSTGRES_PASSWORD missing from $BACKEND_ENV"

psql_local() {
    docker exec -e PGPASSWORD="$POSTGRES_PASSWORD" -i "$PG_CONTAINER" \
        psql -U "$DATABASE_USER" -d "$DATABASE_NAME" "$@"
}

# -------------------------------------------------------------------- status -
if [[ "$STATUS_ONLY" == true ]]; then
    echo; info "Container:"
    docker ps --filter "name=$PG_CONTAINER" \
        --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' || true
    echo; info "Database ($DATABASE_NAME on 127.0.0.1:$POSTGRES_HOST_PORT):"
    if docker exec "$PG_CONTAINER" pg_isready -U "$DATABASE_USER" -d "$DATABASE_NAME" >/dev/null 2>&1; then
        psql_local -tAc "SELECT 'tables:        '||count(*) FROM information_schema.tables WHERE table_schema='public';"
        psql_local -tAc "SELECT 'pgvector:      '||coalesce(max(extversion),'MISSING') FROM pg_extension WHERE extname='vector';"
        psql_local -tAc "SELECT 'tiers:         '||count(*) FROM subscription_tiers;" 2>/dev/null || echo "tiers:         (table missing)"
        psql_local -tAc "SELECT 'organizations: '||count(*) FROM organizations;" 2>/dev/null || true
        psql_local -tAc "SELECT 'documents:     '||count(*) FROM documents;" 2>/dev/null || true
    else
        warn "Postgres not accepting connections"
    fi
    echo; info "Env file:"
    [[ -f "$ENV_FILE" ]] && ok "$ENV_FILE" || warn "$ENV_FILE missing — run ./setup_local.sh"
    exit 0
fi

# --------------------------------------------------------------- containers --
# The backend repo owns this container, so hand off to its script rather than
# duplicating the compose definition here. Two compose files claiming the same
# container name is how you end up with a second, empty database.
state=$(docker inspect "$PG_CONTAINER" --format '{{.State.Status}}' 2>/dev/null || echo missing)
if [[ "$state" != running ]]; then
    warn "$PG_CONTAINER is '$state' — starting it via the backend repo"
    [[ -x "$BACKEND_REPO/scripts/local_exec/setup_infra.sh" ]] \
        || die "Cannot start it: $BACKEND_REPO/scripts/local_exec/setup_infra.sh not executable"
    "$BACKEND_REPO/scripts/local_exec/setup_infra.sh"
    state=$(docker inspect "$PG_CONTAINER" --format '{{.State.Status}}' 2>/dev/null || echo missing)
    [[ "$state" == running ]] || die "$PG_CONTAINER still '$state' after setup_infra.sh"
fi

for _ in $(seq 1 30); do
    health=$(docker inspect "$PG_CONTAINER" --format '{{.State.Health.Status}}' 2>/dev/null || echo starting)
    [[ "$health" == healthy ]] && break
    sleep 2
done
[[ "${health:-}" == healthy ]] || die "$PG_CONTAINER did not become healthy. Check: docker logs $PG_CONTAINER"
ok "Postgres healthy on 127.0.0.1:$POSTGRES_HOST_PORT"

# ---------------------------------------------------------------- pgvector ---
# src/db/repositories/semantic_cache_repository.py disables itself silently
# when the extension is absent, so assert it rather than discover it later.
psql_local -v ON_ERROR_STOP=1 -c "CREATE EXTENSION IF NOT EXISTS vector;" >/dev/null \
    || die "Could not enable pgvector in $DATABASE_NAME"
ok "pgvector present"

# ------------------------------------------------------------------ env file -
if [[ ! -f "$ENV_FILE" ]]; then
    [[ -f "$SEED_ENV" ]] || die "Cannot generate $ENV_FILE: seed file $SEED_ENV not found.
       Point SEED_ENV at an env file holding your API keys."
    info "Generating $ENV_FILE from $(basename "$SEED_ENV")"
    {
        cat <<'HEADER'
# Generated by setup_local.sh — local development only, never commit.
#
# Deliberately holds NO database settings. start_local.sh reads the Postgres
# credentials live from the backend repo's .env.local, so a password rotated
# there does not leave this file stale and this service failing to connect.
HEADER
        echo
        # Drop every Cloud SQL / database line from the seed: those are exactly
        # the settings this local mode replaces.
        grep -vE '^(CLOUD_SQL_[A-Z_]+|DATABASE_(URL|NAME|USER|PASSWORD|ENABLED)|USE_CLOUD_SQL_CONNECTOR|DB_(POOL_SIZE|BACKGROUND_POOL_SIZE|MAX_OVERFLOW))=' "$SEED_ENV"
        cat <<'FOOTER'

# --- local overrides ---------------------------------------------------------
# Service-account key for GCS. A key is used rather than your personal ADC
# because ADC cannot sign URLs without an iam.serviceAccounts.signBlob grant,
# which would break the signed upload URLs that bulk processing hands out.
GCP_SA_KEY_FILE=../biz2bricks_stack/secrets/gcp-sa-key.json

# Port 8001 — the backend API holds 8000.
PORT=8001
FOOTER
    } > "$ENV_FILE"
    chmod 600 "$ENV_FILE"
    ok "Created $ENV_FILE"
else
    info "Using existing $ENV_FILE"
fi

# -------------------------------------------------------------------- schema -
# The backend created the shared biz2bricks_core tables already. This is a
# no-op in the normal case; it exists to catch drift when this repo's models
# move ahead of whatever the backend last created.
info "Reconciling schema with this repo's models"
BEFORE=$(psql_local -tAc "SELECT count(*) FROM information_schema.tables WHERE table_schema='public';" | tr -d '[:space:]')

DB_ENV=(
    "USE_CLOUD_SQL_CONNECTOR=false"
    "CLOUD_SQL_INSTANCE="
    "DATABASE_ENABLED=true"
    "DATABASE_NAME=$DATABASE_NAME"
    "DATABASE_USER=$DATABASE_USER"
    "DATABASE_PASSWORD=$POSTGRES_PASSWORD"
    "DATABASE_URL=postgresql+asyncpg://${DATABASE_USER}:${POSTGRES_PASSWORD}@127.0.0.1:${POSTGRES_HOST_PORT}/${DATABASE_NAME}"
)

(cd "$REPO_ROOT" && env "${DB_ENV[@]}" "$VENV_PYTHON" scripts/db_setup.py setup) >/dev/null 2>&1 \
    || die "Schema reconciliation failed. Re-run verbosely:
       cd $REPO_ROOT && env ${DB_ENV[*]} .venv/bin/python scripts/db_setup.py setup"

AFTER=$(psql_local -tAc "SELECT count(*) FROM information_schema.tables WHERE table_schema='public';" | tr -d '[:space:]')
if [[ "$AFTER" -gt "$BEFORE" ]]; then
    ok "Schema reconciled ($BEFORE -> $AFTER tables)"
else
    ok "Schema already current ($AFTER tables)"
fi

# --------------------------------------------------------------------- tiers -
TIERS=$(psql_local -tAc "SELECT count(*) FROM subscription_tiers;" 2>/dev/null | tr -d '[:space:]' || echo 0)
if [[ "${TIERS:-0}" -eq 0 ]]; then
    info "Seeding subscription tiers"
    (cd "$REPO_ROOT" && env "${DB_ENV[@]}" "$VENV_PYTHON" scripts/seed_tiers.py) >/dev/null 2>&1 || true
    TIERS=$(psql_local -tAc "SELECT count(*) FROM subscription_tiers;" | tr -d '[:space:]')
    [[ "${TIERS:-0}" -gt 0 ]] && ok "Subscription tiers seeded ($TIERS)" || warn "Tier seeding produced no rows"
else
    ok "Subscription tiers present ($TIERS)"
fi

# ------------------------------------------------------------ agent builder --
# Nothing under src/ reads these tables — only scripts/apply_agent_builder_schema.py
# writes them — so this stays opt-in rather than part of the default path.
if [[ "$MIGRATE_AGENT_BUILDER" == true ]]; then
    info "Applying agent_builder schema"
    (cd "$REPO_ROOT" && env "${DB_ENV[@]}" "$VENV_PYTHON" scripts/apply_agent_builder_schema.py) \
        || die "agent_builder migration failed"
    ok "agent_builder tables applied"
fi

# ---------------------------------------------------------------------- done -
ORG_COUNT=$(psql_local -tAc "SELECT count(*) FROM organizations;" | tr -d '[:space:]')
SAMPLE_USER=$(psql_local -tAc "SELECT email FROM users ORDER BY created_at LIMIT 1;" 2>/dev/null | tr -d '[:space:]')

cat <<EOF

${GREEN}Local environment ready.${NC}

  Postgres   127.0.0.1:${POSTGRES_HOST_PORT}   db=${DATABASE_NAME} (shared with the backend API)
  Tables     ${AFTER}          Tiers: ${TIERS}          Organizations: ${ORG_COUNT}
  Org ID     ${BOOTSTRAP_ORG_ID:-<none — see backend setup_infra.sh>}
  Env file   ${ENV_FILE}

Next:
  ./start_local.sh

Every request needs both headers — org alone is rejected by get_org_id:
  -H 'X-Organization-ID: ${BOOTSTRAP_ORG_ID:-<org-uuid>}'
  -H 'X-User-Email: ${SAMPLE_USER:-<user-email>}'

psql:
  docker exec -it ${PG_CONTAINER} psql -U ${DATABASE_USER} -d ${DATABASE_NAME}

Containers belong to the backend repo. Stop them there:
  ${BACKEND_REPO}/scripts/local_exec/stop_infra.sh
EOF
