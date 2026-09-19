#!/usr/bin/env bash
# =============================================================================
# Start, stop and inspect the Document Intelligence AI service as a background
# process, so you get your terminal back.
#
#   ./ai_server.sh start [--port N] [--no-reload]
#   ./ai_server.sh stop
#   ./ai_server.sh restart [--port N] [--no-reload]
#   ./ai_server.sh status
#   ./ai_server.sh logs [-f] [-n N]
#
# This is a wrapper around start_local.sh, which still runs the server in the
# foreground when you want it there. All the environment, credential and
# preflight logic lives there and is not duplicated here.
#
# `stop` stops only this service. The Postgres container is shared with the
# backend API on :8000 — stopping it would break that session — so it is left
# alone. To stop it too, use the backend repo's stop_infra.sh.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="$SCRIPT_DIR/.env.local"
BACKEND_REPO="${B2B_BACKEND_REPO:-$REPO_ROOT/../doc_intelligence_backend_api_v2.0}"
RUN_DIR="$SCRIPT_DIR/.run"
PID_FILE="$RUN_DIR/ai_server.pid"
PORT_FILE="$RUN_DIR/ai_server.port"
LOG_FILE="$RUN_DIR/ai_server.log"
STOP_GRACE_SECONDS=15

RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'; BLUE=$'\033[0;34m'; NC=$'\033[0m'
info() { echo "${BLUE}[INFO]${NC} $*"; }
ok()   { echo "${GREEN}[OK]${NC} $*"; }
warn() { echo "${YELLOW}[WARN]${NC} $*"; }
die()  { echo "${RED}[ERROR]${NC} $*" >&2; exit 1; }

mkdir -p "$RUN_DIR"

# --------------------------------------------------------------- primitives --
# The port is the source of truth for "is it up", not the PID file: with
# --reload uvicorn runs a supervisor plus a worker, and a stale PID file
# outlives a crash. The PID file only records who we started.

running_pid() {  # echoes the recorded pid if that process is alive
    [[ -f "$PID_FILE" ]] || return 1
    local pid; pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && { echo "$pid"; return 0; }
    return 1
}

port_pids() {  # echoes every pid listening on $1, newest last
    lsof -nP -iTCP:"$1" -sTCP:LISTEN -t 2>/dev/null || true
}

resolved_port() {
    if [[ -n "${OPT_PORT:-}" ]]; then echo "$OPT_PORT"; return; fi
    if [[ -f "$PORT_FILE" ]]; then cat "$PORT_FILE"; return; fi
    # Fall back to the configured default so `status` works before a first start.
    local p=""
    [[ -f "$ENV_FILE" ]] && p="$(sed -n 's/^PORT=//p' "$ENV_FILE" | tail -1 | tr -d '"[:space:]')"
    echo "${p:-8001}"
}

health_code() {
    curl -s -m 3 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$1/health" 2>/dev/null || echo 000
}

# -------------------------------------------------------------------- start --
cmd_start() {
    local port; port="$(resolved_port)"

    if pid="$(running_pid)"; then
        die "Already running (pid $pid, port $(cat "$PORT_FILE" 2>/dev/null || echo '?')).
       Use ./ai_server.sh restart, or stop first."
    fi

    # A port held by something we did not start is a different problem, and
    # silently killing a stranger's process is not this script's business.
    local holders; holders="$(port_pids "$port")"
    if [[ -n "$holders" ]]; then
        die "Port $port is already held by pid(s): $(echo "$holders" | tr '\n' ' ')
       That is not a server this script started. Free the port, or pass --port N."
    fi

    [[ -x "$SCRIPT_DIR/start_local.sh" ]] || die "start_local.sh not found or not executable"

    local args=(--port "$port")
    [[ "${OPT_NO_RELOAD:-false}" == true ]] && args+=(--no-reload)

    info "Starting on port $port (log: $LOG_FILE)"
    : > "$LOG_FILE"
    # start_local.sh ends in `exec uvicorn`, so this pid becomes uvicorn itself
    # rather than a shell wrapper that outlives it.
    nohup "$SCRIPT_DIR/start_local.sh" "${args[@]}" >>"$LOG_FILE" 2>&1 &
    local pid=$!
    echo "$pid"  > "$PID_FILE"
    echo "$port" > "$PORT_FILE"

    # Wait for readiness rather than reporting success on a process that is
    # about to die on a bad credential or a missing env file.
    local code=000
    for _ in $(seq 1 60); do
        if ! kill -0 "$pid" 2>/dev/null; then
            rm -f "$PID_FILE"
            echo; warn "Process exited during startup. Last lines:"
            tail -20 "$LOG_FILE" >&2
            die "Failed to start. Full log: $LOG_FILE"
        fi
        code="$(health_code "$port")"
        [[ "$code" == 200 ]] && break
        sleep 1
    done

    [[ "$code" == 200 ]] || {
        warn "Started (pid $pid) but /health did not return 200 within 60s (last: $code)"
        warn "Check: ./ai_server.sh logs"
        exit 1
    }

    ok "Running (pid $pid)"
    grep -E '^  (URL|Docs|Database|Storage|Org ID|Reload) ' "$LOG_FILE" | head -6 || true
    echo
    echo "  Logs    ./ai_server.sh logs -f"
    echo "  Stop    ./ai_server.sh stop"
    echo
    echo "  Wanted the logs in this terminal instead? Stop this, then run"
    echo "  ./start_local.sh — same server, foreground, Ctrl+C to stop."
}

# --------------------------------------------------------------------- stop --
cmd_stop() {
    local port; port="$(resolved_port)"
    local stopped=false

    if pid="$(running_pid)"; then
        info "Stopping pid $pid"
        kill -TERM "$pid" 2>/dev/null || true
        for _ in $(seq 1 "$STOP_GRACE_SECONDS"); do
            kill -0 "$pid" 2>/dev/null || { stopped=true; break; }
            sleep 1
        done
        if [[ "$stopped" != true ]]; then
            warn "Did not exit after ${STOP_GRACE_SECONDS}s — sending SIGKILL"
            kill -KILL "$pid" 2>/dev/null || true
            sleep 1
        fi
        ok "Stopped (pid $pid)"
    else
        info "No recorded process running"
    fi
    rm -f "$PID_FILE"

    # With --reload the supervisor spawns a worker; if the supervisor was killed
    # hard, the worker can survive and keep the port. Sweep by port rather than
    # by process name, so we only ever touch something bound to our own port.
    local leftovers; leftovers="$(port_pids "$port")"
    if [[ -n "$leftovers" ]]; then
        warn "Port $port still held by: $(echo "$leftovers" | tr '\n' ' ') — terminating"
        echo "$leftovers" | xargs kill -TERM 2>/dev/null || true
        sleep 2
        leftovers="$(port_pids "$port")"
        [[ -n "$leftovers" ]] && { echo "$leftovers" | xargs kill -KILL 2>/dev/null || true; sleep 1; }
    fi

    [[ -z "$(port_pids "$port")" ]] || die "Port $port is still held. Check manually: lsof -nP -iTCP:$port"
    rm -f "$PORT_FILE"

    echo
    echo "  b2blocal-postgres left running — the backend API on :8000 shares it."
    echo "  To stop it too: $BACKEND_REPO/scripts/local_exec/stop_infra.sh"
}

# ------------------------------------------------------------------- status --
cmd_status() {
    local port; port="$(resolved_port)"
    local pid; pid="$(running_pid || true)"
    local code; code="$(health_code "$port")"

    echo
    if [[ -n "$pid" && "$code" == 200 ]]; then
        echo "${GREEN}AI service: running${NC}   pid $pid   port $port"
    elif [[ "$code" == 200 ]]; then
        echo "${YELLOW}AI service: running, but not started by this script${NC}   port $port"
        echo "  Holding pid(s): $(port_pids "$port" | tr '\n' ' ')"
    elif [[ -n "$pid" ]]; then
        echo "${YELLOW}AI service: process alive (pid $pid) but /health returned $code${NC}"
    else
        echo "${RED}AI service: not running${NC}   (expected port $port)"
    fi

    if [[ "$code" == 200 ]]; then
        # Passed through the environment and read by a quoted heredoc: nesting
        # quotes inside `python3 -c '...'` silently produced a SyntaxError that
        # 2>/dev/null then hid, so status printed nothing useful.
        HEALTH_JSON="$(curl -s -m 5 "http://127.0.0.1:$port/health")" python3 <<'PY' || true
import json, os
d = json.loads(os.environ["HEALTH_JSON"])
print("  %-15s%s  v%s" % ("status", d.get("status"), d.get("version")))
for name, c in (d.get("components") or {}).items():
    msg = c.get("message")
    print("  %-15s%s%s" % (name, c.get("status"), "  (%s)" % msg if msg else ""))
PY
    fi

    # The backend is a separate service; report it so a confusing "why is my
    # request 404ing" has an obvious first thing to check.
    echo "  backend    :8000 -> $(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/health 2>/dev/null || echo 'not running')"
    local pgstate; pgstate="$(docker inspect b2blocal-postgres --format '{{.State.Status}}' 2>/dev/null || echo missing)"
    echo "  postgres   b2blocal-postgres -> $pgstate"
    echo
    if [[ -f "$LOG_FILE" ]]; then echo "  Log  $LOG_FILE"; fi

    # Deliberate exit contract, so `if ./ai_server.sh status; then ...` works:
    # 0 when the service answers /health, 1 otherwise.
    [[ "$code" == 200 ]]
}

# --------------------------------------------------------------------- logs --
cmd_logs() {
    [[ -f "$LOG_FILE" ]] || die "No log file yet at $LOG_FILE. Start the server first."
    tail ${LOG_FOLLOW:+-f} -n "${LOG_LINES:-50}" "$LOG_FILE"
}

# --------------------------------------------------------------------- main --
COMMAND="${1:-}"; shift || true
OPT_PORT=""; OPT_NO_RELOAD=false; LOG_FOLLOW=""; LOG_LINES=50
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)      OPT_PORT="$2"; shift 2 ;;
        --no-reload) OPT_NO_RELOAD=true; shift ;;
        -f|--follow) LOG_FOLLOW=1; shift ;;
        -n)          LOG_LINES="$2"; shift 2 ;;
        *) die "Unknown option: $1" ;;
    esac
done

case "$COMMAND" in
    start)   cmd_start ;;
    stop)    cmd_stop ;;
    restart) cmd_stop; echo; cmd_start ;;
    status)  cmd_status ;;
    logs)    cmd_logs ;;
    ""|-h|--help) sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//' ;;
    *) die "Unknown command: $COMMAND (expected start, stop, restart, status or logs)" ;;
esac
