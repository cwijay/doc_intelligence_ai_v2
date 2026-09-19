#!/usr/bin/env bash
# =============================================================================
# DESTRUCTIVE: Tear down ALL billable GCP resources linked to cwijay@biz2bricks.ai
# =============================================================================
# Deletes every discovered resource in the projects that bill to YOUR billing
# account, across these categories:
#   - Cloud Run services
#   - Cloud SQL (Postgres) instances   -> drops ALL their databases
#   - Memorystore (Redis) instances
#   - GCS buckets                      -> bucket and all objects inside
#   - Artifact Registry repos          -> all container images
#   - Secret Manager secrets
#   - (defensive) Compute VMs, disks, static IPs, Cloud Functions, GKE clusters
#
# SCOPE = "cwijay ONLY": each project is checked at runtime and SKIPPED unless
# it bills to BILLING_ACCOUNT below. This guarantees projects on other billing
# accounts (e.g. biz2bricksv1 -> 015507-...) are never touched.
#
# DISCOVERY-FIRST: the script enumerates what actually exists and deletes it,
# so it can't miss resources due to stale/renamed identifiers.
#
# SAFE BY DEFAULT: dry-run unless you pass --confirm. Nothing is deleted in
# dry-run mode -- it only prints the plan.
#
# Usage:
#   ./scripts/gcp/teardown.sh                  # dry-run, all eligible projects
#   ./scripts/gcp/teardown.sh --confirm        # delete (asks once)
#   ./scripts/gcp/teardown.sh --confirm --yes  # delete, no prompt
#   ./scripts/gcp/teardown.sh --project X      # restrict to one project
#   ./scripts/gcp/teardown.sh --keep-tfstate   # do NOT delete *-tf-state buckets
# =============================================================================

set -uo pipefail
export CLOUDSDK_CORE_DISABLE_PROMPTS=1   # never block on "enable API? (y/N)"

# ---- configuration ---------------------------------------------------------
# Your billing account. Only projects billing here are eligible for teardown.
BILLING_ACCOUNT="${BILLING_ACCOUNT:-billingAccounts/01BFBA-189CAF-F8E852}"

# Candidate projects (each re-verified against BILLING_ACCOUNT at runtime).
PROJECTS=(
    "biz-2-bricks-dev-v2"
    "biz2bricks-dev-v1"
    "dynamic-reef-473916-n3"
)

# Cloud Run / GKE live in regions; everything observed is us-central1.
REGIONS=("us-central1")

# ---- arg parsing -----------------------------------------------------------
DRY_RUN=1
ASSUME_YES=0
KEEP_TFSTATE=0
ONLY_PROJECT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --confirm)      DRY_RUN=0; shift ;;
        --yes|-y)       ASSUME_YES=1; shift ;;
        --dry-run)      DRY_RUN=1; shift ;;
        --keep-tfstate) KEEP_TFSTATE=1; shift ;;
        --project)      ONLY_PROJECT="${2:-}"; shift 2 ;;
        -h|--help)      grep '^#' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' | head -40; exit 0 ;;
        *) echo "[ERROR] Unknown argument: $1" >&2; exit 1 ;;
    esac
done
[[ -n "$ONLY_PROJECT" ]] && PROJECTS=("$ONLY_PROJECT")

# ---- helpers ---------------------------------------------------------------
log_info()    { echo "[INFO] $*"; }
log_warn()    { echo "[WARN] $*"; }
log_error()   { echo "[ERROR] $*" >&2; }
log_success() { echo "[SUCCESS] $*"; }
section()     { printf "\n  \033[1;36m-- %s --\033[0m\n" "$*"; }

# run "<description>" cmd args...
run() {
    local desc="$1"; shift
    if [[ "$DRY_RUN" -eq 1 ]]; then
        printf "    \033[1;33m[DRY-RUN]\033[0m %s\n" "$desc"
        printf "             $ %s\n" "$*"
    else
        printf "    \033[1;31m[DELETE]\033[0m  %s\n" "$desc"
        if "$@"; then log_success "    done: ${desc}"
        else          log_warn "    failed (continuing): ${desc}"; fi
    fi
}

# ---- preflight -------------------------------------------------------------
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null | grep -q .; then
    log_error "Not authenticated. Run: gcloud auth login"
    exit 1
fi
ACTIVE_ACCOUNT="$(gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null | head -1)"

printf "\n\033[1;31m============================================================\033[0m\n"
printf "\033[1;31m  GCP TEARDOWN  (account: %s)\033[0m\n" "${ACTIVE_ACCOUNT}"
printf "\033[1;31m============================================================\033[0m\n"
printf "Mode          : %s\n" "$([[ "$DRY_RUN" -eq 1 ]] && echo "DRY-RUN (no changes)" || echo "LIVE DELETE")"
printf "Billing scope : %s\n" "${BILLING_ACCOUNT}"
printf "Candidate proj: %s\n" "${PROJECTS[*]}"
printf "tf-state       : %s\n" "$([[ "$KEEP_TFSTATE" -eq 1 ]] && echo "KEEP" || echo "DELETE (use --keep-tfstate to preserve)")"

if [[ "$DRY_RUN" -eq 0 && "$ASSUME_YES" -eq 0 ]]; then
    printf "\n\033[1;31mThis permanently deletes resources in the projects above.\033[0m\n"
    printf "\033[1;31mType DELETE to proceed: \033[0m"
    read -r reply
    [[ "$reply" == "DELETE" ]] || { log_error "Not confirmed. Aborting."; exit 1; }
fi

# ---- per-category teardown -------------------------------------------------
teardown_cloud_run() {
    local p="$1" r svc
    section "Cloud Run services"
    for r in "${REGIONS[@]}"; do
        for svc in $(gcloud run services list --project="$p" --region="$r" \
                        --format="value(metadata.name)" 2>/dev/null); do
            run "Delete Cloud Run ${svc} (${r})" \
                gcloud run services delete "$svc" --project="$p" --region="$r" --quiet
        done
    done
}

teardown_sql() {
    local p="$1" inst
    section "Cloud SQL instances"
    for inst in $(gcloud sql instances list --project="$p" --format="value(name)" 2>/dev/null); do
        run "Disable deletion protection on ${inst}" \
            gcloud sql instances patch "$inst" --project="$p" --no-deletion-protection --quiet
        run "Delete Cloud SQL ${inst} (and all databases)" \
            gcloud sql instances delete "$inst" --project="$p" --quiet
    done
}

teardown_redis() {
    local p="$1" uri region name
    section "Memorystore Redis instances"
    for uri in $(gcloud redis instances list --project="$p" --region="-" --uri 2>/dev/null); do
        # uri: .../projects/P/locations/REGION/instances/NAME
        region="$(echo "$uri" | sed -E 's#.*/locations/([^/]+)/.*#\1#')"
        name="${uri##*/}"
        run "Delete Redis ${name} (${region})" \
            gcloud redis instances delete "$name" --project="$p" --region="$region" --quiet
    done
}

teardown_buckets() {
    local p="$1" b
    section "GCS buckets"
    for b in $(gcloud storage buckets list --project="$p" --format="value(name)" 2>/dev/null); do
        if [[ "$KEEP_TFSTATE" -eq 1 && "$b" == *tf-state* ]]; then
            log_info "    keeping tf-state bucket: gs://${b}"
            continue
        fi
        run "Delete bucket gs://${b} and all contents" \
            gcloud storage rm --recursive "gs://${b}" --project="$p"
    done
}

teardown_artifacts() {
    local p="$1" loc repo
    section "Artifact Registry repos"
    # Full name is projects/P/locations/LOC/repositories/REPO; segment(3)=LOC, segment(5)=REPO.
    while IFS=$'\t' read -r loc repo; do
        [[ -z "$repo" ]] && continue
        run "Delete Artifact repo ${repo} (${loc})" \
            gcloud artifacts repositories delete "$repo" --project="$p" --location="$loc" --quiet
    done < <(gcloud artifacts repositories list --project="$p" \
                --format="value(name.segment(3),name.segment(5))" 2>/dev/null)
}

teardown_secrets() {
    local p="$1" s
    section "Secret Manager secrets"
    for s in $(gcloud secrets list --project="$p" --format="value(name)" 2>/dev/null); do
        run "Delete secret ${s}" \
            gcloud secrets delete "$s" --project="$p" --quiet
    done
}

teardown_functions() {
    local p="$1" f
    section "Cloud Functions"
    for f in $(gcloud functions list --project="$p" --format="value(name)" 2>/dev/null); do
        run "Delete function ${f}" gcloud functions delete "$f" --project="$p" --quiet
    done
}

teardown_compute() {
    local p="$1" line name zone region
    section "Compute Engine (VMs / disks / static IPs)"
    while IFS=$'\t' read -r name zone; do
        [[ -z "$name" ]] && continue
        run "Delete VM ${name} (${zone})" \
            gcloud compute instances delete "$name" --project="$p" --zone="$zone" --quiet
    done < <(gcloud compute instances list --project="$p" --format="value(name,zone)" 2>/dev/null)

    while IFS=$'\t' read -r name zone; do
        [[ -z "$name" ]] && continue
        run "Delete disk ${name} (${zone})" \
            gcloud compute disks delete "$name" --project="$p" --zone="$zone" --quiet
    done < <(gcloud compute disks list --project="$p" --format="value(name,zone)" 2>/dev/null)

    while IFS=$'\t' read -r name region; do
        [[ -z "$name" ]] && continue
        run "Release static IP ${name} (${region:-global})" \
            gcloud compute addresses delete "$name" --project="$p" \
                ${region:+--region="$region"} ${region:+} --quiet
    done < <(gcloud compute addresses list --project="$p" --format="value(name,region)" 2>/dev/null)
}

teardown_gke() {
    local p="$1" line name loc
    section "GKE clusters"
    while IFS=$'\t' read -r name loc; do
        [[ -z "$name" ]] && continue
        run "Delete GKE cluster ${name} (${loc})" \
            gcloud container clusters delete "$name" --project="$p" --location="$loc" --quiet
    done < <(gcloud container clusters list --project="$p" --format="value(name,location)" 2>/dev/null)
}

# ---- main loop -------------------------------------------------------------
for p in "${PROJECTS[@]}"; do
    printf "\n\033[1;35m############################################################\033[0m\n"
    printf "\033[1;35m# PROJECT: %s\033[0m\n" "$p"
    printf "\033[1;35m############################################################\033[0m\n"

    if ! gcloud projects describe "$p" >/dev/null 2>&1; then
        log_warn "No access to project ${p}; skipping."
        continue
    fi

    # Enforce "cwijay ONLY": project must bill to BILLING_ACCOUNT.
    billing="$(gcloud billing projects describe "$p" --format='value(billingAccountName)' 2>/dev/null)"
    if [[ "$billing" != "$BILLING_ACCOUNT" ]]; then
        log_warn "SKIP ${p}: bills to '${billing:-none}', not ${BILLING_ACCOUNT}."
        continue
    fi
    log_info "Eligible: ${p} bills to ${BILLING_ACCOUNT}"

    teardown_cloud_run "$p"
    teardown_functions "$p"
    teardown_gke       "$p"
    teardown_redis     "$p"
    teardown_sql       "$p"
    teardown_artifacts "$p"
    teardown_buckets   "$p"
    teardown_secrets   "$p"
    teardown_compute   "$p"
done

echo ""
if [[ "$DRY_RUN" -eq 1 ]]; then
    log_warn "DRY-RUN complete. No resources were deleted."
    log_warn "Review the plan above, then re-run with --confirm."
else
    log_success "Teardown complete."
fi
