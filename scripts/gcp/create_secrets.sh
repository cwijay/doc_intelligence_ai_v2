#!/usr/bin/env bash
# =============================================================================
# Create secrets in GCP Secret Manager for Document Intelligence AI
# =============================================================================
# Usage: ./create_secrets.sh [--update] [--from-env]
#
# This script creates (or updates) secrets in GCP Secret Manager.
# It prompts for secret values interactively to avoid storing them in files.
#
# Options:
#   --update    Update existing secrets (otherwise skips if exists)
#   --from-env  Read secret values from current environment/.env file
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/config.sh"

UPDATE_MODE=false
FROM_ENV=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --update)
            UPDATE_MODE=true
            shift
            ;;
        --from-env)
            FROM_ENV=true
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Usage: $0 [--update] [--from-env]"
            exit 1
            ;;
    esac
done

# =============================================================================
# Functions
# =============================================================================

secret_exists() {
    local secret_name="$1"
    gcloud secrets describe "${secret_name}" --project="${GCP_PROJECT}" > /dev/null 2>&1
}

create_or_update_secret() {
    local secret_name="$1"
    local secret_value="$2"
    local description="$3"

    if [[ -z "${secret_value}" ]]; then
        log_warn "Skipping empty secret: ${secret_name}"
        return 0
    fi

    if secret_exists "${secret_name}"; then
        if [[ "${UPDATE_MODE}" == "true" ]]; then
            log_info "Updating secret: ${secret_name}"
            echo -n "${secret_value}" | gcloud secrets versions add "${secret_name}" \
                --project="${GCP_PROJECT}" \
                --data-file=-
        else
            log_warn "Secret already exists (use --update to update): ${secret_name}"
            return 0
        fi
    else
        log_info "Creating secret: ${secret_name}"
        echo -n "${secret_value}" | gcloud secrets create "${secret_name}" \
            --project="${GCP_PROJECT}" \
            --replication-policy="automatic" \
            --data-file=- \
            --labels="app=doc-intelligence,env=dev"
    fi
}

prompt_secret() {
    local prompt_text="$1"
    local env_var_name="$2"
    local result_var="$3"

    # If --from-env, try to get from environment
    if [[ "${FROM_ENV}" == "true" ]]; then
        local env_value="${!env_var_name:-}"
        if [[ -n "${env_value}" ]]; then
            log_info "Using ${env_var_name} from environment"
            eval "${result_var}='${env_value}'"
            return 0
        fi
    fi

    # Interactive prompt
    echo -n "${prompt_text}: "
    read -rs value
    echo ""  # New line after hidden input

    if [[ -z "${value}" ]]; then
        log_warn "No value provided for ${env_var_name}"
        eval "${result_var}=''"
        return 0
    fi

    eval "${result_var}='${value}'"
}

grant_secret_access() {
    local secret_name="$1"
    local service_account="$2"

    log_info "Granting access to ${secret_name}"
    gcloud secrets add-iam-policy-binding "${secret_name}" \
        --project="${GCP_PROJECT}" \
        --member="serviceAccount:${service_account}" \
        --role="roles/secretmanager.secretAccessor" \
        --quiet 2>/dev/null || true
}

# =============================================================================
# Main
# =============================================================================

main() {
    echo "============================================================"
    echo "  GCP Secret Manager Setup - Document Intelligence AI"
    echo "============================================================"
    echo ""
    log_info "Project: ${GCP_PROJECT}"
    log_info "Update mode: ${UPDATE_MODE}"
    log_info "From environment: ${FROM_ENV}"
    echo ""

    # Load .env if --from-env is set
    if [[ "${FROM_ENV}" == "true" ]] && [[ -f "${PROJECT_ROOT}/.env" ]]; then
        log_info "Loading environment from ${PROJECT_ROOT}/.env"
        set -a
        source "${PROJECT_ROOT}/.env"
        set +a
    fi

    # Verify gcloud authentication
    check_gcloud_auth
    set_project

    # Enable Secret Manager API if not enabled
    log_info "Ensuring Secret Manager API is enabled..."
    gcloud services enable secretmanager.googleapis.com --project="${GCP_PROJECT}" --quiet

    echo ""
    echo "============================================================"
    echo "  Enter Secret Values"
    echo "============================================================"
    if [[ "${FROM_ENV}" != "true" ]]; then
        echo "Note: Input will be hidden for security"
    fi
    echo ""

    # Prompt for each secret
    prompt_secret "Enter OpenAI API Key (sk-...)" "OPENAI_API_KEY" OPENAI_KEY
    prompt_secret "Enter Google API Key (for Gemini)" "GOOGLE_API_KEY" GOOGLE_KEY
    prompt_secret "Enter LlamaCloud API Key (llx-...)" "LLAMA_CLOUD_API_KEY" LLAMA_KEY
    prompt_secret "Enter Database Password" "DATABASE_PASSWORD" DB_PASSWORD
    prompt_secret "Enter Webhook Secret (for bulk processing, or press Enter to skip)" "BULK_WEBHOOK_SECRET" WEBHOOK_SECRET_VALUE

    echo ""
    echo "============================================================"
    echo "  Creating/Updating Secrets"
    echo "============================================================"

    # Create secrets
    create_or_update_secret "${SECRET_OPENAI_API_KEY}" "${OPENAI_KEY}" "OpenAI API key for GPT models"
    create_or_update_secret "${SECRET_GOOGLE_API_KEY}" "${GOOGLE_KEY}" "Google API key for Gemini"
    create_or_update_secret "${SECRET_LLAMA_API_KEY}" "${LLAMA_KEY}" "LlamaCloud API key for parsing"
    create_or_update_secret "${SECRET_DATABASE_PASSWORD}" "${DB_PASSWORD}" "Cloud SQL database password"

    if [[ -n "${WEBHOOK_SECRET_VALUE}" ]]; then
        create_or_update_secret "${SECRET_WEBHOOK_SECRET}" "${WEBHOOK_SECRET_VALUE}" "Webhook secret for bulk processing"
    fi

    echo ""
    echo "============================================================"
    echo "  Granting Access to Cloud Run Service Account"
    echo "============================================================"

    # Get the default compute service account
    PROJECT_NUMBER=$(get_project_number)
    SERVICE_ACCOUNT="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

    log_info "Service account: ${SERVICE_ACCOUNT}"

    for secret_name in \
        "${SECRET_OPENAI_API_KEY}" \
        "${SECRET_GOOGLE_API_KEY}" \
        "${SECRET_LLAMA_API_KEY}" \
        "${SECRET_DATABASE_PASSWORD}"; do

        if secret_exists "${secret_name}"; then
            grant_secret_access "${secret_name}" "${SERVICE_ACCOUNT}"
        fi
    done

    if secret_exists "${SECRET_WEBHOOK_SECRET}"; then
        grant_secret_access "${SECRET_WEBHOOK_SECRET}" "${SERVICE_ACCOUNT}"
    fi

    echo ""
    echo "============================================================"
    echo "  Summary"
    echo "============================================================"
    log_success "Secret setup complete!"
    echo ""
    log_info "To list secrets:"
    echo "  gcloud secrets list --project=${GCP_PROJECT}"
    echo ""
    log_info "To view a secret value:"
    echo "  gcloud secrets versions access latest --secret=<SECRET_NAME> --project=${GCP_PROJECT}"
    echo ""
    log_info "To update secrets later:"
    echo "  $0 --update"
}

main "$@"
