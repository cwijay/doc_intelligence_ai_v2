#!/usr/bin/env bash
# =============================================================================
# Deploy Document Intelligence AI to GCP Cloud Run
# =============================================================================
# Usage: ./deploy_to_cloud_run.sh [OPTIONS]
#
# Options:
#   --build-only     Only build and push the Docker image
#   --deploy-only    Only deploy (assumes image already exists)
#   --no-traffic     Deploy without routing traffic (for testing)
#   --tag TAG        Use specific image tag (default: timestamp)
#   --help           Show this help message
#
# Examples:
#   ./deploy_to_cloud_run.sh                    # Full build and deploy
#   ./deploy_to_cloud_run.sh --build-only       # Only build image
#   ./deploy_to_cloud_run.sh --deploy-only      # Deploy existing image
#   ./deploy_to_cloud_run.sh --tag v1.0.0       # Use specific tag
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/config.sh"

# Parse arguments
BUILD_ONLY=false
DEPLOY_ONLY=false
NO_TRAFFIC=false
IMAGE_TAG=""

show_help() {
    head -25 "$0" | tail -22
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-only)
            BUILD_ONLY=true
            shift
            ;;
        --deploy-only)
            DEPLOY_ONLY=true
            shift
            ;;
        --no-traffic)
            NO_TRAFFIC=true
            shift
            ;;
        --tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# =============================================================================
# Functions
# =============================================================================

enable_apis() {
    log_info "Ensuring required GCP APIs are enabled..."
    gcloud services enable \
        run.googleapis.com \
        artifactregistry.googleapis.com \
        secretmanager.googleapis.com \
        sqladmin.googleapis.com \
        --project="${GCP_PROJECT}" --quiet
}

create_artifact_registry() {
    log_info "Ensuring Artifact Registry repository exists..."

    if ! gcloud artifacts repositories describe doc-intelligence \
        --location="${GCP_REGION}" \
        --project="${GCP_PROJECT}" > /dev/null 2>&1; then

        log_info "Creating Artifact Registry repository: doc-intelligence"
        gcloud artifacts repositories create doc-intelligence \
            --repository-format=docker \
            --location="${GCP_REGION}" \
            --project="${GCP_PROJECT}" \
            --description="Docker images for Document Intelligence AI"
    else
        log_info "Artifact Registry repository already exists"
    fi
}

configure_docker_auth() {
    log_info "Configuring Docker authentication for Artifact Registry..."
    gcloud auth configure-docker "${GCP_REGION}-docker.pkg.dev" --quiet
}

build_and_push_image() {
    local tag="${IMAGE_TAG:-$(date +%Y%m%d-%H%M%S)}"
    local image_tag="${IMAGE_NAME}:${tag}"
    local image_latest="${IMAGE_NAME}:latest"

    log_info "Building Docker image..."
    log_info "Tag: ${tag}"
    log_info "Image: ${image_tag}"

    cd "${PROJECT_ROOT}"

    # Build the image for linux/amd64 (Cloud Run requirement)
    docker build \
        --platform linux/amd64 \
        --tag "${image_tag}" \
        --tag "${image_latest}" \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        .

    log_info "Pushing image to Artifact Registry..."
    docker push "${image_tag}"
    docker push "${image_latest}"

    # Export for deployment step
    export DEPLOYED_IMAGE="${image_tag}"
    log_success "Image pushed: ${DEPLOYED_IMAGE}"
}

check_secrets_exist() {
    log_info "Verifying secrets exist in Secret Manager..."

    local missing_secrets=()

    for secret_name in \
        "${SECRET_OPENAI_API_KEY}" \
        "${SECRET_GOOGLE_API_KEY}" \
        "${SECRET_LLAMA_API_KEY}" \
        "${SECRET_DATABASE_PASSWORD}"; do

        if ! gcloud secrets describe "${secret_name}" --project="${GCP_PROJECT}" > /dev/null 2>&1; then
            missing_secrets+=("${secret_name}")
        fi
    done

    if [[ ${#missing_secrets[@]} -gt 0 ]]; then
        log_error "Missing secrets in Secret Manager:"
        for secret in "${missing_secrets[@]}"; do
            echo "  - ${secret}"
        done
        echo ""
        log_error "Please run ./create_secrets.sh first to create the required secrets."
        exit 1
    fi

    log_info "All required secrets exist"
}

deploy_to_cloud_run() {
    local image="${DEPLOYED_IMAGE:-${IMAGE_NAME}:latest}"

    log_info "Deploying to Cloud Run..."
    log_info "Service: ${SERVICE_NAME}"
    log_info "Region: ${GCP_REGION}"
    log_info "Image: ${image}"

    # Build environment variables string
    local env_vars=""
    env_vars+="CLOUD_SQL_INSTANCE=${CLOUD_SQL_INSTANCE}"
    env_vars+=",DATABASE_NAME=${DATABASE_NAME}"
    env_vars+=",DATABASE_USER=${DATABASE_USER}"
    env_vars+=",DATABASE_ENABLED=${DATABASE_ENABLED}"
    env_vars+=",USE_CLOUD_SQL_CONNECTOR=${USE_CLOUD_SQL_CONNECTOR}"
    env_vars+=",CLOUD_SQL_IP_TYPE=${CLOUD_SQL_IP_TYPE}"
    env_vars+=",GCS_BUCKET=${GCS_BUCKET}"
    env_vars+=",GCS_PREFIX=${GCS_PREFIX}"
    env_vars+=",PARSED_DIRECTORY=${PARSED_DIRECTORY}"
    env_vars+=",GENERATED_DIRECTORY=${GENERATED_DIRECTORY}"
    env_vars+=",LOG_LEVEL=${LOG_LEVEL}"
    env_vars+=",DEBUG=${DEBUG}"
    env_vars+=",OPENAI_SHEET_MODEL=${OPENAI_SHEET_MODEL}"
    env_vars+=",DOCUMENT_AGENT_MODEL=${DOCUMENT_AGENT_MODEL}"
    env_vars+=",OPENAI_TEMPERATURE=${OPENAI_TEMPERATURE}"
    env_vars+=",DOCUMENT_AGENT_TEMPERATURE=${DOCUMENT_AGENT_TEMPERATURE}"
    env_vars+=",DB_POOL_SIZE=${DB_POOL_SIZE}"
    env_vars+=",DB_BACKGROUND_POOL_SIZE=${DB_BACKGROUND_POOL_SIZE}"
    env_vars+=",DB_MAX_OVERFLOW=${DB_MAX_OVERFLOW}"

    # Build secrets string
    local secrets=""
    secrets+="OPENAI_API_KEY=${SECRET_OPENAI_API_KEY}:latest"
    secrets+=",GOOGLE_API_KEY=${SECRET_GOOGLE_API_KEY}:latest"
    secrets+=",LLAMA_CLOUD_API_KEY=${SECRET_LLAMA_API_KEY}:latest"
    secrets+=",DATABASE_PASSWORD=${SECRET_DATABASE_PASSWORD}:latest"

    # Add webhook secret if it exists
    if gcloud secrets describe "${SECRET_WEBHOOK_SECRET}" --project="${GCP_PROJECT}" > /dev/null 2>&1; then
        secrets+=",BULK_WEBHOOK_SECRET=${SECRET_WEBHOOK_SECRET}:latest"
    fi

    # Get service account
    local project_number
    project_number=$(get_project_number)
    local service_account="${project_number}-compute@developer.gserviceaccount.com"

    # Build the deployment command
    local deploy_args=(
        "${SERVICE_NAME}"
        --project="${GCP_PROJECT}"
        --region="${GCP_REGION}"
        --image="${image}"
        --platform=managed
        --allow-unauthenticated
        --memory="${CLOUD_RUN_MEMORY}"
        --cpu="${CLOUD_RUN_CPU}"
        --min-instances="${CLOUD_RUN_MIN_INSTANCES}"
        --max-instances="${CLOUD_RUN_MAX_INSTANCES}"
        --concurrency="${CLOUD_RUN_CONCURRENCY}"
        --timeout="${CLOUD_RUN_TIMEOUT}"
        --port=8080
        --add-cloudsql-instances="${CLOUD_SQL_INSTANCE}"
        --set-env-vars="${env_vars}"
        --set-secrets="${secrets}"
        --service-account="${service_account}"
    )

    # Add no-traffic flag if specified
    if [[ "${NO_TRAFFIC}" == "true" ]]; then
        deploy_args+=(--no-traffic)
    fi

    # Execute deployment
    gcloud run deploy "${deploy_args[@]}"

    # Get the service URL
    local service_url
    service_url=$(get_service_url)

    echo ""
    echo "============================================================"
    echo "  Deployment Complete!"
    echo "============================================================"
    log_success "Service URL: ${service_url}"
    log_info "Health Check: ${service_url}/health"
    log_info "API Docs: ${service_url}/docs"
}

verify_deployment() {
    log_info "Verifying deployment..."

    local service_url
    service_url=$(get_service_url)

    if [[ -z "${service_url}" ]]; then
        log_warn "Could not get service URL"
        return 1
    fi

    log_info "Checking health endpoint..."

    # Wait a bit for the service to start
    sleep 5

    local health_response
    local retry_count=0
    local max_retries=6

    while [[ ${retry_count} -lt ${max_retries} ]]; do
        if health_response=$(curl -s -f "${service_url}/health" 2>/dev/null); then
            log_success "Health check passed!"
            echo "${health_response}" | python3 -m json.tool 2>/dev/null || echo "${health_response}"
            return 0
        fi

        retry_count=$((retry_count + 1))
        if [[ ${retry_count} -lt ${max_retries} ]]; then
            log_info "Health check attempt ${retry_count}/${max_retries} failed, retrying in 10s..."
            sleep 10
        fi
    done

    log_warn "Health check did not pass after ${max_retries} attempts."
    log_info "The service may still be starting. Check logs with:"
    echo "  gcloud run logs read --service=${SERVICE_NAME} --region=${GCP_REGION} --project=${GCP_PROJECT} --limit=50"
}

# =============================================================================
# Main
# =============================================================================

main() {
    echo "============================================================"
    echo "  Document Intelligence AI - Cloud Run Deployment"
    echo "============================================================"
    echo ""
    log_info "Project: ${GCP_PROJECT}"
    log_info "Region: ${GCP_REGION}"
    log_info "Service: ${SERVICE_NAME}"
    log_info "Build only: ${BUILD_ONLY}"
    log_info "Deploy only: ${DEPLOY_ONLY}"
    echo ""

    # Verify authentication
    check_gcloud_auth
    set_project

    # Enable required APIs
    enable_apis

    if [[ "${DEPLOY_ONLY}" != "true" ]]; then
        check_docker
        create_artifact_registry
        configure_docker_auth
        build_and_push_image
    fi

    if [[ "${BUILD_ONLY}" != "true" ]]; then
        check_secrets_exist
        deploy_to_cloud_run
        verify_deployment
    fi

    echo ""
    echo "============================================================"
    echo "  Done!"
    echo "============================================================"

    if [[ "${BUILD_ONLY}" != "true" ]]; then
        local service_url
        service_url=$(get_service_url)
        echo ""
        log_info "Useful commands:"
        echo ""
        echo "  # View logs"
        echo "  gcloud run logs read --service=${SERVICE_NAME} --region=${GCP_REGION} --limit=100"
        echo ""
        echo "  # Stream logs"
        echo "  gcloud run logs tail --service=${SERVICE_NAME} --region=${GCP_REGION}"
        echo ""
        echo "  # Describe service"
        echo "  gcloud run services describe ${SERVICE_NAME} --region=${GCP_REGION}"
        echo ""
        echo "  # Test health"
        echo "  curl ${service_url}/health"
    fi
}

main "$@"
