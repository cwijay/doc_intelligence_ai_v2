#!/bin/bash
# =============================================================================
# Rollback Cloud Run Service - AI API
# =============================================================================
# Quick rollback to a previous revision of the AI API service.
#
# Usage:
#   ./scripts/gcp/rollback.sh                    # Interactive mode
#   ./scripts/gcp/rollback.sh --env=dev          # List dev revisions
#   ./scripts/gcp/rollback.sh --env=prod         # List prod revisions
#   ./scripts/gcp/rollback.sh --env=dev --revision=REVISION_NAME
# =============================================================================

set -e

# Default configuration
PROJECT_ID="${PROJECT_ID:-biz2bricks-dev-v1}"
REGION="us-central1"
ENV=""
REVISION=""
LIST_ONLY=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --env=*)
            ENV="${1#*=}"
            shift
            ;;
        --revision=*)
            REVISION="${1#*=}"
            shift
            ;;
        --list)
            LIST_ONLY=true
            shift
            ;;
        --project=*)
            PROJECT_ID="${1#*=}"
            shift
            ;;
        --help)
            echo "Usage: $0 [--env=dev|prod] [--revision=REVISION_NAME] [--list] [--project=PROJECT_ID]"
            echo ""
            echo "Options:"
            echo "  --env=dev|prod     Target environment (required)"
            echo "  --revision=NAME    Specific revision to rollback to"
            echo "  --list             Only list revisions, don't rollback"
            echo "  --project=ID       GCP project ID"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate environment
if [[ -z "$ENV" ]]; then
    echo "Error: --env is required (dev or prod)"
    echo "Usage: $0 --env=dev [--revision=REVISION_NAME]"
    exit 1
fi

# Determine service name based on environment
case $ENV in
    dev)
        SERVICE_NAME="document-intelligence-ai-api-dev"
        ;;
    prod)
        SERVICE_NAME="document-intelligence-ai-api-prod"
        ;;
    *)
        echo "Error: Invalid environment '$ENV'. Use 'dev' or 'prod'."
        exit 1
        ;;
esac

echo "============================================="
echo "AI API Rollback Helper"
echo "============================================="
echo "Project: $PROJECT_ID"
echo "Service: $SERVICE_NAME"
echo "Region: $REGION"
echo ""

# Set the project
gcloud config set project "$PROJECT_ID" --quiet

# List recent revisions
echo "Recent revisions:"
echo "-----------------"
REVISIONS=$(gcloud run revisions list \
    --service="$SERVICE_NAME" \
    --region="$REGION" \
    --limit=10 \
    --format="table(metadata.name,status.conditions[0].lastTransitionTime,spec.containers[0].image:label=IMAGE)" \
    2>/dev/null || echo "No revisions found")

echo "$REVISIONS"
echo ""

# If list only mode, exit here
if [[ "$LIST_ONLY" == "true" ]]; then
    exit 0
fi

# Get current revision serving traffic
CURRENT_REVISION=$(gcloud run services describe "$SERVICE_NAME" \
    --region="$REGION" \
    --format="value(status.traffic[0].revisionName)" \
    2>/dev/null || echo "")

if [[ -n "$CURRENT_REVISION" ]]; then
    echo "Current revision: $CURRENT_REVISION"
    echo ""
fi

# If no revision specified, prompt for one
if [[ -z "$REVISION" ]]; then
    echo "Enter the revision name to rollback to (or Ctrl+C to cancel):"
    read -r REVISION
fi

if [[ -z "$REVISION" ]]; then
    echo "No revision specified. Exiting."
    exit 1
fi

# Confirm rollback
echo ""
echo "About to rollback $SERVICE_NAME to revision: $REVISION"
echo "This will route 100% of traffic to this revision."
echo ""
read -p "Continue? (y/N): " CONFIRM

if [[ "$CONFIRM" != "y" && "$CONFIRM" != "Y" ]]; then
    echo "Rollback cancelled."
    exit 0
fi

# Perform rollback
echo ""
echo "Rolling back to $REVISION..."
gcloud run services update-traffic "$SERVICE_NAME" \
    --region="$REGION" \
    --to-revisions="$REVISION=100"

echo ""
echo "============================================="
echo "Rollback Complete!"
echo "============================================="
echo ""
echo "Service: $SERVICE_NAME"
echo "Revision: $REVISION"
echo ""

# Get service URL
SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" \
    --region="$REGION" \
    --format="value(status.url)")

echo "Service URL: $SERVICE_URL"
echo ""
echo "Verify the service is working:"
echo "  curl $SERVICE_URL/health"
