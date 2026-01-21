#!/bin/bash
# =============================================================================
# Setup Cloud Build Triggers for AI API
# =============================================================================
# Creates GitHub-connected Cloud Build triggers for automated deployments.
#
# Prerequisites:
#   1. GitHub repository connected to Cloud Build
#   2. gcloud CLI authenticated with appropriate permissions
#   3. Cloud Build API enabled
#
# Usage:
#   ./scripts/gcp/setup_triggers.sh
#   ./scripts/gcp/setup_triggers.sh --project=my-project --repo=my-org/my-repo
# =============================================================================

set -e

# Default configuration
PROJECT_ID="${PROJECT_ID:-biz2bricks-dev-v1}"
GITHUB_REPO="${GITHUB_REPO:-chaminda/doc_intelligence_ai_v3.0}"
REGION="us-central1"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --project=*)
            PROJECT_ID="${1#*=}"
            shift
            ;;
        --repo=*)
            GITHUB_REPO="${1#*=}"
            shift
            ;;
        --help)
            echo "Usage: $0 [--project=PROJECT_ID] [--repo=GITHUB_OWNER/REPO]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "============================================="
echo "Setting up Cloud Build Triggers for AI API"
echo "============================================="
echo "Project: $PROJECT_ID"
echo "GitHub Repo: $GITHUB_REPO"
echo ""

# Extract owner and repo name
REPO_OWNER=$(echo "$GITHUB_REPO" | cut -d'/' -f1)
REPO_NAME=$(echo "$GITHUB_REPO" | cut -d'/' -f2)

# Set the project
gcloud config set project "$PROJECT_ID"

# -----------------------------------------------------------------------------
# Development Trigger (develop branch)
# -----------------------------------------------------------------------------
echo "Creating development trigger..."

# Check if trigger exists
if gcloud builds triggers describe ai-api-dev-deploy --region="$REGION" &>/dev/null; then
    echo "  Trigger 'ai-api-dev-deploy' already exists. Updating..."
    gcloud builds triggers delete ai-api-dev-deploy --region="$REGION" --quiet
fi

gcloud builds triggers create github \
    --name="ai-api-dev-deploy" \
    --region="$REGION" \
    --repo-owner="$REPO_OWNER" \
    --repo-name="$REPO_NAME" \
    --branch-pattern="^develop$" \
    --build-config="cloudbuild.yaml" \
    --description="Deploy AI API to development environment on develop branch push"

echo "  Created trigger: ai-api-dev-deploy"

# -----------------------------------------------------------------------------
# Production Trigger (master branch)
# -----------------------------------------------------------------------------
echo "Creating production trigger..."

# Check if trigger exists
if gcloud builds triggers describe ai-api-prod-deploy --region="$REGION" &>/dev/null; then
    echo "  Trigger 'ai-api-prod-deploy' already exists. Updating..."
    gcloud builds triggers delete ai-api-prod-deploy --region="$REGION" --quiet
fi

gcloud builds triggers create github \
    --name="ai-api-prod-deploy" \
    --region="$REGION" \
    --repo-owner="$REPO_OWNER" \
    --repo-name="$REPO_NAME" \
    --branch-pattern="^master$" \
    --build-config="cloudbuild.yaml" \
    --substitutions="_ENV=prod,_SERVICE_NAME=document-intelligence-ai-api-prod,_SECRET_SUFFIX=-prod,_LOG_LEVEL=INFO,_MAX_INSTANCES=10" \
    --description="Deploy AI API to production environment on master branch push"

echo "  Created trigger: ai-api-prod-deploy"

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "============================================="
echo "Triggers Created Successfully!"
echo "============================================="
echo ""
echo "Triggers:"
gcloud builds triggers list --region="$REGION" --filter="name~ai-api" --format="table(name,description)"
echo ""
echo "Next steps:"
echo "  1. Push to 'develop' branch to trigger dev deployment"
echo "  2. Push to 'master' branch to trigger prod deployment"
echo ""
echo "Manual trigger:"
echo "  gcloud builds triggers run ai-api-dev-deploy --region=$REGION --branch=develop"
echo "  gcloud builds triggers run ai-api-prod-deploy --region=$REGION --branch=master"
