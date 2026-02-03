#!/bin/bash
# Deploy SmokeNMirror API to AWS Lambda using SAM

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 SmokeNMirror Lambda Deployment Script${NC}"
echo "=========================================="

# Check if SAM CLI is installed
if ! command -v sam &> /dev/null; then
    echo -e "${RED}❌ AWS SAM CLI is not installed.${NC}"
    echo "Install it from: https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html"
    exit 1
fi

# Check if AWS CLI is configured
if ! aws sts get-caller-identity &> /dev/null; then
    echo -e "${RED}❌ AWS CLI is not configured.${NC}"
    echo "Run: aws configure"
    exit 1
fi

# Load environment variables from .env if exists
if [ -f .env ]; then
    echo -e "${YELLOW}📝 Loading environment variables from .env${NC}"
    export $(grep -v '^#' .env | xargs)
fi

# Check required environment variables
REQUIRED_VARS=("FRED_API" "MISTRAL" "groq" "OPENROUTER" "GAI")
MISSING_VARS=()

for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        MISSING_VARS+=("$var")
    fi
done

if [ ${#MISSING_VARS[@]} -ne 0 ]; then
    echo -e "${RED}❌ Missing required environment variables:${NC}"
    printf '   - %s\n' "${MISSING_VARS[@]}"
    echo ""
    echo "Set them in .env file or export them before running this script."
    exit 1
fi

# Get the stage (default: prod)
STAGE="${1:-prod}"
echo -e "${YELLOW}📦 Deploying to stage: ${STAGE}${NC}"

# Build the application
echo -e "${YELLOW}🔨 Building the application...${NC}"
sam build --use-container

# Deploy with parameter overrides
echo -e "${YELLOW}🚀 Deploying to AWS...${NC}"
sam deploy \
    --config-env "$STAGE" \
    --parameter-overrides \
        "FredApiKey=$FRED_API" \
        "MistralApiKey=$MISTRAL" \
        "GroqApiKey=$groq" \
        "OpenRouterApiKey=$OPENROUTER" \
        "GoogleAiKey=$GAI" \
        "FinnhubApiKey=${finhub:-}" \
        "PolygonApiKey=${POLYGON:-}" \
        "BraveApiKey=${BRAVE:-}" \
        "FmpApiKey=${FMP:-}" \
        "StageName=$STAGE" \
    --no-confirm-changeset \
    --no-fail-on-empty-changeset

# Get the API endpoint
echo ""
echo -e "${GREEN}✅ Deployment complete!${NC}"
echo ""

API_URL=$(aws cloudformation describe-stacks \
    --stack-name "smokenmiror-api-${STAGE}" \
    --query 'Stacks[0].Outputs[?OutputKey==`ApiEndpoint`].OutputValue' \
    --output text)

echo -e "${GREEN}📡 API Endpoint: ${API_URL}${NC}"
echo ""
echo "Next steps:"
echo "1. Test the API: curl ${API_URL}/api/health"
echo "2. Update vercel.frontend.json with the API URL"
echo "3. Deploy frontend: vercel --prod"
