"""
AWS Lambda Handler for SmokeNMirror API

This module wraps the Flask application for AWS Lambda using apig-wsgi.
apig-wsgi is a lightweight adapter for WSGI apps (like Flask) on API Gateway.
"""

import logging
import os
import re

# Configure logging for Lambda
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Log Lambda cold start
logger.info("🚀 Lambda cold start - initializing SmokeNMirror API")

# Import the Flask app
from app import app

# Use apig-wsgi to wrap Flask for Lambda (proper WSGI adapter)
try:
    from apig_wsgi import make_lambda_handler

    # Create the Lambda handler for HTTP API (API Gateway v2)
    # Set binary_support=True for proper handling of binary responses
    _apig_handler = make_lambda_handler(app, binary_support=True)

    logger.info("✅ apig-wsgi handler initialized successfully")

except ImportError as e:
    logger.error(f"❌ Failed to import apig-wsgi: {e}")
    logger.error("Make sure 'apig-wsgi' is in requirements.txt")
    _apig_handler = None


def handler(event, context):
    """
    Main Lambda entry point.

    Handles both HTTP API v2 and REST API v1 formats.
    Strips stage prefix from path if present.
    """
    # Log the incoming event for debugging
    logger.info(f"📥 Raw event keys: {list(event.keys())}")

    # Get path info
    raw_path = event.get("rawPath", "")
    path = event.get("path", "")
    stage = event.get("requestContext", {}).get("stage", "")

    logger.info(f"📥 rawPath: {raw_path}, path: {path}, stage: {stage}")

    # HTTP API v2 uses rawPath (should NOT include stage)
    # REST API v1 uses path (MAY include stage)

    # Fix: Strip stage prefix if it's accidentally included in rawPath
    if stage and raw_path.startswith(f"/{stage}/"):
        fixed_path = raw_path[len(f"/{stage}"):]
        event["rawPath"] = fixed_path
        logger.info(f"🔧 Fixed rawPath: {raw_path} -> {fixed_path}")

    # Also fix the path in requestContext.http if present
    if "requestContext" in event and "http" in event["requestContext"]:
        http_path = event["requestContext"]["http"].get("path", "")
        if stage and http_path.startswith(f"/{stage}/"):
            fixed_http_path = http_path[len(f"/{stage}"):]
            event["requestContext"]["http"]["path"] = fixed_http_path
            logger.info(f"🔧 Fixed http.path: {http_path} -> {fixed_http_path}")

    method = event.get("requestContext", {}).get("http", {}).get("method", "unknown")
    final_path = event.get("rawPath", event.get("path", "unknown"))
    logger.info(f"📥 Request: {method} {final_path}")

    if _apig_handler is None:
        return {
            "statusCode": 500,
            "body": '{"error": "Lambda handler not properly configured - apig-wsgi missing"}',
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            }
        }

    try:
        # Invoke the apig-wsgi handler
        response = _apig_handler(event, context)

        # Log response status
        status_code = response.get("statusCode", 200)
        logger.info(f"📤 Response: {status_code}")

        return response

    except Exception as e:
        logger.exception(f"❌ Unhandled exception: {e}")
        return {
            "statusCode": 500,
            "body": '{"error": "Internal server error"}',
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            }
        }


# For local testing
if __name__ == "__main__":
    test_event = {
        "version": "2.0",
        "routeKey": "GET /api/health",
        "rawPath": "/api/health",
        "requestContext": {
            "http": {
                "method": "GET",
                "path": "/api/health",
                "protocol": "HTTP/1.1",
                "sourceIp": "127.0.0.1",
                "userAgent": "test"
            },
            "stage": "prod"
        },
        "headers": {},
        "isBase64Encoded": False
    }

    class MockContext:
        function_name = "test"
        memory_limit_in_mb = 512
        invoked_function_arn = "arn:aws:lambda:us-east-1:123456789:function:test"
        aws_request_id = "test-request-id"
        def get_remaining_time_in_millis(self):
            return 120000

    result = handler(test_event, MockContext())
    print(f"Test result: {result}")
