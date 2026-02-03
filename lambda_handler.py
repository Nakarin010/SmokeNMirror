"""
AWS Lambda Handler for SmokeNMirror API

This module wraps the Flask application for AWS Lambda using Mangum.
Mangum is an ASGI/WSGI adapter that allows Flask apps to run on Lambda.
"""

import logging
import os

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

# Use Mangum to wrap Flask for Lambda
try:
    from mangum import Mangum
    
    # Create the Lambda handler
    # lifespan="off" is recommended for Flask apps
    handler = Mangum(app, lifespan="off")
    
    logger.info("✅ Mangum handler initialized successfully")
    
except ImportError as e:
    logger.error(f"❌ Failed to import Mangum: {e}")
    logger.error("Make sure 'mangum' is in requirements.txt")
    
    # Fallback handler that returns an error
    def handler(event, context):
        return {
            "statusCode": 500,
            "body": '{"error": "Lambda handler not properly configured"}',
            "headers": {
                "Content-Type": "application/json"
            }
        }


def lambda_handler(event, context):
    """
    Main Lambda entry point.
    
    This function is called by AWS Lambda when the function is invoked.
    It delegates to the Mangum handler which wraps the Flask app.
    
    Args:
        event: AWS Lambda event object (API Gateway request)
        context: AWS Lambda context object
        
    Returns:
        API Gateway response dictionary
    """
    # Log request info (sanitized)
    path = event.get("rawPath", event.get("path", "unknown"))
    method = event.get("requestContext", {}).get("http", {}).get("method", "unknown")
    logger.info(f"📥 Request: {method} {path}")
    
    try:
        # Invoke the Mangum handler
        response = handler(event, context)
        
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
    # Simulate a Lambda event for testing
    test_event = {
        "version": "2.0",
        "routeKey": "GET /api/health",
        "rawPath": "/api/health",
        "requestContext": {
            "http": {
                "method": "GET",
                "path": "/api/health"
            }
        },
        "headers": {},
        "isBase64Encoded": False
    }
    
    class MockContext:
        function_name = "test"
        memory_limit_in_mb = 512
        invoked_function_arn = "arn:aws:lambda:us-east-1:123456789:function:test"
        aws_request_id = "test-request-id"
    
    result = lambda_handler(test_event, MockContext())
    print(f"Test result: {result}")
