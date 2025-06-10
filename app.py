"""
Response Engine Flask Application
Main application that coordinates all modules to provide customer response generation services.
"""

import os
import sys
import logging
from datetime import datetime
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from modules.processors import (
    process_customer_request,
    validate_request_data,
    build_test_endpoint_response,
    build_process_endpoint_response,
    build_api_error_response,
)
from modules.processors.data_processor import get_request_summary
from modules.processors.response_builder import get_response_builder
from modules.ai import test_pinecone_connection, test_llm_generation
from modules.database import get_customer_request_data, get_product_data

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

# Global variables for tracking
app_start_time = datetime.now()
request_count = 0


def increment_request_count():
    """Increment and return request count"""
    global request_count
    request_count += 1
    return request_count


@app.before_request
def log_request():
    """Log incoming requests"""
    req_id = increment_request_count()
    logger.info(
        f"Request #{req_id}: {request.method} {request.path} from {request.remote_addr}"
    )


@app.after_request
def add_cors_headers(response):
    """Add CORS headers to all responses"""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return (
        jsonify(
            build_api_error_response(
                "Endpoint not found",
                "not_found",
                404,
                {"path": request.path, "method": request.method},
            )
        ),
        404,
    )


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return (
        jsonify(
            build_api_error_response("Internal server error", "internal_error", 500)
        ),
        500,
    )


@app.route("/", methods=["GET"])
def root():
    """Root endpoint with API information"""
    response_builder = get_response_builder()

    api_info = {
        "service": "Response Engine API",
        "version": response_builder.api_version,
        "status": "running",
        "uptime": str(datetime.now() - app_start_time),
        "total_requests": request_count,
        "endpoints": {
            "health": "GET /health - Health check",
            "test": "GET /test/<request_id> - Test processing without AI response",
            "process": "GET /process/<request_id> - Full processing with AI response",
            "summary": "GET /summary/<request_id> - Quick request summary",
            "debug": {
                "product": "GET /debug/product/<product_id> - Debug product lookup",
                "connections": "GET /debug/connections - Test all connections",
            },
        },
        "documentation": "See README.md for detailed API documentation",
    }

    return jsonify(
        response_builder.build_success_response(
            api_info, "Response Engine API is running"
        )
    )


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    response_builder = get_response_builder()
    return jsonify(response_builder.build_health_response())


@app.route("/test/<request_id>", methods=["GET"])
def test_customer_request(request_id):
    """
    Test endpoint - processes request without AI response generation
    Fast and cost-free testing of the pipeline
    """
    try:
        logger.info(f"Processing test request for ID: {request_id}")

        # Process request without AI response generation
        processing_result = process_customer_request(
            request_id, generate_response=False
        )

        # Build standardized test response
        response = build_test_endpoint_response(request_id, processing_result)

        # Determine HTTP status code
        if response.get("status") == "success":
            status_code = 200
        elif response.get("status") == "out_of_scope":
            status_code = 200  # Still successful processing, just out of scope
        else:
            status_code = response.get("status_code", 400)

        logger.info(
            f"Test request {request_id} completed with status: {response.get('status')}"
        )
        return jsonify(response), status_code

    except Exception as e:
        logger.error(f"Test endpoint error for request {request_id}: {e}")
        import traceback

        logger.error(f"Full traceback: {traceback.format_exc()}")

        error_response = build_api_error_response(
            "Internal server error during testing",
            "test_endpoint_error",
            500,
            {"request_id": request_id, "error_details": str(e)},
        )
        return jsonify(error_response), 500


@app.route("/process/<request_id>", methods=["GET"])
def process_customer_request_full(request_id):
    """
    Main processing endpoint - full pipeline with AI response generation
    This endpoint will make API calls to OpenAI and cost money
    """
    try:
        logger.info(f"Processing full request for ID: {request_id}")

        # Process request with AI response generation
        processing_result = process_customer_request(request_id, generate_response=True)

        # Build standardized process response
        response = build_process_endpoint_response(request_id, processing_result)

        # Determine HTTP status code
        if response.get("status") == "success":
            status_code = 200
        elif response.get("status") == "out_of_scope":
            status_code = 200  # Still successful processing, just out of scope
        else:
            status_code = response.get("status_code", 400)

        # Log generation info
        if processing_result.get("generated_response"):
            generation_method = processing_result.get("generation_method", "unknown")
            response_length = len(processing_result.get("generated_response", ""))
            logger.info(
                f"Generated response for {request_id} using {generation_method} method, length: {response_length}"
            )

        logger.info(
            f"Full processing request {request_id} completed with status: {response.get('status')}"
        )
        return jsonify(response), status_code

    except Exception as e:
        logger.error(f"Process endpoint error for request {request_id}: {e}")
        import traceback

        logger.error(f"Full traceback: {traceback.format_exc()}")

        error_response = build_api_error_response(
            "Internal server error during processing",
            "process_endpoint_error",
            500,
            {"request_id": request_id, "error_details": str(e)},
        )
        return jsonify(error_response), 500


@app.route("/summary/<request_id>", methods=["GET"])
def get_customer_request_summary(request_id):
    """
    Quick summary endpoint - basic request information without heavy processing
    """
    try:
        logger.info(f"Getting summary for request ID: {request_id}")

        # Get request summary
        summary_result = get_request_summary(request_id)

        response_builder = get_response_builder()
        response = response_builder.build_summary_response(summary_result)

        # Determine HTTP status code
        if response.get("status") == "success":
            status_code = 200
        else:
            status_code = response.get("status_code", 400)

        logger.info(
            f"Summary for {request_id} completed with status: {response.get('status')}"
        )
        return jsonify(response), status_code

    except Exception as e:
        logger.error(f"Summary endpoint error for request {request_id}: {e}")

        error_response = build_api_error_response(
            "Error generating summary",
            "summary_endpoint_error",
            500,
            {"request_id": request_id, "error_details": str(e)},
        )
        return jsonify(error_response), 500


@app.route("/debug/product/<product_id>", methods=["GET"])
def debug_product_lookup(product_id):
    """
    Debug endpoint to test product lookup directly
    """
    try:
        logger.info(f"Debug: Looking up product ID {product_id}")

        # Get product data
        product_data = get_product_data(product_id, is_alternative=False)

        debug_data = {
            "product_id": product_id,
            "product_data": product_data,
            "lookup_successful": product_data is not None
            and not product_data.get("error"),
            "debug_timestamp": datetime.now().isoformat(),
        }

        response_builder = get_response_builder()
        response = response_builder.build_debug_response(debug_data)

        return jsonify(response)

    except Exception as e:
        logger.error(f"Debug product endpoint error: {e}")

        error_response = build_api_error_response(
            "Debug product lookup failed",
            "debug_product_error",
            500,
            {"product_id": product_id, "error_details": str(e)},
        )
        return jsonify(error_response), 500


@app.route("/debug/connections", methods=["GET"])
def debug_connections():
    """
    Debug endpoint to test all service connections
    """
    try:
        logger.info("Testing all service connections...")

        # Test Pinecone connection
        pinecone_status = test_pinecone_connection()

        # Test LLM generation
        llm_status = test_llm_generation()

        # Test database connection by trying a simple query
        try:
            test_data = get_customer_request_data("1")  # Test with ID 1
            db_connected = test_data is not None
        except Exception as e:
            db_connected = False
            logger.error(f"Database connection test failed: {e}")

        debug_data = {
            "database": {"connected": db_connected, "note": "Tested with simple query"},
            "pinecone": pinecone_status,
            "llm": llm_status,
            "overall_status": all(
                [
                    db_connected,
                    pinecone_status.get("openai_connected", False),
                    pinecone_status.get("pinecone_connected", False),
                    llm_status.get("test_successful", False),
                ]
            ),
            "test_timestamp": datetime.now().isoformat(),
        }

        response_builder = get_response_builder()
        response = response_builder.build_debug_response(debug_data)

        return jsonify(response)

    except Exception as e:
        logger.error(f"Debug connections endpoint error: {e}")

        error_response = build_api_error_response(
            "Connection testing failed",
            "debug_connections_error",
            500,
            {"error_details": str(e)},
        )
        return jsonify(error_response), 500


@app.route("/validate/<request_id>", methods=["GET"])
def validate_request(request_id):
    """
    Validation endpoint - check if request ID exists and is valid
    """
    try:
        logger.info(f"Validating request ID: {request_id}")

        # Validate request data
        validation_result = validate_request_data(request_id)

        response_builder = get_response_builder()

        if validation_result["status"] == "success":
            validation_data = {
                "request_id": request_id,
                "valid": True,
                "customer_comment_preview": validation_result.get("data", {}).get(
                    "customer_comment", ""
                )[:100],
                "validation_timestamp": datetime.now().isoformat(),
            }
            response = response_builder.build_success_response(
                validation_data, "Request ID is valid"
            )
            status_code = 200
        else:
            response = response_builder.build_validation_error_response(
                validation_result
            )
            status_code = response.get("status_code", 400)

        return jsonify(response), status_code

    except Exception as e:
        logger.error(f"Validation endpoint error for request {request_id}: {e}")

        error_response = build_api_error_response(
            "Validation failed",
            "validation_endpoint_error",
            500,
            {"request_id": request_id, "error_details": str(e)},
        )
        return jsonify(error_response), 500


# Development/testing endpoints
@app.route("/stats", methods=["GET"])
def get_stats():
    """
    Get application statistics
    """
    stats_data = {
        "uptime": str(datetime.now() - app_start_time),
        "total_requests": request_count,
        "start_time": app_start_time.isoformat(),
        "current_time": datetime.now().isoformat(),
        "environment": {
            "python_version": sys.version,
            "flask_env": os.getenv("FLASK_ENV", "production"),
        },
    }

    response_builder = get_response_builder()
    return jsonify(
        response_builder.build_success_response(stats_data, "Application statistics")
    )


if __name__ == "__main__":
    # Application startup
    logger.info("=" * 50)
    logger.info("Starting Response Engine API")
    logger.info("=" * 50)

    # Validate environment variables
    required_env_vars = [
        "OPENAI_API_KEY",
        "PINECONE_API_KEY",
        "PINECONE_ENV",
        "PINECONE_INDEX",
        "DB_HOST",
        "DB_NAME",
        "DB_USER",
        "DB_PASSWORD",
        "BACKEND_DB_HOST",
        "BACKEND_DB_NAME",
        "BACKEND_DB_USER",
        "BACKEND_DB_PASSWORD",
    ]

    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        logger.error("Please check your .env file")
        exit(1)

    logger.info("✅ All required environment variables found")

    # Test connections on startup
    try:
        logger.info("Testing service connections...")

        # Test basic database connection
        test_db = get_customer_request_data("1")
        logger.info("✅ Main database connection successful")

        # Test Pinecone connection
        pinecone_test = test_pinecone_connection()
        if pinecone_test.get("openai_connected") and pinecone_test.get(
            "pinecone_connected"
        ):
            logger.info("✅ Pinecone and OpenAI connections successful")
        else:
            logger.warning("⚠️ Pinecone or OpenAI connection issues detected")

    except Exception as e:
        logger.warning(f"⚠️ Connection test failed: {e}")
        logger.warning("API will start but some endpoints may not work properly")

    # Start the Flask application
    port = int(os.getenv("PORT", 5001))
    debug_mode = os.getenv("FLASK_ENV") == "development"

    logger.info(f"Starting Flask application on port {port}")
    logger.info(f"Debug mode: {debug_mode}")
    logger.info("=" * 50)

    app.run(debug=debug_mode, host="0.0.0.0", port=port)
