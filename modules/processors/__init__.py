"""
Data Processing Module
Handles data validation, processing, and business logic for the response engine.
"""

from .data_processor import DataProcessor, process_customer_request, validate_request_data, initialize_ai_product_identification, get_ai_system_health
from .response_builder import ResponseBuilder, build_test_endpoint_response, build_process_endpoint_response, build_api_error_response
from .product_search import ProductSearchPlaceholder, search_comment_for_products, extract_product_identifiers_from_comment

__all__ = [
    # Data processing
    'DataProcessor',
    'process_customer_request',
    'validate_request_data',
    'initialize_ai_product_identification',
    'get_ai_system_health',

    # Response building
    'ResponseBuilder',
    'build_test_endpoint_response',
    'build_process_endpoint_response',
    'build_api_error_response',

    # Product search (placeholder)
    'ProductSearchPlaceholder',
    'search_comment_for_products',
    'extract_product_identifiers_from_comment'
]