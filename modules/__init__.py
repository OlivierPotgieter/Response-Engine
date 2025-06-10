"""
Response Engine Modules Package
Contains all modular components for the response engine system.
"""

__version__ = "1.1.0"

# Import main module components for easy access
from .database import get_customer_request_data, get_all_products_for_request
from .ai import predict_customer_intent, check_intent_scope
from .processors import (
    process_customer_request,
    validate_request_data,
    search_comment_for_products,
)

__all__ = [
    # Database operations
    "get_customer_request_data",
    "get_all_products_for_request",
    # AI operations
    "predict_customer_intent",
    "check_intent_scope",
    # Processing operations
    "process_customer_request",
    "validate_request_data",
    # Product search operations (placeholder)
    "search_comment_for_products",
]
