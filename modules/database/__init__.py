"""
Database Operations Module
Handles all database connections and operations for the response engine.
"""

from .main_db import MainDatabase, get_customer_request_data, get_custom_response_data
from .backend_db import BackendDatabase, ProductLookupService, get_product_data, get_all_products_for_request

__all__ = [
    # Main database operations
    'MainDatabase',
    'get_customer_request_data',
    'get_custom_response_data',

    # Backend database operations
    'BackendDatabase',
    'ProductLookupService',
    'get_product_data',
    'get_all_products_for_request'
]