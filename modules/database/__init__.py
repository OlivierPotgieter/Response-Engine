"""
Database Operations Module
Handles all database connections and operations for the response engine.
"""

from .main_db import MainDatabase, get_customer_request_data, get_custom_response_data, get_response_example_by_id
from .backend_db import (
    BackendDatabase,
    ProductLookupService,
    get_product_data,
    get_all_products_for_request,
    get_product_intelligence_from_database,
)

def get_category_intelligence_from_database():
    """
    Get category intelligence from database

    Returns:
        Dict with category intelligence data from live database
    """
    try:
        backend_db = BackendDatabase()
        try:
            return backend_db.get_category_intelligence_data()
        finally:
            backend_db.close()
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error getting category intelligence from database: {e}")
        return {}

__all__ = [
    # Main database operations
    "MainDatabase",
    "get_customer_request_data",
    "get_custom_response_data",
    "get_response_example_by_id",  # FIXED: Added missing export
    # Backend database operations
    "BackendDatabase",
    "ProductLookupService",
    "get_product_data",
    "get_all_products_for_request",
    "get_product_intelligence_from_database",
    "get_category_intelligence_from_database",
]