"""
Backend Database Operations Module
Handles all operations for the backend Products database table.
"""

import os
import logging
from typing import Dict, Optional, List
import mysql.connector
from mysql.connector import Error
from datetime import datetime

logger = logging.getLogger(__name__)


class BackendDatabase:
    def __init__(self):
        self.connection = None
        self._connect()

    def _connect(self):
        """Establish connection to backend database"""
        try:
            self.connection = mysql.connector.connect(
                host=os.getenv("BACKEND_DB_HOST"),
                database=os.getenv("BACKEND_DB_NAME"),
                user=os.getenv("BACKEND_DB_USER"),
                password=os.getenv("BACKEND_DB_PASSWORD"),
            )
            logger.info("Backend database connection established")

        except Error as e:
            logger.error(f"Backend database connection error: {e}")
            raise

    def _ensure_connection(self):
        """Ensure database connection is active"""
        if not self.connection or not self.connection.is_connected():
            logger.info("Reconnecting to backend database...")
            self._connect()

    def get_product_details(
        self, product_id: str, is_alternative: bool = False
    ) -> Optional[Dict]:
        """
        Fetch detailed product information from Products table

        Args:
            product_id: The ProductId to look up
            is_alternative: Whether this is an alternative product lookup

        Returns:
            Dict with product details if found, error dict if error, None if not found
        """
        self._ensure_connection()
        cursor = self.connection.cursor(dictionary=True)

        try:
            logger.info(f"Querying backend database for product ID: {product_id}")

            # Query the Products table using correct column names
            query = """
                    SELECT ProductId, \
                           Name, \
                           Sku, \
                           Category, \
                           CostExVat, \
                           Cost, \
                           Price, \
                           SpecialPrice, \
                           CurrentPrice, \
                           IsOnPromotion, \
                           Availability, \
                           LeadTime, \
                           Eta, \
                           IsEol, \
                           IsEnabled, \
                           StockQuantity, \
                           IsStockManaged, \
                           IsInStock, \
                           Created, \
                           Updated, \
                           ManufacturerId, \
                           Status, \
                           Rating, \
                           ReviewCount, \
                           ExpectedDispatch, \
                           ShortDescription, \
                           StockCondition, \
                           Description
                    FROM Products
                    WHERE ProductId = %s \
                    """

            cursor.execute(query, (product_id,))
            result = cursor.fetchone()

            if result:
                logger.info(
                    f"✅ Found product details for {'alternative' if is_alternative else 'main'} product ID {product_id}: {result.get('Name', 'Unknown')}"
                )
                return {
                    "product_id": result.get("ProductId"),
                    "name": result.get("Name"),
                    "sku": result.get("Sku"),
                    "category": result.get("Category"),
                    "cost_ex_vat": (
                        float(result.get("CostExVat", 0))
                        if result.get("CostExVat")
                        else None
                    ),
                    "cost": (
                        float(result.get("Cost", 0)) if result.get("Cost") else None
                    ),
                    "price": (
                        float(result.get("Price", 0)) if result.get("Price") else None
                    ),
                    "special_price": (
                        float(result.get("SpecialPrice", 0))
                        if result.get("SpecialPrice")
                        else None
                    ),
                    "current_price": (
                        float(result.get("CurrentPrice", 0))
                        if result.get("CurrentPrice")
                        else None
                    ),
                    "is_on_promotion": bool(result.get("IsOnPromotion", 0)),
                    "availability": result.get("Availability"),
                    "lead_time": result.get("LeadTime"),
                    "eta": result.get("Eta"),
                    "is_eol": bool(result.get("IsEol", 0)),
                    "is_enabled": bool(result.get("IsEnabled", 0)),
                    "stock_quantity": (
                        float(result.get("StockQuantity", 0))
                        if result.get("StockQuantity")
                        else 0
                    ),
                    "is_stock_managed": bool(result.get("IsStockManaged", 0)),
                    "is_in_stock": bool(result.get("IsInStock", 0)),
                    "created": result.get("Created"),
                    "updated": result.get("Updated"),
                    "manufacturer_id": result.get("ManufacturerId"),
                    "status": result.get("Status"),
                    "rating": (
                        float(result.get("Rating", 0)) if result.get("Rating") else None
                    ),
                    "review_count": result.get("ReviewCount"),
                    "expected_dispatch": result.get("ExpectedDispatch"),
                    "short_description": result.get("ShortDescription"),
                    "stock_condition": result.get("StockCondition"),
                    "description": result.get("Description"),
                    "is_alternative": is_alternative,
                    "query_timestamp": datetime.now().isoformat(),
                }
            else:
                logger.warning(
                    f"❌ No product found in Products table for {'alternative' if is_alternative else 'main'} product ID {product_id}"
                )
                return {
                    "error": "Product not found",
                    "product_id": product_id,
                    "is_alternative": is_alternative,
                    "message": f"No product with ProductId {product_id} found in backend database",
                }

        except Error as e:
            logger.error(f"❌ Backend product query error for ID {product_id}: {e}")
            return {
                "error": "Database query error",
                "error_code": e.errno if hasattr(e, "errno") else None,
                "error_message": str(e),
                "product_id": product_id,
                "is_alternative": is_alternative,
                "query_timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"❌ Unexpected error querying product ID {product_id}: {e}")
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")
            return {
                "error": "Unexpected error",
                "error_message": str(e),
                "product_id": product_id,
                "is_alternative": is_alternative,
                "query_timestamp": datetime.now().isoformat(),
            }
        finally:
            cursor.close()

    def get_products_by_ids(self, product_ids: List[str]) -> Dict:
        """
        Get multiple products by their IDs

        Args:
            product_ids: List of ProductIds to look up

        Returns:
            Dict with products found and any errors
        """
        results = {"products": {}, "errors": {}, "found_count": 0, "not_found_count": 0}

        for product_id in product_ids:
            if product_id:  # Skip None or empty IDs
                product_data = self.get_product_details(str(product_id))
                if product_data and not product_data.get("error"):
                    results["products"][product_id] = product_data
                    results["found_count"] += 1
                else:
                    results["errors"][product_id] = product_data
                    results["not_found_count"] += 1

        return results

    def close(self):
        """Close database connection"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("Backend database connection closed")


class ProductLookupService:
    """
    High-level service for product lookups combining main DB data with backend product details
    """

    def __init__(self):
        self.backend_db = BackendDatabase()

    def get_all_product_details(self, customer_data: Dict) -> Dict:
        """
        Get product details for both main product and alternative if they exist

        Args:
            customer_data: Customer data from main database

        Returns:
            Dict with main_product, alternative_product, and metadata
        """
        try:
            product_details = {
                "main_product": None,
                "alternative_product": None,
                "products_found": [],
                "debug_info": {},
            }

            # Debug: Log what IDs we're looking for
            main_product_id = customer_data.get("product_id")
            alternative_id = customer_data.get("alternative_id")

            logger.info(
                f"Looking for products - Main ID: {main_product_id}, Alternative ID: {alternative_id}"
            )
            product_details["debug_info"]["main_product_id"] = main_product_id
            product_details["debug_info"]["alternative_id"] = alternative_id

            # Check for main product_id
            if main_product_id:
                logger.info(f"Searching for main product with ID: {main_product_id}")
                main_product = self.backend_db.get_product_details(
                    str(main_product_id), is_alternative=False
                )
                if main_product and not main_product.get("error"):
                    product_details["main_product"] = main_product
                    product_details["products_found"].append("main_product")
                    logger.info(
                        f"Found main product: {main_product.get('name', 'Unknown')}"
                    )
                else:
                    logger.warning(f"Main product not found for ID: {main_product_id}")
                    if main_product and main_product.get("error"):
                        product_details["main_product"] = (
                            main_product  # Include error info
                        )

            # Check for alternative_id
            if alternative_id and str(alternative_id).strip():
                logger.info(
                    f"Searching for alternative product with ID: {alternative_id}"
                )
                alternative_product = self.backend_db.get_product_details(
                    str(alternative_id), is_alternative=True
                )
                if alternative_product and not alternative_product.get("error"):
                    product_details["alternative_product"] = alternative_product
                    product_details["products_found"].append("alternative_product")
                    logger.info(
                        f"Found alternative product: {alternative_product.get('name', 'Unknown')}"
                    )
                else:
                    logger.warning(
                        f"Alternative product not found for ID: {alternative_id}"
                    )
                    if alternative_product and alternative_product.get("error"):
                        product_details["alternative_product"] = (
                            alternative_product  # Include error info
                        )

            logger.info(
                f"Product lookup complete. Found: {product_details['products_found']}"
            )
            return product_details

        except Exception as e:
            logger.error(f"Exception in get_all_product_details: {e}")
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")
            return {
                "error": f"Exception in get_all_product_details: {str(e)}",
                "main_product": None,
                "alternative_product": None,
                "products_found": [],
                "traceback": traceback.format_exc(),
            }

    def close(self):
        """Close all database connections"""
        self.backend_db.close()


# Convenience functions for single operations
def get_product_data(product_id: str, is_alternative: bool = False) -> Optional[Dict]:
    """
    Convenience function to get single product data

    Args:
        product_id: The ProductId to look up
        is_alternative: Whether this is an alternative product

    Returns:
        Dict with product data if found, None otherwise
    """
    db = BackendDatabase()
    try:
        return db.get_product_details(product_id, is_alternative)
    finally:
        db.close()


def get_all_products_for_request(customer_data: Dict) -> Dict:
    """
    Convenience function to get all product data for a customer request

    Args:
        customer_data: Customer data from main database

    Returns:
        Dict with all product details
    """
    service = ProductLookupService()
    try:
        return service.get_all_product_details(customer_data)
    finally:
        service.close()
