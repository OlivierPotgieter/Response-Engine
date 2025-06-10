"""
Main Database Operations Module
Handles all operations for the main availability database tables:
- wootware_inventorymanagement_availability_conversion
- wootware_inventorymanagement_availability_custom_body
"""

import os
import logging
from typing import Dict, Optional
import mysql.connector
from mysql.connector import Error
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class MainDatabase:
    def __init__(self):
        self.connection = None
        self._connect()

    def _connect(self):
        """Establish connection to main database"""
        try:
            self.connection = mysql.connector.connect(
                host=os.getenv("DB_HOST"),
                database=os.getenv("DB_NAME"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD")
            )
            logger.info("Main database connection established")

        except Error as e:
            logger.error(f"Main database connection error: {e}")
            raise

    def _ensure_connection(self):
        """Ensure database connection is active"""
        if not self.connection or not self.connection.is_connected():
            logger.info("Reconnecting to main database...")
            self._connect()

    def get_customer_request(self, request_id: str) -> Dict:
        """
        Fetch customer request data from availability_conversion table

        Args:
            request_id: The ID of the request to fetch

        Returns:
            Dict containing status, data, and metadata
        """
        self._ensure_connection()
        cursor = self.connection.cursor(dictionary=True)

        try:
            # Query to get all data from availability_conversion table
            query = """
                    SELECT id, date, customer_email, customer_firstname, customer_telephone, customer_comment, product_id, product_name, product_sku, parent_leadtime, parent_comment, alternative_id, alternative_name, alternative_leadtime, alternative_comment, woot_rep, automated_response, purchased, closed_by, email_sent, date_responded, country, ip, feedback_hash
                    FROM wootware_inventorymanagement_availability_conversion
                    WHERE id = %s \
                    """

            cursor.execute(query, (request_id,))
            result = cursor.fetchone()

            if not result:
                return {
                    "error": f"Request ID {request_id} not found in database",
                    "status": "not_found"
                }

            # Check if automated_response is not 0 (outside scope warning)
            scope_warning = None
            if result.get('automated_response', 0) != 0:
                scope_warning = "WARNING: This query type is outside of scope (automated_response != 0)"

            # Log what data was received vs missing for key fields
            key_fields = [
                'customer_comment', 'product_id', 'product_name', 'parent_leadtime',
                'alternative_id', 'alternative_name', 'alternative_leadtime', 'woot_rep'
            ]

            data_log = {
                "received_fields": [],
                "missing_fields": [],
                "null_fields": []
            }

            for field in key_fields:
                value = result.get(field)
                if value is not None and str(value).strip():
                    data_log["received_fields"].append(field)
                elif value is None:
                    data_log["missing_fields"].append(field)
                else:
                    data_log["null_fields"].append(field)

            logger.info(f"Data retrieval for ID {request_id}: "
                        f"Received: {data_log['received_fields']}, "
                        f"Missing: {data_log['missing_fields']}, "
                        f"Null: {data_log['null_fields']}")

            return {
                "status": "success",
                "data": result,
                "data_log": data_log,
                "scope_warning": scope_warning
            }

        except Error as e:
            logger.error(f"Database query error for ID {request_id}: {e}")
            return {
                "error": f"Database error: {str(e)}",
                "status": "database_error"
            }
        finally:
            cursor.close()

    def get_custom_body_response(self, request_id: str) -> Optional[Dict]:
        """
        Check for existing custom response in the custom_body table

        Args:
            request_id: The ID to look up in custom_body table

        Returns:
            Dict with custom body data if found, None if not found
        """
        self._ensure_connection()
        cursor = self.connection.cursor(dictionary=True)

        try:
            query = """
                    SELECT id, body
                    FROM wootware_inventorymanagement_availability_custom_body
                    WHERE id = %s \
                    """

            cursor.execute(query, (request_id,))
            result = cursor.fetchone()

            if not result:
                logger.info(f"No custom body found for ID {request_id}")
                return None

            # Clean HTML from the body using BeautifulSoup
            raw_body = result.get('body', '')
            if raw_body:
                soup = BeautifulSoup(raw_body, 'html.parser')
                cleaned_body = soup.get_text(separator=' ', strip=True)

                logger.info(f"Found custom body for ID {request_id}, length: {len(cleaned_body)}")

                return {
                    "id": result['id'],
                    "raw_html_body": raw_body,
                    "cleaned_text_body": cleaned_body,
                    "body_length": len(cleaned_body)
                }
            else:
                logger.info(f"Custom body exists for ID {request_id} but is empty")
                return {
                    "id": result['id'],
                    "raw_html_body": "",
                    "cleaned_text_body": "",
                    "body_length": 0
                }

        except Error as e:
            logger.error(f"Custom body query error for ID {request_id}: {e}")
            return None
        finally:
            cursor.close()

    def get_key_fields(self, customer_data: Dict) -> Dict:
        """
        Extract key fields from customer data for easy access

        Args:
            customer_data: Raw customer data from database

        Returns:
            Dict with organized key fields
        """
        return {
            "customer_comment": customer_data.get('customer_comment'),
            "product_id": customer_data.get('product_id'),
            "product_name": customer_data.get('product_name'),
            "parent_leadtime": customer_data.get('parent_leadtime'),
            "alternative_id": customer_data.get('alternative_id'),
            "alternative_name": customer_data.get('alternative_name'),
            "alternative_leadtime": customer_data.get('alternative_leadtime'),
            "woot_rep": customer_data.get('woot_rep')
        }

    def close(self):
        """Close database connection"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("Main database connection closed")


# Convenience function for single operations
def get_customer_request_data(request_id: str) -> Dict:
    """
    Convenience function to get customer request data

    Args:
        request_id: The ID of the request to fetch

    Returns:
        Dict containing request data and metadata
    """
    db = MainDatabase()
    try:
        return db.get_customer_request(request_id)
    finally:
        db.close()


def get_custom_response_data(request_id: str) -> Optional[Dict]:
    """
    Convenience function to get custom response data

    Args:
        request_id: The ID to look up

    Returns:
        Dict with custom response data if found, None otherwise
    """
    db = MainDatabase()
    try:
        return db.get_custom_body_response(request_id)
    finally:
        db.close()