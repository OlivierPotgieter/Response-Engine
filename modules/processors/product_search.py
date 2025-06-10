"""
Product Search Module - Placeholder Implementation
Attempts to identify and search for products mentioned in customer comments.
This is a temporary solution while a dedicated product identification system is being built.
"""

import os
import logging
import re
from typing import Dict, List
from datetime import datetime
import mysql.connector
from mysql.connector import Error

logger = logging.getLogger(__name__)


class ProductSearchPlaceholder:
    def __init__(self):
        """Initialize the placeholder product search system"""
        self.connection = None
        self.brand_patterns = self._build_brand_patterns()
        self.model_patterns = self._build_model_patterns()
        self._connect_to_backend()

    def _connect_to_backend(self):
        """Establish connection to backend database"""
        try:
            self.connection = mysql.connector.connect(
                host=os.getenv("BACKEND_DB_HOST"),
                database=os.getenv("BACKEND_DB_NAME"),
                user=os.getenv("BACKEND_DB_USER"),
                password=os.getenv("BACKEND_DB_PASSWORD"),
            )
            logger.info("Product search connected to backend database")
        except Error as e:
            logger.error(f"Backend database connection error for product search: {e}")
            self.connection = None

    def _ensure_connection(self):
        """Ensure database connection is active"""
        if not self.connection or not self.connection.is_connected():
            logger.info("Reconnecting product search to backend database...")
            self._connect_to_backend()

    def _build_brand_patterns(self) -> List[str]:
        """Build regex patterns for common brand recognition"""
        return [
            r"\bseagate\b",
            r"\bwestern\s+digital\b",
            r"\bwd\b",
            r"\bsamsung\b",
            r"\btoshiba\b",
            r"\bhitachi\b",
            r"\bintel\b",
            r"\bamd\b",
            r"\bnvidia\b",
            r"\basus\b",
            r"\bmsi\b",
            r"\bgigabyte\b",
            r"\bevga\b",
            r"\bcorsair\b",
            r"\blogitech\b",
            r"\brazer\b",
            r"\bkingston\b",
            r"\bcrucial\b",
        ]

    def _build_model_patterns(self) -> List[str]:
        """Build regex patterns for common model number formats"""
        return [
            # Storage drives
            r"\bST\d{4,}[A-Z\d]*\b",  # Seagate models (ST20000NM002H)
            r"\bWD\d{4,}[A-Z\d]*\b",  # WD models
            r"\b[A-Z]{2,}\d{4,}[A-Z\d]*\b",  # General model patterns
            # Graphics cards
            r"\bGTX\s*\d{4}\s*(Ti|Super)?\b",  # GTX 1080, GTX 1080 Ti
            r"\bRTX\s*\d{4}\s*(Ti|Super)?\b",  # RTX 4090, RTX 4080 Super
            r"\bRX\s*\d{4}\s*(XT|X)?\b",  # AMD RX series
            # Processors
            r"\bi[3579]-\d{4,5}[A-Z]*\b",  # Intel processors (i7-12700K)
            r"\bRyzen\s*\d+\s*\d{4}[A-Z]*\b",  # AMD Ryzen
            # Storage capacity indicators
            r"\b\d{1,3}TB\b",  # Storage sizes
            r"\b\d{2,4}GB\b",  # Memory/storage sizes
            # Memory
            r"\bDDR[45]-\d{4}\b",  # Memory types
        ]

    def extract_product_identifiers(self, comment: str) -> Dict:
        """
        Extract potential product identifiers from customer comment

        Args:
            comment: Customer comment text

        Returns:
            Dict with extracted identifiers and confidence scores
        """
        try:
            results = {
                "brands_found": [],
                "models_found": [],
                "capacities_found": [],
                "categories_inferred": [],
                "search_terms": [],
                "confidence_score": 0.0,
                "extraction_timestamp": datetime.now().isoformat(),
            }

            comment_lower = comment.lower()

            # Extract brands
            for pattern in self.brand_patterns:
                matches = re.findall(pattern, comment_lower, re.IGNORECASE)
                if matches:
                    results["brands_found"].extend([m.title() for m in matches])

            # Extract model numbers
            for pattern in self.model_patterns:
                matches = re.findall(pattern, comment, re.IGNORECASE)
                if matches:
                    results["models_found"].extend(matches)

            # Extract capacity indicators
            capacity_patterns = [r"\b\d{1,3}TB\b", r"\b\d{2,4}GB\b"]
            for pattern in capacity_patterns:
                matches = re.findall(pattern, comment, re.IGNORECASE)
                if matches:
                    results["capacities_found"].extend(matches)

            # Infer categories based on keywords
            category_keywords = {
                "HDD": ["hard drive", "hdd", "internal drive", "storage drive"],
                "SSD": ["ssd", "solid state"],
                "Graphics Card": ["graphics", "gpu", "video card", "gtx", "rtx"],
                "Processor": ["processor", "cpu", "intel", "amd", "ryzen"],
                "Memory": ["memory", "ram", "ddr"],
                "Motherboard": ["motherboard", "mobo"],
                "Power Supply": ["power supply", "psu"],
            }

            for category, keywords in category_keywords.items():
                if any(keyword in comment_lower for keyword in keywords):
                    results["categories_inferred"].append(category)

            # Build search terms
            search_terms = []
            search_terms.extend(results["brands_found"])
            search_terms.extend(results["models_found"])
            search_terms.extend(results["capacities_found"])
            results["search_terms"] = list(set(search_terms))  # Remove duplicates

            # Calculate confidence score
            confidence = 0.0
            if results["models_found"]:
                confidence += 0.6  # Model numbers give highest confidence
            if results["brands_found"]:
                confidence += 0.3  # Brands add confidence
            if results["capacities_found"]:
                confidence += 0.1  # Capacity indicators add some confidence

            results["confidence_score"] = min(confidence, 1.0)

            logger.info(
                f"Extracted product identifiers: {len(results['search_terms'])} terms, "
                f"confidence: {confidence:.2f}"
            )

            return results

        except Exception as e:
            logger.error(f"Error extracting product identifiers: {e}")
            return {
                "brands_found": [],
                "models_found": [],
                "capacities_found": [],
                "categories_inferred": [],
                "search_terms": [],
                "confidence_score": 0.0,
                "error": str(e),
                "extraction_timestamp": datetime.now().isoformat(),
            }

    def search_products_by_identifiers(
        self, identifiers: Dict, max_results: int = 5
    ) -> List[Dict]:
        """
        Search backend database for products matching extracted identifiers

        Args:
            identifiers: Results from extract_product_identifiers
            max_results: Maximum number of results to return

        Returns:
            List of matching products with relevance scores
        """
        try:
            if not self.connection:
                logger.warning("No database connection available for product search")
                return []

            self._ensure_connection()
            cursor = self.connection.cursor(dictionary=True)

            search_terms = identifiers.get("search_terms", [])
            if not search_terms:
                logger.info("No search terms available for product search")
                return []

            logger.info(f"Searching for products with terms: {search_terms}")

            # Build dynamic search query
            search_conditions = []
            search_params = []

            for term in search_terms[
                :3
            ]:  # Limit to first 3 terms to avoid overly complex queries
                # Search in Name, Sku, Description
                condition = "(p.Name LIKE %s OR p.Sku LIKE %s OR p.Description LIKE %s)"
                search_conditions.append(condition)
                search_params.extend([f"%{term}%", f"%{term}%", f"%{term}%"])

            if not search_conditions:
                return []

            # Construct query
            base_query = """
                         SELECT p.ProductId, 
                                p.Name, 
                                p.Sku, 
                                p.Category, 
                                p.CurrentPrice,
                                p.IsInStock, 
                                p.IsEnabled, 
                                p.IsEol, 
                                p.LeadTime,
                                p.ShortDescription, 
                                p.Updated, 
                                b.Name as BrandName
                         FROM Products p
                                  LEFT JOIN Brands b ON p.ManufacturerId = b.ManufacturerId
                         WHERE p.IsEnabled = 1 
                           AND ( 
                         """

            query = base_query + " OR ".join(search_conditions) + ")"
            query += " ORDER BY p.IsInStock DESC, p.CurrentPrice ASC"
            query += f" LIMIT {max_results}"

            cursor.execute(query, search_params)
            results = cursor.fetchall()

            # Calculate relevance scores and format results
            formatted_results = []
            for result in results:
                relevance_score = self._calculate_relevance_score(result, identifiers)

                formatted_result = {
                    "product_id": result.get("ProductId"),
                    "name": result.get("Name"),
                    "sku": result.get("Sku"),
                    "brand": result.get("BrandName"),
                    "category": result.get("Category"),
                    "current_price": (
                        float(result.get("CurrentPrice", 0))
                        if result.get("CurrentPrice")
                        else None
                    ),
                    "is_in_stock": bool(result.get("IsInStock", 0)),
                    "is_enabled": bool(result.get("IsEnabled", 0)),
                    "is_eol": bool(result.get("IsEol", 0)),
                    "lead_time": result.get("LeadTime"),
                    "short_description": result.get("ShortDescription"),
                    "relevance_score": relevance_score,
                    "search_timestamp": datetime.now().isoformat(),
                }
                formatted_results.append(formatted_result)

            # Sort by relevance score (highest first)
            formatted_results.sort(key=lambda x: x["relevance_score"], reverse=True)

            logger.info(f"Found {len(formatted_results)} potential product matches")
            return formatted_results

        except Error as e:
            logger.error(f"Database error in product search: {e}")
            return []
        except Exception as e:
            logger.error(f"Error searching products by identifiers: {e}")
            return []
        finally:
            if "cursor" in locals():
                cursor.close()

    def _calculate_relevance_score(self, product: Dict, identifiers: Dict) -> float:
        """
        Calculate relevance score for a product match

        Args:
            product: Product data from database
            identifiers: Extracted identifiers from comment

        Returns:
            Relevance score between 0.0 and 1.0
        """
        score = 0.0

        product_name = (product.get("Name") or "").lower()
        product_sku = (product.get("Sku") or "").lower()
        brand_name = (product.get("BrandName") or "").lower()

        # Check for exact model matches (highest score)
        for model in identifiers.get("models_found", []):
            if model.lower() in product_name or model.lower() in product_sku:
                score += 0.6
                break

        # Check for brand matches
        for brand in identifiers.get("brands_found", []):
            if brand.lower() in brand_name or brand.lower() in product_name:
                score += 0.3
                break

        # Check for capacity matches
        for capacity in identifiers.get("capacities_found", []):
            if capacity.lower() in product_name:
                score += 0.1
                break

        # Bonus for in-stock products
        if product.get("IsInStock"):
            score += 0.05

        # Penalty for EOL products
        if product.get("IsEol"):
            score -= 0.2

        return min(max(score, 0.0), 1.0)  # Clamp between 0 and 1

    def search_comment_for_products(self, comment: str, max_results: int = 3) -> Dict:
        """
        Complete pipeline: extract identifiers and search for products

        Args:
            comment: Customer comment text
            max_results: Maximum number of products to return

        Returns:
            Dict with extraction results and found products
        """
        try:
            logger.info(f"Starting product search for comment: {comment[:100]}...")

            # Step 1: Extract identifiers
            identifiers = self.extract_product_identifiers(comment)

            # Step 2: Search for products if we have good identifiers
            products_found = []
            if (
                identifiers.get("confidence_score", 0) > 0.2
            ):  # Minimum confidence threshold
                products_found = self.search_products_by_identifiers(
                    identifiers, max_results
                )
            else:
                logger.info(
                    f"Confidence score too low ({identifiers.get('confidence_score', 0):.2f}) - skipping database search"
                )

            return {
                "comment_analyzed": comment,
                "identifiers_extracted": identifiers,
                "products_found": products_found,
                "search_successful": len(products_found) > 0,
                "best_match": products_found[0] if products_found else None,
                "search_summary": {
                    "extraction_confidence": identifiers.get("confidence_score", 0),
                    "products_found_count": len(products_found),
                    "search_terms_used": identifiers.get("search_terms", []),
                    "categories_inferred": identifiers.get("categories_inferred", []),
                },
                "search_timestamp": datetime.now().isoformat(),
                "note": "This is a placeholder product search implementation",
            }

        except Exception as e:
            logger.error(f"Error in complete product search pipeline: {e}")
            return {
                "comment_analyzed": comment,
                "search_successful": False,
                "error": str(e),
                "search_timestamp": datetime.now().isoformat(),
                "note": "Product search failed - placeholder implementation",
            }

    def close(self):
        """Close database connection"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("Product search database connection closed")


# Global instance (lazy-loaded) - SINGLETON PATTERN
_product_search_instance = None


def get_product_search() -> ProductSearchPlaceholder:
    """
    Get a global product search instance (singleton pattern)

    This ensures only ONE instance of ProductSearchPlaceholder exists across the entire application,
    which is important for database connections and performance.

    Returns:
        ProductSearchPlaceholder instance
    """
    global _product_search_instance
    if _product_search_instance is None:
        _product_search_instance = ProductSearchPlaceholder()
    return _product_search_instance


def search_comment_for_products(comment: str, max_results: int = 3) -> Dict:
    """
    Convenience function to search for products mentioned in a comment

    This is a STANDALONE function (not a class method) that internally uses
    the singleton ProductSearchPlaceholder instance.

    Args:
        comment: Customer comment text
        max_results: Maximum number of products to return

    Returns:
        Dict with search results
    """
    try:
        search_engine = get_product_search()
        return search_engine.search_comment_for_products(comment, max_results)
    except Exception as e:
        logger.error(f"Error in convenience function search_comment_for_products: {e}")
        return {
            "comment_analyzed": comment,
            "search_successful": False,
            "error": f"Convenience function error: {str(e)}",
            "search_timestamp": datetime.now().isoformat(),
            "note": "Product search convenience function failed",
        }


def extract_product_identifiers_from_comment(comment: str) -> Dict:
    """
    Convenience function to extract product identifiers from comment

    This is also a STANDALONE function that uses the singleton instance.

    Args:
        comment: Customer comment text

    Returns:
        Dict with extracted identifiers
    """
    try:
        search_engine = get_product_search()
        return search_engine.extract_product_identifiers(comment)
    except Exception as e:
        logger.error(
            f"Error in convenience function extract_product_identifiers_from_comment: {e}"
        )
        return {
            "brands_found": [],
            "models_found": [],
            "capacities_found": [],
            "categories_inferred": [],
            "search_terms": [],
            "confidence_score": 0.0,
            "error": f"Convenience function error: {str(e)}",
            "extraction_timestamp": datetime.now().isoformat(),
        }


def test_product_search(test_comment: str = "I need a Seagate 4TB hard drive") -> Dict:
    """
    NEW: Test function to validate product search functionality

    Args:
        test_comment: Comment to test with

    Returns:
        Dict with test results
    """
    try:
        logger.info(f"Testing product search with: {test_comment}")

        # Test the full pipeline
        search_results = search_comment_for_products(test_comment, max_results=3)

        # Test just extraction
        extraction_results = extract_product_identifiers_from_comment(test_comment)

        return {
            "test_successful": True,
            "test_comment": test_comment,
            "extraction_results": extraction_results,
            "search_results": search_results,
            "extraction_confidence": extraction_results.get("confidence_score", 0),
            "products_found": len(search_results.get("products_found", [])),
            "best_match": search_results.get("best_match"),
            "test_timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error in product search test: {e}")
        return {
            "test_successful": False,
            "test_comment": test_comment,
            "error": str(e),
            "test_timestamp": datetime.now().isoformat(),
        }
