"""
Backend Database Operations Module - CLEAN VERSION
Handles all operations for the backend Products database table.
"""

import os
import logging
from typing import Dict, Optional, List
import mysql.connector
from mysql.connector import Error
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Product:
    """Product data class with correct manufacturer information"""

    product_id: int
    name: str
    sku: str
    category: str
    manufacturer_id: int
    manufacturer_name: str
    current_price: float
    is_enabled: bool
    is_eol: bool
    is_in_stock: bool
    search_text: str
    short_description: str
    description: str
    popularity: float
    created: str
    updated: str


@dataclass
class Manufacturer:
    """Manufacturer data class for the correct table"""

    id: int
    name: str


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

    def get_product_details(self, product_id: str, is_alternative: bool = False) -> Optional[Dict]:
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
                    SELECT ProductId, 
                           Name, 
                           Sku, 
                           Category, 
                           CostExVat, 
                           Cost, 
                           Price, 
                           SpecialPrice, 
                           CurrentPrice, 
                           IsOnPromotion, 
                           Availability, 
                           LeadTime, 
                           Eta, 
                           IsEol, 
                           IsEnabled, 
                           StockQuantity, 
                           IsStockManaged, 
                           IsInStock, 
                           Created, 
                           Updated, 
                           ManufacturerId, 
                           Status, 
                           Rating, 
                           ReviewCount, 
                           ExpectedDispatch, 
                           ShortDescription, 
                           StockCondition, 
                           Description
                    FROM Products
                    WHERE ProductId = %s
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
                    "cost_ex_vat": float(result.get("CostExVat", 0)) if result.get("CostExVat") else None,
                    "cost": float(result.get("Cost", 0)) if result.get("Cost") else None,
                    "price": float(result.get("Price", 0)) if result.get("Price") else None,
                    "special_price": float(result.get("SpecialPrice", 0)) if result.get("SpecialPrice") else None,
                    "current_price": float(result.get("CurrentPrice", 0)) if result.get("CurrentPrice") else None,
                    "is_on_promotion": bool(result.get("IsOnPromotion", 0)),
                    "availability": result.get("Availability"),
                    "lead_time": result.get("LeadTime"),
                    "eta": result.get("Eta"),
                    "is_eol": bool(result.get("IsEol", 0)),
                    "is_enabled": bool(result.get("IsEnabled", 0)),
                    "stock_quantity": float(result.get("StockQuantity", 0)) if result.get("StockQuantity") else 0,
                    "is_stock_managed": bool(result.get("IsStockManaged", 0)),
                    "is_in_stock": bool(result.get("IsInStock", 0)),
                    "created": result.get("Created"),
                    "updated": result.get("Updated"),
                    "manufacturer_id": result.get("ManufacturerId"),
                    "status": result.get("Status"),
                    "rating": float(result.get("Rating", 0)) if result.get("Rating") else None,
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

    def get_category_intelligence_data(self) -> Dict:
        """
        Build category intelligence from Products and Manufacturers tables
        Replaces category_intelligence.json with live database queries

        Returns:
            Dictionary with category intelligence data
        """
        self._ensure_connection()
        cursor = self.connection.cursor(dictionary=True)

        try:
            logger.info("🏗️ Building category intelligence from database...")

            # Get all categories with their products and manufacturers
            query = """
                    SELECT p.Category, 
                           p.Name   as ProductName, 
                           p.SearchText, 
                           m.Name   as ManufacturerName, 
                           COUNT(*) as ProductCount
                    FROM Products p
                             LEFT JOIN Manufacturers m ON p.ManufacturerId = m.Id
                    WHERE p.IsEnabled = 1
                      AND p.Category IS NOT NULL
                      AND p.Category != ''
                      AND p.Name IS NOT NULL
                      AND p.Name != ''
                    GROUP BY p.Category, m.Name, p.Name, p.SearchText
                    ORDER BY p.Category, ProductCount DESC
                    """

            cursor.execute(query)
            results = cursor.fetchall()

            if not results:
                logger.warning("No category data found in database")
                return {}

            # Process results into category intelligence structure
            categories_data = {}
            category_products = {}

            for row in results:
                category = row['Category']
                manufacturer = row['ManufacturerName'] or 'Unknown'
                product_name = row['ProductName'] or ''
                search_text = row['SearchText'] or ''

                # Initialize category if not exists
                if category not in categories_data:
                    categories_data[category] = {
                        'manufacturers': set(),
                        'product_names': [],
                        'search_texts': []
                    }
                    category_products[category] = 0

                # Collect data for this category
                categories_data[category]['manufacturers'].add(manufacturer.lower())
                categories_data[category]['product_names'].append(product_name.lower())
                if search_text:
                    categories_data[category]['search_texts'].append(search_text.lower())
                category_products[category] += 1

            # Build final intelligence structure
            intelligence = {
                "categories": {},
                "created": datetime.now().isoformat(),
                "source": "live_database",
                "total_categories": len(categories_data),
                "note": "Category intelligence generated from Products and Manufacturers tables"
            }

            for category, data in categories_data.items():
                # Extract keywords from product names and search text
                keywords = self._extract_category_keywords(
                    data['product_names'] + data['search_texts']
                )

                # Extract patterns (model numbers, technical specs)
                patterns = self._extract_category_patterns(
                    data['product_names'] + data['search_texts']
                )

                # Convert manufacturers set to sorted list
                manufacturers = sorted(list(data['manufacturers']))

                intelligence["categories"][category] = {
                    "keywords": keywords,
                    "patterns": patterns,
                    "manufacturers": manufacturers,
                    "product_count": category_products[category]
                }

            logger.info(f"✅ Built category intelligence for {len(intelligence['categories'])} categories")
            return intelligence

        except Error as e:
            logger.error(f"❌ Database error building category intelligence: {e}")
            return {}
        except Exception as e:
            logger.error(f"❌ Error building category intelligence: {e}")
            return {}
        finally:
            cursor.close()

    def _extract_category_keywords(self, texts: List[str]) -> List[str]:
        """
        Extract relevant keywords from product names and search texts

        Args:
            texts: List of product names and search texts

        Returns:
            List of relevant keywords for the category
        """
        import re
        from collections import Counter

        # Common tech keywords that might be relevant
        tech_keywords = {
            'gpu', 'graphics', 'card', 'video', 'gaming', 'rtx', 'gtx', 'radeon', 'geforce',
            'cpu', 'processor', 'intel', 'amd', 'ryzen', 'core',
            'motherboard', 'mobo', 'mainboard', 'socket', 'chipset',
            'memory', 'ram', 'ddr4', 'ddr5', 'dimm',
            'storage', 'ssd', 'hdd', 'nvme', 'drive', 'hard', 'solid', 'state',
            'monitor', 'display', 'screen', '4k', '1440p', '1080p', 'hz', 'curved',
            'keyboard', 'mechanical', 'optical', 'wireless', 'rgb',
            'mouse', 'laser', 'sensor',
            'power', 'supply', 'psu', 'watt', 'modular',
            'case', 'tower', 'chassis', 'atx', 'mini', 'micro'
        }

        # Count word frequency across all texts
        word_counts = Counter()

        for text in texts:
            if not text:
                continue
            # Extract words (alphanumeric, 3+ characters)
            words = re.findall(r'\b[a-zA-Z0-9]{3,}\b', text.lower())
            for word in words:
                if word in tech_keywords:
                    word_counts[word] += 1

        # Return most common relevant keywords (top 15)
        return [word for word, count in word_counts.most_common(15)]

    def _extract_category_patterns(self, texts: List[str]) -> List[str]:
        """
        Extract technical patterns (model numbers, specs) from texts

        Args:
            texts: List of product names and search texts

        Returns:
            List of regex patterns found in the category
        """
        import re

        patterns = set()

        # Common tech patterns
        pattern_regexes = [
            r'\b(rtx|gtx)\s*\d{4}\b',  # RTX 4090, GTX 1080
            r'\b(radeon)\s*(rx|r)\s*\d{4}\b',  # Radeon RX 7900
            r'\b(ryzen)\s*\d+\s*\d{4}[a-z]*\b',  # Ryzen 7 5800X
            r'\b(core)\s*(i[3579])\s*\d{4,5}[a-z]*\b',  # Core i7-12700K
            r'\b\d+gb\b',  # Memory/storage sizes
            r'\b\d+tb\b',  # Storage sizes
            r'\b(ddr[45])\s*\d{4}\b',  # Memory types
            r'\b\d+hz\b',  # Monitor refresh rates
            r'\b\d+w\b',  # Power ratings
            r'\b\d+\"\b',  # Screen sizes
        ]

        for text in texts:
            if not text:
                continue
            text_lower = text.lower()

            for pattern_regex in pattern_regexes:
                matches = re.finditer(pattern_regex, text_lower)
                for match in matches:
                    # Store the actual pattern found, not the regex
                    patterns.add(match.group(0))

        # Return sorted list of unique patterns (limit to prevent bloat)
        return sorted(list(patterns))[:20]

    def get_all_manufacturers(self) -> List[Manufacturer]:
        """
        Get all manufacturers from the database

        Returns:
            List of Manufacturer objects
        """
        self._ensure_connection()
        cursor = self.connection.cursor(dictionary=True)

        try:
            # Build query based on actual table structure
            query = """
                SELECT *
                FROM Manufacturers
                WHERE Name IS NOT NULL 
                  AND Name != ''
                ORDER BY Name
            """

            cursor.execute(query)
            results = cursor.fetchall()

            manufacturers = []
            for row in results:
                # Create Manufacturer object
                manufacturer = Manufacturer(
                    id=row.get("Id") or row.get("id"),
                    name=row.get("Name") or row.get("name") or "",
                )
                manufacturers.append(manufacturer)

            logger.info(f"Retrieved {len(manufacturers)} manufacturers")
            return manufacturers

        except Error as e:
            logger.error(f"Error retrieving manufacturers: {e}")
            return []
        finally:
            cursor.close()

    def get_all_products(self, enabled_only: bool = True) -> List[Product]:
        """
        Get all products with CORRECT manufacturer information

        Args:
            enabled_only: Only return enabled products

        Returns:
            List of Product objects with manufacturer names
        """
        self._ensure_connection()
        cursor = self.connection.cursor(dictionary=True)

        try:
            # Now joining with Manufacturers table instead of Brands
            where_clause = "WHERE p.Name IS NOT NULL AND p.Name != ''"
            if enabled_only:
                where_clause += " AND p.IsEnabled = 1"

            query = f"""
                SELECT 
                    p.ProductId,
                    p.Name,
                    p.Sku,
                    p.Category,
                    p.ManufacturerId,
                    p.CurrentPrice,
                    p.IsEnabled,
                    p.IsEol,
                    p.IsInStock,
                    p.SearchText,
                    p.ShortDescription,
                    p.Description,
                    p.Popularity,
                    p.Created,
                    p.Updated,
                    COALESCE(m.Name, 'Unknown') as ManufacturerName
                FROM Products p
                LEFT JOIN Manufacturers m ON p.ManufacturerId = m.Id
                {where_clause}
                ORDER BY p.Name
            """

            cursor.execute(query)
            results = cursor.fetchall()

            products = []
            for row in results:
                product = Product(
                    product_id=row["ProductId"],
                    name=row["Name"] or "",
                    sku=row["Sku"] or "",
                    category=row["Category"] or "",
                    manufacturer_id=row["ManufacturerId"] or 0,
                    manufacturer_name=row["ManufacturerName"] or "Unknown",
                    current_price=float(row["CurrentPrice"] or 0),
                    is_enabled=bool(row["IsEnabled"]),
                    is_eol=bool(row["IsEol"]),
                    is_in_stock=bool(row["IsInStock"]),
                    search_text=row["SearchText"] or "",
                    short_description=row["ShortDescription"] or "",
                    description=row["Description"] or "",
                    popularity=float(row["Popularity"] or 0),
                    created=str(row["Created"] or ""),
                    updated=str(row["Updated"] or ""),
                )
                products.append(product)

            logger.info(f"Retrieved {len(products)} products with manufacturer information")
            return products

        except Error as e:
            logger.error(f"Error retrieving products: {e}")
            return []
        finally:
            cursor.close()

    def get_categories(self) -> List[str]:
        """
        Get all unique categories

        Returns:
            List of category names
        """
        self._ensure_connection()
        cursor = self.connection.cursor()

        try:
            query = """
                    SELECT DISTINCT Category, COUNT(*) as ProductCount
                    FROM Products
                    WHERE Category IS NOT NULL
                      AND Category != ''
                      AND IsEnabled = 1
                    GROUP BY Category
                    ORDER BY ProductCount DESC, Category
                    """

            cursor.execute(query)
            results = cursor.fetchall()

            categories = [row[0] for row in results]
            logger.info(f"Found {len(categories)} categories")
            return categories

        except Error as e:
            logger.error(f"Error retrieving categories: {e}")
            return []
        finally:
            cursor.close()

    def get_manufacturer_names(self) -> List[str]:
        """
        Get all manufacturer names

        Returns:
            List of manufacturer names
        """
        self._ensure_connection()
        cursor = self.connection.cursor()

        try:
            query = """
                SELECT DISTINCT m.Name
                FROM Manufacturers m
                INNER JOIN Products p ON m.Id = p.ManufacturerId
                WHERE m.Name IS NOT NULL 
                  AND m.Name != ''
                  AND p.IsEnabled = 1
                ORDER BY m.Name
            """

            cursor.execute(query)
            results = cursor.fetchall()

            manufacturers = [row[0] for row in results]
            logger.info(f"Found {len(manufacturers)} active manufacturers")
            return manufacturers

        except Error as e:
            logger.error(f"Error retrieving manufacturer names: {e}")
            return []
        finally:
            cursor.close()

    def get_database_stats(self) -> Dict:
        """
        Get database statistics with CORRECT manufacturer information

        Returns:
            Dictionary with database stats
        """
        self._ensure_connection()
        cursor = self.connection.cursor(dictionary=True)

        try:
            stats = {}

            # Total products
            cursor.execute("SELECT COUNT(*) as total FROM Products")
            stats["total_products"] = cursor.fetchone()["total"]

            # Enabled products
            cursor.execute("SELECT COUNT(*) as enabled FROM Products WHERE IsEnabled = 1")
            stats["enabled_products"] = cursor.fetchone()["enabled"]

            # In stock products
            cursor.execute("SELECT COUNT(*) as in_stock FROM Products WHERE IsInStock = 1 AND IsEnabled = 1")
            stats["in_stock_products"] = cursor.fetchone()["in_stock"]

            # Total manufacturers
            cursor.execute("SELECT COUNT(*) as total FROM Manufacturers")
            stats["total_manufacturers"] = cursor.fetchone()["total"]

            # Active manufacturers (with products)
            cursor.execute("""
                SELECT COUNT(DISTINCT m.Id) as active 
                FROM Manufacturers m 
                INNER JOIN Products p ON m.Id = p.ManufacturerId 
                WHERE p.IsEnabled = 1
            """)
            stats["active_manufacturers"] = cursor.fetchone()["active"]

            # Categories
            cursor.execute("""
                SELECT COUNT(DISTINCT Category) as categories 
                FROM Products 
                WHERE Category IS NOT NULL AND Category != '' AND IsEnabled = 1
            """)
            stats["categories"] = cursor.fetchone()["categories"]

            logger.info(f"Database stats: {stats}")
            return stats

        except Error as e:
            logger.error(f"Error getting database stats: {e}")
            return {}
        finally:
            cursor.close()

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

            logger.info(f"Looking for products - Main ID: {main_product_id}, Alternative ID: {alternative_id}")
            product_details["debug_info"]["main_product_id"] = main_product_id
            product_details["debug_info"]["alternative_id"] = alternative_id

            # Check for main product_id
            if main_product_id:
                logger.info(f"Searching for main product with ID: {main_product_id}")
                main_product = self.backend_db.get_product_details(str(main_product_id), is_alternative=False)
                if main_product and not main_product.get("error"):
                    product_details["main_product"] = main_product
                    product_details["products_found"].append("main_product")
                    logger.info(f"Found main product: {main_product.get('name', 'Unknown')}")
                else:
                    logger.warning(f"Main product not found for ID: {main_product_id}")
                    if main_product and main_product.get("error"):
                        product_details["main_product"] = main_product

            # Check for alternative_id
            if alternative_id and str(alternative_id).strip():
                logger.info(f"Searching for alternative product with ID: {alternative_id}")
                alternative_product = self.backend_db.get_product_details(str(alternative_id), is_alternative=True)
                if alternative_product and not alternative_product.get("error"):
                    product_details["alternative_product"] = alternative_product
                    product_details["products_found"].append("alternative_product")
                    logger.info(f"Found alternative product: {alternative_product.get('name', 'Unknown')}")
                else:
                    logger.warning(f"Alternative product not found for ID: {alternative_id}")
                    if alternative_product and alternative_product.get("error"):
                        product_details["alternative_product"] = alternative_product

            logger.info(f"Product lookup complete. Found: {product_details['products_found']}")
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

def get_product_intelligence_from_database() -> Dict:
    """
    Convenience function to get product intelligence from database
    Replaces product_intelligence.json file dependency

    Returns:
        Dict with product intelligence data from live database
    """
    try:
        from ..product_identification.product_intelligence_builder import ProductIntelligenceBuilder

        builder = ProductIntelligenceBuilder()
        try:
            return builder.get_intelligence_from_database()
        finally:
            builder.close()

    except Exception as e:
        logger.error(f"Error getting product intelligence from database: {e}")
        return {}


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


if __name__ == "__main__":
    # Test the clean database connector
    import logging
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    print("🔍 Testing Clean Database Connector:")
    print(f"  DB Host: {os.getenv('BACKEND_DB_HOST', 'NOT SET')}")
    print(f"  DB User: {os.getenv('BACKEND_DB_USER', 'NOT SET')}")
    print(f"  DB Password: {'SET' if os.getenv('BACKEND_DB_PASSWORD') else 'NOT SET'}")
    print()

    try:
        db = BackendDatabase()

        # Test database stats
        print("📊 Database Statistics:")
        stats = db.get_database_stats()
        for key, value in stats.items():
            print(f"  {key}: {value:,}")

        # Test manufacturers
        print("\n🏭 Top 5 Manufacturers:")
        manufacturers = db.get_manufacturer_names()[:5]
        for i, manufacturer in enumerate(manufacturers, 1):
            print(f"  {i}. {manufacturer}")

        # Test categories
        print("\n📂 Top 5 Categories:")
        categories = db.get_categories()[:5]
        for i, category in enumerate(categories, 1):
            print(f"  {i}. {category}")

        db.close()

        print("\n✅ Clean Database Connector Test Completed!")

    except Exception as e:
        print(f"❌ Error testing database: {e}")
        import traceback
        traceback.print_exc()