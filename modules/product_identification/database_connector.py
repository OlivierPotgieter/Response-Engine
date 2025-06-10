"""
Fixed Database Connector for Product Identifier
Now correctly uses Manufacturers table instead of Brands table
"""

import os
import logging
from typing import Dict, List, Optional, Tuple
import mysql.connector
from mysql.connector import Error
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class Product:
    """Product data class with correct manufacturer information"""
    product_id: int
    name: str
    sku: str
    category: str
    manufacturer_id: int
    manufacturer_name: str  # Changed from brand_name to manufacturer_name
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
    # Add other fields from Manufacturers table as needed
    # You might want to inspect the Manufacturers table structure and add relevant fields


class DatabaseConnector:
    """Fixed database connector using Manufacturers table"""

    def __init__(self):
        """Initialize database connection"""
        self.connection = None
        self._connect()

    def _connect(self):
        """Establish connection to backend_portal database"""
        try:
            host = os.getenv("BACKEND_DB_HOST", "localhost")
            user = os.getenv("BACKEND_DB_USER")
            password = os.getenv("BACKEND_DB_PASSWORD")

            logger.info(f"Attempting to connect to: {host} as user: {user}")

            if not user:
                raise ValueError("BACKEND_DB_USER environment variable not set")
            if not password:
                logger.warning("BACKEND_DB_PASSWORD environment variable not set (using empty password)")

            self.connection = mysql.connector.connect(
                host=host,
                database="backend_portal",
                user=user,
                password=password or "",
                charset='utf8mb4',
                autocommit=True,
                connect_timeout=10
            )
            logger.info("Connected to backend_portal database")
        except Error as e:
            logger.error(f"Database connection error: {e}")
            logger.error(f"Connection parameters: host={host}, user={user}, database=backend_portal")
            raise

    def _ensure_connection(self):
        """Ensure database connection is active"""
        try:
            if not self.connection or not self.connection.is_connected():
                logger.info("Reconnecting to database...")
                self._connect()
        except Error:
            self._connect()

    def get_all_manufacturers(self) -> List[Manufacturer]:
        """
        Get all manufacturers from the database

        Returns:
            List of Manufacturer objects
        """
        self._ensure_connection()
        cursor = self.connection.cursor(dictionary=True)

        try:
            # First, let's see what columns exist in the Manufacturers table
            cursor.execute("DESCRIBE Manufacturers")
            columns = cursor.fetchall()

            logger.info("Manufacturers table structure:")
            for col in columns:
                logger.info(f"  {col['Field']}: {col['Type']}")

            # Build query based on actual table structure
            # Assuming at minimum we have Id and Name columns
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
                # Create Manufacturer object - adapt based on actual table structure
                manufacturer = Manufacturer(
                    id=row.get('Id') or row.get('id'),  # Handle different case
                    name=row.get('Name') or row.get('name') or '',
                    # Add other fields as they exist in your table
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
            # FIXED: Now joining with Manufacturers table instead of Brands
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
                    product_id=row['ProductId'],
                    name=row['Name'] or '',
                    sku=row['Sku'] or '',
                    category=row['Category'] or '',
                    manufacturer_id=row['ManufacturerId'] or 0,
                    manufacturer_name=row['ManufacturerName'] or 'Unknown',  # Now from Manufacturers table
                    current_price=float(row['CurrentPrice'] or 0),
                    is_enabled=bool(row['IsEnabled']),
                    is_eol=bool(row['IsEol']),
                    is_in_stock=bool(row['IsInStock']),
                    search_text=row['SearchText'] or '',
                    short_description=row['ShortDescription'] or '',
                    description=row['Description'] or '',
                    popularity=float(row['Popularity'] or 0),
                    created=str(row['Created'] or ''),
                    updated=str(row['Updated'] or '')
                )
                products.append(product)

            logger.info(f"Retrieved {len(products)} products with manufacturer information")
            return products

        except Error as e:
            logger.error(f"Error retrieving products: {e}")
            return []
        finally:
            cursor.close()

    def get_products_by_category(self, category: str, enabled_only: bool = True) -> List[Product]:
        """
        Get products by category with CORRECT manufacturer information

        Args:
            category: Category name to filter by
            enabled_only: Only return enabled products

        Returns:
            List of Product objects in the category
        """
        self._ensure_connection()
        cursor = self.connection.cursor(dictionary=True)

        try:
            where_clause = "WHERE p.Category = %s AND p.Name IS NOT NULL AND p.Name != ''"
            params = [category]

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
                    p.Popularity,
                    COALESCE(m.Name, 'Unknown') as ManufacturerName
                FROM Products p
                LEFT JOIN Manufacturers m ON p.ManufacturerId = m.Id
                {where_clause}
                ORDER BY p.Popularity DESC, p.Name
            """

            cursor.execute(query, params)
            results = cursor.fetchall()

            products = []
            for row in results:
                product = Product(
                    product_id=row['ProductId'],
                    name=row['Name'] or '',
                    sku=row['Sku'] or '',
                    category=row['Category'] or '',
                    manufacturer_id=row['ManufacturerId'] or 0,
                    manufacturer_name=row['ManufacturerName'] or 'Unknown',
                    current_price=float(row['CurrentPrice'] or 0),
                    is_enabled=bool(row['IsEnabled']),
                    is_eol=bool(row['IsEol']),
                    is_in_stock=bool(row['IsInStock']),
                    search_text=row['SearchText'] or '',
                    short_description=row['ShortDescription'] or '',
                    description='',  # Not needed for category view
                    popularity=float(row['Popularity'] or 0),
                    created='',
                    updated=''
                )
                products.append(product)

            logger.info(f"Retrieved {len(products)} products in category '{category}'")
            return products

        except Error as e:
            logger.error(f"Error retrieving products by category: {e}")
            return []
        finally:
            cursor.close()

    def get_products_by_manufacturer(self, manufacturer_name: str, enabled_only: bool = True) -> List[Product]:
        """
        Get products by manufacturer name (FIXED method name and logic)

        Args:
            manufacturer_name: Manufacturer name to filter by
            enabled_only: Only return enabled products

        Returns:
            List of Product objects from the manufacturer
        """
        self._ensure_connection()
        cursor = self.connection.cursor(dictionary=True)

        try:
            where_clause = "WHERE m.Name = %s AND p.Name IS NOT NULL AND p.Name != ''"
            params = [manufacturer_name]

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
                    p.Popularity,
                    m.Name as ManufacturerName
                FROM Products p
                INNER JOIN Manufacturers m ON p.ManufacturerId = m.Id
                {where_clause}
                ORDER BY p.Popularity DESC, p.Name
            """

            cursor.execute(query, params)
            results = cursor.fetchall()

            products = []
            for row in results:
                product = Product(
                    product_id=row['ProductId'],
                    name=row['Name'] or '',
                    sku=row['Sku'] or '',
                    category=row['Category'] or '',
                    manufacturer_id=row['ManufacturerId'] or 0,
                    manufacturer_name=row['ManufacturerName'] or '',
                    current_price=float(row['CurrentPrice'] or 0),
                    is_enabled=bool(row['IsEnabled']),
                    is_eol=bool(row['IsEol']),
                    is_in_stock=bool(row['IsInStock']),
                    search_text=row['SearchText'] or '',
                    short_description=row['ShortDescription'] or '',
                    description='',
                    popularity=float(row['Popularity'] or 0),
                    created='',
                    updated=''
                )
                products.append(product)

            logger.info(f"Retrieved {len(products)} products from manufacturer '{manufacturer_name}'")
            return products

        except Error as e:
            logger.error(f"Error retrieving products by manufacturer: {e}")
            return []
        finally:
            cursor.close()

    def get_manufacturer_names(self) -> List[str]:
        """
        Get all manufacturer names (FIXED: was get_brand_names)

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
            stats['total_products'] = cursor.fetchone()['total']

            # Enabled products
            cursor.execute("SELECT COUNT(*) as enabled FROM Products WHERE IsEnabled = 1")
            stats['enabled_products'] = cursor.fetchone()['enabled']

            # In stock products
            cursor.execute("SELECT COUNT(*) as in_stock FROM Products WHERE IsInStock = 1 AND IsEnabled = 1")
            stats['in_stock_products'] = cursor.fetchone()['in_stock']

            # Total manufacturers (FIXED)
            cursor.execute("SELECT COUNT(*) as total FROM Manufacturers")
            stats['total_manufacturers'] = cursor.fetchone()['total']

            # Active manufacturers (with products)
            cursor.execute("""
                SELECT COUNT(DISTINCT m.Id) as active 
                FROM Manufacturers m 
                INNER JOIN Products p ON m.Id = p.ManufacturerId 
                WHERE p.IsEnabled = 1
            """)
            stats['active_manufacturers'] = cursor.fetchone()['active']

            # Categories
            cursor.execute("""
                SELECT COUNT(DISTINCT Category) as categories 
                FROM Products 
                WHERE Category IS NOT NULL AND Category != '' AND IsEnabled = 1
            """)
            stats['categories'] = cursor.fetchone()['categories']

            logger.info(f"Database stats: {stats}")
            return stats

        except Error as e:
            logger.error(f"Error getting database stats: {e}")
            return {}
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
                    ORDER BY ProductCount DESC, Category \
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

    def search_products(self, search_term: str, limit: int = 50) -> List[Product]:
        """
        Search products by name, SKU, or search text with CORRECT manufacturer information

        Args:
            search_term: Term to search for
            limit: Maximum number of results

        Returns:
            List of matching Product objects
        """
        self._ensure_connection()
        cursor = self.connection.cursor(dictionary=True)

        try:
            search_pattern = f"%{search_term}%"

            query = """
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
                    p.Popularity,
                    COALESCE(m.Name, 'Unknown') as ManufacturerName
                FROM Products p
                LEFT JOIN Manufacturers m ON p.ManufacturerId = m.Id
                WHERE p.IsEnabled = 1
                  AND (
                    p.Name LIKE %s OR 
                    p.Sku LIKE %s OR 
                    p.SearchText LIKE %s OR
                    m.Name LIKE %s
                  )
                ORDER BY p.Popularity DESC, p.Name
                LIMIT %s
            """

            cursor.execute(query, [search_pattern, search_pattern, search_pattern, search_pattern, limit])
            results = cursor.fetchall()

            products = []
            for row in results:
                product = Product(
                    product_id=row['ProductId'],
                    name=row['Name'] or '',
                    sku=row['Sku'] or '',
                    category=row['Category'] or '',
                    manufacturer_id=row['ManufacturerId'] or 0,
                    manufacturer_name=row['ManufacturerName'] or 'Unknown',
                    current_price=float(row['CurrentPrice'] or 0),
                    is_enabled=bool(row['IsEnabled']),
                    is_eol=bool(row['IsEol']),
                    is_in_stock=bool(row['IsInStock']),
                    search_text=row['SearchText'] or '',
                    short_description=row['ShortDescription'] or '',
                    description='',
                    popularity=float(row['Popularity'] or 0),
                    created='',
                    updated=''
                )
                products.append(product)

            logger.info(f"Search for '{search_term}' returned {len(products)} results")
            return products

        except Error as e:
            logger.error(f"Error searching products: {e}")
            return []
        finally:
            cursor.close()

    def close(self):
        """Close database connection"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("Database connection closed")


# Test the fixed database connector
if __name__ == "__main__":
    import logging
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    print("🔍 Testing Fixed Database Connector:")
    print(f"  DB Host: {os.getenv('BACKEND_DB_HOST', 'NOT SET')}")
    print(f"  DB User: {os.getenv('BACKEND_DB_USER', 'NOT SET')}")
    print(f"  DB Password: {'SET' if os.getenv('BACKEND_DB_PASSWORD') else 'NOT SET'}")
    print()

    try:
        db = DatabaseConnector()

        # Test database stats with corrected manufacturer info
        print("📊 Database Statistics (Fixed):")
        stats = db.get_database_stats()
        for key, value in stats.items():
            print(f"  {key}: {value:,}")

        # Test manufacturers (instead of brands)
        print(f"\n🏭 Top 10 Manufacturers:")
        manufacturers = db.get_manufacturer_names()[:10]
        for i, manufacturer in enumerate(manufacturers, 1):
            print(f"  {i}. {manufacturer}")

        # Test categories
        print(f"\n📂 Top 10 Categories:")
        categories = db.get_categories()[:10]
        for i, category in enumerate(categories, 1):
            print(f"  {i}. {category}")

        # Test search with manufacturer info
        print(f"\n🔍 Search Test (ASUS):")
        search_results = db.search_products("ASUS", limit=5)
        for product in search_results:
            print(f"  - {product.name} ({product.manufacturer_name}) - ${product.current_price}")

        db.close()

        print("\n✅ Fixed Database Connector Test Completed!")
        print("🚀 Ready to rebuild product intelligence with correct manufacturer data!")

    except Exception as e:
        print(f"❌ Error testing database: {e}")
        import traceback
        traceback.print_exc()