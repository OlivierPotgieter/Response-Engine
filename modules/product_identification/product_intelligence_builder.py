"""
Updated Product Intelligence Builder
Fixed to use Manufacturers table instead of Brands table
"""

import logging
import json
import re
from typing import Dict, List
from datetime import datetime
from collections import Counter, defaultdict
from dotenv import load_dotenv
from database_connector import DatabaseConnector, Product, Manufacturer

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class ProductIntelligenceBuilder:
    """Build comprehensive product intelligence from database using Manufacturers"""

    def __init__(self):
        """Initialize the intelligence builder"""
        self.db = DatabaseConnector()
        self.intelligence = {}

    def build_complete_intelligence(self) -> Dict:
        """
        Build complete product intelligence with FIXED manufacturer data

        Returns:
            Comprehensive intelligence dictionary
        """
        logger.info(
            "🧠 Building complete product intelligence with manufacturer data..."
        )

        start_time = datetime.now()

        # Get all data from database with CORRECT manufacturer information
        all_products_raw = self.db.get_all_products(enabled_only=True)
        all_manufacturers = (
            self.db.get_all_manufacturers()
        )  # Changed from get_all_brands
        categories = self.db.get_categories()
        self.db.get_manufacturer_names()  # Changed from get_brand_names

        # Filter out problematic products
        all_products = self._filter_quality_products(all_products_raw)

        logger.info(
            f"📊 Retrieved {len(all_products_raw)} total products, filtered to {len(all_products)} quality products"
        )
        logger.info(
            f"📊 {len(all_manufacturers)} manufacturers, {len(categories)} categories"
        )

        # Build intelligence components with FIXED manufacturer references
        intelligence = {
            "metadata": {
                "created": start_time.isoformat(),
                "total_products_raw": len(all_products_raw),
                "total_products_filtered": len(all_products),
                "enabled_products": len(all_products),
                "total_manufacturers": len(
                    all_manufacturers
                ),  # Changed from total_brands
                "total_categories": len(categories),
            },
            "manufacturers": self._analyze_manufacturers(
                all_manufacturers, all_products
            ),  # Changed from brands
            "categories": self._analyze_categories(categories, all_products),
            "products": self._build_product_index(all_products),
            "search_patterns": self._extract_search_patterns(all_products),
            "statistics": self._calculate_statistics(all_products),
        }

        self.intelligence = intelligence

        build_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"✅ Intelligence built in {build_time:.2f} seconds")

        return intelligence

    def _filter_quality_products(self, products: List[Product]) -> List[Product]:
        """Filter out low-quality products that might hurt our AI model"""
        filtered_products = []

        excluded_manufacturers = {
            "unknown",
            "n/a",
            "not specified",
            "",
        }  # Changed from brands
        excluded_categories = {"default", "unknown", "uncategorized", "n/a", ""}

        for product in products:
            # Skip products with unknown/missing manufacturers
            if (
                product.manufacturer_name.lower().strip() in excluded_manufacturers
            ):  # Changed from brand_name
                continue

            # Skip products with default/unknown categories
            if product.category.lower().strip() in excluded_categories:
                continue

            # Skip products with very short/empty names
            if len(product.name.strip()) < 5:
                continue

            filtered_products.append(product)

        logger.info(f"🔍 Filtered {len(products)} -> {len(filtered_products)} products")
        logger.info(
            f"   Removed {len(products) - len(filtered_products)} low-quality products"
        )

        return filtered_products

    def _analyze_manufacturers(
        self, manufacturers: List[Manufacturer], products: List[Product]
    ) -> Dict:
        """Analyze manufacturer information and create searchable variations"""
        logger.info("🏭 Analyzing manufacturers...")

        manufacturer_analysis = {}

        # Group products by manufacturer
        products_by_manufacturer = defaultdict(list)
        for product in products:
            products_by_manufacturer[product.manufacturer_name].append(product)

        for manufacturer in manufacturers:
            manufacturer_products = products_by_manufacturer.get(manufacturer.name, [])

            # Create variations for better matching
            variations = self._create_manufacturer_variations(manufacturer.name)

            # Get categories this manufacturer appears in
            manufacturer_categories = list(
                set(p.category for p in manufacturer_products if p.category)
            )

            # Calculate average price for this manufacturer
            prices = [
                p.current_price for p in manufacturer_products if p.current_price > 0
            ]
            avg_price = sum(prices) / len(prices) if prices else 0

            manufacturer_analysis[manufacturer.name] = {
                "id": manufacturer.id,
                "name": manufacturer.name,
                "variations": variations,
                "product_count": len(manufacturer_products),
                "categories": manufacturer_categories,
                "average_price": round(avg_price, 2),
                "price_range": {
                    "min": min(prices) if prices else 0,
                    "max": max(prices) if prices else 0,
                },
                "top_products": [
                    {"id": p.product_id, "name": p.name, "popularity": p.popularity}
                    for p in sorted(
                        manufacturer_products, key=lambda x: x.popularity, reverse=True
                    )[:5]
                ],
            }

        logger.info(f"✅ Analyzed {len(manufacturer_analysis)} manufacturers")
        return manufacturer_analysis

    def _create_manufacturer_variations(self, manufacturer_name: str) -> List[str]:
        """Create search variations for a manufacturer name"""
        if not manufacturer_name:
            return []

        variations = [manufacturer_name.lower()]

        # Add without common suffixes
        name_lower = manufacturer_name.lower()
        suffixes = [
            " inc",
            " corp",
            " ltd",
            " limited",
            " technology",
            " tech",
            " co",
            " corporation",
        ]
        for suffix in suffixes:
            if name_lower.endswith(suffix):
                clean_name = name_lower.replace(suffix, "").strip()
                if clean_name:
                    variations.append(clean_name)

        # Add acronyms for multi-word manufacturers
        words = manufacturer_name.split()
        if len(words) > 1:
            acronym = "".join(
                word[0].lower() for word in words if word and len(word) > 0
            )
            if len(acronym) >= 2:
                variations.append(acronym)

        # Add common tech manufacturer abbreviations
        tech_abbreviations = {
            "nvidia corporation": ["nvidia", "nv"],
            "advanced micro devices": ["amd"],
            "asustek computer": ["asus"],
            "micro-star international": ["msi"],
            "gigabyte technology": ["gigabyte", "gb"],
            "corsair memory": ["corsair"],
            "logitech international": ["logitech", "logi"],
            "cooler master technology": ["cooler master", "cm"],
            "western digital corporation": ["western digital", "wd"],
            "seagate technology": ["seagate"],
            "samsung electronics": ["samsung"],
            "kingston technology": ["kingston"],
            "intel corporation": ["intel"],
            "hewlett-packard": ["hp"],
            "dell technologies": ["dell"],
            "lenovo group": ["lenovo"],
            "apple inc": ["apple"],
            "microsoft corporation": ["microsoft", "ms"],
        }

        for full_name, abbrevs in tech_abbreviations.items():
            if full_name in name_lower or any(
                part in name_lower for part in full_name.split()
            ):
                variations.extend(abbrevs)

        return list(set(variations))

    def _analyze_categories(
        self, categories: List[str], products: List[Product]
    ) -> Dict:
        """Analyze categories and extract patterns with manufacturer data"""
        logger.info("📂 Analyzing categories...")

        category_analysis = {}

        for category in categories:
            # Get products in this category
            category_products = [p for p in products if p.category == category]

            if not category_products:
                continue

            # Analyze manufacturers in this category (FIXED from brands)
            manufacturer_distribution = Counter(
                p.manufacturer_name for p in category_products
            )

            # Analyze price ranges
            prices = [p.current_price for p in category_products if p.current_price > 0]

            # Extract common terms from product names
            common_terms = self._extract_common_terms(
                [p.name for p in category_products]
            )

            # Extract model patterns
            model_patterns = self._extract_model_patterns(
                [p.name for p in category_products]
            )

            category_analysis[category] = {
                "name": category,
                "product_count": len(category_products),
                "manufacturer_distribution": dict(
                    manufacturer_distribution.most_common(10)
                ),  # Changed from brand_distribution
                "price_statistics": {
                    "min": min(prices) if prices else 0,
                    "max": max(prices) if prices else 0,
                    "average": sum(prices) / len(prices) if prices else 0,
                    "median": sorted(prices)[len(prices) // 2] if prices else 0,
                },
                "common_terms": dict(common_terms.most_common(20)),
                "model_patterns": model_patterns,
                "top_products": [
                    {
                        "id": p.product_id,
                        "name": p.name,
                        "manufacturer": p.manufacturer_name,  # Changed from brand
                        "popularity": p.popularity,
                    }
                    for p in sorted(
                        category_products, key=lambda x: x.popularity, reverse=True
                    )[:10]
                ],
            }

        logger.info(f"✅ Analyzed {len(category_analysis)} categories")
        return category_analysis

    def _build_product_index(self, products: List[Product]) -> List[Dict]:
        """Build searchable product index with manufacturer data"""
        logger.info("📖 Building product search index...")

        search_index = []

        for product in products:
            # Create comprehensive searchable text
            searchable_terms = [
                product.name,
                product.sku,
                product.manufacturer_name,  # Changed from brand_name
                product.category,
                product.search_text,
                product.short_description,
            ]

            # Filter and clean terms
            clean_terms = []
            for term in searchable_terms:
                if term and term.strip():
                    clean_terms.append(term.strip())

            searchable_text = " ".join(clean_terms).lower()

            # Determine price range category
            price_range = "unknown"
            if product.current_price > 0:
                if product.current_price < 100:
                    price_range = "budget"
                elif product.current_price < 500:
                    price_range = "mid"
                elif product.current_price < 1500:
                    price_range = "high"
                else:
                    price_range = "premium"

            index_entry = {
                "product_id": product.product_id,
                "name": product.name,
                "sku": product.sku,
                "manufacturer": product.manufacturer_name,  # Changed from brand
                "manufacturer_id": product.manufacturer_id,  # Changed from brand_id
                "category": product.category,
                "price": product.current_price,
                "price_range": price_range,
                "popularity": product.popularity,
                "is_enabled": product.is_enabled,
                "is_eol": product.is_eol,
                "is_in_stock": product.is_in_stock,
                "searchable_text": searchable_text,
            }

            search_index.append(index_entry)

        logger.info(f"✅ Built search index with {len(search_index)} products")
        return search_index

    def _extract_search_patterns(self, products: List[Product]) -> Dict:
        """Extract search patterns across all products"""
        logger.info("🔍 Extracting search patterns...")

        all_names = [p.name for p in products if p.name]
        all_search_text = [p.search_text for p in products if p.search_text]

        patterns = {
            "gpu_patterns": self._extract_gpu_patterns(all_names),
            "cpu_patterns": self._extract_cpu_patterns(all_names),
            "memory_patterns": self._extract_memory_patterns(
                all_names + all_search_text
            ),
            "storage_patterns": self._extract_storage_patterns(
                all_names + all_search_text
            ),
            "common_keywords": self._extract_tech_keywords(all_names + all_search_text),
        }

        return patterns

    def _extract_common_terms(self, product_names: List[str]) -> Counter:
        """Extract common terms from product names"""
        terms = Counter()

        for name in product_names:
            if not name:
                continue

            # Clean and split the name
            cleaned = re.sub(r"[^\w\s]", " ", name.lower())
            words = cleaned.split()

            for word in words:
                # Filter out very short words and numbers
                if len(word) >= 3 and not word.isdigit():
                    terms[word] += 1

        return terms

    def _extract_model_patterns(self, product_names: List[str]) -> List[str]:
        """Extract model number patterns from product names"""
        patterns = set()

        # Common tech product patterns
        model_regexes = [
            r"\b(rtx|gtx)\s*(\d{4})\s*(ti|super)?\b",  # RTX 4090, GTX 1080 Ti
            r"\b(radeon)\s*(rx|r)\s*(\d{4})\s*(xt|x)?\b",  # Radeon RX 7900 XT
            r"\b(ryzen)\s*(\d+)\s*(\d{4}[a-z]*)\b",  # Ryzen 7 5800X
            r"\b(core)\s*(i[3579])\s*(\d{4,5}[a-z]*)\b",  # Core i7-12700K
            r"\b(\d{1,2}gb)\b",  # Memory sizes
            r"\b(\d+)\s*(tb|gb)\b",  # Storage sizes
            r"\b(ddr[45])\s*(\d{4})\b",  # Memory types
            r"\b(geforce)\s*(rtx|gtx)\s*(\d{4})\b",  # GeForce RTX 4090
        ]

        for name in product_names:
            if not name:
                continue

            name_lower = name.lower()
            for regex in model_regexes:
                matches = re.finditer(regex, name_lower)
                for match in matches:
                    pattern = match.group(0).strip()
                    if pattern:
                        patterns.add(pattern)

        return sorted(list(patterns))

    def _extract_gpu_patterns(self, product_names: List[str]) -> List[str]:
        """Extract GPU-specific patterns"""
        patterns = set()

        gpu_regexes = [
            r"\b(rtx|gtx)\s*(\d{4})\s*(ti|super)?\b",
            r"\b(radeon)\s*(rx|r)\s*(\d{4})\s*(xt|x)?\b",
            r"\b(geforce)\s*(rtx|gtx)\s*(\d{4})\b",
            r"\b(nvidia|amd)\s*(rtx|gtx|radeon)\b",
        ]

        for name in product_names:
            if not name:
                continue
            name_lower = name.lower()
            for regex in gpu_regexes:
                matches = re.finditer(regex, name_lower)
                for match in matches:
                    patterns.add(match.group(0).strip())

        return sorted(list(patterns))

    def _extract_cpu_patterns(self, product_names: List[str]) -> List[str]:
        """Extract CPU-specific patterns"""
        patterns = set()

        cpu_regexes = [
            r"\b(ryzen)\s*(\d+)\s*(\d{4}[a-z]*)\b",
            r"\b(core)\s*(i[3579])\s*(\d{4,5}[a-z]*)\b",
            r"\b(intel|amd)\s*(core|ryzen)\b",
            r"\b(\d+)th\s*gen\b",
        ]

        for name in product_names:
            if not name:
                continue
            name_lower = name.lower()
            for regex in cpu_regexes:
                matches = re.finditer(regex, name_lower)
                for match in matches:
                    patterns.add(match.group(0).strip())

        return sorted(list(patterns))

    def _extract_memory_patterns(self, texts: List[str]) -> List[str]:
        """Extract memory-related patterns"""
        patterns = set()

        memory_regexes = [
            r"\b(\d{1,2}gb)\s*(ddr[45])?\b",
            r"\b(ddr[45])\s*(\d{4})\b",
            r"\b(\d+)\s*x\s*(\d+gb)\b",
        ]

        for text in texts:
            if not text:
                continue
            text_lower = text.lower()
            for regex in memory_regexes:
                matches = re.finditer(regex, text_lower)
                for match in matches:
                    patterns.add(match.group(0).strip())

        return sorted(list(patterns))

    def _extract_storage_patterns(self, texts: List[str]) -> List[str]:
        """Extract storage-related patterns"""
        patterns = set()

        storage_regexes = [
            r"\b(\d+)\s*(tb|gb)\s*(ssd|hdd|nvme)?\b",
            r"\b(ssd|hdd|nvme)\s*(\d+)\s*(tb|gb)\b",
            r"\b(m\.?2|sata|pcie)\s*(ssd|nvme)?\b",
        ]

        for text in texts:
            if not text:
                continue
            text_lower = text.lower()
            for regex in storage_regexes:
                matches = re.finditer(regex, text_lower)
                for match in matches:
                    patterns.add(match.group(0).strip())

        return sorted(list(patterns))

    def _extract_tech_keywords(self, texts: List[str]) -> List[str]:
        """Extract common tech keywords"""
        keywords = Counter()

        tech_terms = [
            "gaming",
            "rgb",
            "wireless",
            "mechanical",
            "optical",
            "laser",
            "ultrawide",
            "4k",
            "1440p",
            "144hz",
            "240hz",
            "curved",
            "bluetooth",
            "usb",
            "hdmi",
            "displayport",
            "thunderbolt",
            "wifi",
            "ethernet",
            "gigabit",
            "pcie",
            "atx",
            "mini-itx",
            "modular",
            "gold",
            "bronze",
            "platinum",
            "efficiency",
        ]

        for text in texts:
            if not text:
                continue
            text_lower = text.lower()
            for term in tech_terms:
                if term in text_lower:
                    keywords[term] += 1

        return [term for term, count in keywords.most_common(50)]

    def _calculate_statistics(self, products: List[Product]) -> Dict:
        """Calculate various statistics about the product catalog"""

        # Price statistics
        prices = [p.current_price for p in products if p.current_price > 0]

        # Category distribution
        category_dist = Counter(p.category for p in products if p.category)

        # Manufacturer distribution (FIXED from brand)
        manufacturer_dist = Counter(
            p.manufacturer_name for p in products if p.manufacturer_name
        )

        # Stock statistics
        in_stock_count = sum(1 for p in products if p.is_in_stock)
        eol_count = sum(1 for p in products if p.is_eol)

        stats = {
            "total_products": len(products),
            "in_stock_products": in_stock_count,
            "eol_products": eol_count,
            "price_statistics": {
                "min_price": min(prices) if prices else 0,
                "max_price": max(prices) if prices else 0,
                "average_price": sum(prices) / len(prices) if prices else 0,
                "median_price": sorted(prices)[len(prices) // 2] if prices else 0,
            },
            "category_distribution": dict(category_dist.most_common(20)),
            "manufacturer_distribution": dict(
                manufacturer_dist.most_common(20)
            ),  # Changed from brand_distribution
            "stock_ratio": in_stock_count / len(products) if products else 0,
            "eol_ratio": eol_count / len(products) if products else 0,
        }

        return stats
    #Removed the following two functions :
    #save_intelligence
    #load_intelligence

    #Will add the following functions :
    #get_intelligence_from_database
    #cache_intelligence_in_memory
    #invalidate_cache
    def get_intelligence_from_database(self) -> Dict:
        """
        Get product intelligence directly from database (replaces file loading)

        Returns:
            Intelligence dictionary built from current database state
        """
        try:
            logger.info("🔄 Building product intelligence from live database data...")

            # Always build fresh from database - no file dependency
            intelligence = self.build_complete_intelligence()

            if intelligence and "error" not in intelligence:
                logger.info("✅ Product intelligence built successfully from database")
                return intelligence
            else:
                logger.error("❌ Failed to build product intelligence from database")
                return {}

        except Exception as e:
            logger.error(f"❌ Error building intelligence from database: {e}")
            return {}

    def cache_intelligence_in_memory(self) -> Dict:
        """
        Build and cache intelligence in memory with timestamp for freshness checking

        Returns:
            Cached intelligence data with metadata
        """
        try:
            # Check if we have recent cached data
            if (hasattr(self, '_cached_intelligence') and
                    hasattr(self, '_cache_timestamp') and
                    (datetime.now() - self._cache_timestamp).total_seconds() < 3600):  # 1 hour cache

                logger.info("📋 Using cached product intelligence (less than 1 hour old)")
                return self._cached_intelligence

            # Build fresh intelligence from database
            logger.info("🏗️ Building fresh product intelligence from database...")
            intelligence = self.get_intelligence_from_database()

            # Cache the result
            self._cached_intelligence = intelligence
            self._cache_timestamp = datetime.now()

            logger.info(f"💾 Cached product intelligence with {len(intelligence.get('products', []))} products")
            return intelligence

        except Exception as e:
            logger.error(f"❌ Error caching intelligence: {e}")
            return {}

    def invalidate_cache(self):
        """Invalidate the cached intelligence to force rebuild"""
        if hasattr(self, '_cached_intelligence'):
            delattr(self, '_cached_intelligence')
        if hasattr(self, '_cache_timestamp'):
            delattr(self, '_cache_timestamp')
        logger.info("🗑️ Product intelligence cache invalidated")

    def get_summary(self) -> Dict:
        """Get intelligence summary"""
        if not self.intelligence:
            return {"error": "No intelligence data available"}

        metadata = self.intelligence.get("metadata", {})
        stats = self.intelligence.get("statistics", {})

        return {
            "created": metadata.get("created"),
            "total_products_raw": metadata.get("total_products_raw", 0),
            "total_products_filtered": metadata.get("total_products_filtered", 0),
            "total_manufacturers": metadata.get(
                "total_manufacturers", 0
            ),  # Changed from total_brands
            "total_categories": metadata.get("total_categories", 0),
            "filter_ratio": round(
                (
                    metadata.get("total_products_filtered", 0)
                    / metadata.get("total_products_raw", 1)
                )
                * 100,
                1,
            ),
            "price_range": {
                "min": stats.get("price_statistics", {}).get("min_price", 0),
                "max": stats.get("price_statistics", {}).get("max_price", 0),
                "average": round(
                    stats.get("price_statistics", {}).get("average_price", 0), 2
                ),
            },
            "top_categories": list(stats.get("category_distribution", {}).keys())[:5],
            "top_manufacturers": list(
                stats.get("manufacturer_distribution", {}).keys()
            )[
                :5
            ],  # Changed from top_brands
            "stock_ratio": round(stats.get("stock_ratio", 0) * 100, 1),
        }

    def close(self):
        """Close database connection"""
        self.db.close()


# Convenience functions

#Removed old build_product intelligence, but I just updated the def
def build_product_intelligence(cache_in_memory: bool = True) -> Dict:
    """
    Build product intelligence from database (replaces file-based approach)

    Args:
        cache_in_memory: Whether to use memory caching for performance

    Returns:
        Intelligence dictionary built from current database state
    """
    builder = ProductIntelligenceBuilder()

    try:
        if cache_in_memory:
            intelligence = builder.cache_intelligence_in_memory()
        else:
            intelligence = builder.get_intelligence_from_database()

        return intelligence

    finally:
        builder.close()


#Removed old Convenience FUnction get_intelligence_summary

#Added new get_intelligence_summary and invalidate_product_intelligence_cache
def get_intelligence_summary() -> Dict:
    """Get intelligence summary from live database (replaces file loading)"""
    builder = ProductIntelligenceBuilder()

    try:
        # Always get fresh data for summary
        intelligence = builder.get_intelligence_from_database()

        if intelligence and "error" not in intelligence:
            # Build summary from live data
            metadata = intelligence.get('metadata', {})
            stats = intelligence.get('statistics', {})

            return {
                'created': metadata.get('created'),
                'total_products_raw': metadata.get('total_products_raw', 0),
                'total_products_filtered': metadata.get('total_products_filtered', 0),
                'total_manufacturers': metadata.get('total_manufacturers', 0),
                'total_categories': metadata.get('total_categories', 0),
                'filter_ratio': round(
                    (metadata.get('total_products_filtered', 0) / metadata.get('total_products_raw', 1)) * 100, 1),
                'price_range': {
                    'min': stats.get('price_statistics', {}).get('min_price', 0),
                    'max': stats.get('price_statistics', {}).get('max_price', 0),
                    'average': round(stats.get('price_statistics', {}).get('average_price', 0), 2)
                },
                'top_categories': list(stats.get('category_distribution', {}).keys())[:5],
                'top_manufacturers': list(stats.get('manufacturer_distribution', {}).keys())[:5],
                'stock_ratio': round(stats.get('stock_ratio', 0) * 100, 1),
                'data_source': 'live_database',
                'cache_available': hasattr(builder, '_cached_intelligence')
            }
        else:
            return {
                'error': 'Failed to build intelligence from database',
                'data_source': 'database_error'
            }
    finally:
        builder.close()

def invalidate_product_intelligence_cache() -> Dict:
    """
    Invalidate product intelligence cache to force fresh rebuild

    Returns:
        Status of cache invalidation
    """
    builder = ProductIntelligenceBuilder()

    try:
        builder.invalidate_cache()
        return {
            'success': True,
            'message': 'Product intelligence cache invalidated',
            'next_access_will_rebuild': True
        }
    finally:
        builder.close()


if __name__ == "__main__":
    # Test the updated intelligence builder
    import logging
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    try:
        print("🚀 Building Product Intelligence with Manufacturer Data...")
        intelligence = build_product_intelligence(save_to_file=True)

        print("\n✅ Intelligence Built Successfully!")

        # Print summary
        summary = get_intelligence_summary()
        print("\n📊 Intelligence Summary:")
        print(f"  Products (Raw): {summary.get('total_products_raw', 0):,}")
        print(f"  Products (Filtered): {summary.get('total_products_filtered', 0):,}")
        print(f"  Filter Efficiency: {summary.get('filter_ratio', 0)}% kept")
        print(
            f"  Manufacturers: {summary.get('total_manufacturers', 0):,}"
        )  # Changed from Brands
        print(f"  Categories: {summary.get('total_categories', 0):,}")
        print(
            f"  Price Range: ${summary.get('price_range', {}).get('min', 0):,.2f} - ${summary.get('price_range', {}).get('max', 0):,.2f}"
        )
        print(
            f"  Average Price: ${summary.get('price_range', {}).get('average', 0):,.2f}"
        )
        print(f"  Stock Ratio: {summary.get('stock_ratio', 0)}%")

        print("\n🔥 Top Categories:")
        for i, category in enumerate(summary.get("top_categories", []), 1):
            print(f"    {i}. {category}")

        print("\n🏭 Top Manufacturers:")  # Changed from Brands
        for i, manufacturer in enumerate(summary.get("top_manufacturers", []), 1):
            print(f"    {i}. {manufacturer}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
