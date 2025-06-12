"""
Category Intelligence for Product Identification
Detects likely product categories from customer queries using pattern matching and keywords
"""

import re
import json
import logging
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class CategoryDetector:
    """Detects product categories from text queries"""

    def __init__(self, intelligence_data: Optional[Dict] = None):
        """
        Initialize category detector

        Args:
            intelligence_data: Pre-loaded intelligence data (optional)
        """
        self.intelligence = intelligence_data or self._load_default_intelligence()

    def detect_categories(
        self, query: str, confidence_threshold: float = 0.3
    ) -> Dict[str, float]:
        """
        Detect likely categories from a query

        Args:
            query: Customer query text
            confidence_threshold: Minimum confidence to include category

        Returns:
            Dict mapping category names to confidence scores
        """
        if not query or not query.strip():
            return {}

        query_lower = query.lower().strip()
        category_scores = {}

        categories = self.intelligence.get("categories", {})

        for category_name, category_data in categories.items():
            score = self._calculate_category_score(query_lower, category_data)

            if score >= confidence_threshold:
                category_scores[category_name] = score

        # Sort by confidence and return top categories
        sorted_categories = dict(
            sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
        )

        # Limit to top 3 categories to avoid noise
        return dict(list(sorted_categories.items())[:3])

    def _calculate_category_score(self, query: str, category_data: Dict) -> float:
        """
        Calculate confidence score for a category based on query

        Args:
            query: Lowercase query text
            category_data: Category intelligence data

        Returns:
            Confidence score between 0.0 and 1.0
        """
        score = 0.0

        # Check exact keyword matches
        keywords = category_data.get("keywords", [])
        keyword_matches = sum(1 for keyword in keywords if keyword in query)
        if keywords:
            score += (keyword_matches / len(keywords)) * 0.6

        # Check pattern matches (model numbers, etc.)
        patterns = category_data.get("patterns", [])
        pattern_matches = 0
        for pattern in patterns:
            try:
                if re.search(pattern, query, re.IGNORECASE):
                    pattern_matches += 1
            except re.error:
                continue  # Skip invalid regex patterns

        if patterns:
            score += (pattern_matches / len(patterns)) * 0.3

        # Check brand mentions
        brands = category_data.get("brands", [])
        brand_matches = sum(1 for brand in brands if brand in query)
        if brands:
            score += (brand_matches / len(brands)) * 0.1

        return min(score, 1.0)  # Cap at 1.0

    def _load_default_intelligence(self) -> Dict:
        """Load default category intelligence if no external data provided"""
        return {
            "categories": {
                "Graphics Cards": {
                    "keywords": [
                        "gpu",
                        "graphics",
                        "video card",
                        "rtx",
                        "gtx",
                        "radeon",
                        "geforce",
                    ],
                    "patterns": [
                        r"\brtx\s*\d{4}\b",
                        r"\bgtx\s*\d{4}\b",
                        r"\bradeon\s*rx\s*\d{4}\b",
                    ],
                    "brands": ["nvidia", "amd", "asus", "msi", "gigabyte", "evga"],
                },
                "Processors": {
                    "keywords": [
                        "cpu",
                        "processor",
                        "ryzen",
                        "intel",
                        "core",
                        "i3",
                        "i5",
                        "i7",
                        "i9",
                    ],
                    "patterns": [
                        r"\bryzen\s*\d+\s*\d{4}[a-z]*\b",
                        r"\bcore\s*i[3579]-\d{4,5}[a-z]*\b",
                    ],
                    "brands": ["amd", "intel"],
                },
                "Memory": {
                    "keywords": ["ram", "memory", "ddr4", "ddr5", "dimm"],
                    "patterns": [r"\b\d+gb\s*ddr[45]\b", r"\bddr[45]-\d{4}\b"],
                    "brands": ["corsair", "kingston", "crucial", "gskill", "teamgroup"],
                },
                "Storage": {
                    "keywords": [
                        "ssd",
                        "hdd",
                        "nvme",
                        "storage",
                        "drive",
                        "hard drive",
                        "solid state",
                    ],
                    "patterns": [
                        r"\b\d+gb\b",
                        r"\b\d+tb\b",
                        r"\bnvme\b",
                        r"\bsata\b",
                        r"\bm\.?2\b",
                    ],
                    "brands": [
                        "samsung",
                        "western digital",
                        "seagate",
                        "crucial",
                        "wd",
                    ],
                },
                "Motherboards": {
                    "keywords": [
                        "motherboard",
                        "mobo",
                        "mainboard",
                        "socket",
                        "chipset",
                    ],
                    "patterns": [
                        r"\bam[45]\b",
                        r"\blga\d{4}\b",
                        r"\bx\d{3}\b",
                        r"\bb\d{3}\b",
                    ],
                    "brands": ["asus", "msi", "gigabyte", "asrock"],
                },
                "Power Supplies": {
                    "keywords": [
                        "psu",
                        "power supply",
                        "watt",
                        "watts",
                        "modular",
                        "80+",
                    ],
                    "patterns": [r"\b\d+w\b", r"\b\d+\s*watts?\b", r"\b80\+\b"],
                    "brands": ["corsair", "evga", "seasonic", "cooler master"],
                },
                "Monitors": {
                    "keywords": [
                        "monitor",
                        "display",
                        "screen",
                        "4k",
                        "1440p",
                        "144hz",
                        "curved",
                    ],
                    "patterns": [
                        r"\b\d+\s*inch\b",
                        r"\b\d+hz\b",
                        r"\b4k\b",
                        r"\b1440p\b",
                    ],
                    "brands": ["asus", "acer", "lg", "samsung", "dell"],
                },
                "Keyboards": {
                    "keywords": ["keyboard", "mechanical", "rgb", "gaming", "wireless"],
                    "patterns": [r"\bmechanical\b", r"\brgb\b", r"\bwireless\b"],
                    "brands": ["logitech", "razer", "corsair", "steelseries"],
                },
                "Mice": {
                    "keywords": [
                        "mouse",
                        "gaming mouse",
                        "wireless",
                        "optical",
                        "laser",
                    ],
                    "patterns": [r"\bmouse\b", r"\bwireless\b", r"\boptical\b"],
                    "brands": ["logitech", "razer", "corsair", "steelseries"],
                },
                "Cases": {
                    "keywords": [
                        "case",
                        "chassis",
                        "tower",
                        "atx",
                        "mini-itx",
                        "tempered glass",
                    ],
                    "patterns": [r"\batx\b", r"\bmini-?itx\b", r"\bmid-?tower\b"],
                    "brands": ["corsair", "nzxt", "fractal design", "cooler master"],
                },
            },
            "created": datetime.now().isoformat(),
            "note": "Default category intelligence for product identification",
        }

#Replacing the entire Class CategoryIntelligenceManager
class CategoryIntelligenceManager:
    """Manages category intelligence data from database instead of files"""

    @staticmethod
    def load_intelligence(filepath: str = None) -> Optional[Dict]:
        """
        UPDATED: Load category intelligence from database instead of file

        Args:
            filepath: Ignored (kept for backwards compatibility)

        Returns:
            Intelligence data from database or None if failed
        """
        try:
            # Import here to avoid circular imports
            from ..database import get_category_intelligence_from_database

            logger.info("📊 Loading category intelligence from database...")
            intelligence = get_category_intelligence_from_database()

            if intelligence and intelligence.get('categories'):
                categories_count = len(intelligence['categories'])
                logger.info(f"✅ Category intelligence loaded from database: {categories_count} categories")
                return intelligence
            else:
                logger.warning("❌ No category intelligence data found in database")
                return None

        except Exception as e:
            logger.error(f"❌ Failed to load category intelligence from database: {e}")
            return None

    @staticmethod
    def save_intelligence(intelligence: Dict, filepath: str = None) -> bool:
        """
        DEPRECATED: Saving to file no longer supported
        Category intelligence is now generated from database on-demand

        Args:
            intelligence: Intelligence data (ignored)
            filepath: File path (ignored)

        Returns:
            False (operation not supported)
        """
        logger.warning("⚠️ save_intelligence() is deprecated - category intelligence is now database-driven")
        logger.info("💡 Use get_category_intelligence_from_database() to get fresh data")
        return False

    @staticmethod
    def create_default_intelligence(filepath: str = None) -> Dict:
        """
        ✅ UPDATED: Get intelligence from database instead of creating default file

        Args:
            filepath: Ignored (kept for backwards compatibility)

        Returns:
            Intelligence data from database
        """
        try:
            from ..database import get_category_intelligence_from_database

            logger.info("🏗️ Creating category intelligence from database...")
            intelligence = get_category_intelligence_from_database()

            if intelligence:
                logger.info("✅ Category intelligence created from database")
                return intelligence
            else:
                logger.error("❌ Failed to create category intelligence from database")
                return {}

        except Exception as e:
            logger.error(f"❌ Error creating category intelligence: {e}")
            return {}

    @staticmethod
    def refresh_intelligence() -> Dict:
        """
        ✅ NEW: Force refresh category intelligence from database

        Returns:
            Fresh intelligence data from database
        """
        try:
            from ..database import get_category_intelligence_from_database

            logger.info("🔄 Refreshing category intelligence from database...")
            intelligence = get_category_intelligence_from_database()

            if intelligence:
                categories_count = len(intelligence.get('categories', {}))
                logger.info(f"✅ Category intelligence refreshed: {categories_count} categories")
                return intelligence
            else:
                logger.error("❌ Failed to refresh category intelligence")
                return {}

        except Exception as e:
            logger.error(f"❌ Error refreshing category intelligence: {e}")
            return {}


# Convenience functions

#RReplacing the Convenience functions
def detect_categories_from_query(query: str, intelligence_file: str = None) -> Dict[str, float]:
    """
    UPDATED: Detect categories from query using database intelligence

    Args:
        query: Customer query text
        intelligence_file: Ignored (kept for backwards compatibility)

    Returns:
        Dict mapping category names to confidence scores
    """
    # Load intelligence from database instead of file
    intelligence = CategoryIntelligenceManager.load_intelligence()

    # Create detector with database intelligence
    detector = CategoryDetector(intelligence)

    return detector.detect_categories(query)


def initialize_category_intelligence(filepath: str = None) -> bool:
    """
    UPDATED: Validate category intelligence from database

    Args:
        filepath: Ignored (kept for backwards compatibility)

    Returns:
        True if database intelligence is available, False otherwise
    """
    try:
        intelligence = CategoryIntelligenceManager.load_intelligence()

        if intelligence and intelligence.get('categories'):
            categories_count = len(intelligence['categories'])
            logger.info(f"✅ Category intelligence validated: {categories_count} categories from database")
            return True
        else:
            logger.error("❌ Category intelligence validation failed - no data from database")
            return False

    except Exception as e:
        logger.error(f"❌ Failed to validate category intelligence: {e}")
        return False

#Newly added
def refresh_category_intelligence() -> Dict:
    """
    NEW: Refresh category intelligence from database

    Returns:
        Dict with refresh results
    """
    try:
        intelligence = CategoryIntelligenceManager.refresh_intelligence()

        if intelligence:
            return {
                'success': True,
                'message': 'Category intelligence refreshed from database',
                'categories': len(intelligence.get('categories', {})),
                'source': 'live_database'
            }
        else:
            return {
                'success': False,
                'message': 'Failed to refresh category intelligence from database',
                'categories': 0
            }

    except Exception as e:
        logger.error(f"Error refreshing category intelligence: {e}")
        return {
            'success': False,
            'message': f'Refresh failed: {str(e)}',
            'categories': 0
        }

if __name__ == "__main__":
    # Test the category detection
    import logging

    logging.basicConfig(level=logging.INFO)

    print("🧠 Testing Category Intelligence")
    print("=" * 40)

    # Initialize with default intelligence
    intelligence = CategoryIntelligenceManager.create_default_intelligence()

    # Create detector
    detector = CategoryDetector(intelligence)

    # Test queries
    test_queries = [
        "Looking for RTX 4090",
        "Need DDR5 memory 32GB",
        "AMD Ryzen 9 7950X processor",
        "Gaming mechanical keyboard",
        "4K monitor 144Hz",
        "1TB NVMe SSD",
        "650W modular PSU",
    ]

    for query in test_queries:
        categories = detector.detect_categories(query)
        print(f"\nQuery: '{query}'")

        if categories:
            for category, confidence in categories.items():
                print(f"  {category}: {confidence:.3f}")
        else:
            print("  No categories detected")

    print("\n✅ Category intelligence test completed")
