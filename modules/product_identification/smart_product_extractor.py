"""
Smart Product Extractor with Category Intelligence
Uses learned category patterns for better product matching
"""

import re
import logging
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from embedding_service import EmbeddingService
from pinecone_manager import PineconeManager
from category_intelligence import CategoryDetector, CategoryIntelligenceManager
from config import ProductIdentifierConfig

logger = logging.getLogger(__name__)


@dataclass
class SmartProductMatch:
    """Enhanced product match with category intelligence"""
    product_id: int
    name: str
    brand: str
    category: str
    confidence: float
    category_confidence: float
    match_reason: str
    metadata: Dict
    sku: Optional[str] = None
    price_range: Optional[str] = None


@dataclass
class SmartExtractionResult:
    """Enhanced extraction result"""
    products: List[SmartProductMatch]
    extraction_method: str
    query_processed: str
    detected_categories: Dict[str, float]
    category_searches: Dict[str, int]
    total_matches: int
    processing_time: float


class SmartProductExtractor:
    """Category-aware product extractor using learned intelligence"""

    def __init__(self):
        """Initialize the smart extractor"""
        logger.info("🧠 Initializing Smart Product Extractor...")

        # Initialize core services
        self.embedding_service = EmbeddingService()
        self.pinecone_manager = PineconeManager()
        self.config = ProductIdentifierConfig.SEARCH_CONFIG

        # Load category intelligence
        self.category_intelligence = CategoryIntelligenceManager.load_intelligence()
        if not self.category_intelligence:
            logger.warning("⚠️ No category intelligence found. Run category_intelligence.py first!")
            self.category_detector = None
        else:
            self.category_detector = CategoryDetector(self.category_intelligence)
            logger.info(f"✅ Loaded intelligence for {len(self.category_intelligence)} categories")

        # Load embedding cache
        self.embedding_service.load_embeddings_cache()

        # Connect to Pinecone
        if not self.pinecone_manager.connect_to_index():
            logger.error("❌ Failed to connect to Pinecone")
            raise Exception("Pinecone connection failed")

    def extract_products_smart(self, comment: str, max_products: int = 5) -> SmartExtractionResult:
        """
        Extract products using category intelligence

        Args:
            comment: Customer comment
            max_products: Maximum products to return

        Returns:
            SmartExtractionResult with category-aware matches
        """
        start_time = time.time()

        logger.info(f"🔍 Smart extraction: {comment[:50]}...")

        # Preprocess comment
        processed_comment = self.embedding_service.preprocess_comment(comment)

        # Detect likely categories
        detected_categories = {}
        category_searches = {}

        if self.category_detector:
            detected_categories = self.category_detector.detect_categories(processed_comment)
            logger.info(f"📂 Detected categories: {detected_categories}")

        # Generate embedding for the comment
        comment_embedding = self.embedding_service.generate_embedding(processed_comment)
        if not comment_embedding:
            logger.error("❌ Failed to generate embedding")
            return self._empty_result(processed_comment)

        # Search strategy based on detected categories
        all_matches = []

        if detected_categories:
            # Category-aware search
            all_matches = self._category_aware_search(
                comment_embedding, detected_categories, max_products * 2
            )
            category_searches = {cat: len([m for m in all_matches if m.category == cat])
                                 for cat in detected_categories.keys()}
            extraction_method = "category_aware"
        else:
            # Fallback to global search
            logger.info("🌍 No categories detected, using global search")
            all_matches = self._global_search_with_scoring(comment_embedding, max_products * 2)
            extraction_method = "global_fallback"

        # Re-rank results using category confidence
        final_matches = self._rerank_with_category_intelligence(
            all_matches, detected_categories, processed_comment
        )

        # Limit results
        final_matches = final_matches[:max_products]

        # Calculate processing time
        processing_time = time.time() - start_time

        logger.info(f"✅ Smart extraction completed: {len(final_matches)} products in {processing_time:.2f}s")

        return SmartExtractionResult(
            products=final_matches,
            extraction_method=extraction_method,
            query_processed=processed_comment,
            detected_categories=detected_categories,
            category_searches=category_searches,
            total_matches=len(all_matches),
            processing_time=processing_time
        )

    def _category_aware_search(self, embedding: List[float],
                               detected_categories: Dict[str, float],
                               max_results: int) -> List[SmartProductMatch]:
        """Search with category awareness"""

        all_matches = []

        # Search each detected category
        for category, category_confidence in detected_categories.items():
            # Calculate how many results to get from this category
            category_weight = category_confidence
            category_results = max(2, int(max_results * category_weight))

            logger.info(f"🔍 Searching {category} (conf: {category_confidence:.2f}, results: {category_results})")

            # Search with category filter
            search_results = self.pinecone_manager.search_products_with_filters(
                query_embedding=embedding,
                category=category,
                enabled_only=True,
                top_k=category_results
            )

            # Convert to SmartProductMatch objects
            for result in search_results:
                metadata = result['metadata']

                match = SmartProductMatch(
                    product_id=result['product_id'],
                    name=metadata.get('name', ''),
                    brand=metadata.get('brand', ''),
                    category=metadata.get('category', ''),
                    confidence=result['confidence'],
                    category_confidence=category_confidence,
                    match_reason=f"category_aware_{category.lower().replace(' ', '_')}",
                    metadata=metadata,
                    sku=metadata.get('sku'),
                    price_range=metadata.get('price_range')
                )

                all_matches.append(match)

        return all_matches

    def _global_search_with_scoring(self, embedding: List[float], max_results: int) -> List[SmartProductMatch]:
        """Global search with enhanced scoring"""

        search_results = self.pinecone_manager.search_products(
            query_embedding=embedding,
            top_k=max_results
        )

        matches = []
        for result in search_results:
            metadata = result['metadata']

            match = SmartProductMatch(
                product_id=result['product_id'],
                name=metadata.get('name', ''),
                brand=metadata.get('brand', ''),
                category=metadata.get('category', ''),
                confidence=result['confidence'],
                category_confidence=0.5,  # Neutral category confidence
                match_reason="global_search",
                metadata=metadata,
                sku=metadata.get('sku'),
                price_range=metadata.get('price_range')
            )

            matches.append(match)

        return matches

    def _rerank_with_category_intelligence(self, matches: List[SmartProductMatch],
                                           detected_categories: Dict[str, float],
                                           query: str) -> List[SmartProductMatch]:
        """Re-rank results using category intelligence and query analysis"""

        if not detected_categories:
            # No category intelligence, just sort by confidence
            return sorted(matches, key=lambda x: x.confidence, reverse=True)

        # Calculate enhanced scores
        for match in matches:
            # Base score is the Pinecone confidence
            enhanced_score = match.confidence

            # Boost if category matches detected categories
            if match.category in detected_categories:
                category_boost = detected_categories[match.category]
                enhanced_score *= (1.0 + category_boost)
                match.match_reason += f"_category_boost_{category_boost:.2f}"

            # Apply query-specific boosts
            query_boost = self._calculate_query_boost(match, query)
            enhanced_score *= query_boost

            # Update confidence with enhanced score
            match.confidence = min(0.99, enhanced_score)  # Cap at 0.99

        # Sort by enhanced confidence
        return sorted(matches, key=lambda x: x.confidence, reverse=True)

    def _calculate_query_boost(self, match: SmartProductMatch, query: str) -> float:
        """Calculate query-specific boost factors"""
        boost = 1.0
        query_lower = query.lower()
        name_lower = match.name.lower()

        # Exact model number match
        if self._has_exact_model_match(query_lower, name_lower):
            boost *= 1.5

        # Brand name mentioned
        if match.brand.lower() in query_lower:
            boost *= 1.2

        # Technical terms alignment
        if self._has_technical_alignment(query_lower, name_lower):
            boost *= 1.1

        return boost

    def _has_exact_model_match(self, query: str, product_name: str) -> bool:
        """Check if query contains exact model numbers found in product name"""

        # Extract potential model numbers from query
        query_models = re.findall(r'\b[a-zA-Z]*\d{3,}[a-zA-Z0-9]*\b', query)
        product_models = re.findall(r'\b[a-zA-Z]*\d{3,}[a-zA-Z0-9]*\b', product_name)

        # Check for exact matches
        for query_model in query_models:
            for product_model in product_models:
                if query_model.lower() == product_model.lower():
                    return True

        return False

    def _has_technical_alignment(self, query: str, product_name: str) -> bool:
        """Check if technical terms align between query and product"""

        tech_terms = [
            'gaming', 'professional', 'workstation', 'rgb', 'wireless',
            'mechanical', 'optical', '4k', '1440p', '144hz', '240hz'
        ]

        query_terms = set(term for term in tech_terms if term in query)
        product_terms = set(term for term in tech_terms if term in product_name)

        # Check for overlap
        return len(query_terms.intersection(product_terms)) > 0

    def _empty_result(self, processed_query: str) -> SmartExtractionResult:
        """Return empty result"""
        return SmartExtractionResult(
            products=[],
            extraction_method="failed",
            query_processed=processed_query,
            detected_categories={},
            category_searches={},
            total_matches=0,
            processing_time=0.0
        )

    def get_category_suggestions(self, query: str) -> Dict[str, float]:
        """Get category suggestions for a query"""
        if not self.category_detector:
            return {}

        return self.category_detector.detect_categories(query)


# Test function
def test_smart_extractor():
    """Test the smart extractor"""

    # Initialize
    extractor = SmartProductExtractor()

    # Test queries
    test_queries = [
        "9800X3D",
        "RTX 4090",
        "AMD Ryzen processor",
        "gaming keyboard",
        "32GB DDR4",
        "1TB NVMe SSD"
    ]

    print("🧪 Testing Smart Product Extractor")
    print("=" * 50)

    for query in test_queries:
        print(f"\n🔍 Testing: '{query}'")

        result = extractor.extract_products_smart(query)

        print(f"   Method: {result.extraction_method}")
        print(f"   Categories: {result.detected_categories}")
        print(f"   Products found: {len(result.products)}")
        print(f"   Processing time: {result.processing_time:.2f}s")

        if result.products:
            top_match = result.products[0]
            print(f"   Top match: {top_match.name} (conf: {top_match.confidence:.3f})")


if __name__ == "__main__":
    # Test the smart extractor
    import logging

    logging.basicConfig(level=logging.INFO)

    test_smart_extractor()