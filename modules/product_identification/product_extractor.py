"""
Product Extractor - Main inference engine for extracting products from customer comments
Combines embedding search with rule-based fallbacks for robust product identification
"""

import logging
import re
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from collections import Counter

from config import ProductIdentifierConfig
from embedding_service import EmbeddingService
from pinecone_manager import PineconeManager

logger = logging.getLogger(__name__)


@dataclass
class ProductMatch:
    """Data class for product matches"""

    product_id: int
    name: str
    brand: str
    category: str
    confidence: float
    match_reason: str
    metadata: Dict[str, Any]
    sku: Optional[str] = None
    price_range: Optional[str] = None


@dataclass
class ExtractionResult:
    """Data class for extraction results"""

    products: List[ProductMatch]
    extraction_method: str
    query_processed: str
    total_matches: int
    confidence_distribution: Dict[str, int]
    processing_time: float
    fallback_used: bool = False


class ProductExtractor:
    """Main product extraction engine"""

    def __init__(self):
        """Initialize the product extractor"""
        self.embedding_service = EmbeddingService()
        self.pinecone_manager = PineconeManager()
        self.config = ProductIdentifierConfig.SEARCH_CONFIG

        # Load any cached embeddings
        self.embedding_service.load_embeddings_cache()

        # Ensure Pinecone connection
        if not self.pinecone_manager.connect_to_index():
            logger.warning(
                "Failed to connect to Pinecone index. Some features may not work."
            )

    def extract_products_from_comment(
        self,
        comment: str,
        max_products: Optional[int] = None,
        category_filter: Optional[str] = None,
        brand_filter: Optional[str] = None,
    ) -> ExtractionResult:
        """
        Extract products from a customer comment

        Args:
            comment: Customer comment text
            max_products: Maximum number of products to return
            category_filter: Optional category filter
            brand_filter: Optional brand filter

        Returns:
            ExtractionResult with matched products
        """
        start_time = time.time()

        if max_products is None:
            max_products = self.config["max_products_per_comment"]

        logger.info(f"Extracting products from comment: {comment[:100]}...")

        # Preprocess the comment
        processed_comment = self.embedding_service.preprocess_comment(comment)

        # Try vector search first
        extraction_result = self._vector_search_extraction(
            processed_comment, max_products, category_filter, brand_filter
        )

        # If vector search fails or returns low-confidence results, try fallback methods
        if not extraction_result.products or all(
            p.confidence < self.config["confidence_threshold"]["medium"]
            for p in extraction_result.products
        ):

            logger.info("Vector search yielded poor results, trying fallback methods")
            fallback_result = self._fallback_extraction(
                comment, max_products, category_filter, brand_filter
            )

            if fallback_result.products:
                # Combine results, prioritizing vector search but including fallback
                all_products = extraction_result.products + fallback_result.products

                # Remove duplicates and sort by confidence
                unique_products = self._deduplicate_products(all_products)
                extraction_result.products = unique_products[:max_products]
                extraction_result.fallback_used = True
                extraction_result.extraction_method = "vector_search_with_fallback"

        # Calculate final statistics
        processing_time = time.time() - start_time
        extraction_result.processing_time = processing_time
        extraction_result.query_processed = processed_comment
        extraction_result.total_matches = len(extraction_result.products)
        extraction_result.confidence_distribution = (
            self._calculate_confidence_distribution(extraction_result.products)
        )

        logger.info(
            f"Extraction completed in {processing_time:.2f}s. Found {len(extraction_result.products)} products"
        )

        return extraction_result

    def _vector_search_extraction(
        self,
        processed_comment: str,
        max_products: int,
        category_filter: Optional[str] = None,
        brand_filter: Optional[str] = None,
    ) -> ExtractionResult:
        """
        Extract products using vector similarity search

        Args:
            processed_comment: Preprocessed comment text
            max_products: Maximum number of products to return
            category_filter: Optional category filter
            brand_filter: Optional brand filter

        Returns:
            ExtractionResult from vector search
        """
        try:
            # Generate embedding for the comment
            comment_embedding = self.embedding_service.generate_embedding(
                processed_comment
            )

            if not comment_embedding:
                logger.error("Failed to generate embedding for comment")
                return ExtractionResult(
                    products=[],
                    extraction_method="vector_search_failed",
                    query_processed=processed_comment,
                    total_matches=0,
                    confidence_distribution={},
                    processing_time=0.0,
                )

            # Search Pinecone
            search_results = self.pinecone_manager.search_products_with_filters(
                query_embedding=comment_embedding,
                category=category_filter,
                brand=brand_filter,
                enabled_only=True,
                top_k=max_products * 2,  # Get more results for better filtering
            )

            # Convert to ProductMatch objects
            products = []
            for result in search_results:
                if result["confidence"] >= self.config["confidence_threshold"]["low"]:
                    metadata = result["metadata"]

                    product_match = ProductMatch(
                        product_id=result["product_id"],
                        name=metadata.get("name", ""),
                        brand=metadata.get("brand", ""),
                        category=metadata.get("category", ""),
                        confidence=result["confidence"],
                        match_reason="vector_similarity",
                        metadata=metadata,
                        sku=metadata.get("sku"),
                        price_range=metadata.get("price_range"),
                    )
                    products.append(product_match)

            # Sort by confidence and limit results
            products.sort(key=lambda x: x.confidence, reverse=True)
            products = products[:max_products]

            return ExtractionResult(
                products=products,
                extraction_method="vector_search",
                query_processed=processed_comment,
                total_matches=len(products),
                confidence_distribution={},
                processing_time=0.0,
            )

        except Exception as e:
            logger.error(f"Vector search extraction failed: {e}")
            return ExtractionResult(
                products=[],
                extraction_method="vector_search_error",
                query_processed=processed_comment,
                total_matches=0,
                confidence_distribution={},
                processing_time=0.0,
            )

    def _fallback_extraction(
        self,
        original_comment: str,
        max_products: int,
        category_filter: Optional[str] = None,
        brand_filter: Optional[str] = None,
    ) -> ExtractionResult:
        """
        Fallback extraction using rule-based pattern matching

        Args:
            original_comment: Original comment text
            max_products: Maximum number of products to return
            category_filter: Optional category filter
            brand_filter: Optional brand filter

        Returns:
            ExtractionResult from fallback methods
        """
        logger.info("Using fallback extraction methods")

        products = []
        comment_lower = original_comment.lower()

        # Extract specific model patterns
        model_patterns = self._extract_model_patterns_from_text(comment_lower)

        for pattern in model_patterns:
            # Search for products matching this pattern
            pattern_products = self._search_by_pattern(
                pattern, category_filter, brand_filter
            )
            products.extend(pattern_products)

        # Extract brand mentions
        brand_mentions = self._extract_brand_mentions(comment_lower)

        for brand in brand_mentions:
            # Get popular products from this brand
            brand_products = self._get_popular_brand_products(brand, category_filter)
            products.extend(brand_products)

        # Remove duplicates and assign confidence scores
        unique_products = self._deduplicate_products(products)

        # Assign confidence based on match specificity
        for product in unique_products:
            if "specific_model" in product.match_reason:
                product.confidence = 0.7
            elif "brand_mention" in product.match_reason:
                product.confidence = 0.5
            else:
                product.confidence = 0.3

        # Sort and limit
        unique_products.sort(key=lambda x: x.confidence, reverse=True)
        unique_products = unique_products[:max_products]

        return ExtractionResult(
            products=unique_products,
            extraction_method="fallback_patterns",
            query_processed=original_comment,
            total_matches=len(unique_products),
            confidence_distribution={},
            processing_time=0.0,
        )

    def _extract_model_patterns_from_text(self, text: str) -> List[str]:
        """Extract specific model patterns from text"""
        patterns = []

        # GPU patterns
        gpu_patterns = [
            r"\b(rtx|gtx)\s*(\d{4})\s*(ti|super)?\b",
            r"\b(radeon)\s*(rx|r)\s*(\d{4})\s*(xt|x)?\b",
        ]

        # CPU patterns
        cpu_patterns = [
            r"\b(ryzen)\s*(\d+)\s*(\d{4}[a-z]*)\b",
            r"\b(core)\s*(i[3579])\s*(\d{4,5}[a-z]*)\b",
        ]

        all_patterns = gpu_patterns + cpu_patterns

        for pattern in all_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                patterns.append(match.group(0).strip())

        return patterns

    def _search_by_pattern(
        self,
        pattern: str,
        category_filter: Optional[str] = None,
        brand_filter: Optional[str] = None,
    ) -> List[ProductMatch]:
        """Search for products matching a specific pattern"""
        # This would ideally search the original product intelligence data
        # For now, we'll simulate some results
        products = []

        # This is a simplified implementation - in practice, you'd search
        # through your product intelligence data loaded from product_identifier.py

        return products

    def _extract_brand_mentions(self, text: str) -> List[str]:
        """Extract brand mentions from text"""
        brands = []

        common_brands = [
            "nvidia",
            "amd",
            "intel",
            "asus",
            "msi",
            "gigabyte",
            "evga",
            "corsair",
            "logitech",
            "razer",
            "steelseries",
            "hyperx",
        ]

        for brand in common_brands:
            if brand in text:
                brands.append(brand)

        return brands

    def _get_popular_brand_products(
        self, brand: str, category_filter: Optional[str] = None
    ) -> List[ProductMatch]:
        """Get popular products from a specific brand"""
        # This would search Pinecone for popular products from the brand
        # For now, return empty list
        return []

    def _deduplicate_products(self, products: List[ProductMatch]) -> List[ProductMatch]:
        """Remove duplicate products from list"""
        seen_ids = set()
        unique_products = []

        for product in products:
            if product.product_id not in seen_ids:
                seen_ids.add(product.product_id)
                unique_products.append(product)

        return unique_products

    def _calculate_confidence_distribution(
        self, products: List[ProductMatch]
    ) -> Dict[str, int]:
        """Calculate confidence distribution for products"""
        distribution = {"high": 0, "medium": 0, "low": 0}

        for product in products:
            if product.confidence >= self.config["confidence_threshold"]["high"]:
                distribution["high"] += 1
            elif product.confidence >= self.config["confidence_threshold"]["medium"]:
                distribution["medium"] += 1
            else:
                distribution["low"] += 1

        return distribution

    def extract_products_batch(
        self, comments: List[str], max_products_per_comment: Optional[int] = None
    ) -> List[ExtractionResult]:
        """
        Extract products from multiple comments in batch

        Args:
            comments: List of customer comments
            max_products_per_comment: Maximum products per comment

        Returns:
            List of ExtractionResult objects
        """
        if max_products_per_comment is None:
            max_products_per_comment = self.config["max_products_per_comment"]

        results = []

        logger.info(f"Processing batch of {len(comments)} comments")

        for i, comment in enumerate(comments):
            logger.info(f"Processing comment {i+1}/{len(comments)}")

            try:
                result = self.extract_products_from_comment(
                    comment, max_products_per_comment
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process comment {i+1}: {e}")
                # Add empty result for failed comment
                results.append(
                    ExtractionResult(
                        products=[],
                        extraction_method="error",
                        query_processed=comment,
                        total_matches=0,
                        confidence_distribution={},
                        processing_time=0.0,
                    )
                )

        return results

    def get_extraction_statistics(
        self, results: List[ExtractionResult]
    ) -> Dict[str, Any]:
        """
        Get statistics for a batch of extraction results

        Args:
            results: List of ExtractionResult objects

        Returns:
            Dictionary with extraction statistics
        """
        stats = {
            "total_comments": len(results),
            "successful_extractions": 0,
            "total_products_found": 0,
            "average_products_per_comment": 0.0,
            "average_processing_time": 0.0,
            "confidence_distribution": {"high": 0, "medium": 0, "low": 0},
            "extraction_methods": Counter(),
            "fallback_usage": 0,
        }

        total_processing_time = 0.0

        for result in results:
            if result.products:
                stats["successful_extractions"] += 1
                stats["total_products_found"] += len(result.products)

            total_processing_time += result.processing_time
            stats["extraction_methods"][result.extraction_method] += 1

            if result.fallback_used:
                stats["fallback_usage"] += 1

            # Aggregate confidence distribution
            for level, count in result.confidence_distribution.items():
                stats["confidence_distribution"][level] += count

        # Calculate averages
        if stats["total_comments"] > 0:
            stats["average_products_per_comment"] = (
                stats["total_products_found"] / stats["total_comments"]
            )
            stats["average_processing_time"] = (
                total_processing_time / stats["total_comments"]
            )

        # Convert Counter to dict
        stats["extraction_methods"] = dict(stats["extraction_methods"])

        return stats

    def suggest_products(
        self, partial_query: str, max_suggestions: int = 5
    ) -> List[ProductMatch]:
        """
        Suggest products based on partial query (for autocomplete/suggestions)

        Args:
            partial_query: Partial product search query
            max_suggestions: Maximum number of suggestions

        Returns:
            List of suggested products
        """
        try:
            # Preprocess the partial query
            processed_query = self.embedding_service.preprocess_text(partial_query)

            # Generate embedding
            query_embedding = self.embedding_service.generate_embedding(processed_query)

            if not query_embedding:
                return []

            # Search for suggestions
            search_results = self.pinecone_manager.search_products(
                query_embedding=query_embedding, top_k=max_suggestions
            )

            # Convert to ProductMatch objects
            suggestions = []
            for result in search_results:
                metadata = result["metadata"]

                suggestion = ProductMatch(
                    product_id=result["product_id"],
                    name=metadata.get("name", ""),
                    brand=metadata.get("brand", ""),
                    category=metadata.get("category", ""),
                    confidence=result["confidence"],
                    match_reason="suggestion",
                    metadata=metadata,
                    sku=metadata.get("sku"),
                    price_range=metadata.get("price_range"),
                )
                suggestions.append(suggestion)

            return suggestions

        except Exception as e:
            logger.error(f"Failed to generate suggestions: {e}")
            return []

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the product extractor

        Returns:
            Health check results
        """
        health_status = {
            "embedding_service": False,
            "pinecone_connection": False,
            "cache_status": {},
            "errors": [],
        }

        try:
            # Test embedding service
            test_embedding = self.embedding_service.generate_embedding("test")
            health_status["embedding_service"] = test_embedding is not None

            # Get cache status
            health_status["cache_status"] = self.embedding_service.get_cache_stats()

        except Exception as e:
            health_status["errors"].append(f"Embedding service error: {e}")

        try:
            # Test Pinecone connection
            pinecone_health = self.pinecone_manager.health_check()
            health_status["pinecone_connection"] = pinecone_health["index_connection"]

            if pinecone_health["errors"]:
                health_status["errors"].extend(pinecone_health["errors"])

        except Exception as e:
            health_status["errors"].append(f"Pinecone connection error: {e}")

        return health_status


if __name__ == "__main__":
    # Test the product extractor
    ProductIdentifierConfig.setup_logging()

    try:
        extractor = ProductExtractor()

        # Perform health check
        health = extractor.health_check()
        print("🏥 Product Extractor Health Check:")
        print(f"   Embedding Service: {'✅' if health['embedding_service'] else '❌'}")
        print(
            f"   Pinecone Connection: {'✅' if health['pinecone_connection'] else '❌'}"
        )

        if health["errors"]:
            print("   Errors:")
            for error in health["errors"]:
                print(f"     - {error}")

        if health["cache_status"]:
            cache = health["cache_status"]
            print("\n💾 Cache Status:")
            print(f"   Cache size: {cache.get('cache_size', 0)} embeddings")
            print(f"   Model: {cache.get('model_used', 'Unknown')}")

        # Test extraction with sample comments
        print("\n🧪 Testing Product Extraction:")

        sample_comments = [
            "Looking for a good RTX 4090 graphics card for gaming",
            "Need a new Ryzen 7 5800X processor for my build",
            "What's the best gaming keyboard under $100?",
            "AMD Radeon RX 7900 XT vs RTX 4080 comparison",
        ]

        for i, comment in enumerate(sample_comments):
            print(f"\n   Test {i+1}: {comment}")

            try:
                result = extractor.extract_products_from_comment(comment)

                if result.products:
                    print(
                        f"     ✅ Found {len(result.products)} products in {result.processing_time:.2f}s"
                    )
                    for product in result.products[:2]:  # Show top 2
                        print(f"       - {product.name} ({product.confidence:.2f})")
                else:
                    print(f"     ❌ No products found ({result.extraction_method})")

            except Exception as e:
                print(f"     ❌ Extraction failed: {e}")

        # Test suggestions
        print("\n💡 Testing Product Suggestions:")
        suggestions = extractor.suggest_products("RTX 40", max_suggestions=3)

        if suggestions:
            print(f"   Found {len(suggestions)} suggestions for 'RTX 40':")
            for suggestion in suggestions:
                print(f"     - {suggestion.name} ({suggestion.confidence:.2f})")
        else:
            print("   No suggestions found")

    except Exception as e:
        print(f"❌ Error testing product extractor: {e}")
