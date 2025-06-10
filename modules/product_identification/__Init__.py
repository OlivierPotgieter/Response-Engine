"""
Product Identification Module
AI-powered product identification from customer comments using embeddings and vector search.
"""

from .config import ProductIdentifierConfig
from .smart_product_extractor import SmartProductExtractor, SmartProductMatch, SmartExtractionResult
from .product_extractor import ProductExtractor, ProductMatch, ExtractionResult
from .build_system import ProductIdentifierSystemBuilder, SystemStatus


# Main interface functions that replace the old product_search.py
def search_comment_for_products(comment: str, max_results: int = 3) -> dict:
    """
    REPLACEMENT for the old search_comment_for_products function.
    This maintains the exact same interface so existing code continues to work.

    Args:
        comment: Customer comment text
        max_results: Maximum number of products to return

    Returns:
        Dict with search results in the same format as before
    """
    try:
        # Initialize the smart extractor (will use cached data if available)
        extractor = SmartProductExtractor()

        # Extract products using the advanced AI system
        result = extractor.extract_products_smart(comment, max_products=max_results)

        # Convert to the expected format for backward compatibility
        products_found = []
        for product in result.products:
            products_found.append({
                "product_id": product.product_id,
                "name": product.name,
                "sku": product.sku,
                "brand": product.brand,  # Note: now from Manufacturers table
                "category": product.category,
                "current_price": product.metadata.get("current_price"),
                "is_in_stock": product.metadata.get("is_in_stock"),
                "relevance_score": product.confidence,
                "search_timestamp": "current",
                "match_reason": product.match_reason
            })

        return {
            "comment_analyzed": comment,
            "identifiers_extracted": {
                "search_terms": list(result.detected_categories.keys()),
                "confidence_score": max(result.detected_categories.values()) if result.detected_categories else 0,
                "categories_inferred": list(result.detected_categories.keys())
            },
            "products_found": products_found,
            "search_successful": len(products_found) > 0,
            "best_match": products_found[0] if products_found else None,
            "search_summary": {
                "extraction_confidence": max(result.detected_categories.values()) if result.detected_categories else 0,
                "products_found_count": len(products_found),
                "search_terms_used": list(result.detected_categories.keys()),
                "categories_inferred": list(result.detected_categories.keys()),
                "extraction_method": result.extraction_method,
                "processing_time": result.processing_time
            },
            "search_timestamp": "current",
            "note": "AI-powered product identification using embeddings and vector search"
        }

    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"AI product identification failed: {e}")

        # Return the same error format as the old system
        return {
            "comment_analyzed": comment,
            "search_successful": False,
            "error": str(e),
            "search_timestamp": "current",
            "note": "AI product identification failed - system may need initialization"
        }


def extract_product_identifiers_from_comment(comment: str) -> dict:
    """
    REPLACEMENT for the old extract_product_identifiers_from_comment function.
    Maintains backward compatibility.

    Args:
        comment: Customer comment text

    Returns:
        Dict with extracted identifiers
    """
    try:
        # Use the smart extractor to detect categories and patterns
        extractor = SmartProductExtractor()

        # Get category suggestions (this is like identifier extraction)
        detected_categories = extractor.get_category_suggestions(comment)

        return {
            "brands_found": [],  # Legacy field - could be populated if needed
            "models_found": [],  # Legacy field - could be populated if needed
            "capacities_found": [],  # Legacy field - could be populated if needed
            "categories_inferred": list(detected_categories.keys()),
            "search_terms": list(detected_categories.keys()),
            "confidence_score": max(detected_categories.values()) if detected_categories else 0,
            "extraction_timestamp": "current",
            "note": "AI-powered identifier extraction using category intelligence"
        }

    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Product identifier extraction failed: {e}")

        return {
            "brands_found": [],
            "models_found": [],
            "capacities_found": [],
            "categories_inferred": [],
            "search_terms": [],
            "confidence_score": 0.0,
            "error": str(e),
            "extraction_timestamp": "current"
        }


def initialize_product_identification_system(force_rebuild: bool = False) -> dict:
    """
    Initialize the AI product identification system.
    This builds the required data files from your database.

    Args:
        force_rebuild: Whether to rebuild even if data exists

    Returns:
        Dict with initialization results
    """
    try:
        from .build_system import ProductIdentifierSystemBuilder

        builder = ProductIdentifierSystemBuilder(force_rebuild=force_rebuild)
        result = builder.build_complete_system()

        return {
            "success": result.success,
            "message": result.message,
            "details": result.details,
            "time_taken": result.processing_time,
            "files_created": result.files_created
        }

    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"System initialization failed: {e}")

        return {
            "success": False,
            "message": f"System initialization failed: {str(e)}",
            "error": str(e)
        }


def get_system_health() -> dict:
    """
    Check the health of the product identification system.

    Returns:
        Dict with system health information
    """
    try:
        # Check if required files exist
        import os
        required_files = [
            "product_intelligence.json",
            "category_intelligence.json",
            "product_embeddings_cache.pkl"
        ]

        files_exist = {}
        for file in required_files:
            files_exist[file] = os.path.exists(file)

        # Try to initialize the extractor
        system_ready = False
        error_message = None

        try:
            extractor = SmartProductExtractor()
            system_ready = True
        except Exception as e:
            error_message = str(e)

        return {
            "system_ready": system_ready,
            "files_exist": files_exist,
            "all_files_present": all(files_exist.values()),
            "error": error_message,
            "note": "System is ready if all files exist and extractor initializes"
        }

    except Exception as e:
        return {
            "system_ready": False,
            "error": str(e),
            "note": "Health check failed"
        }


__all__ = [
    # Configuration
    'ProductIdentifierConfig',

    # Extractors
    'SmartProductExtractor',
    'ProductExtractor',
    'SmartProductMatch',
    'ProductMatch',
    'SmartExtractionResult',
    'ExtractionResult',

    # System management
    'ProductIdentifierSystemBuilder',
    'SystemStatus',

    # Backward-compatible interface functions
    'search_comment_for_products',
    'extract_product_identifiers_from_comment',

    # System management functions
    'initialize_product_identification_system',
    'get_system_health'
]