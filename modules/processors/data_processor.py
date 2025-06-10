"""
Data Processing Module - Enhanced with AI Product Identification
Handles data validation, processing, and business logic coordination with smart data prioritization and EOL detection.
Now includes advanced AI-powered product identification when no product_id/alternative_id is available.
"""

import logging
import re
from typing import Dict, Optional, List
from datetime import datetime

# Import our other modules
from ..database import get_customer_request_data, get_custom_response_data, get_all_products_for_request
from ..ai import (
    check_intent_scope,
    search_with_full_context,
    generate_response_with_examples,
    generate_enhanced_response_with_context,
    generate_simple_response_for_comment
)

# Import the NEW AI product identification system
try:
    from ..product_identification import (
        search_comment_for_products,
        extract_product_identifiers_from_comment,
        initialize_product_identification_system,
        get_system_health
    )
    AI_PRODUCT_IDENTIFICATION_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ AI Product Identification system loaded successfully")
except ImportError as e:
    AI_PRODUCT_IDENTIFICATION_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ AI Product Identification not available: {e}")

    # Fallback functions with same interface
    def search_comment_for_products(comment: str, max_results: int = 3) -> Dict:
        return {
            "search_successful": False,
            "products_found": [],
            "best_match": None,
            "search_summary": {
                "extraction_confidence": 0,
                "search_terms_used": []
            },
            "note": "AI Product Identification system not available - run initialization"
        }

    def extract_product_identifiers_from_comment(comment: str) -> Dict:
        return {
            "brands_found": [],
            "models_found": [],
            "capacities_found": [],
            "categories_inferred": [],
            "search_terms": [],
            "confidence_score": 0.0,
            "note": "AI Product Identification system not available"
        }


class DataProcessor:
    def __init__(self):
        """Initialize the data processor with FIXED keyword detection and AI product identification"""
        self.out_of_scope_intents = ["Compatibility or Upgrade", "Product Recommendation"]

        # FIXED: Use regex patterns with proper word boundaries
        self.pricing_patterns = [
            r'\bR\s*\d+',           # R followed by numbers (R1000, R 1000)
            r'\bR\d+',              # R immediately followed by numbers (R1000)
            r'\bprice\b',           # price as whole word
            r'\bcost\b',            # cost as whole word
            r'\bpricing\b',         # pricing as whole word
            r'\bexpensive\b',       # expensive as whole word
            r'\bcheap\b',           # cheap as whole word
            r'\bspecial\b',         # special as whole word
            r'\bdiscount\b',        # discount as whole word
            r'\bpromotion\b',       # promotion as whole word
            r'\bpromo\b',           # promo as whole word
            r'\bdeal\b',            # deal as whole word
            r'\bsale\b',            # sale as whole word
            r'\boffer\b',           # offer as whole word
            r'\bquote\b',           # quote as whole word
            r'\bquotation\b'        # quotation as whole word
        ]

        self.stock_patterns = [
            r'\bstock\b',           # stock as whole word
            r'\bavailable\b',       # available as whole word
            r'\bavailability\b',    # availability as whole word
            r'\bin\s+stock\b',      # "in stock" as phrase
            r'\bout\s+of\s+stock\b', # "out of stock" as phrase
            r'\bwhen\s+available\b', # "when available" as phrase
            r'\beta\b',             # eta as whole word (not "meta")
            r'\blead\s+time\b',     # "lead time" as phrase
            r'\bdispatch\b',        # dispatch as whole word
            r'\bready\b',           # ready as whole word
            r'\bhave\s+it\b',       # "have it" as phrase
            r'\bon\s+hand\b',       # "on hand" as phrase
            r'\bdelivery\b',        # delivery as whole word
            r'\bshipping\b'         # shipping as whole word
        ]

    def _get_intent_based_needs(self, predicted_intent: str) -> Dict:
        """Get data needs based on intent classification (takes precedence)"""
        intent_mapping = {
            "Pricing Inquiry": {"pricing": True, "stock": False},
            "Stock Availability": {"pricing": False, "stock": True},
            "Quotation Request": {"pricing": True, "stock": True},
            "Warranty Inquiry": {"pricing": False, "stock": False},
            "General Inquiry": {"pricing": False, "stock": False},
            "Order Assistance": {"pricing": False, "stock": False},
            "Returns and Issues": {"pricing": False, "stock": False},
            "Shipping and Collection": {"pricing": False, "stock": False}
        }

        return intent_mapping.get(predicted_intent, {"pricing": False, "stock": False})

    def _get_keyword_based_needs(self, customer_comment: str) -> Dict:
        """Get data needs based on keyword detection using FIXED regex patterns"""
        comment_lower = customer_comment.lower()

        pricing_matches = []
        stock_matches = []

        # Check pricing patterns with regex
        for pattern in self.pricing_patterns:
            matches = re.findall(pattern, comment_lower, re.IGNORECASE)
            if matches:
                pricing_matches.extend(matches)

        # Check stock patterns with regex
        for pattern in self.stock_patterns:
            matches = re.findall(pattern, comment_lower, re.IGNORECASE)
            if matches:
                stock_matches.extend(matches)

        return {
            "pricing": len(pricing_matches) > 0,
            "stock": len(stock_matches) > 0,
            "pricing_matches": list(set(pricing_matches)),  # Remove duplicates
            "stock_matches": list(set(stock_matches))
        }

    def detect_data_needs(self, customer_comment: str, predicted_intent: str) -> Dict:
        """FIXED: Detect data needs with intent-first approach and proper keyword detection"""
        try:
            # Layer 1: Intent-Based Detection (TAKES PRECEDENCE)
            intent_based_needs = self._get_intent_based_needs(predicted_intent)

            # Layer 2: Keyword-Based Detection (FIXED with regex)
            keyword_based_needs = self._get_keyword_based_needs(customer_comment)

            # FIXED: Intent determines primary needs, keywords can enhance
            needs_pricing = intent_based_needs.get("pricing", False) or keyword_based_needs.get("pricing", False)
            needs_stock = intent_based_needs.get("stock", False) or keyword_based_needs.get("stock", False)

            # CRITICAL: Intent determines prompt strategy, NOT keywords
            prompt_strategy = self._determine_prompt_strategy(predicted_intent, needs_pricing, needs_stock)

            detection_result = {
                "needs_pricing": needs_pricing,
                "needs_stock": needs_stock,
                "needs_real_time_data": needs_pricing or needs_stock,
                "prompt_strategy": prompt_strategy,
                "detection_layers": {
                    "intent_based": intent_based_needs,
                    "keyword_based": {
                        "pricing": keyword_based_needs.get("pricing", False),
                        "stock": keyword_based_needs.get("stock", False),
                        "pricing_matches": keyword_based_needs.get("pricing_matches", []),
                        "stock_matches": keyword_based_needs.get("stock_matches", [])
                    }
                },
                "detection_reason": self._build_detection_reason(
                    intent_based_needs, keyword_based_needs, predicted_intent
                ),
                "predicted_intent": predicted_intent
            }

            logger.info(f"FIXED Data needs detection: Intent={predicted_intent}, "
                        f"Pricing={needs_pricing}, Stock={needs_stock}, "
                        f"Strategy={prompt_strategy}")

            return detection_result

        except Exception as e:
            logger.error(f"Error in fixed data needs detection: {e}")
            return {
                "needs_pricing": False,
                "needs_stock": False,
                "needs_real_time_data": False,
                "prompt_strategy": "fallback",
                "error": str(e),
                "predicted_intent": predicted_intent
            }

    def _build_detection_reason(self, intent_needs: Dict, keyword_needs: Dict, predicted_intent: str) -> str:
        """Build human-readable detection reasoning with intent precedence"""
        reasons = []

        # Intent-based reasons (primary)
        if intent_needs.get("pricing", False):
            reasons.append(f"Intent '{predicted_intent}' requires pricing data")
        if intent_needs.get("stock", False):
            reasons.append(f"Intent '{predicted_intent}' requires stock data")

        # Keyword-based reasons (secondary)
        pricing_matches = keyword_needs.get("pricing_matches", [])
        stock_matches = keyword_needs.get("stock_matches", [])

        if pricing_matches:
            reasons.append(f"Pricing keywords detected: {', '.join(pricing_matches[:3])}")
        if stock_matches:
            reasons.append(f"Stock keywords detected: {', '.join(stock_matches[:3])}")

        if not reasons:
            return f"Intent '{predicted_intent}' - no special data requirements detected"

        return "; ".join(reasons)

    def _determine_prompt_strategy(self, predicted_intent: str, needs_pricing: bool, needs_stock: bool) -> str:
        """Determine prompt strategy based on INTENT, not just keywords"""
        # Intent-based strategy (takes precedence)
        intent_strategies = {
            "Warranty Inquiry": "warranty_focused",
            "Pricing Inquiry": "pricing_focused",
            "Stock Availability": "stock_focused",
            "Quotation Request": "combined_pricing_stock",
            "General Inquiry": "general_helpful",
            "Order Assistance": "order_focused",
            "Returns and Issues": "support_focused",
            "Shipping and Collection": "shipping_focused"
        }

        base_strategy = intent_strategies.get(predicted_intent, "general_helpful")

        # Only modify for special cases
        if base_strategy == "general_helpful" and needs_pricing and needs_stock:
            return "combined_pricing_stock"
        elif base_strategy == "general_helpful" and needs_pricing:
            return "pricing_enhanced"
        elif base_strategy == "general_helpful" and needs_stock:
            return "stock_enhanced"

        return base_strategy

    def determine_viable_product(self, product_details: Dict, customer_data: Dict) -> Dict:
        """FIXED: Determine which product to use based on viability and availability"""
        try:
            main_product = product_details.get("main_product")
            alternative_product = product_details.get("alternative_product")

            # Check if we have alternative_id in customer data (sales staff recommendation)
            alternative_id = customer_data.get('alternative_id')
            alternative_leadtime = customer_data.get('alternative_leadtime', '').strip()

            result = {
                "primary_product": None,
                "secondary_product": None,
                "selection_reason": "No product data available",
                "main_product_viable": False,
                "alternative_product_viable": False,
                "has_alternative_recommendation": bool(alternative_id and alternative_leadtime),
                "has_product_data": False
            }

            # FIXED: Check if we have any actual product data
            has_main_product_data = main_product and not main_product.get('error')
            has_alt_product_data = alternative_product and not alternative_product.get('error')
            result["has_product_data"] = has_main_product_data or has_alt_product_data

            # Check main product viability (only if we have data)
            if has_main_product_data:
                main_viable = self._is_product_viable(main_product)
                result["main_product_viable"] = main_viable
                logger.info(f"Main product {main_product.get('product_id')} viable: {main_viable}")
            else:
                logger.info("No main product data available for viability check")

            # Check alternative product viability (only if we have data)
            if has_alt_product_data:
                alt_viable = self._is_product_viable(alternative_product)
                result["alternative_product_viable"] = alt_viable
                logger.info(f"Alternative product {alternative_product.get('product_id')} viable: {alt_viable}")
            else:
                logger.info("No alternative product data available for viability check")

            # FIXED: Decision logic - don't assume EOL without product data
            if not result["has_product_data"]:
                result["selection_reason"] = "No product data available - may need product lookup"
                logger.info("No product data available - cannot determine viability")

            elif result["has_alternative_recommendation"] and result["alternative_product_viable"]:
                result["primary_product"] = alternative_product
                result["secondary_product"] = main_product if result["main_product_viable"] else None
                result["selection_reason"] = "Using sales staff recommended alternative product (viable)"
                logger.info("Selected alternative product (staff recommendation + viable)")

            elif result["main_product_viable"] and not result["has_alternative_recommendation"]:
                result["primary_product"] = main_product
                result["secondary_product"] = alternative_product if result["alternative_product_viable"] else None
                result["selection_reason"] = "Using main product (viable, no alternative recommendation)"
                logger.info("Selected main product (viable, no alternative)")

            elif result["alternative_product_viable"] and not result["main_product_viable"]:
                result["primary_product"] = alternative_product
                result["secondary_product"] = None
                result["selection_reason"] = "Using alternative product (main product not viable)"
                logger.info("Selected alternative product (main not viable)")

            elif result["main_product_viable"] and result["has_alternative_recommendation"]:
                result["primary_product"] = alternative_product
                result["secondary_product"] = main_product
                result["selection_reason"] = "Using sales staff recommended alternative (both viable)"
                logger.info("Selected alternative product (staff recommendation, both viable)")

            elif result["has_product_data"]:
                result["selection_reason"] = "Products found but not viable (EOL or discontinued)"
                logger.warning("Product data available but no viable products found")

            return result

        except Exception as e:
            logger.error(f"Error determining viable product: {e}")
            return {
                "primary_product": None,
                "secondary_product": None,
                "selection_reason": f"Error in product selection: {str(e)}",
                "main_product_viable": False,
                "alternative_product_viable": False,
                "has_alternative_recommendation": False,
                "has_product_data": False
            }

    def _is_product_viable(self, product: Dict) -> bool:
        """Check if a product is viable for pricing/stock information"""
        try:
            if not product or product.get('error'):
                logger.info("No product data available for viability check")
                return False

            # Check if product is EOL
            is_eol = product.get('is_eol', False)
            if is_eol:
                logger.info(f"Product {product.get('product_id')} is EOL")
                return False

            # Check if product is enabled
            is_enabled = product.get('is_enabled', False)
            if not is_enabled:
                logger.info(f"Product {product.get('product_id')} is not enabled")
                return False

            # Check lead time - but only for out-of-stock products
            is_in_stock = product.get('is_in_stock', False)
            if is_in_stock:
                logger.info(f"Product {product.get('product_id')} is in stock (viable)")
                return True

            # If not in stock, check lead time
            lead_time = product.get('lead_time')
            if lead_time is None or lead_time == "":
                logger.info(f"Product {product.get('product_id')} has no lead time and is out of stock (likely EOL)")
                return False

            # Has lead time and is enabled (viable even if out of stock)
            logger.info(f"Product {product.get('product_id')} has lead time '{lead_time}' (viable)")
            return True

        except Exception as e:
            logger.error(f"Error checking product viability: {e}")
            return False

    def _detect_external_comment_pattern(self, customer_comment: str) -> bool:
        """Detect if comment came from external method (explains missing product_id)"""
        external_patterns = [
            r'^I need assistance with my order\.',
            r'^I\'m looking for a specific component',
            r'^I need help with',
            r'^I\'m interested in',
        ]

        # Check for prefab intro + comment separator
        comment_separator_pattern = r'\n\nComment:\n\n'

        has_separator = bool(re.search(comment_separator_pattern, customer_comment))
        has_prefab_intro = any(re.match(pattern, customer_comment.strip(), re.IGNORECASE)
                               for pattern in external_patterns)

        return has_separator and has_prefab_intro

    def attempt_product_lookup_from_comment(self, customer_comment: str, customer_data: Dict) -> Dict:
        """
        ENHANCED: Attempt to find products mentioned in customer comment using AI
        Now uses the advanced AI product identification system instead of placeholder
        """
        try:
            # Only attempt if we don't have product_id or alternative_id
            product_id = customer_data.get('product_id')
            alternative_id = customer_data.get('alternative_id')

            if product_id or alternative_id:
                logger.info("Product IDs available - skipping AI product search")
                return {
                    "search_attempted": False,
                    "reason": "Product IDs already available",
                    "has_suggestions": False,
                    "ai_system_available": AI_PRODUCT_IDENTIFICATION_AVAILABLE
                }

            logger.info("No product IDs available - attempting AI product search from comment")

            # Check if AI system is available and healthy
            if AI_PRODUCT_IDENTIFICATION_AVAILABLE:
                system_health = get_system_health()
                if not system_health.get("system_ready", False):
                    logger.warning("AI system not ready - may need initialization")
                    return {
                        "search_attempted": False,
                        "reason": "AI system not ready - may need initialization",
                        "has_suggestions": False,
                        "ai_system_available": True,
                        "system_health": system_health,
                        "note": "Run initialize_product_identification_system() to set up the AI system"
                    }

            # Use the advanced AI product search
            search_results = search_comment_for_products(customer_comment, max_results=3)

            has_suggestions = search_results.get("search_successful", False)
            suggestions = []

            if has_suggestions:
                products_found = search_results.get("products_found", [])
                for product in products_found[:2]:  # Limit to top 2 suggestions
                    suggestion = {
                        "product_id": product.get("product_id"),
                        "name": product.get("name"),
                        "sku": product.get("sku"),
                        "brand": product.get("brand"),
                        "category": product.get("category"),
                        "relevance_score": product.get("relevance_score"),
                        "is_in_stock": product.get("is_in_stock"),
                        "current_price": product.get("current_price"),
                        "match_reason": product.get("match_reason", "AI_similarity")
                    }
                    suggestions.append(suggestion)

            return {
                "search_attempted": True,
                "ai_system_available": AI_PRODUCT_IDENTIFICATION_AVAILABLE,
                "search_results": search_results,
                "has_suggestions": has_suggestions,
                "suggestions": suggestions,
                "best_match": search_results.get("best_match"),
                "confidence": search_results.get("search_summary", {}).get("extraction_confidence", 0),
                "search_terms": search_results.get("search_summary", {}).get("search_terms_used", []),
                "extraction_method": search_results.get("search_summary", {}).get("extraction_method", "unknown"),
                "processing_time": search_results.get("search_summary", {}).get("processing_time", 0),
                "note": "AI-powered product identification using embeddings and vector search"
            }

        except Exception as e:
            logger.error(f"Error in AI product lookup from comment: {e}")
            return {
                "search_attempted": True,
                "ai_system_available": AI_PRODUCT_IDENTIFICATION_AVAILABLE,
                "error": str(e),
                "has_suggestions": False,
                "suggestions": [],
                "note": "AI product search failed - check system health and initialization"
            }

    def build_context_with_real_data(self, customer_comment: str, intent: str, product_details: Dict, examples: List, data_needs: Dict, customer_data: Dict) -> Dict:
        """ENHANCED: Build context with FIXED product viability logic and external comment detection"""
        try:
            # Detect if this is an external comment
            is_external_comment = self._detect_external_comment_pattern(customer_comment)

            context = {
                "customer_comment": customer_comment,
                "predicted_intent": intent,
                "data_needs": data_needs,
                "prioritization_strategy": "standard",
                "is_external_comment": is_external_comment
            }

            # Determine viable product selection with FIXED logic
            product_selection = self.determine_viable_product(product_details, customer_data)
            context["product_selection"] = product_selection

            # FIXED: Don't prioritize real-time data if no product data available
            if data_needs.get("needs_real_time_data", False) and product_selection.get("has_product_data", False):
                context["prioritization_strategy"] = "real_time_priority"
                context["real_time_data"] = self._extract_real_time_data_with_viability(
                    product_selection, data_needs
                )
                context["examples"] = examples[:1] if examples else []
                context["examples_usage"] = "tone_reference_only"
                logger.info(f"Using real-time data prioritization with {product_selection['selection_reason']}")

            elif data_needs.get("needs_real_time_data", False) and not product_selection.get("has_product_data", False):
                # FIXED: Need real-time data but don't have it - fall back to examples
                context["prioritization_strategy"] = "examples_with_lookup_suggestion"
                context["examples"] = examples[:3]  # Use more examples
                context["examples_usage"] = "full_context_with_lookup_note"
                context["real_time_data"] = {"note": "No product data available - may need lookup"}
                logger.info("Real-time data needed but not available - using examples with lookup suggestion")

            else:
                context["prioritization_strategy"] = "example_priority"
                context["examples"] = examples[:3]
                context["examples_usage"] = "full_context"
                context["real_time_data"] = self._extract_real_time_data_with_viability(
                    product_selection, data_needs, include_all=False
                )
                logger.info("Using standard example-based prioritization strategy")

            # Always include product details but mark their availability
            context["product_details"] = product_details
            context["context_timestamp"] = datetime.now().isoformat()

            return context

        except Exception as e:
            logger.error(f"Error building context with real data: {e}")
            return {
                "customer_comment": customer_comment,
                "predicted_intent": intent,
                "examples": examples[:2],  # Fallback
                "product_details": product_details,
                "error": str(e),
                "prioritization_strategy": "fallback"
            }

    def _extract_real_time_data_with_viability(self, product_selection: Dict, data_needs: Dict, include_all: bool = True) -> Dict:
        """Extract and format real-time data based on viable product selection"""
        real_time_data = {
            "primary_product": {},
            "secondary_product": {},
            "product_selection_info": product_selection,
            "data_freshness": datetime.now().isoformat()
        }

        # Process primary (viable) product
        primary_product = product_selection.get("primary_product")
        if primary_product and not primary_product.get("error"):
            real_time_data["primary_product"] = self._format_product_real_time_data(
                primary_product, data_needs, include_all
            )

        # Process secondary product if available
        secondary_product = product_selection.get("secondary_product")
        if secondary_product and not secondary_product.get("error"):
            real_time_data["secondary_product"] = self._format_product_real_time_data(
                secondary_product, data_needs, include_all
            )

        return real_time_data

    def _format_product_real_time_data(self, product: Dict, data_needs: Dict, include_all: bool = True) -> Dict:
        """Format product data focusing on real-time aspects"""
        formatted_data = {
            "product_name": product.get("name"),
            "sku": product.get("sku"),
            "product_id": product.get("product_id"),
            "is_viable": self._is_product_viable(product),
            "viability_reason": self._get_viability_reason(product)
        }

        # Only include pricing/stock data if product is viable
        if formatted_data["is_viable"]:
            # Always include pricing if needed or if include_all is True
            if data_needs.get("needs_pricing", False) or include_all:
                formatted_data["pricing"] = {
                    "current_price": product.get("current_price"),
                    "special_price": product.get("special_price"),
                    "is_on_promotion": product.get("is_on_promotion", False),
                    "price_updated": product.get("updated")
                }

            # Always include stock if needed or if include_all is True
            if data_needs.get("needs_stock", False) or include_all:
                formatted_data["stock"] = {
                    "is_in_stock": product.get("is_in_stock", False),
                    "stock_quantity": product.get("stock_quantity", 0),
                    "availability": product.get("availability"),
                    "lead_time": product.get("lead_time"),
                    "eta": product.get("eta"),
                    "expected_dispatch": product.get("expected_dispatch"),
                    "stock_updated": product.get("updated")
                }
        else:
            # For non-viable products, indicate why they shouldn't be used
            formatted_data["warning"] = "Product not viable for current pricing/stock information"

        return formatted_data

    def _get_viability_reason(self, product: Dict) -> str:
        """Get human-readable reason for product viability status"""
        is_eol = product.get('is_eol', False)
        is_enabled = product.get('is_enabled', False)
        lead_time = product.get('lead_time')
        is_in_stock = product.get('is_in_stock', False)

        if is_eol:
            return "Product is End of Life (EOL)"
        if not is_enabled:
            return "Product is not enabled"
        if lead_time is None or lead_time == "":
            return "No lead time available (likely discontinued)"
        if is_in_stock:
            return "Product is in stock"
        if lead_time and lead_time.strip():
            return f"Product available with lead time: {lead_time}"

        return "Unknown viability status"

    def validate_request_id(self, request_id: str) -> Dict:
        """Validate that a request ID is properly formatted"""
        try:
            # Basic validation
            if not request_id:
                return {
                    "valid": False,
                    "error": "Request ID is empty",
                    "error_type": "empty_id"
                }

            if not str(request_id).strip():
                return {
                    "valid": False,
                    "error": "Request ID contains only whitespace",
                    "error_type": "whitespace_id"
                }

            # Try to convert to int to validate it's numeric
            try:
                int(request_id)
            except ValueError:
                return {
                    "valid": False,
                    "error": f"Request ID '{request_id}' is not a valid number",
                    "error_type": "invalid_format"
                }

            return {
                "valid": True,
                "cleaned_id": str(request_id).strip()
            }

        except Exception as e:
            logger.error(f"Error validating request ID '{request_id}': {e}")
            return {
                "valid": False,
                "error": f"Validation error: {str(e)}",
                "error_type": "validation_exception"
            }

    def get_request_data(self, request_id: str) -> Dict:
        """
        Get and validate customer request data

        Args:
            request_id: The request ID to fetch

        Returns:
            Dict with request data and validation info
        """
        try:
            # Validate request ID first
            validation = self.validate_request_id(request_id)
            if not validation["valid"]:
                return {
                    "status": "invalid_request_id",
                    "error": validation["error"],
                    "error_type": validation["error_type"],
                    "request_id": request_id
                }

            cleaned_id = validation["cleaned_id"]

            # Get customer data from database
            db_result = get_customer_request_data(cleaned_id)

            if db_result["status"] != "success":
                return db_result

            customer_data = db_result["data"]

            # Validate customer comment exists
            customer_comment = customer_data.get('customer_comment', '')
            if not customer_comment or not customer_comment.strip():
                return {
                    "status": "no_customer_comment",
                    "error": "No customer comment found for this request",
                    "request_id": cleaned_id,
                    "data": customer_data
                }

            # Add processed timestamp
            db_result["processed_timestamp"] = datetime.now().isoformat()
            db_result["request_id_validated"] = cleaned_id

            return db_result

        except Exception as e:
            logger.error(f"Error getting request data for ID {request_id}: {e}")
            return {
                "status": "processing_error",
                "error": f"Error processing request: {str(e)}",
                "request_id": request_id
            }

    def check_scope_and_intent(self, customer_comment: str) -> Dict:
        """
        Check intent and scope for a customer comment

        Args:
            customer_comment: The customer's comment

        Returns:
            Dict with intent and scope information
        """
        try:
            # Get intent and scope check
            intent_result = check_intent_scope(customer_comment)

            predicted_intent = intent_result.get('predicted_intent')
            scope_check = intent_result.get('scope_check', {})

            # Check if intent is out of scope
            if scope_check.get('is_out_of_scope', False):
                return {
                    "status": "out_of_scope",
                    "predicted_intent": predicted_intent,
                    "message": "This Query is outside of scope",
                    "reason": scope_check.get('reason', f"Intent '{predicted_intent}' is not supported"),
                    "supported_intents": scope_check.get('supported_intents', []),
                    "scope_check": scope_check
                }

            return {
                "status": "in_scope",
                "predicted_intent": predicted_intent,
                "scope_check": scope_check,
                "message": "Intent is within scope for processing"
            }

        except Exception as e:
            logger.error(f"Error checking intent/scope for comment '{customer_comment[:50]}...': {e}")
            return {
                "status": "intent_error",
                "error": f"Error checking intent: {str(e)}",
                "predicted_intent": "General Inquiry"  # Fallback
            }

    def get_product_context(self, customer_data: Dict) -> Dict:
        """
        Get product context from backend database

        Args:
            customer_data: Customer data from main database

        Returns:
            Dict with product information
        """
        try:
            product_details = get_all_products_for_request(customer_data)

            # Log what products were found
            products_found = product_details.get('products_found', [])
            logger.info(f"Product context: Found {len(products_found)} products: {products_found}")

            return {
                "status": "success",
                "product_details": product_details,
                "has_main_product": "main_product" in products_found,
                "has_alternative_product": "alternative_product" in products_found,
                "products_found_count": len(products_found)
            }

        except Exception as e:
            logger.error(f"Error getting product context: {e}")
            return {
                "status": "product_error",
                "error": f"Error getting product context: {str(e)}",
                "product_details": {
                    "main_product": None,
                    "alternative_product": None,
                    "products_found": []
                }
            }

    def get_similar_responses_context(self, customer_comment: str) -> Dict:
        """
        Get similar responses context from Pinecone

        Args:
            customer_comment: Customer's comment

        Returns:
            Dict with similar responses and examples
        """
        try:
            # Get comprehensive search results
            search_result = search_with_full_context(customer_comment, top_k=3)

            return {
                "status": "success",
                "search_result": search_result,
                "similar_responses_found": search_result.get('similar_responses_found', 0),
                "has_examples": search_result.get('has_labeled_context', False),
                "examples_count": len(search_result.get('response_examples', []))
            }

        except Exception as e:
            logger.error(f"Error getting similar responses for '{customer_comment[:50]}...': {e}")
            return {
                "status": "search_error",
                "error": f"Error searching similar responses: {str(e)}",
                "search_result": {
                    "similar_responses_found": 0,
                    "response_examples": []
                }
            }

    def get_existing_response_context(self, request_id: str) -> Dict:
        """
        Get existing custom response if available

        Args:
            request_id: The request ID to check

        Returns:
            Dict with existing response information
        """
        try:
            custom_response = get_custom_response_data(request_id)

            if custom_response:
                return {
                    "status": "found",
                    "has_existing_response": True,
                    "existing_response": custom_response.get('cleaned_text_body', ''),
                    "existing_response_length": custom_response.get('body_length', 0),
                    "raw_html": custom_response.get('raw_html_body', '')
                }
            else:
                return {
                    "status": "not_found",
                    "has_existing_response": False,
                    "existing_response": None,
                    "existing_response_length": 0
                }

        except Exception as e:
            logger.error(f"Error getting existing response for ID {request_id}: {e}")
            return {
                "status": "error",
                "error": f"Error checking existing response: {str(e)}",
                "has_existing_response": False
            }

    def process_full_request(self, request_id: str, generate_response: bool = True) -> Dict:
        """
        Process a complete customer request with ENHANCED product search capabilities
        """
        try:
            logger.info(f"Processing full request for ID: {request_id}")

            # 1. Get and validate request data
            request_result = self.get_request_data(request_id)
            if request_result["status"] != "success":
                return request_result

            customer_data = request_result["data"]
            customer_comment = customer_data.get('customer_comment', '')

            # EARLY EXIT 1: Check automated_response field FIRST
            automated_response = customer_data.get('automated_response', 0)
            if automated_response != 0:
                logger.info(f"EARLY EXIT: Request {request_id} has automated_response={automated_response}")
                return {
                    "status": "out_of_scope",
                    "request_id": request_id,
                    "customer_comment": customer_comment,
                    "early_exit": True,
                    "exit_reason": "automated_response_flag",
                    "automated_response_flag": automated_response,
                    "message": f"Query marked as automated_response={automated_response} - no response generated",
                    "processing_timestamp": datetime.now().isoformat()
                }

            # 2. Check intent and scope
            intent_result = self.check_scope_and_intent(customer_comment)

            # EARLY EXIT 2: Check if intent is out of scope
            if intent_result["status"] == "out_of_scope":
                logger.info(
                    f"EARLY EXIT: Request {request_id} is out of scope - intent: {intent_result.get('predicted_intent')}")
                return {
                    "status": "out_of_scope",
                    "request_id": request_id,
                    "customer_comment": customer_comment,
                    "intent_result": intent_result,
                    "request_data": request_result,
                    "early_exit": True,
                    "exit_reason": "intent_out_of_scope",
                    "message": "Query is outside of scope - no response generated",
                    "processing_timestamp": datetime.now().isoformat()
                }

            # If we reach here, query is IN SCOPE - continue with normal processing
            predicted_intent = intent_result.get("predicted_intent")
            logger.info(f"Request {request_id} is IN SCOPE - continuing with intent: {predicted_intent}")

            # 3. ENHANCED: Detect data needs using FIXED dual-layer detection
            data_needs = self.detect_data_needs(customer_comment, predicted_intent)
            logger.info(f"Data needs detected: {data_needs.get('detection_reason')}")

            # 4. Get product context
            product_result = self.get_product_context(customer_data)

            # 4.5. ENHANCED: Attempt AI product search from comment if no product data
            product_search_result = self.attempt_product_lookup_from_comment(customer_comment, customer_data)

            # 5. Get similar responses context
            similar_responses_result = self.get_similar_responses_context(customer_comment)

            # 6. Get existing response context
            existing_response_result = self.get_existing_response_context(request_id)

            # 7. ENHANCED: Build context with real-time data prioritization and product viability
            examples = similar_responses_result.get('search_result', {}).get('response_examples', [])
            enhanced_context = self.build_context_with_real_data(
                customer_comment, predicted_intent,
                product_result.get('product_details', {}),
                examples, data_needs, customer_data
            )

            # 7.5. ENHANCED: Add AI product search results to context
            enhanced_context["product_search_result"] = product_search_result

            # 8. Generate AI response if requested with enhanced context and viability logic
            generated_response = None
            generation_method = None

            if generate_response:
                try:
                    # Extract rep name for signature
                    woot_rep = customer_data.get('woot_rep', '').strip()
                    logger.info(f"Generating response with rep signature: {woot_rep}")

                    # FIXED: Use intent-based generation strategy
                    prompt_strategy = data_needs.get("prompt_strategy", "general_helpful")

                    if prompt_strategy in ["warranty_focused", "pricing_focused", "stock_focused",
                                           "combined_pricing_stock"]:
                        # Use enhanced context for specialized prompts
                        generated_response = generate_enhanced_response_with_context(
                            customer_comment, enhanced_context, woot_rep)
                        generation_method = f"enhanced_{prompt_strategy}"

                        logger.info(f"Generated response using enhanced {prompt_strategy} strategy")

                    elif examples:
                        # Use examples for general queries
                        generated_response = generate_response_with_examples(
                            customer_comment, examples, woot_rep
                        )
                        generation_method = "examples_based"
                        logger.info("Generated response using examples method")

                    else:
                        # Fallback to simple generation
                        generated_response = generate_simple_response_for_comment(
                            customer_comment, predicted_intent, woot_rep
                        )
                        generation_method = "simple_fallback"
                        logger.info("Generated response using simple fallback method")

                except Exception as e:
                    logger.error(f"Error generating response: {e}")
                    woot_rep = customer_data.get('woot_rep', '').strip()
                    fallback_response = "I apologize, but I'm having trouble generating a response right now. Please contact our sales team directly."

                    # Even fallback responses get rep signature
                    if woot_rep:
                        generated_response = f"{fallback_response}\n\nKind Regards,\n{woot_rep}"
                    else:
                        generated_response = f"{fallback_response}\n\nKind Regards,\nWootware Sales Team"
                    generation_method = "error_fallback"

            # 9. Build comprehensive result with ENHANCED product search tracking
            result = {
                "status": "success",
                "request_id": request_id,
                "customer_comment": customer_comment,
                "request_data": request_result,
                "intent_result": intent_result,
                "data_needs_analysis": data_needs,
                "enhanced_context": enhanced_context,
                "product_result": product_result,
                "product_search_result": product_search_result,  # ENHANCED
                "similar_responses_result": similar_responses_result,
                "existing_response_result": existing_response_result,
                "generated_response": generated_response,
                "generation_method": generation_method,
                "processing_summary": {
                    "scope_warning": request_result.get("scope_warning"),
                    "predicted_intent": predicted_intent,
                    "prompt_strategy": data_needs.get("prompt_strategy"),
                    "needs_real_time_data": data_needs.get("needs_real_time_data", False),
                    "needs_pricing": data_needs.get("needs_pricing", False),
                    "needs_stock": data_needs.get("needs_stock", False),
                    "prioritization_strategy": enhanced_context.get("prioritization_strategy"),
                    "product_selection": enhanced_context.get("product_selection", {}),
                    "products_found": product_result.get("products_found_count", 0),
                    "product_search_attempted": product_search_result.get("search_attempted", False),
                    "product_suggestions_found": product_search_result.get("has_suggestions", False),
                    "similar_responses_found": similar_responses_result.get("similar_responses_found", 0),
                    "has_existing_response": existing_response_result.get("has_existing_response", False),
                    "response_generated": generated_response is not None,
                    "early_exit": False ,
                    "ai_product_search_attempted": product_search_result.get("search_attempted", False),  # ENHANCED
                    "ai_product_suggestions_found": product_search_result.get("has_suggestions", False),  # ENHANCED
                    "ai_system_available": product_search_result.get("ai_system_available", False),  # ENHANCED
                    # Made it through all processing
                },
                "processing_timestamp": datetime.now().isoformat()
            }

            logger.info(
                f"Successfully processed request {request_id} with strategy: {data_needs.get('prompt_strategy')}")
            return result

        except Exception as e:
            logger.error(f"Error in full request processing for ID {request_id}: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")

            return {
                "status": "processing_error",
                "request_id": request_id,
                "error": f"Processing error: {str(e)}",
                "early_exit": False,
                "processing_timestamp": datetime.now().isoformat()
            }


# Global processor instance (lazy-loaded)
_data_processor_instance = None


def get_data_processor() -> DataProcessor:
    """
    Get a global data processor instance (singleton pattern)

    Returns:
        DataProcessor instance
    """
    global _data_processor_instance
    if _data_processor_instance is None:
        _data_processor_instance = DataProcessor()
    return _data_processor_instance


# Convenience functions
def validate_request_data(request_id: str) -> Dict:
    """
    Convenience function to validate request data

    Args:
        request_id: The request ID to validate

    Returns:
        Dict with validation results
    """
    processor = get_data_processor()
    return processor.get_request_data(request_id)


def process_customer_request(request_id: str, generate_response: bool = True) -> Dict:
    """
    Convenience function to process a complete customer request with enhanced real-time data and product viability

    Args:
        request_id: The request ID to process
        generate_response: Whether to generate an AI response

    Returns:
        Dict with complete processing results
    """
    processor = get_data_processor()
    return processor.process_full_request(request_id, generate_response)


def get_request_summary(request_id: str) -> Dict:
    """
    Convenience function to get a quick summary of a request

    Args:
        request_id: The request ID to summarize

    Returns:
        Dict with request summary
    """
    processor = get_data_processor()

    # Get basic request data
    request_result = processor.get_request_data(request_id)
    if request_result["status"] != "success":
        return request_result

    customer_data = request_result["data"]
    customer_comment = customer_data.get('customer_comment', '')

    # Get intent
    intent_result = processor.check_scope_and_intent(customer_comment)

    return {
        "status": "success",
        "request_id": request_id,
        "customer_comment": customer_comment[:100] + "..." if len(customer_comment) > 100 else customer_comment,
        "predicted_intent": intent_result.get("predicted_intent"),
        "is_in_scope": intent_result.get("status") == "in_scope",
        "product_id": customer_data.get('product_id'),
        "alternative_id": customer_data.get('alternative_id'),
        "woot_rep": customer_data.get('woot_rep'),
        "summary_timestamp": datetime.now().isoformat()
    }


# NEW: Convenience function for data needs detection
def detect_query_data_needs(customer_comment: str, predicted_intent: str) -> Dict:
    """
    Convenience function to detect data needs for a query

    Args:
        customer_comment: Customer's comment
        predicted_intent: Predicted intent

    Returns:
        Dict with data needs detection results
    """
    processor = get_data_processor()
    return processor.detect_data_needs(customer_comment, predicted_intent)


# NEW: Convenience function for product viability check
def determine_viable_product_for_request(customer_data: Dict, product_details: Dict) -> Dict:
    """
    Convenience function to determine viable product for a request

    Args:
        customer_data: Customer data from main database
        product_details: Product data from backend

    Returns:
        Dict with viable product selection results
    """
    processor = get_data_processor()
    return processor.determine_viable_product(product_details, customer_data)


# NEW: Convenience functions for AI product identification system management
def initialize_ai_product_identification(force_rebuild: bool = False) -> Dict:
    """
    ENHANCED: Initialize the AI product identification system

    Args:
        force_rebuild: Whether to rebuild even if data exists

    Returns:
        Dict with initialization results
    """
    if not AI_PRODUCT_IDENTIFICATION_AVAILABLE:
        return {
            "success": False,
            "message": "AI Product Identification system not available",
            "error": "Module not imported correctly"
        }

    try:
        return initialize_product_identification_system(force_rebuild=force_rebuild)
    except Exception as e:
        logger.error(f"Failed to initialize AI product identification: {e}")
        return {
            "success": False,
            "message": f"Initialization failed: {str(e)}",
            "error": str(e)
        }


def get_ai_system_health() -> Dict:
    """
    ENHANCED: Get health status of AI product identification system

    Returns:
        Dict with system health information
    """
    if not AI_PRODUCT_IDENTIFICATION_AVAILABLE:
        return {
            "ai_system_available": False,
            "system_ready": False,
            "error": "AI Product Identification module not available"
        }

    try:
        health = get_system_health()
        health["ai_system_available"] = True
        return health
    except Exception as e:
        logger.error(f"Failed to get AI system health: {e}")
        return {
            "ai_system_available": True,
            "system_ready": False,
            "error": str(e)
        }