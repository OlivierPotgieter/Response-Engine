"""
Response Builder Module - Enhanced with Real-Time Data Tracking
Handles response formatting, structure, and presentation with metadata for real-time data usage.
"""

import logging
from typing import Dict
from datetime import datetime

logger = logging.getLogger(__name__)


class ResponseBuilder:
    def __init__(self):
        """Initialize the enhanced response builder"""
        self.api_version = "1.2.0"  # Incremented for fixed real-time data features

    def build_success_response(self, data: Dict, message: str = None) -> Dict:
        """
        Build a standardized success response

        Args:
            data: The response data
            message: Optional success message

        Returns:
            Formatted success response
        """
        response = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "api_version": self.api_version,
            "data": data,
        }

        if message:
            response["message"] = message

        return response

    def build_error_response(
        self,
        error: str,
        error_type: str = "general_error",
        status_code: int = 500,
        details: Dict = None,
    ) -> Dict:
        """
        Build a standardized error response

        Args:
            error: Error message
            error_type: Type of error
            status_code: HTTP status code
            details: Additional error details

        Returns:
            Formatted error response
        """
        response = {
            "status": "error",
            "error": error,
            "error_type": error_type,
            "status_code": status_code,
            "timestamp": datetime.now().isoformat(),
            "api_version": self.api_version,
        }

        if details:
            response["details"] = details

        return response

    def build_out_of_scope_response(self, request_id: str, intent_result: Dict) -> Dict:
        """
        Build response for out-of-scope requests

        Args:
            request_id: The request ID
            intent_result: Intent analysis results

        Returns:
            Formatted out-of-scope response
        """
        return {
            "status": "out_of_scope",
            "request_id": request_id,
            "message": "This Query is outside of scope",
            "predicted_intent": intent_result.get("predicted_intent"),
            "reason": intent_result.get("reason"),
            "supported_intents": intent_result.get("supported_intents", []),
            "timestamp": datetime.now().isoformat(),
            "api_version": self.api_version,
        }

    def _extract_real_time_data_metadata(self, processing_result: Dict) -> Dict:
        """
        Extract metadata about real-time data usage for tracking and validation

        Args:
            processing_result: Complete processing results

        Returns:
            Real-time data usage metadata
        """
        metadata = {
            "real_time_data_used": False,
            "data_sources_prioritized": [],
            "pricing_data_source": "none",
            "stock_data_source": "none",
            "data_freshness": None,
            "detection_method": "none",
            "prioritization_strategy": "standard",
        }

        # Extract data needs analysis
        data_needs = processing_result.get("data_needs_analysis", {})
        if data_needs:
            metadata["real_time_data_used"] = data_needs.get(
                "needs_real_time_data", False
            )
            metadata["detection_method"] = data_needs.get("detection_reason", "Unknown")

            if data_needs.get("needs_pricing", False):
                metadata["pricing_data_source"] = "real_time_backend"
                metadata["data_sources_prioritized"].append("pricing")

            if data_needs.get("needs_stock", False):
                metadata["stock_data_source"] = "real_time_backend"
                metadata["data_sources_prioritized"].append("stock")

        # Extract enhanced context info
        enhanced_context = processing_result.get("enhanced_context", {})
        if enhanced_context:
            metadata["prioritization_strategy"] = enhanced_context.get(
                "prioritization_strategy", "standard"
            )

            real_time_data = enhanced_context.get("real_time_data", {})
            if real_time_data:
                metadata["data_freshness"] = real_time_data.get("data_freshness")

        return metadata

    def _build_data_source_summary(self, processing_result: Dict) -> Dict:
        """
        Build summary of data sources used in response generation

        Args:
            processing_result: Complete processing results

        Returns:
            Data source usage summary
        """
        summary = {
            "examples_used": 0,
            "examples_purpose": "none",
            "product_data_used": False,
            "real_time_pricing": False,
            "real_time_stock": False,
            "generation_method": processing_result.get("generation_method", "unknown"),
        }

        # Check enhanced context for data usage
        enhanced_context = processing_result.get("enhanced_context", {})
        if enhanced_context:
            examples = enhanced_context.get("examples", [])
            summary["examples_used"] = len(examples)
            summary["examples_purpose"] = enhanced_context.get("examples_usage", "none")

            real_time_data = enhanced_context.get("real_time_data", {})
            if real_time_data:
                primary_product = real_time_data.get("primary_product", {})
                if primary_product.get("pricing"):
                    summary["real_time_pricing"] = True
                if primary_product.get("stock"):
                    summary["real_time_stock"] = True

                summary["product_data_used"] = bool(primary_product)

        # Fallback to check similar responses
        similar_responses = processing_result.get("similar_responses_result", {})
        if similar_responses and not summary["examples_used"]:
            examples = similar_responses.get("search_result", {}).get(
                "response_examples", []
            )
            summary["examples_used"] = len(examples)
            summary["examples_purpose"] = "full_context" if examples else "none"

        return summary

    def _determine_fallback_reason(self, processing_result: Dict) -> str:
        """
        NEW: Determine why fallback generation was used

        Args:
            processing_result: Complete processing results

        Returns:
            Human-readable fallback reason
        """
        generation_method = processing_result.get("generation_method", "unknown")

        if "error_fallback" in generation_method:
            return "Error occurred during response generation"
        elif "simple_fallback" in generation_method:
            return "No examples available, used simple generation"
        elif "examples_with_lookup_suggestion" in processing_result.get(
            "enhanced_context", {}
        ).get("prioritization_strategy", ""):
            return "Real-time data needed but not available"
        elif (
            not processing_result.get("similar_responses_result", {})
            .get("search_result", {})
            .get("response_examples", [])
        ):
            return "No similar responses found in Pinecone"
        else:
            return "Standard generation method used"

    def build_test_response(self, request_id: str, processing_result: Dict) -> Dict:
        """
        ENHANCED: Build response for test endpoint with early exit handling

        Args:
            request_id: The request ID
            processing_result: Results from data processing

        Returns:
            Formatted test response with enhanced metadata
        """
        # Handle early exits (NEW)
        if processing_result.get("early_exit", False):
            exit_reason = processing_result.get("exit_reason", "unknown")

            if exit_reason == "automated_response_flag":
                return {
                    "status": "out_of_scope",
                    "request_id": request_id,
                    "early_exit": True,
                    "exit_reason": "automated_response_flag",
                    "automated_response_value": processing_result.get(
                        "automated_response_flag"
                    ),
                    "message": "Request marked for automated response - exited early without processing",
                    "processing_note": "No Pinecone searches or OpenAI calls were made",
                    "cost_savings": "Saved API costs by early exit",
                    "timestamp": datetime.now().isoformat(),
                    "api_version": self.api_version,
                }

            elif exit_reason == "intent_out_of_scope":
                return self.build_out_of_scope_response(
                    request_id, processing_result.get("intent_result", {})
                )

        # Handle different processing statuses
        if processing_result["status"] == "out_of_scope":
            return self.build_out_of_scope_response(
                request_id, processing_result.get("intent_result", {})
            )

        if processing_result["status"] != "success":
            return self.build_error_response(
                processing_result.get("error", "Processing failed"),
                processing_result.get("status", "processing_error"),
                details={"request_id": request_id},
            )

        # Extract data for enhanced test response
        request_data = processing_result.get("request_data", {})
        intent_result = processing_result.get("intent_result", {})
        data_needs_analysis = processing_result.get("data_needs_analysis", {})
        enhanced_context = processing_result.get("enhanced_context", {})
        product_result = processing_result.get("product_result", {})

        existing_response_result = processing_result.get("existing_response_result", {})
        processing_summary = processing_result.get("processing_summary", {})

        # Build key fields summary
        customer_data = request_data.get("data", {})
        key_fields = {
            "customer_comment": customer_data.get("customer_comment"),
            "product_id": customer_data.get("product_id"),
            "product_name": customer_data.get("product_name"),
            "parent_leadtime": customer_data.get("parent_leadtime"),
            "alternative_id": customer_data.get("alternative_id"),
            "alternative_name": customer_data.get("alternative_name"),
            "alternative_leadtime": customer_data.get("alternative_leadtime"),
            "woot_rep": customer_data.get("woot_rep"),
        }

        # Build enhanced intent scope check
        scope_check = intent_result.get("scope_check", {})
        intent_scope_check = {
            "is_out_of_scope": scope_check.get("is_out_of_scope", False),
            "message": intent_result.get("message", "Unknown"),
            "predicted_intent": intent_result.get("predicted_intent"),
        }

        # ENHANCED: Build real-time data detection summary with fixed logic
        data_detection_summary = {
            "needs_real_time_data": data_needs_analysis.get(
                "needs_real_time_data", False
            ),
            "needs_pricing": data_needs_analysis.get("needs_pricing", False),
            "needs_stock": data_needs_analysis.get("needs_stock", False),
            "detection_method": data_needs_analysis.get(
                "detection_reason", "No detection performed"
            ),
            "detection_layers": data_needs_analysis.get("detection_layers", {}),
            "prompt_strategy": data_needs_analysis.get(
                "prompt_strategy", "general_helpful"
            ),
            "prioritization_strategy": enhanced_context.get(
                "prioritization_strategy", "standard"
            ),
        }

        # ENHANCED: Build real-time data availability summary with product data tracking
        real_time_data_availability = {
            "has_product_data": False,
            "main_product_available": False,
            "alternative_product_available": False,
            "pricing_data_fresh": False,
            "stock_data_fresh": False,
            "data_timestamp": None,
            "is_external_comment": enhanced_context.get("is_external_comment", False),
        }

        real_time_data = enhanced_context.get("real_time_data", {})
        product_selection = enhanced_context.get("product_selection", {})

        if product_selection:
            real_time_data_availability["has_product_data"] = product_selection.get(
                "has_product_data", False
            )

        if real_time_data:
            primary_product_data = real_time_data.get("primary_product", {})
            secondary_product_data = real_time_data.get("secondary_product", {})

            real_time_data_availability.update(
                {
                    "main_product_available": bool(primary_product_data),
                    "alternative_product_available": bool(secondary_product_data),
                    "pricing_data_fresh": bool(primary_product_data.get("pricing")),
                    "stock_data_fresh": bool(primary_product_data.get("stock")),
                    "data_timestamp": real_time_data.get("data_freshness"),
                }
            )

        # NEW: Add product search results summary
        product_search_summary = {
            "search_attempted": False,
            "has_suggestions": False,
            "confidence": 0,
            "search_terms": [],
        }

        product_search_result = processing_result.get("product_search_result", {})
        if product_search_result:
            product_search_summary.update(
                {
                    "search_attempted": product_search_result.get(
                        "search_attempted", False
                    ),
                    "has_suggestions": product_search_result.get(
                        "has_suggestions", False
                    ),
                    "confidence": product_search_result.get("confidence", 0),
                    "search_terms": product_search_result.get("search_terms", []),
                }
            )

        # Build product details summary with viability info
        product_details = product_result.get("product_details", {})

        # Build existing response summary
        existing_response_summary = {
            "found": existing_response_result.get("has_existing_response", False),
            "response": existing_response_result.get("existing_response"),
            "length": existing_response_result.get("existing_response_length", 0),
        }

        test_data = {
            "request_id": request_id,
            "scope_warning": request_data.get("scope_warning"),
            "data_log": request_data.get("data_log", {}),
            "predicted_intent": intent_result.get("predicted_intent"),
            "intent_scope_check": intent_scope_check,
            "data_detection_summary": data_detection_summary,
            "real_time_data_availability": real_time_data_availability,
            "product_search_summary": product_search_summary,  # NEW
            "product_details": product_details,
            "existing_custom_response": existing_response_summary,
            "all_database_fields": customer_data,
            "key_fields_summary": key_fields,
            "processing_summary": processing_summary,
            "enhanced_features": {
                "real_time_data_integration": "active",
                "dual_layer_detection": "enhanced_with_regex",
                "smart_prioritization": "intent_based",
                "eol_logic": "fixed",
                "external_comment_detection": "active",
                "product_search": "placeholder_active",
                "early_exit_optimization": "active",
            },
            "testing_note": "This is a TEST endpoint with FIXED real-time data detection - no LLM calls or expensive operations were performed",
        }

        return self.build_success_response(
            test_data, "Enhanced test processing completed successfully"
        )

    def build_process_response(self, request_id: str, processing_result: Dict) -> Dict:
        """
        ENHANCED: Build response for full processing endpoint with early exit and fallback tracking

        Args:
            request_id: The request ID
            processing_result: Results from full processing

        Returns:
            Formatted process response with enhanced metadata
        """
        # Handle early exits (NEW)
        if processing_result.get("early_exit", False):
            exit_reason = processing_result.get("exit_reason", "unknown")

            if exit_reason == "automated_response_flag":
                return {
                    "status": "out_of_scope",
                    "request_id": request_id,
                    "early_exit": True,
                    "exit_reason": "automated_response_flag",
                    "automated_response_value": processing_result.get(
                        "automated_response_flag"
                    ),
                    "message": "Request marked for automated response - exited early without processing",
                    "cost_savings": "Avoided Pinecone searches and OpenAI API calls",
                    "timestamp": datetime.now().isoformat(),
                    "api_version": self.api_version,
                }

            elif exit_reason == "intent_out_of_scope":
                return self.build_out_of_scope_response(
                    request_id, processing_result.get("intent_result", {})
                )

        # Handle different processing statuses
        if processing_result["status"] == "out_of_scope":
            return self.build_out_of_scope_response(
                request_id, processing_result.get("intent_result", {})
            )

        if processing_result["status"] != "success":
            return self.build_error_response(
                processing_result.get("error", "Processing failed"),
                processing_result.get("status", "processing_error"),
                details={"request_id": request_id},
            )

        # Extract all processing results
        request_data = processing_result.get("request_data", {})
        intent_result = processing_result.get("intent_result", {})
        data_needs_analysis = processing_result.get("data_needs_analysis", {})
        enhanced_context = processing_result.get("enhanced_context", {})
        product_result = processing_result.get("product_result", {})

        existing_response_result = processing_result.get("existing_response_result", {})
        processing_summary = processing_result.get("processing_summary", {})

        customer_data = request_data.get("data", {})

        # Build key fields
        key_fields = {
            "customer_comment": customer_data.get("customer_comment"),
            "product_id": customer_data.get("product_id"),
            "product_name": customer_data.get("product_name"),
            "parent_leadtime": customer_data.get("parent_leadtime"),
            "alternative_id": customer_data.get("alternative_id"),
            "alternative_name": customer_data.get("alternative_name"),
            "alternative_leadtime": customer_data.get("alternative_leadtime"),
            "woot_rep": customer_data.get("woot_rep"),
        }

        # Build Pinecone results summary
        pinecone_results = {
            "similar_responses_found": 0,
            "top_matches": [],
            "response_examples": [],
        }

        # ENHANCED: Extract real-time data metadata with fixed logic
        real_time_metadata = self._extract_real_time_data_metadata(processing_result)

        # ENHANCED: Build data source summary with fallback tracking
        data_source_summary = self._build_data_source_summary(processing_result)

        # Build response comparison
        response_comparison = None
        if existing_response_result.get("has_existing_response"):
            generated_response = processing_result.get("generated_response", "")
            existing_response = existing_response_result.get("existing_response", "")

            response_comparison = {
                "has_existing_response": True,
                "existing_response": existing_response,
                "existing_response_length": len(existing_response),
                "generated_response_length": len(generated_response),
                "comparison_note": "This shows what the sales staff originally wrote vs our AI-generated response",
            }
        else:
            response_comparison = {
                "has_existing_response": False,
                "comparison_note": "No existing staff response found for comparison",
            }

        # ENHANCED: Build comprehensive real-time data summary with fixed logic
        real_time_data_summary = {
            "detection_results": {
                "needs_pricing": data_needs_analysis.get("needs_pricing", False),
                "needs_stock": data_needs_analysis.get("needs_stock", False),
                "detection_reason": data_needs_analysis.get(
                    "detection_reason", "No detection"
                ),
                "detection_layers": data_needs_analysis.get("detection_layers", {}),
                "prompt_strategy": data_needs_analysis.get(
                    "prompt_strategy", "general_helpful"
                ),
            },
            "data_prioritization": {
                "strategy_used": enhanced_context.get(
                    "prioritization_strategy", "standard"
                ),
                "real_time_data_prioritized": real_time_metadata.get(
                    "real_time_data_used", False
                ),
                "examples_limited": enhanced_context.get("examples_usage")
                == "tone_reference_only",
                "has_product_data": enhanced_context.get("product_selection", {}).get(
                    "has_product_data", False
                ),
                "is_external_comment": enhanced_context.get(
                    "is_external_comment", False
                ),
            },
            "data_sources": data_source_summary,
            "metadata": real_time_metadata,
            "fallback_info": {
                "generation_method": processing_result.get(
                    "generation_method", "unknown"
                ),
                "used_fallback": "fallback"
                in processing_result.get("generation_method", ""),
                "fallback_reason": self._determine_fallback_reason(processing_result),
            },
        }

        # ENHANCED: Build product viability summary
        product_viability_summary = {
            "has_product_data": enhanced_context.get("product_selection", {}).get(
                "has_product_data", False
            ),
            "main_product_viable": enhanced_context.get("product_selection", {}).get(
                "main_product_viable", False
            ),
            "alternative_product_viable": enhanced_context.get(
                "product_selection", {}
            ).get("alternative_product_viable", False),
            "selection_reason": enhanced_context.get("product_selection", {}).get(
                "selection_reason", "No product selection performed"
            ),
            "is_external_comment": enhanced_context.get("is_external_comment", False),
        }

        # NEW: Build product search summary for process endpoint
        product_search_summary = {
            "search_attempted": False,
            "has_suggestions": False,
            "suggestions_used": False,
            "confidence": 0,
        }

        product_search_result = processing_result.get("product_search_result", {})
        if product_search_result:
            product_search_summary.update(
                {
                    "search_attempted": product_search_result.get(
                        "search_attempted", False
                    ),
                    "has_suggestions": product_search_result.get(
                        "has_suggestions", False
                    ),
                    "suggestions_used": len(
                        product_search_result.get("suggestions", [])
                    )
                    > 0,
                    "confidence": product_search_result.get("confidence", 0),
                }
            )

        # Build complete enhanced response data
        process_data = {
            "request_id": request_id,
            "scope_warning": request_data.get("scope_warning"),
            "data_log": request_data.get("data_log", {}),
            "key_fields": key_fields,
            "all_fields": customer_data,
            "predicted_intent": intent_result.get("predicted_intent"),
            "real_time_data_summary": real_time_data_summary,
            "product_viability_summary": product_viability_summary,
            "product_search_summary": product_search_summary,  # NEW
            "product_search_result": processing_result.get("product_search_result", {}),
            "product_details": product_result.get("product_details", {}),
            "pinecone_results": pinecone_results,
            "generated_response": processing_result.get("generated_response"),
            "generation_method": processing_result.get("generation_method"),
            "response_comparison": response_comparison,
            "processing_summary": processing_summary,
            "api_enhancements": {
                "version": self.api_version,
                "features": [
                    "real_time_data_integration",
                    "dual_layer_detection_with_regex",
                    "intent_based_prompt_selection",
                    "smart_data_prioritization",
                    "enhanced_prompt_engineering",
                    "fixed_eol_logic",
                    "external_comment_detection",
                    "graceful_fallback_handling",
                    "product_search_capabilities",
                    "early_exit_optimization",
                ],
            },
        }

        return self.build_success_response(
            process_data,
            "Request processed successfully with FIXED real-time data integration",
        )

    def build_health_response(self) -> Dict:
        """
        Build health check response

        Returns:
            Health check response
        """
        return self.build_success_response(
            {
                "service": "Response Engine API",
                "status": "healthy",
                "uptime": "running",
                "version": self.api_version,
                "enhancements": [
                    "Real-time data integration",
                    "Dual-layer detection system with regex",
                    "Intent-based prompt strategies",
                    "Smart data prioritization",
                    "Product viability logic",
                    "External comment detection",
                    "Early exit optimization",
                ],
            },
            "Service is healthy with enhanced features",
        )

    def build_validation_error_response(self, validation_result: Dict) -> Dict:
        """
        Build response for validation errors

        Args:
            validation_result: Validation results with errors

        Returns:
            Formatted validation error response
        """
        error_type = validation_result.get("error_type", "validation_error")

        # Map error types to HTTP status codes
        status_code_map = {
            "empty_id": 400,
            "whitespace_id": 400,
            "invalid_format": 400,
            "not_found": 404,
            "database_error": 500,
            "validation_exception": 500,
        }

        status_code = status_code_map.get(error_type, 400)

        return self.build_error_response(
            validation_result.get("error", "Validation failed"),
            error_type,
            status_code,
            details=validation_result,
        )

    def build_summary_response(self, summary_result: Dict) -> Dict:
        """
        Build response for request summary

        Args:
            summary_result: Summary results

        Returns:
            Formatted summary response
        """
        if summary_result["status"] != "success":
            return self.build_error_response(
                summary_result.get("error", "Summary generation failed"),
                summary_result.get("status", "summary_error"),
                details=summary_result,
            )

        summary_data = {
            "request_id": summary_result.get("request_id"),
            "customer_comment_preview": summary_result.get("customer_comment"),
            "predicted_intent": summary_result.get("predicted_intent"),
            "is_in_scope": summary_result.get("is_in_scope"),
            "product_id": summary_result.get("product_id"),
            "alternative_id": summary_result.get("alternative_id"),
            "woot_rep": summary_result.get("woot_rep"),
        }

        return self.build_success_response(summary_data, "Request summary generated")

    def build_debug_response(self, debug_data: Dict) -> Dict:
        """
        Build response for debug endpoints

        Args:
            debug_data: Debug information

        Returns:
            Formatted debug response
        """
        return self.build_success_response(debug_data, "Debug information retrieved")

    def extract_key_metrics(self, processing_result: Dict) -> Dict:
        """
        Extract enhanced key metrics from processing results for monitoring

        Args:
            processing_result: Full processing results

        Returns:
            Enhanced key metrics dictionary with real-time data tracking
        """
        processing_summary = processing_result.get("processing_summary", {})
        data_needs_analysis = processing_result.get("data_needs_analysis", {})
        enhanced_context = processing_result.get("enhanced_context", {})

        # Extract real-time data metadata
        real_time_metadata = self._extract_real_time_data_metadata(processing_result)

        return {
            "request_processed": processing_result.get("status") == "success",
            "intent_predicted": processing_summary.get("predicted_intent"),
            "is_in_scope": processing_result.get("status") != "out_of_scope",
            "products_found": processing_summary.get("products_found", 0),
            "similar_responses_found": processing_summary.get(
                "similar_responses_found", 0
            ),
            "has_existing_response": processing_summary.get(
                "has_existing_response", False
            ),
            "ai_response_generated": processing_summary.get(
                "response_generated", False
            ),
            "generation_method": processing_result.get("generation_method"),
            "processing_time": processing_result.get("processing_timestamp"),
            "early_exit": processing_result.get("early_exit", False),
            # Enhanced metrics
            "real_time_data_used": real_time_metadata.get("real_time_data_used", False),
            "needs_pricing": processing_summary.get("needs_pricing", False),
            "needs_stock": processing_summary.get("needs_stock", False),
            "prioritization_strategy": processing_summary.get(
                "prioritization_strategy", "standard"
            ),
            "prompt_strategy": data_needs_analysis.get(
                "prompt_strategy", "general_helpful"
            ),
            "detection_method": data_needs_analysis.get("detection_reason", "none"),
            "data_freshness": enhanced_context.get("real_time_data", {}).get(
                "data_freshness"
            ),
            "pricing_data_source": real_time_metadata.get(
                "pricing_data_source", "none"
            ),
            "stock_data_source": real_time_metadata.get("stock_data_source", "none"),
            "has_product_data": enhanced_context.get("product_selection", {}).get(
                "has_product_data", False
            ),
            "is_external_comment": enhanced_context.get("is_external_comment", False),
            "product_search_attempted": processing_summary.get(
                "product_search_attempted", False
            ),
            # Enhancement tracking
            "api_version": self.api_version,
            "enhanced_features_active": True,
        }


# Global response builder instance (lazy-loaded)
_response_builder_instance = None


def get_response_builder() -> ResponseBuilder:
    """
    Get a global response builder instance (singleton pattern)

    Returns:
        ResponseBuilder instance
    """
    global _response_builder_instance
    if _response_builder_instance is None:
        _response_builder_instance = ResponseBuilder()
    return _response_builder_instance


# Convenience functions
def build_api_success_response(data: Dict, message: str = None) -> Dict:
    """
    Convenience function to build success response

    Args:
        data: Response data
        message: Optional message

    Returns:
        Formatted success response
    """
    builder = get_response_builder()
    return builder.build_success_response(data, message)


def build_api_error_response(
    error: str,
    error_type: str = "general_error",
    status_code: int = 500,
    details: Dict = None,
) -> Dict:
    """
    Convenience function to build error response

    Args:
        error: Error message
        error_type: Error type
        status_code: HTTP status code
        details: Additional details

    Returns:
        Formatted error response
    """
    builder = get_response_builder()
    return builder.build_error_response(error, error_type, status_code, details)


def build_test_endpoint_response(request_id: str, processing_result: Dict) -> Dict:
    """
    Convenience function to build enhanced test endpoint response

    Args:
        request_id: Request ID
        processing_result: Processing results

    Returns:
        Formatted test response with enhanced metadata
    """
    builder = get_response_builder()
    return builder.build_test_response(request_id, processing_result)


def build_process_endpoint_response(request_id: str, processing_result: Dict) -> Dict:
    """
    Convenience function to build enhanced process endpoint response

    Args:
        request_id: Request ID
        processing_result: Processing results

    Returns:
        Formatted process response with enhanced metadata
    """
    builder = get_response_builder()
    return builder.build_process_response(request_id, processing_result)


def extract_response_metrics(processing_result: Dict) -> Dict:
    """
    Convenience function to extract enhanced metrics

    Args:
        processing_result: Processing results

    Returns:
        Enhanced key metrics with real-time data tracking
    """
    builder = get_response_builder()
    return builder.extract_key_metrics(processing_result)
