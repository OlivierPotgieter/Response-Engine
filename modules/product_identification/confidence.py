"""
Confidence System for Product Identification and Response Generation
Implements AGENTS.md Phase 1 confidence thresholds and validation logic
"""

import logging
from typing import Dict, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceThresholds:
    """Confidence thresholds as specified in AGENTS.md"""

    HIGH_CONFIDENCE = 0.85  # Direct product match, use without hesitation
    MEDIUM_CONFIDENCE = 0.70  # Good match, use with slight hedging
    LOW_CONFIDENCE = 0.50  # Uncertain, offer multiple options
    MINIMUM_THRESHOLD = 0.50  # Below this, don't attempt product-specific responses


def should_use_extracted_product(extraction_result: Dict) -> Tuple[bool, str]:
    """
    Determine if extracted product is reliable enough to use
    Implementation of AGENTS.md specification lines 75-87

    Args:
        extraction_result: Product extraction result with confidence score

    Returns:
        Tuple of (should_use: bool, confidence_level: str)
    """
    confidence = extraction_result.get("confidence", 0)

    logger.info(f"Evaluating product extraction confidence: {confidence:.3f}")

    if confidence >= ConfidenceThresholds.HIGH_CONFIDENCE:
        return True, "high_confidence"
    elif confidence >= ConfidenceThresholds.MEDIUM_CONFIDENCE:
        return True, "medium_confidence"
    elif confidence >= ConfidenceThresholds.LOW_CONFIDENCE:
        return True, "low_confidence"
    else:
        logger.info(
            f"Confidence {confidence:.3f} below minimum threshold {ConfidenceThresholds.MINIMUM_THRESHOLD}"
        )
        return False, "insufficient_confidence"


def get_confidence_level(confidence_score: float) -> str:
    """Get confidence level string from numeric score"""
    if confidence_score >= ConfidenceThresholds.HIGH_CONFIDENCE:
        return "high"
    elif confidence_score >= ConfidenceThresholds.MEDIUM_CONFIDENCE:
        return "medium"
    elif confidence_score >= ConfidenceThresholds.LOW_CONFIDENCE:
        return "low"
    else:
        return "insufficient"


class ResponseValidator:
    """
    Response validation layer as specified in AGENTS.md lines 122-132
    """

    def validate_response(self, response: str, context: Dict) -> Tuple[bool, Dict]:
        """
        Validate response quality and appropriateness

        Args:
            response: Generated response text
            context: Response generation context

        Returns:
            Tuple of (is_valid: bool, validation_checks: Dict)
        """
        checks = {
            "has_greeting": self._check_greeting(response),
            "addresses_intent": self._check_intent_coverage(response, context),
            "appropriate_confidence": self._check_confidence_language(
                response, context
            ),
            "includes_next_steps": self._check_actionable(response),
            "length_appropriate": len(response.split()) > 20,
        }

        is_valid = all(checks.values())
        logger.info(
            f"Response validation: {'PASSED' if is_valid else 'FAILED'} - {checks}"
        )

        return is_valid, checks

    def _check_greeting(self, response: str) -> bool:
        """Check if response has appropriate greeting"""
        greeting_indicators = ["thank", "hi", "hello", "good", "appreciate"]
        response_lower = response.lower()
        return any(indicator in response_lower for indicator in greeting_indicators)

    def _check_intent_coverage(self, response: str, context: Dict) -> bool:
        """Check if response addresses the predicted intent"""
        predicted_intent = context.get("predicted_intent", "").lower()
        response_lower = response.lower()

        intent_keywords = {
            "pricing": ["price", "cost", "r", "pricing"],
            "stock": ["stock", "available", "availability", "in stock"],
            "warranty": ["warranty", "guarantee", "cover"],
            "shipping": ["shipping", "delivery", "collection"],
        }

        if predicted_intent in intent_keywords:
            keywords = intent_keywords[predicted_intent]
            return any(keyword in response_lower for keyword in keywords)

        return True  # Default to true for general inquiries

    def _check_confidence_language(self, response: str, context: Dict) -> bool:
        """Check if confidence language matches extraction confidence"""
        confidence = context.get("product_search_result", {}).get("confidence", 0)
        response_lower = response.lower()

        if confidence >= ConfidenceThresholds.HIGH_CONFIDENCE:
            uncertain_words = ["might", "appears", "seems", "possibly", "maybe"]
            return not any(word in response_lower for word in uncertain_words)
        elif confidence >= ConfidenceThresholds.MEDIUM_CONFIDENCE:
            hedging_words = ["appears", "seems", "based on", "likely"]
            return any(word in response_lower for word in hedging_words)

        return True  # Low confidence can use any language

    def _check_actionable(self, response: str) -> bool:
        """Check if response includes next steps or actionable information"""
        actionable_indicators = [
            "contact",
            "call",
            "email",
            "visit",
            "let me know",
            "please",
        ]
        response_lower = response.lower()
        return any(indicator in response_lower for indicator in actionable_indicators)
