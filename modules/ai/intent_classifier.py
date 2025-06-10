"""
Intent Classification Module
Handles customer comment intent prediction using trained ML models.
"""

import os
import logging
import pandas as pd
import joblib
from typing import Dict, Optional, List
import re

logger = logging.getLogger(__name__)


class IntentClassifier:
    def __init__(self, model_path: str = "intent_inference_pipeline.pkl"):
        """
        Initialize the intent classifier

        Args:
            model_path: Path to the trained intent classification pipeline
        """
        self.model_path = model_path
        self.pipeline = None
        self.intent_labels = {
            0: "Compatibility or Upgrade",
            1: "General Inquiry",
            2: "Order Assistance",
            3: "Pricing Inquiry",
            4: "Product Recommendation",
            5: "Quotation Request",
            6: "Returns and Issues",
            7: "Shipping and Collection",
            8: "Stock Availability",
            9: "Warranty Inquiry"
        }
        self.out_of_scope_intents = ["Compatibility or Upgrade", "Product Recommendation"]
        self._load_model()

    def _load_model(self):
        """Load the trained intent classification pipeline"""
        try:
            self.pipeline = joblib.load(self.model_path)
            logger.info(f"Intent classification pipeline loaded from {self.model_path}")
        except FileNotFoundError:
            logger.error(f"Intent classification model not found at {self.model_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading intent classification model: {e}")
            raise

    def _featurize_comment(self, comment: str) -> pd.DataFrame:
        """
        Extract features from customer comment for intent prediction

        Args:
            comment: Customer comment text

        Returns:
            DataFrame with extracted features
        """
        # Replicating the featurization logic from the training script
        df = pd.DataFrame({'customer_comment': [comment]})
        text = df["customer_comment"].str.lower()

        df["comment_length"] = text.str.len()
        df["has_question"] = text.str.contains(r"\?").astype(int)
        df["greeting"] = text.str.contains(r"\b(?:hi|hello|hey)\b").astype(int)
        df["stock_availability"] = text.str.contains(r"\b(?:stock|available)\b").astype(int)
        df["delivery_fee_note"] = text.str.contains(r"\b(?:delivery|shipping fee)\b").astype(int)
        df["lead_time_notice"] = text.str.contains(r"\b(?:lead time|estimate)\b").astype(int)
        df["closing_statement"] = text.str.contains(r"\b(?:thanks|thank you|regards)\b").astype(int)
        df["product_link"] = text.str.contains(r"https?://").astype(int)
        df["alternative_recommendation"] = text.str.contains(r"\balternative\b").astype(int)
        df["pricing_disclaimer"] = text.str.contains(r"\bprice\s*is\b").astype(int)
        df["replacement_item"] = text.str.contains(r"\breplace\b|\bswap\b").astype(int)
        df["artifact_type"] = 0  # Default value

        # Return only the columns required by the pipeline
        return df[[
            "customer_comment", "delivery_fee_note", "lead_time_notice",
            "closing_statement", "product_link", "alternative_recommendation",
            "pricing_disclaimer", "stock_availability", "artifact_type",
            "comment_length", "replacement_item", "greeting", "has_question"
        ]]

    def predict_intent(self, comment: str) -> str:
        """
        Predict the intent of a customer comment

        Args:
            comment: Customer comment text

        Returns:
            Predicted intent as string
        """
        try:
            if not comment or not comment.strip():
                logger.warning("Empty comment provided for intent prediction")
                return "General Inquiry"

            # Extract features
            features_df = self._featurize_comment(comment)

            # Predict using the pipeline
            prediction = self.pipeline.predict(features_df)[0]

            # Map to intent label
            intent_label = self.intent_labels.get(int(prediction), "General Inquiry")

            logger.info(f"Predicted intent '{intent_label}' for comment: {comment[:50]}...")
            return intent_label

        except Exception as e:
            logger.error(f"Intent prediction error for comment '{comment[:50]}...': {e}")
            return "General Inquiry"  # Fallback

    def is_intent_in_scope(self, intent: str) -> bool:
        """
        Check if an intent is within the system's scope

        Args:
            intent: Intent string to check

        Returns:
            True if intent is in scope, False if out of scope
        """
        return intent not in self.out_of_scope_intents

    def check_intent_scope(self, intent: str) -> Dict:
        """
        Check intent scope and return detailed information

        Args:
            intent: Intent string to check

        Returns:
            Dict with scope information
        """
        is_in_scope = self.is_intent_in_scope(intent)

        if is_in_scope:
            return {
                "is_out_of_scope": False,
                "message": "Intent is within scope for processing",
                "intent": intent
            }
        else:
            return {
                "is_out_of_scope": True,
                "message": "This Query is outside of scope",
                "reason": f"Intent '{intent}' is not supported by this system",
                "intent": intent,
                "supported_intents": [
                    intent for intent in self.intent_labels.values()
                    if intent not in self.out_of_scope_intents
                ]
            }

    def predict_with_scope_check(self, comment: str) -> Dict:
        """
        Predict intent and perform scope check in one operation

        Args:
            comment: Customer comment text

        Returns:
            Dict with intent prediction and scope information
        """
        intent = self.predict_intent(comment)
        scope_info = self.check_intent_scope(intent)

        return {
            "predicted_intent": intent,
            "scope_check": scope_info,
            "comment_preview": comment[:100] + "..." if len(comment) > 100 else comment
        }

    def get_supported_intents(self) -> List[str]:
        """
        Get list of supported (in-scope) intents

        Returns:
            List of supported intent strings
        """
        return [
            intent for intent in self.intent_labels.values()
            if intent not in self.out_of_scope_intents
        ]

    def get_out_of_scope_intents(self) -> List[str]:
        """
        Get list of out-of-scope intents

        Returns:
            List of out-of-scope intent strings
        """
        return self.out_of_scope_intents.copy()

    def get_all_intents(self) -> Dict:
        """
        Get all intent information

        Returns:
            Dict with all intent categories
        """
        return {
            "all_intents": list(self.intent_labels.values()),
            "supported_intents": self.get_supported_intents(),
            "out_of_scope_intents": self.get_out_of_scope_intents(),
            "intent_mappings": self.intent_labels
        }


# Global classifier instance (lazy-loaded)
_classifier_instance = None


def get_classifier() -> IntentClassifier:
    """
    Get a global classifier instance (singleton pattern)

    Returns:
        IntentClassifier instance
    """
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = IntentClassifier()
    return _classifier_instance


# Convenience functions
def predict_customer_intent(comment: str) -> str:
    """
    Convenience function to predict customer intent

    Args:
        comment: Customer comment text

    Returns:
        Predicted intent as string
    """
    classifier = get_classifier()
    return classifier.predict_intent(comment)


def check_intent_scope(comment: str) -> Dict:
    """
    Convenience function to predict intent and check scope

    Args:
        comment: Customer comment text

    Returns:
        Dict with intent prediction and scope information
    """
    classifier = get_classifier()
    return classifier.predict_with_scope_check(comment)


def is_comment_in_scope(comment: str) -> bool:
    """
    Convenience function to check if a comment's intent is in scope

    Args:
        comment: Customer comment text

    Returns:
        True if intent is in scope, False otherwise
    """
    classifier = get_classifier()
    intent = classifier.predict_intent(comment)
    return classifier.is_intent_in_scope(intent)