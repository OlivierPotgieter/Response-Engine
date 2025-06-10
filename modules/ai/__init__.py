"""
AI Operations Module
Handles all AI-related operations including intent classification,
vector search, and LLM response generation.
"""

from .intent_classifier import (
    IntentClassifier,
    predict_customer_intent,
    check_intent_scope,
)
from .pinecone_client import (
    PineconeClient,
    search_similar_comments,
    get_response_examples_for_comment,
    search_with_full_context,
    test_pinecone_connection,
)
from .llm_generator import (
    LLMGenerator,
    generate_response_with_examples,
    generate_enhanced_response_with_context,
    generate_simple_response_for_comment,
    test_llm_generation,
)

__all__ = [
    # Intent classification
    "IntentClassifier",
    "predict_customer_intent",
    "check_intent_scope",
    # Vector search
    "PineconeClient",
    "search_similar_comments",
    "get_response_examples_for_comment",
    "search_with_full_context",
    "test_pinecone_connection",
    # LLM generation (now with rep signature support)
    "LLMGenerator",
    "generate_response_with_examples",
    "generate_enhanced_response_with_context",
    "generate_simple_response_for_comment",
    "test_llm_generation",
]
