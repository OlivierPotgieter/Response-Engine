#!/usr/bin/env python3
"""
Debug script to trace confidence flow through the entire pipeline
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.processors.data_processor import DataProcessor
from modules.ai.intent_classifier import predict_customer_intent, is_comment_in_scope

def debug_confidence_flow():
    """Debug the confidence calculation through the entire pipeline"""
    
    customer_comment = "Good afternoon, i am looking for a crosshair extreme vi motherboard you had for sale 2 years ago, i bought one but would need another it was the am4 platform x370 chipset."
    
    print("=== DEBUGGING CONFIDENCE FLOW ===")
    print(f"Comment: {customer_comment[:80]}...")
    print()
    
    print("1. INTENT CLASSIFICATION:")
    intent = predict_customer_intent(customer_comment)
    in_scope = is_comment_in_scope(customer_comment)
    print(f"   Predicted intent: {intent}")
    print(f"   In scope: {in_scope}")
    print()
    
    print("2. PRODUCT IDENTIFICATION:")
    processor = DataProcessor()
    try:
        product_search_result = processor.attempt_product_lookup_from_comment(customer_comment)
        print(f"   Search attempted: {product_search_result.get('search_attempted', False)}")
        print(f"   Has suggestions: {product_search_result.get('has_suggestions', False)}")
        print(f"   Confidence: {product_search_result.get('confidence', 'NOT SET')}")
        print(f"   Confidence level: {product_search_result.get('confidence_level', 'NOT SET')}")
        
        if product_search_result.get('best_match'):
            best_match = product_search_result['best_match']
            print(f"   Best match: {best_match.get('name', 'Unknown')}")
            print(f"   Relevance score: {best_match.get('relevance_score', 'NOT SET')}")
        
        print(f"   Full result keys: {list(product_search_result.keys())}")
        print()
        
    except Exception as e:
        print(f"   ERROR in product identification: {e}")
        print()
    
    print("=== END DEBUG ===")

if __name__ == "__main__":
    debug_confidence_flow()
