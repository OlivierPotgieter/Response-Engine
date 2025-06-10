#!/usr/bin/env python3
"""
Debug script to trace complete confidence flow through the entire pipeline
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.processors.data_processor import DataProcessor
from modules.ai.intent_classifier import predict_customer_intent, is_comment_in_scope
from modules.product_identification.confidence import should_use_extracted_product

def debug_complete_confidence_flow():
    """Debug the complete confidence calculation through the entire pipeline"""
    
    customer_comment = "Good afternoon, i am looking for a crosshair extreme vi motherboard you had for sale 2 years ago, i bought one but would need another it was the am4 platform x370 chipset."
    
    print("=== COMPLETE CONFIDENCE FLOW DEBUG ===")
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
        customer_data = {
            "customer_comment": customer_comment,
            "customer_email": "test@example.com",
            "woot_rep": "CJ"
        }
        
        product_search_result = processor.attempt_product_lookup_from_comment(
            customer_comment, customer_data
        )
        
        print(f"   Search attempted: {product_search_result.get('search_attempted', False)}")
        print(f"   Has suggestions: {product_search_result.get('has_suggestions', False)}")
        print(f"   Confidence: {product_search_result.get('confidence', 'NOT SET')}")
        print(f"   Confidence level: {product_search_result.get('confidence_level', 'NOT SET')}")
        
        if product_search_result.get('best_match'):
            best_match = product_search_result['best_match']
            print(f"   Best match: {best_match.get('name', 'Unknown')}")
            print(f"   Relevance score: {best_match.get('relevance_score', 'NOT SET')}")
        
        print()
        print("3. CONFIDENCE THRESHOLD LOGIC DEBUG:")
        search_results = product_search_result.get('search_results', {})
        if search_results:
            print(f"   Raw search results confidence: {search_results.get('confidence', 'NOT SET')}")
            should_use, conf_level = should_use_extracted_product(search_results)
            print(f"   should_use_extracted_product result: should_use={should_use}, level={conf_level}")
            
            test_confidence = {"confidence": 0.692847526}
            should_use_test, conf_level_test = should_use_extracted_product(test_confidence)
            print(f"   Test with 0.692847526: should_use={should_use_test}, level={conf_level_test}")
        
        print()
        print("4. IDENTIFIERS EXTRACTED DEBUG:")
        identifiers = search_results.get('identifiers_extracted', {})
        print(f"   Identifiers confidence_score: {identifiers.get('confidence_score', 'NOT SET')}")
        
        print()
        print("5. RESPONSE GENERATION REQUIREMENTS:")
        should_use_for_pricing = product_search_result.get('should_use_for_pricing', False)
        print(f"   should_use_for_pricing: {should_use_for_pricing}")
        print(f"   Required for full response: medium_confidence or high_confidence")
        
        print()
        print("=== END COMPLETE DEBUG ===")
        
    except Exception as e:
        print(f"   ERROR in product identification: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_complete_confidence_flow()
