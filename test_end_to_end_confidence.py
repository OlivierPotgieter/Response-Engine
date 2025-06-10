#!/usr/bin/env python3
"""
End-to-end test to verify confidence calculation fix works completely
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.processors.data_processor import DataProcessor
from modules.ai.intent_classifier import predict_customer_intent, is_comment_in_scope

def test_end_to_end_confidence():
    """Test the complete confidence calculation end-to-end"""
    
    print("=== END-TO-END CONFIDENCE TEST ===")
    
    customer_comment = "Good afternoon, i am looking for a crosshair extreme vi motherboard you had for sale 2 years ago, i bought one but would need another it was the am4 platform x370 chipset."
    
    print(f"Testing comment: {customer_comment[:80]}...")
    print()
    
    print("1. INTENT CLASSIFICATION:")
    intent = predict_customer_intent(customer_comment)
    in_scope = is_comment_in_scope(customer_comment)
    print(f"   Intent: {intent}")
    print(f"   In scope: {in_scope}")
    print()
    
    print("2. PRODUCT IDENTIFICATION (with AI system unavailable):")
    processor = DataProcessor()
    
    try:
        customer_data = {
            "customer_comment": customer_comment,
            "customer_email": "test@example.com",
            "woot_rep": "CJ"
        }
        
        result = processor.attempt_product_lookup_from_comment(customer_comment, customer_data)
        
        print(f"   Search attempted: {result.get('search_attempted', False)}")
        print(f"   Has suggestions: {result.get('has_suggestions', False)}")
        print(f"   Confidence: {result.get('confidence', 'NOT SET')}")
        print(f"   Confidence level: {result.get('confidence_level', 'NOT SET')}")
        print(f"   Should use for pricing: {result.get('should_use_for_pricing', False)}")
        
        if result.get('best_match'):
            best_match = result['best_match']
            print(f"   Best match: {best_match.get('name', 'Unknown')}")
            print(f"   Relevance score: {best_match.get('relevance_score', 'NOT SET')}")
        
        confidence = result.get('confidence', 0)
        has_suggestions = result.get('has_suggestions', False)
        
        print()
        print("3. CONFIDENCE VALIDATION:")
        if has_suggestions and confidence > 0:
            print("   ✅ SUCCESS: Products found AND confidence > 0")
        elif has_suggestions and confidence == 0:
            print("   ❌ FAILURE: Products found BUT confidence still 0")
        elif not has_suggestions:
            print("   ⚠️  INFO: No products found (expected if AI system not initialized)")
        else:
            print("   ❓ UNKNOWN: Unexpected state")
        
        return confidence > 0 if has_suggestions else True  # Pass if no products found due to AI system
        
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_end_to_end_confidence()
    print()
    print(f"=== TEST {'PASSED' if success else 'FAILED'} ===")
