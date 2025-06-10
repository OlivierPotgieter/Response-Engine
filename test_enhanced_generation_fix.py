#!/usr/bin/env python3
"""
Test script to verify enhanced generation fix with AI product identification
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.processors.data_processor import DataProcessor
from modules.ai.llm_generator import LLMGenerator

def test_enhanced_generation_fix():
    """Test that enhanced generation works with AI-identified products"""
    
    print("=== ENHANCED GENERATION FIX TEST ===")
    
    customer_comment = "I'm looking for a specific component\n\nComment:\n\nHi there\n\nWould like to find out of Wootware is planning on importing the Gigabyte MO27Q28G 280hz Primary RGB Tandem WOLED monitor?"
    
    processor = DataProcessor()
    
    data_needs = processor.detect_data_needs(customer_comment, "General Inquiry")
    print(f"Prompt strategy: {data_needs.get('prompt_strategy')}")
    
    enhanced_context = {
        "predicted_intent": "General Inquiry",
        "data_needs": data_needs,
        "product_search_result": {
            "best_match": {
                "name": "Gigabyte AORUS FI27Q-P 27\" WQHD (2560x1440) 165Hz 1ms IPS Gaming Desktop Monitor",
                "product_id": 35883,
                "relevance_score": 0.648269117,
                "sku": "WO_GIGABYTE FI27QP",
                "category": "Monitor"
            },
            "confidence": 0.748269117,
            "has_suggestions": True
        },
        "examples": []
    }
    
    enhanced_context = processor._map_product_search_to_real_time_data(enhanced_context, {})
    
    print(f"\nEnhanced context keys after mapping: {list(enhanced_context.keys())}")
    print(f"Has real_time_data: {'real_time_data' in enhanced_context}")
    print(f"Has product_selection: {'product_selection' in enhanced_context}")
    
    try:
        generator = LLMGenerator()
        response = generator.generate_enhanced_response(customer_comment, enhanced_context, "Dylan")
        print(f"\nGenerated response length: {len(response)}")
        print(f"Generated response preview: {response[:200]}...")
        
        is_substantial = len(response) > 100 and "Kind Regards" not in response[:50]
        print(f"Is response substantial: {is_substantial}")
        
        return is_substantial
        
    except Exception as e:
        print(f"Error in enhanced generation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_generation_fix()
    print(f"\n=== TEST {'PASSED' if success else 'FAILED'} ===")
