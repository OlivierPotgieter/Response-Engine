#!/usr/bin/env python3
"""
Test script to verify the complete enhanced generation flow
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.processors.data_processor import DataProcessor

def test_complete_enhanced_flow():
    """Test the complete enhanced generation flow with AI-identified products"""
    
    print("=== COMPLETE ENHANCED FLOW TEST ===")
    
    customer_comment = "Good afternoon, i am looking for a crosshair extreme vi motherboard you had for sale 2 years ago, i bought one but would need another it was the am4 platform x370 chipset."
    predicted_intent = "Pricing Inquiry"
    
    processor = DataProcessor()
    
    try:
        print("1. Testing data needs detection...")
        data_needs = processor.detect_data_needs(customer_comment, predicted_intent)
        print(f"   Prompt strategy: {data_needs.get('prompt_strategy')}")
        print(f"   Needs pricing: {data_needs.get('needs_pricing')}")
        
        print("\n2. Testing product search simulation...")
        mock_product_search_result = {
            "search_successful": True,
            "best_match": {
                "name": "Asus ROG Crosshair VI Extreme AM4 X370 ATX Motherboard",
                "product_id": 12345,
                "relevance_score": 0.593,
                "sku": "WO_ASUS_CROSSHAIR_VI_EXTREME",
                "category": "Motherboard"
            },
            "confidence": 0.748,
            "has_suggestions": True,
            "suggestions": []
        }
        
        print(f"   Mock search confidence: {mock_product_search_result['confidence']}")
        print(f"   Mock best match: {mock_product_search_result['best_match']['name']}")
        
        print("\n3. Testing context building with real data...")
        context = {
            "predicted_intent": predicted_intent,
            "data_needs": data_needs,
            "product_search_result": mock_product_search_result,
            "examples": []
        }
        
        enhanced_context = processor._map_product_search_to_real_time_data(context, {})
        
        print(f"   Context has real_time_data: {'real_time_data' in enhanced_context}")
        print(f"   Context has product_selection: {'product_selection' in enhanced_context}")
        
        if 'real_time_data' in enhanced_context:
            rtd = enhanced_context['real_time_data']
            if 'primary_product' in rtd:
                pp = rtd['primary_product']
                print(f"   Primary product viable: {pp.get('is_viable', False)}")
                print(f"   Primary product name: {pp.get('product_name', 'N/A')}")
        
        print("\n4. Testing strategy selection...")
        prompt_strategy = data_needs.get('prompt_strategy')
        enhanced_strategies = [
            "warranty_focused",
            "pricing_focused", 
            "stock_focused",
            "combined_pricing_stock",
            "pricing_enhanced",
            "stock_enhanced"
        ]
        
        uses_enhanced = prompt_strategy in enhanced_strategies
        print(f"   Strategy '{prompt_strategy}' uses enhanced generation: {uses_enhanced}")
        
        if uses_enhanced:
            print("   ✅ Should trigger enhanced generation path")
        else:
            print("   ❌ Will use simple generation path")
        
        print("\n5. Summary:")
        print(f"   - Data needs detected: {bool(data_needs)}")
        print(f"   - Product search simulated: {mock_product_search_result['search_successful']}")
        print(f"   - Context mapping applied: {'real_time_data' in enhanced_context}")
        print(f"   - Enhanced generation triggered: {uses_enhanced}")
        
        success = (
            bool(data_needs) and
            mock_product_search_result['search_successful'] and
            'real_time_data' in enhanced_context and
            uses_enhanced
        )
        
        return success
        
    except Exception as e:
        print(f"❌ Error in complete flow test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_complete_enhanced_flow()
    print(f"\n=== COMPLETE FLOW TEST {'PASSED' if success else 'FAILED'} ===")
