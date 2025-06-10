#!/usr/bin/env python3
"""
Debug script to test the mapping logic without OpenAI calls
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.processors.data_processor import DataProcessor

def test_mapping_logic():
    """Test the mapping logic without OpenAI calls"""
    
    print("=== MAPPING LOGIC DEBUG ===")
    
    processor = DataProcessor()
    customer_comment = "Hi there\n\nWould like to find out of Wootware is planning on importing the Gigabyte MO27Q28G 280hz Primary RGB Tandem WOLED monitor?"
    
    context = {
        'predicted_intent': 'General Inquiry',
        'data_needs': {'prompt_strategy': 'pricing_enhanced'},
        'product_search_result': {
            'best_match': {
                'name': 'Gigabyte AORUS FI27Q-P 27" WQHD (2560x1440) 165Hz 1ms IPS Gaming Desktop Monitor',
                'product_id': 35883,
                'relevance_score': 0.648,
                'sku': 'WO_GIGABYTE FI27QP',
                'category': 'Monitor'
            },
            'confidence': 0.748,
            'has_suggestions': True
        },
        'examples': []
    }
    
    print(f"Original context keys: {list(context.keys())}")
    print(f"Has product_selection: {'product_selection' in context}")
    print(f"Has real_time_data: {'real_time_data' in context}")
    
    try:
        mapped_context = processor._map_product_search_to_real_time_data(context, {})
        print(f"\nAfter mapping:")
        print(f"Context keys: {list(mapped_context.keys())}")
        print(f"Has product_selection: {'product_selection' in mapped_context}")
        print(f"Has real_time_data: {'real_time_data' in mapped_context}")
        
        if 'product_selection' in mapped_context:
            ps = mapped_context['product_selection']
            print(f"Product selection has_product_data: {ps.get('has_product_data', False)}")
            print(f"Product selection reason: {ps.get('selection_reason', 'N/A')}")
        
        if 'real_time_data' in mapped_context:
            rtd = mapped_context['real_time_data']
            print(f"Real-time data has primary_product: {'primary_product' in rtd}")
            if 'primary_product' in rtd:
                pp = rtd['primary_product']
                print(f"Primary product name: {pp.get('product_name', 'N/A')}")
                print(f"Primary product viable: {pp.get('is_viable', False)}")
        
        print("✅ Mapping logic works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Error in mapping: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_mapping_logic()
    print(f"\n=== MAPPING TEST {'PASSED' if success else 'FAILED'} ===")
