#!/usr/bin/env python3
"""
Debug script to test enhanced generation logic without OpenAI API calls
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class MockOpenAI:
    def __init__(self, *args, **kwargs):
        pass

import modules.ai.llm_generator
modules.ai.llm_generator.OpenAI = MockOpenAI

from modules.ai.llm_generator import LLMGenerator

def test_enhanced_generation_logic():
    """Test enhanced generation logic without API calls"""
    
    print("=== ENHANCED GENERATION DEBUG ===")
    
    try:
        generator = LLMGenerator()
        print("✅ LLMGenerator initialized successfully")
        
        context = {
            'real_time_data': {
                'primary_product': {
                    'product_name': 'Gigabyte AORUS FI27Q-P 27" Monitor',
                    'sku': 'WO_GIGABYTE FI27QP',
                    'product_id': 35883,
                    'category': 'Monitor',
                    'is_viable': True,
                    'viability_reason': 'AI-identified with 64.8% relevance',
                    'pricing': {
                        'current_price': 'TBD',
                        'note': 'Pricing information not available - recommend contacting sales'
                    },
                    'stock': {
                        'is_in_stock': None,
                        'note': 'Stock information not available - recommend contacting sales'
                    }
                },
                'secondary_product': {},
                'data_source': 'ai_product_identification'
            },
            'product_search_result': {
                'best_match': {
                    'name': 'Gigabyte AORUS FI27Q-P 27" Monitor',
                    'sku': 'WO_GIGABYTE FI27QP',
                    'category': 'Monitor'
                },
                'confidence': 0.748
            },
            'examples': []
        }
        
        print(f"Context keys: {list(context.keys())}")
        
        print("\n--- Testing viability analysis ---")
        viability = generator._analyze_product_viability_for_prompt(context)
        print(f"Primary product viable: {viability.get('primary_product_viable', False)}")
        print(f"Has pricing data: {viability.get('has_pricing_data', False)}")
        print(f"Should provide pricing: {viability.get('should_provide_pricing', False)}")
        print(f"Viability keys: {list(viability.keys())}")
        
        print("\n--- Testing prompt building ---")
        try:
            prompt_parts = []
            customer_comment = "Hi there\n\nWould like to find out of Wootware is planning on importing the Gigabyte MO27Q28G 280hz Primary RGB Tandem WOLED monitor?"
            
            if not viability.get('primary_product_viable', False):
                product_search_result = context.get("product_search_result", {})
                best_match = product_search_result.get("best_match", {})
                
                if best_match and product_search_result.get("confidence", 0) > 0.5:
                    print("✅ AI-identified product logic should trigger")
                    print(f"Best match: {best_match.get('name', 'N/A')}")
                    print(f"Confidence: {product_search_result.get('confidence', 0):.1%}")
                else:
                    print("❌ AI-identified product logic will not trigger")
            else:
                print("✅ Primary product is viable - normal flow")
            
            print("✅ Enhanced generation logic analysis complete")
            return True
            
        except Exception as e:
            print(f"❌ Error in prompt building: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"❌ Error initializing generator: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_generation_logic()
    print(f"\n=== ENHANCED GENERATION TEST {'PASSED' if success else 'FAILED'} ===")
