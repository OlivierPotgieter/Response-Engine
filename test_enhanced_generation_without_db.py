#!/usr/bin/env python3
"""
Test enhanced generation logic without database dependencies
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def mock_get_customer_request_data(request_id):
    """Mock customer request data for testing"""
    return {
        "id": int(request_id),
        "customer_comment": "Good afternoon, i am looking for a crosshair extreme vi motherboard you had for sale 2 years ago, i bought one but would need another it was the am4 platform x370 chipset.",
        "predicted_intent": "Pricing Inquiry",
        "rep_name": "Dylan",
        "created_at": "2024-01-15 10:30:00"
    }

def mock_get_custom_response_data(request_id):
    """Mock existing response data"""
    return None  # No existing response

import modules.database.main_db
modules.database.main_db.get_customer_request_data = mock_get_customer_request_data
modules.database.main_db.get_custom_response_data = mock_get_custom_response_data

class MockOpenAI:
    def __init__(self, *args, **kwargs):
        pass
    
    class chat:
        class completions:
            @staticmethod
            def create(*args, **kwargs):
                class MockResponse:
                    class choices:
                        class message:
                            content = """Good afternoon,

Thank you for your inquiry about the Asus ROG Crosshair VI Extreme motherboard.

🔍 PRODUCT IDENTIFICATION RESULTS:
Based on your description, I found: Asus ROG Crosshair VI Extreme AM4 X370 ATX Motherboard
• SKU: WO_ASUS_CROSSHAIR_VI_EXTREME
• Category: Motherboard
• Match Confidence: 74.8%

⚠️ PRICING AVAILABILITY NOTICE:
While I found a matching product, current pricing and availability information is not immediately available.
I recommend contacting our sales team for the most up-to-date pricing and stock information.

The Asus ROG Crosshair VI Extreme was indeed a popular choice for AM4 X370 platform builds. This motherboard featured premium components and excellent overclocking capabilities.

For the most current availability and pricing information, please contact our sales team at sales@wootware.co.za or call us directly.

Kind Regards,
Dylan"""
                        
                    choices = [choices()]
                
                return MockResponse()

import modules.ai.llm_generator
modules.ai.llm_generator.OpenAI = MockOpenAI

def mock_search_comment_for_products(comment, intent):
    """Mock product search results"""
    return {
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

import modules.processors.data_processor
modules.processors.data_processor.search_comment_for_products = mock_search_comment_for_products

from modules.processors.data_processor import DataProcessor

def test_enhanced_generation_without_db():
    """Test enhanced generation with mocked dependencies"""
    
    print("=== ENHANCED GENERATION TEST (NO DB) ===")
    
    try:
        processor = DataProcessor()
        
        print("1. Testing full request processing...")
        result = processor.process_full_request("251004")
        
        print(f"Processing successful: {result.get('success', False)}")
        print(f"Generation method: {result.get('generation_method', 'NOT_SET')}")
        
        generated_response = result.get('generated_response', '')
        print(f"Response length: {len(generated_response)}")
        print(f"Response preview: {generated_response[:200]}...")
        
        is_substantial = len(generated_response) > 100
        uses_enhanced = result.get('generation_method') != 'simple_fallback'
        has_product_info = 'Asus ROG Crosshair' in generated_response
        
        print(f"\nValidation:")
        print(f"  - Substantial response (>100 chars): {is_substantial}")
        print(f"  - Uses enhanced generation: {uses_enhanced}")
        print(f"  - Contains product information: {has_product_info}")
        
        success = is_substantial and uses_enhanced and has_product_info
        
        if success:
            print("✅ Enhanced generation produces full responses!")
        else:
            print("❌ Enhanced generation still producing minimal responses")
            
        return success
        
    except Exception as e:
        print(f"❌ Error in enhanced generation test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_generation_without_db()
    print(f"\n=== ENHANCED GENERATION TEST {'PASSED' if success else 'FAILED'} ===")
