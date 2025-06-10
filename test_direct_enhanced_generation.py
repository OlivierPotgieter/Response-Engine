#!/usr/bin/env python3
"""
Direct test of enhanced generation methods with mapped context
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class MockOpenAI:
    def __init__(self, *args, **kwargs):
        pass
    
    class chat:
        class completions:
            @staticmethod
            def create(*args, **kwargs):
                messages = kwargs.get('messages', [])
                prompt_content = ""
                for msg in messages:
                    if msg.get('role') == 'user':
                        prompt_content = msg.get('content', '')
                        break
                
                print(f"LLM Prompt length: {len(prompt_content)}")
                print(f"Prompt preview: {prompt_content[:300]}...")
                
                should_trigger = ("PRICING INQUIRY" in prompt_content and 
                    ("Asus ROG Crosshair" in prompt_content or "crosshair extreme vi" in prompt_content.lower() or 
                     "motherboard" in prompt_content.lower()))
                
                print(f"MockOpenAI trigger check: {should_trigger}")
                print(f"  - Has PRICING INQUIRY: {'PRICING INQUIRY' in prompt_content}")
                print(f"  - Has motherboard: {'motherboard' in prompt_content.lower()}")
                print(f"  - Has crosshair: {'crosshair' in prompt_content.lower()}")
                
                if should_trigger:
                    response_content = """Good afternoon,

Thank you for your inquiry about the Asus ROG Crosshair VI Extreme motherboard.

Based on your description, I found: Asus ROG Crosshair VI Extreme AM4 X370 ATX Motherboard
• SKU: WO_ASUS_CROSSHAIR_VI_EXTREME  
• Category: Motherboard
• Match Confidence: 74.8%

While I found a matching product, current pricing and availability information is not immediately available.
I recommend contacting our sales team for the most up-to-date pricing and stock information.

The Asus ROG Crosshair VI Extreme was indeed a popular choice for AM4 X370 platform builds. This motherboard featured premium components and excellent overclocking capabilities, making it ideal for enthusiast builds.

For the most current availability and pricing information, please contact our sales team at sales@wootware.co.za or call us directly. We'll be happy to help you find another unit or suggest suitable alternatives if needed."""
                    
                    print(f"MockOpenAI returning comprehensive response, length: {len(response_content)}")
                else:
                    response_content = """Good afternoon,

Thank you for your inquiry about the motherboard.

I understand you're looking for a specific component, but I don't have immediate access to detailed product information for that particular model.

I recommend contacting our sales team directly for the most up-to-date information on availability and pricing."""
                    
                    print(f"MockOpenAI returning fallback response, length: {len(response_content)}")
                
                class MockMessage:
                    def __init__(self, content):
                        self.content = content
                
                class MockChoice:
                    def __init__(self, content):
                        self.message = MockMessage(content)
                
                class MockResponse:
                    def __init__(self, content):
                        self.choices = [MockChoice(content)]
                
                return MockResponse(response_content)

import modules.ai.llm_generator
modules.ai.llm_generator.OpenAI = MockOpenAI

from modules.ai.llm_generator import LLMGenerator
from modules.processors.data_processor import DataProcessor

def test_direct_enhanced_generation():
    """Test enhanced generation methods directly with proper context"""
    
    print("=== DIRECT ENHANCED GENERATION TEST ===")
    
    try:
        generator = LLMGenerator()
        processor = DataProcessor()
        
        print("1. Creating enhanced context with mapped data...")
        
        customer_comment = "Good afternoon, i am looking for a crosshair extreme vi motherboard you had for sale 2 years ago, i bought one but would need another it was the am4 platform x370 chipset."
        
        enhanced_context = {
            "predicted_intent": "Pricing Inquiry",
            "data_needs": {
                "prompt_strategy": "pricing_enhanced",
                "needs_pricing": True
            },
            "product_search_result": {
                "best_match": {
                    "name": "Asus ROG Crosshair VI Extreme AM4 X370 ATX Motherboard",
                    "product_id": 12345,
                    "relevance_score": 0.593,
                    "sku": "WO_ASUS_CROSSHAIR_VI_EXTREME",
                    "category": "Motherboard"
                },
                "confidence": 0.748,
                "has_suggestions": True
            },
            "product_selection": {
                "primary_product": {
                    "product_id": 12345,
                    "product_name": "Asus ROG Crosshair VI Extreme AM4 X370 ATX Motherboard",
                    "sku": "WO_ASUS_CROSSHAIR_VI_EXTREME",
                    "category": "Motherboard",
                    "relevance_score": 0.593,
                    "is_viable": True,
                    "viability_reason": "AI-identified product with 59.3% relevance"
                },
                "has_product_data": True,
                "selection_reason": "Using AI-identified product match"
            },
            "real_time_data": {
                "primary_product": {
                    "product_name": "Asus ROG Crosshair VI Extreme AM4 X370 ATX Motherboard",
                    "sku": "WO_ASUS_CROSSHAIR_VI_EXTREME",
                    "product_id": 12345,
                    "category": "Motherboard",
                    "is_viable": True,
                    "viability_reason": "AI-identified with 59.3% relevance",
                    "pricing": {
                        "current_price": "TBD",
                        "note": "Pricing information not available - recommend contacting sales"
                    },
                    "stock": {
                        "is_in_stock": None,
                        "note": "Stock information not available - recommend contacting sales"
                    }
                },
                "secondary_product": {},
                "data_source": "ai_product_identification"
            },
            "examples": []
        }
        
        print("2. Testing viability analysis...")
        viability = generator._analyze_product_viability_for_prompt(enhanced_context)
        print(f"   Primary product viable: {viability.get('primary_product_viable', False)}")
        print(f"   Has pricing data: {viability.get('has_pricing_data', False)}")
        
        print("3. Testing enhanced response generation...")
        
        import logging
        logging.basicConfig(level=logging.INFO)
        
        original_generate = generator.generate_enhanced_response
        
        def debug_generate_enhanced_response(customer_comment, context, woot_rep=None):
            try:
                import modules.ai.llm_generator
                original_client_create = generator.client.chat.completions.create
                
                def debug_client_create(*args, **kwargs):
                    print(f"DEBUG: OpenAI call with args: {len(args)}, kwargs keys: {list(kwargs.keys())}")
                    response = original_client_create(*args, **kwargs)
                    generated_reply = response.choices[0].message.content.strip()
                    print(f"DEBUG: OpenAI returned response length: {len(generated_reply)}")
                    print(f"DEBUG: OpenAI response preview: {generated_reply[:200]}...")
                    
                    is_minimal = generator._is_response_too_minimal(generated_reply)
                    print(f"DEBUG: Raw response detected as minimal: {is_minimal}")
                    
                    if not is_minimal:
                        print(f"DEBUG: Response passed minimal check, should proceed to signature appending")
                        final_reply = generator._append_rep_signature(generated_reply, "Dylan")
                        print(f"DEBUG: After signature appending, length: {len(final_reply)}")
                        print(f"DEBUG: Final reply preview: {final_reply[:200]}...")
                    
                    return response
                
                generator.client.chat.completions.create = debug_client_create
                result = original_generate(customer_comment, context, woot_rep)
                generator.client.chat.completions.create = original_client_create
                return result
                
            except Exception as e:
                print(f"DEBUG: Error in debug wrapper: {e}")
                return original_generate(customer_comment, context, woot_rep)
        
        generator.generate_enhanced_response = debug_generate_enhanced_response
        
        response = generator.generate_enhanced_response(customer_comment, enhanced_context, "Dylan")
        
        print(f"\nGenerated response:")
        print(f"Length: {len(response)} characters")
        print(f"Preview: {response[:200]}...")
        
        print(f"\n4. Testing minimal response detection...")
        is_minimal = generator._is_response_too_minimal(response)
        print(f"   Response detected as minimal: {is_minimal}")
        
        # Test what happens if we remove the signature
        response_without_sig = response.replace("Kind Regards,\nDylan", "").strip()
        print(f"   Response without signature length: {len(response_without_sig)}")
        is_minimal_without_sig = generator._is_response_too_minimal(response_without_sig)
        print(f"   Response without signature detected as minimal: {is_minimal_without_sig}")
        
        is_substantial = len(response) > 100
        has_product_info = "Asus ROG Crosshair" in response or "Crosshair VI Extreme" in response
        has_confidence_info = "74.8%" in response or "Match Confidence" in response
        not_minimal = "Kind Regards" not in response[:50]  # Not just greeting + signature
        
        print(f"\nValidation:")
        print(f"  - Substantial length (>100 chars): {is_substantial}")
        print(f"  - Contains product information: {has_product_info}")
        print(f"  - Contains confidence information: {has_confidence_info}")
        print(f"  - Not minimal response: {not_minimal}")
        
        success = is_substantial and has_product_info and not_minimal
        
        if success:
            print("✅ Enhanced generation produces comprehensive responses!")
        else:
            print("❌ Enhanced generation still has issues")
            
        return success
        
    except Exception as e:
        print(f"❌ Error in direct enhanced generation test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_direct_enhanced_generation()
    print(f"\n=== DIRECT ENHANCED GENERATION TEST {'PASSED' if success else 'FAILED'} ===")
