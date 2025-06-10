#!/usr/bin/env python3
"""
Test the Pinecone response_examples fix
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

def test_pinecone_csv_loading():
    """Test if labeled_full_replies.csv loads correctly"""
    print("🔍 Testing Pinecone CSV Loading")
    print("=" * 50)
    
    try:
        from modules.ai.pinecone_client import get_pinecone_client
        
        client = get_pinecone_client()
        print(f"✅ Labeled replies loaded: {len(client.labeled_replies_lookup)}")
        
        for reply_id, data in client.labeled_replies_lookup.items():
            if "crosshair" in data.get("customer_comment", "").lower():
                print(f"✅ Found crosshair example: ID {reply_id}")
                break
        else:
            print("⚠️ No crosshair example found in loaded data")
            
        return len(client.labeled_replies_lookup) > 0
        
    except Exception as e:
        print(f"❌ Pinecone client test failed: {e}")
        return False

def test_response_builder_fix():
    """Test if response builder uses similar_responses_result correctly"""
    print("\n🔧 Testing Response Builder Fix")
    print("=" * 50)
    
    try:
        from modules.processors.response_builder import ResponseBuilder
        
        mock_processing_result = {
            "status": "success",
            "request_data": {"data": {"customer_comment": "test comment"}},
            "intent_result": {"predicted_intent": "Pricing Inquiry"},
            "data_needs_analysis": {"needs_pricing": True, "detection_reason": "test"},
            "enhanced_context": {},
            "product_result": {},
            "similar_responses_result": {
                "similar_responses_found": 3,
                "search_result": {
                    "response_examples": [
                        {"id": "205065", "text": "Example response 1"},
                        {"id": "202524", "text": "Example response 2"},
                        {"id": "196012", "text": "Example response 3"}
                    ],
                    "top_matches": [
                        {"id": "205065", "score": 0.7855},
                        {"id": "202524", "score": 0.7846},
                        {"id": "196012", "score": 0.7795}
                    ]
                }
            },
            "existing_response_result": {"has_existing_response": False},
            "generated_response": "Test response",
            "generation_method": "enhanced_pricing_focused",
            "processing_summary": {}
        }
        
        builder = ResponseBuilder()
        response = builder.build_process_response("test_id", mock_processing_result)
        
        pinecone_results = response["data"]["pinecone_results"]
        
        if pinecone_results["similar_responses_found"] == 3:
            print("✅ Similar responses count correctly extracted")
        else:
            print(f"❌ Expected 3 similar responses, got {pinecone_results['similar_responses_found']}")
            
        if len(pinecone_results["response_examples"]) > 0:
            print(f"✅ Response examples extracted: {len(pinecone_results['response_examples'])}")
            return True
        else:
            print("❌ Response examples still empty")
            return False
            
    except Exception as e:
        print(f"❌ Response builder test failed: {e}")
        return False

if __name__ == "__main__":
    csv_ok = test_pinecone_csv_loading()
    builder_ok = test_response_builder_fix()
    
    print(f"\n{'✅ SUCCESS' if all([csv_ok, builder_ok]) else '❌ ISSUES FOUND'}: Pinecone response fix test")
    print(f"CSV loading: {'✅' if csv_ok else '❌'}, Response builder: {'✅' if builder_ok else '❌'}")
