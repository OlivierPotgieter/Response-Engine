#!/usr/bin/env python3
"""
Debug script to reproduce the exact JSON case and trace confidence flow
"""

import sys
import os
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.processors.data_processor import DataProcessor
from modules.ai.intent_classifier import predict_customer_intent, is_comment_in_scope
from modules.product_identification.confidence import should_use_extracted_product

def debug_json_case():
    """Debug the exact case from the JSON attachment"""
    
    print("=== DEBUGGING JSON CASE ===")
    
    json_path = "/home/ubuntu/attachments/3218ced3-0d3a-46fc-baf7-6c3312f17518/No+product+foudn+zero+conf.json"
    
    try:
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        
        print("JSON Data Keys:", list(json_data.keys()))
        
        customer_comment = json_data.get("customer_comment", "")
        if not customer_comment:
            for key in ["comment", "query", "request"]:
                if key in json_data:
                    customer_comment = json_data[key]
                    break
        
        print(f"Customer comment: {customer_comment[:100]}...")
        
        product_search = json_data.get("product_search_result", {})
        print(f"Product search attempted: {product_search.get('search_attempted', False)}")
        print(f"Has suggestions: {product_search.get('has_suggestions', False)}")
        print(f"Confidence: {product_search.get('confidence', 'NOT SET')}")
        print(f"Confidence level: {product_search.get('confidence_level', 'NOT SET')}")
        
        suggestions = product_search.get("suggestions", [])
        print(f"Number of suggestions: {len(suggestions)}")
        
        if suggestions:
            best_suggestion = suggestions[0]
            print(f"Best suggestion: {best_suggestion.get('name', 'Unknown')}")
            print(f"Relevance score: {best_suggestion.get('relevance_score', 'NOT SET')}")
        
        if product_search.get('confidence') is not None:
            test_confidence = {"confidence": product_search['confidence']}
            should_use, confidence_level = should_use_extracted_product(test_confidence)
            print(f"Threshold test: should_use={should_use}, level={confidence_level}")
        
        print()
        print("=== ANALYSIS ===")
        print("The issue appears to be that despite finding products with good relevance,")
        print("the confidence calculation is not properly flowing through the system.")
        
    except Exception as e:
        print(f"Error loading JSON: {e}")
        
        print("Using fallback test case...")
        customer_comment = "Good afternoon, i am looking for a crosshair extreme vi motherboard you had for sale 2 years ago, i bought one but would need another it was the am4 platform x370 chipset."
        
        print(f"Comment: {customer_comment[:80]}...")
        
        intent = predict_customer_intent(customer_comment)
        in_scope = is_comment_in_scope(customer_comment)
        print(f"Intent: {intent}, In scope: {in_scope}")

if __name__ == "__main__":
    debug_json_case()
