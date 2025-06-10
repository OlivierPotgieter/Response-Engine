#!/usr/bin/env python3
"""
Test script to verify confidence calculation fix
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.processors.data_processor import process_customer_request
from modules.ai.intent_classifier import predict_customer_intent, is_comment_in_scope

def test_confidence_fix():
    """Test the confidence calculation fix with the failing case"""
    
    customer_comment = "Good afternoon, i am looking for a crosshair extreme vi motherboard you had for sale 2 years ago, i bought one but would need another it was the am4 platform x370 chipset."
    
    print("Testing confidence calculation fix...")
    print(f"Comment: {customer_comment[:50]}...")
    
    intent = predict_customer_intent(customer_comment)
    in_scope = is_comment_in_scope(customer_comment)
    
    print(f"Predicted intent: {intent}")
    print(f"In scope: {in_scope}")
    
    assert intent == "Pricing Inquiry", f"Expected 'Pricing Inquiry', got '{intent}'"
    assert in_scope == True, f"Expected True for in_scope, got {in_scope}"
    
    print("✅ Confidence fix test passed!")

if __name__ == "__main__":
    test_confidence_fix()
