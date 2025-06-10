#!/usr/bin/env python3
"""
Test script for Phase 1 confidence system implementation
"""

import sys
import os
sys.path.append('/home/ubuntu/repos/Response-Engine/modules/product_identification')

from modules.product_identification.confidence import should_use_extracted_product, ResponseValidator, ConfidenceThresholds

def test_confidence_thresholds():
    """Test confidence threshold logic"""
    test_cases = [
        (0.9, True, "high_confidence"),
        (0.75, True, "medium_confidence"), 
        (0.55, True, "low_confidence"),
        (0.4, False, "insufficient_confidence")
    ]
    
    print("🧪 Testing Confidence Thresholds")
    print("=" * 40)
    
    for confidence, expected_use, expected_level in test_cases:
        result = {"confidence": confidence, "best_match": {"name": "Test Product"}}
        should_use, level = should_use_extracted_product(result)
        
        status = "✅" if (should_use == expected_use and level == expected_level) else "❌"
        print(f"{status} Confidence {confidence}: use={should_use}, level={level}")

def test_response_validator():
    """Test response validation"""
    validator = ResponseValidator()
    
    test_response = "Thank you for your inquiry. The RTX 4090 is currently priced at R25,999. Please let me know if you need any additional information."
    test_context = {
        "predicted_intent": "pricing",
        "product_search_result": {"confidence": 0.85}
    }
    
    is_valid, checks = validator.validate_response(test_response, test_context)
    
    print(f"\n🧪 Testing Response Validator")
    print("=" * 40)
    print(f"Response valid: {is_valid}")
    print(f"Validation checks: {checks}")

if __name__ == "__main__":
    test_confidence_thresholds()
    test_response_validator()
