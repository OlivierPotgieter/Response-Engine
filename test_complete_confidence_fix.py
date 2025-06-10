#!/usr/bin/env python3
"""
Test script to verify the complete confidence calculation fix
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.product_identification.confidence import should_use_extracted_product, ConfidenceThresholds

def test_complete_confidence_fix():
    """Test the complete confidence calculation fix with the failing case"""
    
    print("=== COMPLETE CONFIDENCE FIX TEST ===")
    print()
    
    print("1. CONFIDENCE THRESHOLD LOGIC TEST:")
    test_cases = [
        (0.692847526, "low_confidence"),     # From JSON attachment
        (0.85, "high_confidence"),           # High confidence
        (0.70, "medium_confidence"),         # Medium confidence boundary
        (0.50, "low_confidence"),            # Low confidence boundary
        (0.49, "insufficient_confidence"),   # Below minimum
    ]
    
    all_passed = True
    for confidence, expected_level in test_cases:
        test_input = {"confidence": confidence}
        should_use, confidence_level = should_use_extracted_product(test_input)
        
        status = "✅ PASS" if confidence_level == expected_level else "❌ FAIL"
        print(f"   Confidence {confidence:6.3f}: {status} - got '{confidence_level}', expected '{expected_level}'")
        
        if confidence_level != expected_level:
            all_passed = False
    
    print()
    print("2. SPECIFIC JSON CASE TEST:")
    json_confidence = {"confidence": 0.692847526}
    should_use, confidence_level = should_use_extracted_product(json_confidence)
    
    print(f"   JSON confidence 0.692847526:")
    print(f"   - should_use: {should_use}")
    print(f"   - confidence_level: {confidence_level}")
    print(f"   - Expected: should_use=True, confidence_level='low_confidence'")
    
    json_passed = should_use == True and confidence_level == "low_confidence"
    print(f"   - Result: {'✅ PASS' if json_passed else '❌ FAIL'}")
    
    print()
    print("3. RESPONSE GENERATION REQUIREMENTS:")
    print(f"   - should_use_for_pricing requires: medium_confidence or high_confidence")
    print(f"   - For 0.69 confidence (low_confidence): should_use_for_pricing = False")
    print(f"   - But should still generate full response, not minimal greeting")
    
    print()
    overall_status = "✅ ALL TESTS PASSED" if all_passed and json_passed else "❌ SOME TESTS FAILED"
    print(f"=== {overall_status} ===")
    
    return all_passed and json_passed

if __name__ == "__main__":
    test_complete_confidence_fix()
