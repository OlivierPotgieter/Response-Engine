#!/usr/bin/env python3
"""
Test script to verify confidence threshold logic works correctly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.product_identification.confidence import should_use_extracted_product, ConfidenceThresholds

def test_confidence_threshold_logic():
    """Test the confidence threshold logic with various values"""
    
    print("=== CONFIDENCE THRESHOLD LOGIC TEST ===")
    print(f"HIGH_CONFIDENCE threshold: {ConfidenceThresholds.HIGH_CONFIDENCE}")
    print(f"MEDIUM_CONFIDENCE threshold: {ConfidenceThresholds.MEDIUM_CONFIDENCE}")
    print(f"LOW_CONFIDENCE threshold: {ConfidenceThresholds.LOW_CONFIDENCE}")
    print(f"MINIMUM_THRESHOLD: {ConfidenceThresholds.MINIMUM_THRESHOLD}")
    print()
    
    test_cases = [
        0.692847526,  # From JSON attachment
        0.85,         # High confidence boundary
        0.84,         # Just below high
        0.70,         # Medium confidence boundary
        0.69,         # Just below medium
        0.50,         # Low confidence boundary
        0.49,         # Just below minimum
        0.0,          # Zero confidence
        1.0,          # Maximum confidence
    ]
    
    for confidence in test_cases:
        test_input = {"confidence": confidence}
        should_use, confidence_level = should_use_extracted_product(test_input)
        print(f"Confidence {confidence:6.3f}: should_use={should_use:5}, level={confidence_level}")
    
    print()
    print("=== SPECIFIC TEST CASE FROM JSON ===")
    json_confidence = {"confidence": 0.692847526}
    should_use, confidence_level = should_use_extracted_product(json_confidence)
    print(f"JSON confidence 0.692847526: should_use={should_use}, level={confidence_level}")
    
    expected_level = "low_confidence"
    if confidence_level != expected_level:
        print(f"❌ ERROR: Expected '{expected_level}', got '{confidence_level}'")
    else:
        print(f"✅ CORRECT: Confidence level is '{confidence_level}' as expected")
    
    print()
    print("=== END THRESHOLD TEST ===")

if __name__ == "__main__":
    test_confidence_threshold_logic()
