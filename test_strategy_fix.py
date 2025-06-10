#!/usr/bin/env python3
"""
Test script to verify strategy matching fix
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.processors.data_processor import DataProcessor

def test_strategy_matching():
    """Test that pricing_enhanced strategy triggers enhanced generation"""
    
    print("=== STRATEGY MATCHING FIX TEST ===")
    
    customer_comment = "I'm looking for a specific component\n\nComment:\n\nHi there\n\nWould like to find out of Wootware is planning on importing the Gigabyte MO27Q28G 280hz Primary RGB Tandem WOLED monitor?\nGigabyte is launching this at $500 MSRP, looks like it could be fantastic one to bring in if the price is right.\nArticle below\nhttps://tftcentral.co.uk/news/gigabyte-announce-mo27q28g-with-a-27-primary-rgb-tandem-woled-panel"
    
    processor = DataProcessor()
    
    data_needs = processor.detect_data_needs(customer_comment, "General Inquiry")
    print(f"Prompt strategy: {data_needs.get('prompt_strategy')}")
    print(f"Needs pricing: {data_needs.get('needs_pricing')}")
    
    assert data_needs.get('prompt_strategy') == 'pricing_enhanced', f"Expected pricing_enhanced, got {data_needs.get('prompt_strategy')}"
    
    print("✅ Strategy matching fix verified!")
    return True

if __name__ == "__main__":
    test_strategy_matching()
