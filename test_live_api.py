#!/usr/bin/env python3
"""
Test script to verify the complete fix with live API
"""

import json
import requests
import time
import subprocess
import sys

def test_live_api():
    """Test the actual API endpoint with the failing case"""
    
    print("=== LIVE API TEST ===")
    
    print("Starting Flask app...")
    proc = subprocess.Popen([sys.executable, "app.py"], 
                          stdout=subprocess.PIPE, 
                          stderr=subprocess.PIPE)
    
    time.sleep(8)
    
    try:
        print("Testing request ID 251004...")
        response = requests.get("http://localhost:5001/process/251004", timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            generation_method = data['data'].get('generation_method', 'NOT_FOUND')
            response_length = len(data['data'].get('generated_response', ''))
            prompt_strategy = data['data'].get('processing_summary', {}).get('prompt_strategy', 'NOT_FOUND')
            confidence = data['data'].get('product_search_result', {}).get('confidence', 'NOT_FOUND')
            
            print(f"Generation method: {generation_method}")
            print(f"Response length: {response_length}")
            print(f"Prompt strategy: {prompt_strategy}")
            print(f"Confidence: {confidence}")
            
            if generation_method != 'simple_fallback' and response_length > 50:
                print("✅ SUCCESS: Enhanced generation is now working!")
                return True
            else:
                print("❌ FAILURE: Still using simple_fallback or short response")
                return False
        else:
            print(f"❌ API request failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing API: {e}")
        return False
    finally:
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    success = test_live_api()
    print()
    print(f"=== TEST {'PASSED' if success else 'FAILED'} ===")
