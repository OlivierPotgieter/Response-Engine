#!/usr/bin/env python3
"""
Test script to verify startup behavior is fixed
"""

import subprocess
import time
import signal
import os

def test_startup_behavior():
    """Test that app.py starts without executing LLM queries"""
    
    print("=== STARTUP BEHAVIOR TEST ===")
    
    try:
        process = subprocess.Popen(
            ["python", "app.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd="/home/ubuntu/repos/Response-Engine"
        )
        
        time.sleep(5)
        
        process.terminate()
        output, _ = process.communicate(timeout=5)
        
        print("Startup output:")
        print(output)
        
        has_llm_queries = "HTTP Request: POST https://api.openai.com" in output
        has_import_errors = "cannot import name" in output
        starts_successfully = "Starting Flask application" in output
        
        print(f"\nValidation:")
        print(f"  - No LLM queries during startup: {not has_llm_queries}")
        print(f"  - No import errors: {not has_import_errors}")
        print(f"  - Starts successfully: {starts_successfully}")
        
        success = not has_llm_queries and not has_import_errors and starts_successfully
        
        return success
        
    except Exception as e:
        print(f"Error testing startup: {e}")
        return False

if __name__ == "__main__":
    success = test_startup_behavior()
    print(f"\n=== STARTUP TEST {'PASSED' if success else 'FAILED'} ===")
