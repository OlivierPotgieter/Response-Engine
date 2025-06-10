#!/usr/bin/env python3
"""
Debug script to identify the 500 error in Flask app
"""

import sys
import os
import subprocess
import time
import requests
import json

def debug_flask_500():
    """Debug the 500 error by running Flask with detailed logging"""
    
    print("=== FLASK 500 ERROR DEBUG ===")
    
    print("Starting Flask app with debug logging...")
    
    env = os.environ.copy()
    env['FLASK_ENV'] = 'development'
    env['FLASK_DEBUG'] = '1'
    
    proc = subprocess.Popen(
        [sys.executable, "app.py"], 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        cwd="/home/ubuntu/repos/Response-Engine"
    )
    
    time.sleep(8)
    
    try:
        print("Making request to /process/251004...")
        response = requests.get("http://localhost:5001/process/251004", timeout=30)
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 500:
            print("500 error occurred - checking Flask logs...")
            
            proc.poll()
            if proc.stdout:
                output = proc.stdout.read()
                if output:
                    print("Flask output:")
                    print("-" * 50)
                    print(output)
                    print("-" * 50)
        else:
            print(f"Response content: {response.text[:500]}...")
            
    except Exception as e:
        print(f"Request failed: {e}")
        
    finally:
        print("Terminating Flask app...")
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    debug_flask_500()
