#!/usr/bin/env python3
"""
Test and install sklearn if needed
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

def test_and_install_sklearn():
    """Test if sklearn is available and install if needed"""
    try:
        import sklearn
        print(f"✅ sklearn available: {sklearn.__version__}")
        return True
    except ImportError as e:
        print(f"❌ sklearn import error: {e}")
        print("Installing sklearn...")
        import subprocess
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'scikit-learn'], check=True)
        import sklearn
        print(f"✅ sklearn installed: {sklearn.__version__}")
        return True

if __name__ == "__main__":
    test_and_install_sklearn()
