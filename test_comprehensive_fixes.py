#!/usr/bin/env python3
"""
Comprehensive test for warranty response bug fixes
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

def test_sklearn_availability():
    """Test if sklearn is properly installed and working"""
    print("🔍 Testing sklearn Availability")
    print("=" * 50)
    
    try:
        import sklearn
        print(f"✅ sklearn available: {sklearn.__version__}")
        
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import Pipeline
        print("✅ sklearn modules import successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ sklearn import error: {e}")
        return False

def test_intent_classification():
    """Test intent classification with keyword fallback"""
    print("\n🤖 Testing Intent Classification")
    print("=" * 50)
    
    try:
        from modules.ai.intent_classifier import IntentClassifier
        
        classifier = IntentClassifier()
        
        warranty_comment = "what warranty does this product have?"
        result = classifier.predict_intent(warranty_comment)
        print(f"Warranty inquiry: '{warranty_comment}' -> '{result}'")
        
        if result == "Warranty Inquiry":
            print("✅ Warranty inquiry correctly classified")
            return True
        else:
            print(f"⚠️ Warranty inquiry misclassified as: {result}")
            return False
            
    except Exception as e:
        print(f"❌ Intent classifier error: {e}")
        return False

def test_pinecone_loading():
    """Test if Pinecone client loads CSV file correctly"""
    print("\n📊 Testing Pinecone CSV Loading")
    print("=" * 50)
    
    try:
        from modules.ai.pinecone_client import get_pinecone_client
        
        client = get_pinecone_client()
        print(f"Labeled replies loaded: {len(client.labeled_replies_lookup)}")
        
        if len(client.labeled_replies_lookup) > 0:
            print("✅ Sample warranty responses available for Pinecone matching")
            return True
        else:
            print("⚠️ No labeled replies loaded - will return empty results")
            return False
            
    except Exception as e:
        print(f"⚠️ Pinecone client test failed (expected without API keys): {e}")
        return False

if __name__ == "__main__":
    sklearn_ok = test_sklearn_availability()
    intent_ok = test_intent_classification()
    pinecone_ok = test_pinecone_loading()
    
    print(f"\n{'✅ SUCCESS' if all([sklearn_ok, intent_ok, pinecone_ok]) else '❌ ISSUES FOUND'}: Comprehensive system test")
    print(f"sklearn: {'✅' if sklearn_ok else '❌'}, intent_classification: {'✅' if intent_ok else '❌'}, pinecone_loading: {'✅' if pinecone_ok else '❌'}")
