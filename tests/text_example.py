# test_app.py
import sys
import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
BASE_URL = "http://localhost:5001"
TEST_REQUEST_ID = "137807"
TEST_PRODUCT_ID = "33796"


def test_endpoint(endpoint, method="GET", data=None, expected_status=200):
    """
    Test a single endpoint

    Args:
        endpoint: API endpoint to test
        method: HTTP method
        data: Request data for POST requests
        expected_status: Expected HTTP status code

    Returns:
        Response object or None if failed
    """
    url = f"{BASE_URL}{endpoint}"

    try:
        print(f"Testing {method} {endpoint}...")

        if method == "GET":
            response = requests.get(url, timeout=30)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=30)
        else:
            print(f"❌ Unsupported method: {method}")
            return None

        # Check status code
        if response.status_code == expected_status:
            print(f"✅ Status: {response.status_code}")
        else:
            print(f"⚠️ Status: {response.status_code} (expected: {expected_status})")

        # Try to parse JSON response
        try:
            json_response = response.json()
            print(f"✅ JSON Response received")

            # Show key information
            if 'status' in json_response:
                print(f"   API Status: {json_response['status']}")
            if 'message' in json_response:
                print(f"   Message: {json_response['message'][:100]}...")
            if 'data' in json_response and isinstance(json_response['data'], dict):
                print(f"   Data Keys: {list(json_response['data'].keys())[:5]}")

            return json_response

        except json.JSONDecodeError:
            print(f"❌ Invalid JSON response")
            print(f"   Raw response: {response.text[:200]}...")
            return None

    except requests.ConnectionError:
        print(f"❌ Connection error - is the server running on {BASE_URL}?")
        return None
    except requests.Timeout:
        print(f"❌ Request timeout")
        return None
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return None


def test_basic_endpoints():
    """Test basic endpoints that don't require complex processing"""
    print("Testing basic endpoints...")
    print("=" * 50)

    # Test root endpoint
    response = test_endpoint("/")
    if response and response.get('status') == 'success':
        data = response.get('data', {})
        print(f"   Service: {data.get('service')}")
        print(f"   Version: {data.get('version')}")
        print(f"   Total Requests: {data.get('total_requests')}")

    print()

    # Test health endpoint
    response = test_endpoint("/health")
    if response and response.get('status') == 'success':
        data = response.get('data', {})
        print(f"   Health Status: {data.get('status')}")

    print()

    # Test stats endpoint
    response = test_endpoint("/stats")
    if response and response.get('status') == 'success':
        data = response.get('data', {})
        print(f"   Uptime: {data.get('uptime')}")

    print()


def test_debug_endpoints():
    """Test debug endpoints"""
    print("Testing debug endpoints...")
    print("=" * 50)

    # Test connections debug
    response = test_endpoint("/debug/connections")
    if response and response.get('status') == 'success':
        data = response.get('data', {})
        print(f"   Overall Status: {data.get('overall_status')}")

        # Show connection details
        if 'database' in data:
            print(f"   Database: {'✅' if data['database'].get('connected') else '❌'}")
        if 'pinecone' in data:
            pinecone = data['pinecone']
            print(f"   Pinecone: {'✅' if pinecone.get('pinecone_connected') else '❌'}")
            print(f"   OpenAI: {'✅' if pinecone.get('openai_connected') else '❌'}")
        if 'llm' in data:
            llm = data['llm']
            print(f"   LLM Generation: {'✅' if llm.get('test_successful') else '❌'}")

    print()

    # Test product debug
    response = test_endpoint(f"/debug/product/{TEST_PRODUCT_ID}")
    if response and response.get('status') == 'success':
        data = response.get('data', {})
        print(f"   Product Lookup: {'✅' if data.get('lookup_successful') else '❌'}")
        product_data = data.get('product_data', {})
        if product_data and not product_data.get('error'):
            print(f"   Product Name: {product_data.get('name', 'N/A')}")

    print()


def test_validation_endpoint():
    """Test validation endpoint"""
    print("Testing validation endpoint...")
    print("=" * 50)

    # Test with valid ID
    response = test_endpoint(f"/validate/{TEST_REQUEST_ID}")
    if response:
        if response.get('status') == 'success':
            data = response.get('data', {})
            print(f"   Valid: {data.get('valid')}")
            print(f"   Comment Preview: {data.get('customer_comment_preview', 'N/A')[:50]}...")
        else:
            print(f"   Error: {response.get('error', 'Unknown error')}")

    print()

    # Test with invalid ID
    response = test_endpoint("/validate/invalid123", expected_status=400)
    if response:
        print(f"   Invalid ID Error: {response.get('error', 'N/A')}")

    print()


def test_summary_endpoint():
    """Test summary endpoint"""
    print("Testing summary endpoint...")
    print("=" * 50)

    response = test_endpoint(f"/summary/{TEST_REQUEST_ID}")
    if response and response.get('status') == 'success':
        data = response.get('data', {})
        print(f"   Request ID: {data.get('request_id')}")
        print(f"   Intent: {data.get('predicted_intent')}")
        print(f"   In Scope: {data.get('is_in_scope')}")
        print(f"   Product ID: {data.get('product_id')}")
        print(f"   Rep: {data.get('woot_rep')}")

    print()


def test_test_endpoint():
    """Test the test endpoint (no AI generation)"""
    print("Testing test endpoint (no AI generation)...")
    print("=" * 50)

    response = test_endpoint(f"/test/{TEST_REQUEST_ID}")
    if response:
        if response.get('status') == 'success':
            data = response.get('data', {})
            print(f"   Request ID: {data.get('request_id')}")
            print(f"   Intent: {data.get('predicted_intent')}")

            # Check key sections
            print(f"   Has Product Details: {'product_details' in data}")
            print(f"   Has Data Log: {'data_log' in data}")
            print(f"   Has Processing Summary: {'processing_summary' in data}")

            # Show processing summary
            summary = data.get('processing_summary', {})
            print(f"   Products Found: {summary.get('products_found', 0)}")
            print(f"   Similar Responses: {summary.get('similar_responses_found', 0)}")

        elif response.get('status') == 'out_of_scope':
            print(f"   Out of Scope: {response.get('predicted_intent')}")
            print(f"   Reason: {response.get('reason', 'N/A')}")
        else:
            print(f"   Error: {response.get('error', 'Unknown error')}")

    print()


def test_process_endpoint():
    """Test the process endpoint (with AI generation)"""
    print("Testing process endpoint (with AI generation)...")
    print("=" * 50)

    # Ask user confirmation since this costs money
    confirmation = input("This will make OpenAI API calls and cost money. Continue? (y/n): ")
    if confirmation.lower() not in ['y', 'yes']:
        print("Skipping process endpoint test.")
        return

    print("Making API call to process endpoint...")
    response = test_endpoint(f"/process/{TEST_REQUEST_ID}")

    if response:
        if response.get('status') == 'success':
            data = response.get('data', {})
            print(f"   Request ID: {data.get('request_id')}")
            print(f"   Intent: {data.get('predicted_intent')}")
            print(f"   Generation Method: {data.get('generation_method', 'N/A')}")

            # Check generated response
            generated_response = data.get('generated_response')
            if generated_response:
                print(f"   Generated Response Length: {len(generated_response)}")
                print(f"   Response Preview: {generated_response[:100]}...")

            # Check response comparison
            comparison = data.get('response_comparison', {})
            if comparison.get('has_existing_response'):
                print(f"   Has Existing Response: ✅")
                print(f"   Existing Response Length: {comparison.get('existing_response_length', 0)}")
            else:
                print(f"   Has Existing Response: ❌")

            # Check Pinecone results
            pinecone = data.get('pinecone_results', {})
            print(f"   Similar Responses Found: {pinecone.get('similar_responses_found', 0)}")

        elif response.get('status') == 'out_of_scope':
            print(f"   Out of Scope: {response.get('predicted_intent')}")
            print(f"   Reason: {response.get('reason', 'N/A')}")
        else:
            print(f"   Error: {response.get('error', 'Unknown error')}")

    print()


def test_error_handling():
    """Test error handling with invalid requests"""
    print("Testing error handling...")
    print("=" * 50)

    # Test 404
    response = test_endpoint("/nonexistent", expected_status=404)
    if response:
        print(f"   404 Error: {response.get('error', 'N/A')}")

    print()

    # Test invalid request ID
    response = test_endpoint("/test/999999999", expected_status=404)
    if response:
        print(f"   Invalid ID Error: {response.get('error', 'N/A')}")

    print()


def run_comprehensive_test():
    """Run comprehensive API test suite"""
    print("Response Engine API Test Suite")
    print("=" * 60)
    print(f"Testing server at: {BASE_URL}")
    print(f"Test Request ID: {TEST_REQUEST_ID}")
    print(f"Test Product ID: {TEST_PRODUCT_ID}")
    print("=" * 60)

    try:
        # Test basic connectivity first
        print("\n1. BASIC CONNECTIVITY")
        test_basic_endpoints()

        print("\n2. DEBUG ENDPOINTS")
        test_debug_endpoints()

        print("\n3. VALIDATION")
        test_validation_endpoint()

        print("\n4. SUMMARY")
        test_summary_endpoint()

        print("\n5. TEST PROCESSING")
        test_test_endpoint()

        print("\n6. FULL PROCESSING")
        test_process_endpoint()

        print("\n7. ERROR HANDLING")
        test_error_handling()

        print("\n" + "=" * 60)
        print("✅ API test suite completed!")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n❌ Test suite interrupted by user")
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_comprehensive_test()
