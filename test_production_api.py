"""
Test the production API endpoints
"""
import requests
import json

def test_production_api():
    """Comprehensive API testing"""
    base_url = "http://localhost:5000"
    
    print("🧪 Testing FinSaathi Production API...")
    print("=" * 60)
    
    # Test 1: Health check
    print("\n1️⃣ Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            health_data = response.json()
            print(f"Models loaded: {health_data['models_loaded']}")
            print(f"Model status: {health_data['model_status']}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
    
    # Test 2: SMS categorization (your trained model)
    print("\n2️⃣ Testing SMS categorization...")
    test_cases = [
        {
            "name": "Healthcare (Clinical Laboratory)",
            "sms": "A/c *5678 debited Rs. 970.00 on 10-05-25 to UMA CLINICAL LABORATORY. UPI:882918376710"
        },
        {
            "name": "Shopping (Myntra)",
            "sms": "Rs.1250.00 debited from A/c XX1234 on 15-Oct-25 to MYNTRA FASHION STORE for online purchase"
        },
        {
            "name": "Fuel (Petrol Pump)",
            "sms": "Payment of Rs.45.50 made to INDIAN OIL PETROL PUMP on 15-Oct-25 via UPI"
        },
        {
            "name": "Food & Dining (Starbucks)",
            "sms": "Account debited Rs.285.50 on 15-Oct-25 at STARBUCKS COFFEE STORE for Card ending 1234"
        },
        {
            "name": "Utilities (Power Company)",
            "sms": "A/c debited Rs.471.00 on 23-05-25 to EASTERN POWER DISTRIBUTION COMPANY LIMITED OF ANDHRA PRADESH"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        try:
            response = requests.post(
                f"{base_url}/api/categorize/sms",
                json={"sms_text": test_case["sms"]},
                headers={'Content-Type': 'application/json'}
            )
            
            print(f"\n   📱 Test {i}: {test_case['name']}")
            if response.status_code == 200:
                result = response.json()
                categorization = result['result']
                print(f"   ✅ Category: {categorization['category']}")
                print(f"   🎯 Confidence: {categorization['confidence']:.2f}")
                print(f"   🔧 Method: {categorization['method']}")
                print(f"   🏪 Merchant: {categorization['merchant_detected']}")
            else:
                print(f"   ❌ Failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"   ❌ Request failed: {e}")
    
    # Test 3: Batch processing
    print("\n3️⃣ Testing batch processing...")
    try:
        batch_data = {
            "sms_list": [
                "A/c debited Rs.500 to APOLLO PHARMACY",
                "Payment Rs.1200 to AMAZON INDIA",
                "UPI to UBER INDIA Rs.150"
            ]
        }
        
        response = requests.post(
            f"{base_url}/api/categorize/batch",
            json=batch_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Processed: {result['processed_count']} SMS messages")
            for item in result['results'][:2]:  # Show first 2 results
                if item['success']:
                    cat = item['result']
                    print(f"   📱 SMS {item['index'] + 1}: {cat['category']} ({cat['confidence']:.2f})")
        else:
            print(f"   ❌ Batch failed: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Batch test failed: {e}")
    
    # Test 4: Built-in test endpoint
    print("\n4️⃣ Testing built-in test endpoint...")
    try:
        response = requests.get(f"{base_url}/api/test")
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Built-in tests completed")
            for i, test_result in enumerate(result['test_results'][:2], 1):
                if 'category' in test_result:
                    print(f"   📱 Test {i}: {test_result['category']} ({test_result['confidence']:.2f})")
        else:
            print(f"   ❌ Built-in test failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Built-in test failed: {e}")
    
    print("\n" + "=" * 60)
    print("✅ API Testing Complete!")

if __name__ == '__main__':
    test_production_api()