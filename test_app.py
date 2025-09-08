import requests
import sys

# Start the Flask app first (you'll need to run this manually)
base_url = "http://127.0.0.1:5000"

def test_prediction(test_name, data):
    print(f"\n{'='*50}")
    print(f"TEST: {test_name}")
    print(f"{'='*50}")
    print(f"Input data: {data}")
    
    try:
        response = requests.post(f"{base_url}/predict", data=data)
        print(f"Status Code: {response.status_code}")
        
        # Look for error messages in the response
        if "Error" in response.text or "error" in response.text.lower():
            # Extract the error message from the HTML
            import re
            error_pattern = r'<div class="result-text">(.*?)</div>'
            match = re.search(error_pattern, response.text)
            if match:
                error_msg = match.group(1)
                print(f"ERROR FOUND: {error_msg}")
            else:
                print("Error detected in response but couldn't extract exact message")
        else:
            # Extract success message
            import re
            success_pattern = r'<div class="result-text">(.*?)</div>'
            match = re.search(success_pattern, response.text)
            if match:
                result_msg = match.group(1)
                print(f"SUCCESS: {result_msg}")
            else:
                print("Response received but couldn't extract message")
                
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to Flask app. Make sure it's running on http://127.0.0.1:5000")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False
    
    return True

# Test cases
test_cases = [
    # Valid input
    ("Valid Input", {
        'date': '2024-05-15',
        'region': 'California',
        'total_volume': '50000',
        'total_bags': '25000',
        'small_bags': '20000',
        'large_bags': '4000',
        'xlarge_bags': '1000'
    }),
    
    # Missing date
    ("Missing Date", {
        'region': 'California',
        'total_volume': '50000',
        'total_bags': '25000',
        'small_bags': '20000',
        'large_bags': '4000',
        'xlarge_bags': '1000'
    }),
    
    # Invalid date format
    ("Invalid Date", {
        'date': 'invalid-date',
        'region': 'California',
        'total_volume': '50000',
        'total_bags': '25000',
        'small_bags': '20000',
        'large_bags': '4000',
        'xlarge_bags': '1000'
    }),
    
    # Missing region
    ("Missing Region", {
        'date': '2024-05-15',
        'total_volume': '50000',
        'total_bags': '25000',
        'small_bags': '20000',
        'large_bags': '4000',
        'xlarge_bags': '1000'
    }),
    
    # Invalid region
    ("Invalid Region", {
        'date': '2024-05-15',
        'region': 'InvalidRegion',
        'total_volume': '50000',
        'total_bags': '25000',
        'small_bags': '20000',
        'large_bags': '4000',
        'xlarge_bags': '1000'
    }),
    
    # Invalid numeric inputs
    ("Invalid Numbers", {
        'date': '2024-05-15',
        'region': 'California',
        'total_volume': 'invalid',
        'total_bags': 'abc',
        'small_bags': '20000',
        'large_bags': '4000',
        'xlarge_bags': '1000'
    }),
    
    # Negative numbers
    ("Negative Numbers", {
        'date': '2024-05-15',
        'region': 'California',
        'total_volume': '-50000',
        'total_bags': '25000',
        'small_bags': '20000',
        'large_bags': '4000',
        'xlarge_bags': '1000'
    }),
    
    # Empty form
    ("Empty Form", {}),
]

if __name__ == "__main__":
    print("Flask App Tester - This will help identify exact error messages")
    print("Make sure to run 'python app.py' in another terminal first!")
    input("Press Enter when Flask app is running...")
    
    for test_name, test_data in test_cases:
        if not test_prediction(test_name, test_data):
            print("Stopping tests due to connection error")
            break
        
    print(f"\n{'='*50}")
    print("All tests completed!")
    print(f"{'='*50}")
