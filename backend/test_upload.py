import requests
import sys

def test_upload():
    url = "http://127.0.0.1:5000/api/upload"
    
    with open(r"c:\Users\AYMAN KHAN\Hackday\frontend\sample_dataset.csv", "rb") as f:
        csv_content = f.read()
    files = {
        'file': ('sample_dataset.csv', csv_content, 'text/csv')
    }
    
    try:
        print(f"Sending POST to {url}...")
        response = requests.post(url, files=files)
        print(f"Status Code: {response.status_code}")
        print("Response JSON:")
        try:
            print(response.json())
        except:
            print(response.text)
    except Exception as e:
        print(f"Connection error: {e}")

if __name__ == "__main__":
    test_upload()
