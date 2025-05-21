import requests
import json

register_url = "http://127.0.0.1:8000/auth/register"
register_data = {"email": "testuser@example.com", "password": "testpassword"}

try:
    register_response = requests.post(register_url, json=register_data)
    register_response.raise_for_status()
    token_data = register_response.json()
    access_token = token_data.get("access_token")
    print("Registration Successful!")
    print(f"Access Token: {access_token}")
    YOUR_ACCESS_TOKEN = access_token # Store the token for later use
except requests.exceptions.RequestException as e:
    print(f"Error registering user: {e}")
    print(register_response.text)
    YOUR_ACCESS_TOKEN = None