import requests
import json

run_agent_url = "http://127.0.0.1:8000/run-agent"
headers = {"Content-Type": "application/json"}
payload = {
    "system_prompt": "You are a helpful test agent.",
    "goal": "Process this: Test Input with test_tool",
    "max_steps": 5
}

try:
    run_response = requests.post(run_agent_url, headers=headers, json=payload)
    run_response.raise_for_status()
    print("Run Agent Response:")
    print(run_response.json())
except requests.exceptions.RequestException as e:
    print(f"Error running agent: {e}")
    