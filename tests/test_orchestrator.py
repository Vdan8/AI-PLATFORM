import requests
import json
import asyncio
import os
import sys

# Add project root to sys.path if test script is not in the same directory as app
# Example: if test_orchestrator_dynamic.py is in 'backend/'
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
# If in 'project_root/scripts/' and app is in 'project_root/backend/app'
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))


BASE_URL = "http://localhost:8000" # Ensure this matches your running backend's port

async def send_prompt(prompt: str):
    url = f"{BASE_URL}/api/v1/prompt" # IMPORTANT: Adjust this to your actual API endpoint for processing prompts
    headers = {"Content-Type": "application/json"}
    data = {"user_prompt": prompt}

    print(f"\n--- Sending Prompt: '{prompt}' ---")
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status() # Raise an exception for HTTP errors (e.g., 4xx or 5xx)
        print("Response Status Code:", response.status_code)
        print("Response Body:")
        print(json.dumps(response.json(), indent=2))
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        if e.response:
            print("Error Response Body:", e.response.text)
        print("-" * 50) # Separator for errors

async def main():
    print("Starting dynamic orchestration tests...")

    # --- Test Cases ---

    # 1. Prompt that requires NEW TOOL GENERATION (e.g., factorial)
    # The LLM should recognize it doesn't have this capability and generate a tool.
    print("\n--- Test Case 1: Generate a new tool for factorial calculation ---")
    await send_prompt("I need a tool that can calculate the factorial of a given integer. It should take one integer parameter called 'number'.")
    # Give it a moment to process and save the tool
    await asyncio.sleep(2) # Adjust based on your system's speed

    # 2. Use the newly generated factorial tool
    # The LLM should now see the 'calculate_factorial' tool in its registry.
    print("\n--- Test Case 2: Use the newly generated factorial tool ---")
    await send_prompt("What is the factorial of 5?")
    await asyncio.sleep(2)

    # 3. A prompt that requires another new tool (e.g., summarizing text)
    print("\n--- Test Case 3: Generate a new tool for text summarization ---")
    await send_prompt("Can you create a tool to summarize long articles? It needs a 'text' parameter (string).")
    await asyncio.sleep(2)

    # 4. General conversational prompt (should result in 'respond' action)
    print("\n--- Test Case 4: General conversational query ---")
    await send_prompt("Hello, how are you today?")
    await asyncio.sleep(2)

    # 5. Complex sequential task (might generate multiple tools or chain existing/new ones)
    # This assumes a "weather" tool doesn't exist initially. The LLM might generate it, then use it.
    print("\n--- Test Case 5: Find weather and compare (likely multi-step/generation) ---")
    await send_prompt("Find the weather in London and tell me if it's above 15 degrees Celsius. The weather tool needs a 'location' parameter.")
    await asyncio.sleep(5) # Give more time for multi-step tasks

    # 6. Test case where required parameters are missing after tool identification
    print("\n--- Test Case 6: Missing required parameters ---")
    await send_prompt("Tell me the weather.") # Should ask for a location (assuming get_current_weather requires 'location')
    await asyncio.sleep(2)

    print("\nAll dynamic orchestration tests completed.")

if __name__ == "__main__":
    asyncio.run(main())