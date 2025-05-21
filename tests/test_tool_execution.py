# test_tool_execution.py
from app.services.tool_loader import load_tools

if __name__ == "__main__":
    tools = load_tools()
    print("Available tools:", list(tools.keys()))  # Verify loading
    
    # Test specific tool
    if "test_tool" in tools:
        result = tools["test_tool"]["function"]({"input": "test_data"})
        print("Tool execution result:", result)
    else:
        print("Error: test_tool not found")