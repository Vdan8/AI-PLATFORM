# backend/app/services/tool_registry.py

from typing import Dict, Any
from .tool_loader import load_tools

# Load all tools dynamically
tool_registry: Dict[str, Dict[str, Any]] = load_tools()

def get_tool_definitions() -> list[Dict[str, Any]]:
    """
    Returns a list of OpenAI-compatible tool definitions.
    """
    tool_definitions = []
    for tool in tool_registry.values():
        if (
            "metadata" in tool
            and isinstance(tool["metadata"], dict)
            and "name" in tool["metadata"]
            and "description" in tool["metadata"]
        ):
            parameters = tool["metadata"].get("parameters", {})
            properties = {}
            required = []
            for param_name, param_details in parameters.items():
                if isinstance(param_details, dict):
                    properties[param_name] = {
                        "type": param_details.get("type", "string"),  # Default to string
                        "description": param_details.get("description", ""),
                    }
                    if param_details.get("required", False):
                        required.append(param_name)

            definition = {
                "type": "function",
                "function": {
                    "name": tool["metadata"]["name"],
                    "description": tool["metadata"]["description"],
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            }
            tool_definitions.append(definition)
        else:
            print(f"⚠️ Tool {tool.get('module', 'unknown')} has incomplete metadata and will be skipped.")
    return tool_definitions

def call_tool(name: str, arguments: dict) -> Any:
    """
    Executes the registered tool by name, passing in arguments.
    """
    if name not in tool_registry:
        raise ValueError(f"Tool '{name}' not found in registry.")

    tool_data = tool_registry[name]
    tool_function = tool_data.get("function")
    if not callable(tool_function):
        raise AttributeError(f"Tool '{name}' has no callable 'function'.")

    return tool_function(arguments)