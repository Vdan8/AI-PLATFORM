# backend/app/services/tool_registry.py

from typing import Dict, Any
from app.services.tool_loader import load_python_tools

# Load all tools dynamically from Python files in the tools directory
tool_registry: Dict[str, Dict[str, Any]] = load_python_tools()

def get_tool_definitions() -> list[Dict[str, Any]]:
    """
    Returns a list of OpenAI-compatible tool definitions.
    """
    return [tool["definition"] for tool in tool_registry.values()]

def call_tool(name: str, arguments: dict) -> Any:
    """
    Executes the registered tool by name, passing in arguments.
    """
    if name not in tool_registry:
        raise ValueError(f"Tool '{name}' not found in registry.")
    
    module = tool_registry[name]["module"]
    
    if not hasattr(module, "run"):
        raise AttributeError(f"Tool '{name}' has no 'run' method.")

    return module.run(arguments)
