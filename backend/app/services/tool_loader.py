# app/services/tool_loader.py
import importlib.util
import inspect
from pathlib import Path
import yaml
from typing import Dict, Any, Optional

class ToolLoader:
    def __init__(self):
        self.tool_dir = Path(__file__).parent.parent / "tools"
        self._cache = None

    def _parse_metadata(self, docstring: str) -> Dict[str, Any]:
        """Extracts YAML metadata from docstring"""
        if not docstring or "TOOL_DEFINITION" not in docstring:
            raise ValueError("Missing TOOL_DEFINITION marker")
        
        yaml_content = docstring.split("TOOL_DEFINITION")[1].strip()
        try:
            metadata = yaml.safe_load(yaml_content)
            if not isinstance(metadata, dict):
                raise ValueError("Metadata must be a dictionary")
            return metadata
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML: {str(e)}")

    def _validate_tool_module(self, module, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Flexible validation supporting both old and new tools"""
        # Try preferred function names in order
        for func_name in ["execute", "run"]:
            if hasattr(module, func_name) and callable(getattr(module, func_name)):
                if func_name == "run" and metadata is None:
                    print(f"⚠️ Deprecation: Tool {module.__name__} uses 'run()'. Migrate to 'execute()'")
                return True
        raise AttributeError("Tool requires either 'execute()' or 'run()' function")

    def load_tools(self, use_cache: bool = True) -> Dict[str, Any]:
        """Main loader with comprehensive error handling"""
        if use_cache and self._cache is not None:
            return self._cache
            
        tools = {}
        
        for tool_file in self.tool_dir.glob("*.py"):
            if tool_file.name.startswith("_"):
                continue
                
            module_name = f"app.tools.{tool_file.stem}"
            spec = importlib.util.spec_from_file_location(module_name, str(tool_file))
            module = importlib.util.module_from_spec(spec)
            
            try:
                spec.loader.exec_module(module)
                metadata = None
                
                # Parse metadata if available
                if module.__doc__:
                    try:
                        metadata = self._parse_metadata(module.__doc__)
                    except ValueError as e:
                        print(f"⚠️ Metadata error in {tool_file.stem}: {str(e)}")
                
                self._validate_tool_module(module, metadata)
                
                tool_name = metadata["name"] if metadata else tool_file.stem
                tools[tool_name] = {
                    "function": getattr(module, "execute", getattr(module, "run")),
                    "metadata": metadata or {"name": tool_name},
                    "module": module_name
                }
                
            except Exception as e:
                print(f"⚠️ Failed to load {tool_file.stem}: {str(e)}")
                continue
                
        self._cache = tools
        return tools

# Singleton instance
tool_loader = ToolLoader()

# Explicit export
load_tools = tool_loader.load_tools