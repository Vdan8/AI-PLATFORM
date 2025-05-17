# backend/app/services/tool_loader.py

import importlib.util
import inspect
import yaml
from pathlib import Path

TOOL_DIR = Path(__file__).parent.parent / "tools"

def parse_docstring(doc: str):
    try:
        # Attempt to parse YAML-style metadata from top of docstring
        metadata = yaml.safe_load(doc)
        if not isinstance(metadata, dict):
            raise ValueError("Docstring did not parse to dict")
        return metadata
    except Exception as e:
        raise ValueError(f"Invalid tool docstring format: {e}")

def load_python_tools():
    tool_registry = {}

    for file in TOOL_DIR.glob("*.py"):
        module_name = file.stem
        spec = importlib.util.spec_from_file_location(module_name, str(file))
        module = importlib.util.module_from_spec(spec)

        try:
            spec.loader.exec_module(module)
        except Exception as e:
            print(f"❌ Failed to load tool {module_name}: {e}")
            continue

        if not hasattr(module, "run") or not callable(module.run):
            print(f"⚠️ Tool {module_name} does not have a callable 'run(args)' function.")
            continue

        doc = inspect.getdoc(module)
        if not doc:
            print(f"⚠️ Tool {module_name} missing docstring metadata.")
            continue

        try:
            meta = parse_docstring(doc)
            if not isinstance(meta, dict):
                print(f"⚠️ Docstring in {module_name} did not parse to a dict.")
                continue
        except ValueError as e:
            print(f"⚠️ Failed to parse docstring in {module_name}: {e}")
            continue

        name = meta.get("name")
        if not name:
            print(f"⚠️ Tool {module_name} missing 'name' in docstring.")
            continue

        description = meta.get("description", "No description provided.")
        parameters = meta.get("parameters", {})

        tool_registry[name] = {
            "definition": {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": parameters,
                        "required": list(parameters.keys())
                    }
                }
            },
            "module": module
        }


    return tool_registry
