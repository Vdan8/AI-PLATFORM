# app/tools/write_note.py

import os
import json

def execute(input: dict) -> dict:
    note = input.get("note", "No note provided")
    return {
        "status": "success",
        "output": f"Note received: {note}"
    }

if __name__ == "__main__":
    args = json.loads(os.environ.get("TOOL_ARGUMENTS", "{}"))
    result = execute(args)
    print(json.dumps(result))  # <- sandbox expects this output
