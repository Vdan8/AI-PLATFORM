"""TOOL_DEFINITION
name: test_tool
description: Example test tool
parameters:
  input:
    type: string
    required: true
"""
def execute(args: dict) -> str:
    """Processes input data"""
    return f"Processed: {args['input']}"


