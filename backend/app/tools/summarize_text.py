"""
name: summarize_text
description: Summarizes a block of text by truncating it.
parameters:
  text:
    type: string
"""

def run(args):
    return args["text"][:50] + "..."  # Dummy logic
