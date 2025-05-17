import json
import os
from datetime import datetime

LOG_FILE_PATH = "app/logs/trace_log.jsonl"

def log_trace(step_data: dict):
    """Appends structured execution metadata to a trace log."""
    step_data["timestamp"] = datetime.utcnow().isoformat()
    with open(LOG_FILE_PATH, "a") as f:
        f.write(json.dumps(step_data) + "\n")
