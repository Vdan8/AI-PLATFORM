import json
import os
from datetime import datetime
import logging

LOG_FILE_PATH = "app/logs/trace_log.jsonl"

# Initialize a basic logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a handler to write to the log file
file_handler = logging.FileHandler(LOG_FILE_PATH)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def log_trace(step_data: dict):
    """Appends structured execution metadata to a trace log."""
    step_data["timestamp"] = datetime.utcnow().isoformat()
    with open(LOG_FILE_PATH, "a") as f:
        f.write(json.dumps(step_data) + "\n")

# Example of using the logger within this module (optional)
# logger.info("Logger initialized.")