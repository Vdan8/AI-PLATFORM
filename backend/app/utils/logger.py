import json
import os
from datetime import datetime
import logging

# Dynamically resolve the absolute path to the log file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # backend/app/
LOGS_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE_PATH = os.path.join(LOGS_DIR, "trace_log.jsonl")

# Ensure the logs directory exists
os.makedirs(LOGS_DIR, exist_ok=True)

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a handler to write to the log file
file_handler = logging.FileHandler(LOG_FILE_PATH)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Avoid adding duplicate handlers
if not logger.hasHandlers():
    logger.addHandler(file_handler)

def log_trace(step_data: dict):
    """Appends structured execution metadata to a trace log."""
    step_data["timestamp"] = datetime.utcnow().isoformat()
    with open(LOG_FILE_PATH, "a") as f:
        f.write(json.dumps(step_data) + "\n")
