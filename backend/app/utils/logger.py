# backend/app/utils/logger.py
import json
import os
from datetime import datetime
import logging
from typing import Dict, Any, Optional

# --- Configuration for General Application Logger ---
# BASE_DIR points to 'backend/app/'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

APP_LOG_FILE_PATH = os.path.join(LOGS_DIR, "app_log.log")
TRACE_LOG_FILE_PATH = os.path.join(LOGS_DIR, "trace_log.jsonl")

# Initialize general application logger
# This logger will typically be used for general application events and errors
app_logger = logging.getLogger(__name__)
app_logger.setLevel(logging.INFO) # Default level for app logs (e.g., INFO, WARNING, ERROR)

# Console handler for app_logger (useful for immediate feedback)
console_handler = logging.StreamHandler()
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
console_handler.setFormatter(console_formatter)
app_logger.addHandler(console_handler)

# File handler for general app_logger
app_file_handler = logging.FileHandler(APP_LOG_FILE_PATH)
app_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
app_file_handler.setFormatter(app_formatter)
app_logger.addHandler(app_file_handler)


# --- Custom JSON Formatter for Trace Logs ---
class JsonFormatter(logging.Formatter):
    """Formats log records as JSON, specifically for trace logs."""
    def format(self, record):
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            # Access custom 'context' attribute if present in the log record
            "context": getattr(record, 'context', None)
        }
        return json.dumps(log_entry)

# --- Configuration for Structured Trace Logger ---
# This logger is specifically for detailed, structured tracing of agent operations
trace_logger_instance = logging.getLogger("trace_logger")
trace_logger_instance.setLevel(logging.INFO) # Set to DEBUG for more detailed traces

# Prevent trace messages from propagating to the root logger (and thus app_logger)
# This ensures trace logs only go to the trace file handler.
trace_logger_instance.propagate = False

# File handler for the trace log (JSONL format)
trace_file_handler = logging.FileHandler(TRACE_LOG_FILE_PATH)
trace_file_handler.setFormatter(JsonFormatter())
trace_logger_instance.addHandler(trace_file_handler)


# --- Trace Logging Utility Class ---
class TraceLogger:
    """
    A utility class to simplify logging structured events to the trace logger.
    """
    def __init__(self, logger_instance: logging.Logger):
        self.logger = logger_instance

    async def log_event(self, event_name: str, context: Optional[Dict[str, Any]] = None):
        """
        Logs a structured event with a name and optional context dictionary.
        The context will be included in the 'context' field of the JSON log entry.
        This method is async to potentially support async I/O in the future,
        though current logging module calls are blocking.
        """
        # The 'extra' dictionary is how you pass custom attributes to log records
        # These attributes are then accessible by custom formatters (like JsonFormatter)
        extra_data = {'context': context} if context is not None else {}
        self.logger.info(event_name, extra=extra_data)

        # Also log to the general app_logger for immediate console feedback during development.
        # This can be removed in production or if you only want traces in the JSONL file.
        app_logger.info(f"[TRACE] {event_name}: {json.dumps(context)}")


# Instantiate the trace logger service for easy import elsewhere.
# Other modules will import `trace_logger_service` to log structured events.
trace_logger_service = TraceLogger(trace_logger_instance)

# Expose the general application logger as 'logger' for convenience.
# Other modules will import `logger` for general app logs (errors, info messages).
logger = app_logger