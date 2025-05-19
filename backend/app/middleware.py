# app/middleware.py
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
from app.utils.logger import logger  # Import your configured logger
import time
import traceback  # Import traceback for detailed error info
from typing import Callable


class LoggingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: Callable):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        try:
            response = await call_next(request)
        except Exception as e:
            # Format the exception with traceback for detailed logging
            error_message = f"Unhandled exception: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_message)  # Log the detailed error
            response = JSONResponse(
                status_code=500, content={"detail": "Internal Server Error"}
            )
        finally:
            process_time = time.time() - start_time
            logger.info(
                f"Request: {request.method} {request.url.path} - Response: {response.status_code} - Process Time: {process_time:.4f}s"
            )
        return response