import docker  # Import docker library
import asyncio
import os
import shutil
import json
import textwrap
import inspect
import tempfile
from contextlib import asynccontextmanager
import logging
import platform  # For platform-specific Docker checks
from uuid import uuid4  # Import uuid4 for generating random call_ids

from docker.errors import ImageNotFound, ContainerError, APIError
from typing import Dict, Any, Union, Optional
from pathlib import Path

from app.schemas.tool import MCPToolCall, MCPToolResponse
from backend.config.config import settings
from app.utils.logger import logger  # Assuming app_logger is the main logger you want to use here
from app.services.tool_loader import tool_loader_service
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.base import get_db as get_async_db_session  # Alias to avoid naming conflict with service logger


# Get a module-specific logger for SandboxService
logger = logging.getLogger(__name__)


class SandboxService:
    _instance = None  # Singleton instance holder
    client: docker.DockerClient = None
    _is_initialized: bool = False  # Flag to track if initialization has completed
    _docker_available: bool = False  # Flag to track Docker availability
    _container_ready: bool = False  # Flag to track if container setup is ready

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SandboxService, cls).__new__(cls)
            cls._instance.client = None
            cls._instance._is_initialized = False
            cls._instance._docker_available = False
            cls._instance._container_ready = False
        return cls._instance

    def __init__(self):
        # __init__ might be called multiple times if the singleton is imported often,
        # but the actual initialization logic should run only once via _is_initialized flag.
        if not self._is_initialized:
            # We don't call initialize() directly here because it's async and
            # relies on the FastAPI lifespan event to be awaited.
            pass

    async def initialize(self):
        """
        Initializes basic service setup but defers Docker client creation until runtime.
        This is called once during application startup (FastAPI lifespan event).
        """
        if self._is_initialized:
            logger.debug("SandboxService already initialized.")
            return

        logger.info("Initializing SandboxService...")
        try:
            # Only initialize basic directory setup during startup
            self.sandbox_base_dir = settings.SANDBOX_BASE_DIR
            os.makedirs(self.sandbox_base_dir, exist_ok=True)
            logger.info(f"Sandbox base directory ensured: {self.sandbox_base_dir}")

            self._is_initialized = True
            logger.info("✅ SandboxService initialized successfully (Docker will be initialized on first use).")

        except Exception as e:
            logger.critical(f"🛑 An unexpected error occurred during sandbox initialization: {e}")
            raise

    async def ensure_container_ready(self):
        """
        Ensures Docker client and container setup is ready for use.
        This method is called lazily when Docker functionality is actually needed.
        """
        if self._container_ready and self._docker_available and self.client:
            return  # Already ready

        logger.info("Initializing Docker client and container setup...")
        try:
            # Run blocking Docker setup in a separate thread
            await asyncio.to_thread(self._setup_docker_sync)

            if not self._docker_available:
                raise RuntimeError("Docker daemon is not available. Sandbox service cannot start.")

            self._container_ready = True
            logger.info("✅ Docker container setup completed successfully.")

        except docker.errors.DockerException as e:
            logger.critical(f"🛑 Failed to initialize Docker client due to Docker error: {e}")
            raise
        except Exception as e:
            logger.critical(f"🛑 An unexpected error occurred during Docker initialization: {e}")
            raise

    def _setup_docker_sync(self):
        """
        Internal method to set up Docker client and check availability.
        This is a synchronous call run in a thread pool.
        """
        try:
            self.client = docker.from_env()
            # Attempt a simple Docker operation like ping to confirm connectivity
            self.client.ping()
            self._docker_available = True
            logger.info("✅ Successfully connected to Docker daemon.")
            
            # Verify Docker image existence
            try:
                self.client.images.get(settings.SANDBOX_IMAGE)
                logger.info(f"✅ Required Docker image '{settings.SANDBOX_IMAGE}' found.")
            except ImageNotFound:
                logger.critical(f"🛑 Docker image '{settings.SANDBOX_IMAGE}' not found. Please build or pull it.")
                self._docker_available = False
                # Optionally, raise an error here if missing image is a critical blocker
                # raise RuntimeError(f"Required Docker image '{settings.SANDBOX_IMAGE}' not found.")

        except docker.errors.APIError as e:
            self._docker_available = False
            logger.critical(f"🛑 Docker API error: {e}. Is Docker daemon running and accessible? Check permissions.")
            # For specific Docker issues on Linux/macOS, provide hints
            if platform.system() in ["Linux", "Darwin"]:
                logger.critical("   - On Linux: Ensure Docker daemon is running and user is in 'docker' group (`sudo usermod -aG docker $USER && newgrp docker`)")
                logger.critical("   - On macOS: Ensure Docker Desktop is running.")
        except Exception as e:
            self._docker_available = False
            logger.critical(f"🛑 Failed to connect to Docker daemon: {e}. Please ensure Docker is installed and running.")
            logger.debug("Full traceback for Docker connection error:", exc_info=True)

    def _check_docker_availability_sync(self):
        """
        Legacy method - now replaced by _setup_docker_sync for lazy initialization.
        Kept for compatibility but redirects to new method.
        """
        self._setup_docker_sync()

    async def shutdown(self):
        """
        Shuts down the Docker client and cleans up any lingering resources.
        This is called during application shutdown (FastAPI lifespan event).
        """
        if self.client:
            try:
                # Ensure client is properly closed if it was initialized
                # client.close() is typically synchronous, so await to_thread
                await asyncio.to_thread(self.client.close)
                self.client = None
                self._is_initialized = False
                self._docker_available = False
                self._container_ready = False
                logger.info("👋 Docker client shut down and SandboxService reset.")
            except Exception as e:
                logger.error(f"Error during Docker client shutdown: {e}", exc_info=True)

    def wrap_tool_function(self, tool_function_code: str) -> str:
        """
        Wraps a given tool function code with execution logic to run inside the sandbox.
        It handles both synchronous and asynchronous functions and prints JSON output.
        """
        code = tool_function_code.strip()
        if code.startswith("```python"):
            code = code[len("```python"):].strip()
        if code.endswith("```"):
            code = code[:-3].strip()
        code = textwrap.dedent(code)

        # Detect function name (more robust check)
        first_line = next((line for line in code.splitlines() if line.startswith("def ") or line.startswith("async def ")), None)
        if not first_line:
            raise RuntimeError("Found no function definition in tool script.\n--- Tool Script ---\n" + code)
        func_name = first_line.split("def ")[1].split("(")[0].strip()

        # Check if it's async
        is_async = first_line.strip().startswith("async def")

        if is_async:
            wrapper = (
                "import os\n"
                "import json\n"
                "import asyncio\n\n"
                f"{code}\n\n"
                "async def run_tool_main():\n"
                '    tool_args_str = os.environ.get("TOOL_ARGUMENTS", "{}")\n' # Get the raw string [cite: 148]
                "    try:\n"
                '        args = json.loads(tool_args_str)\n'
                f"        result = await globals()['{func_name}'](**args)\n"
                '        print(json.dumps({"status": "success", "output": result}))\n'
                "    except Exception as e:\n"
                "        import traceback\n"
                '        print(json.dumps({"status": "error", "message": str(e), "traceback": traceback.format_exc()}))\n\n'
                "if __name__ == \"__main__\":\n"
                "    asyncio.run(run_tool_main())\n"
            )
        else:
            wrapper = (
                "import os\n"
                "import json\n\n"
                f"{code}\n\n"
                "if __name__ == \"__main__\":\n"
                '    tool_args_str = os.environ.get("TOOL_ARGUMENTS", "{}")\n' # Get the raw string
                '    args = json.loads(tool_args_str)\n'
                "    try:\n"
                f"        result = globals()['{func_name}'](**args)\n"
                '        print(json.dumps({"status": "success", "output": result}))\n'
                "    except Exception as e:\n"
                "        import traceback\n"
                '        print(json.dumps({"status": "error", "message": str(e), "traceback": traceback.format_exc()}))\n'
            )
        return textwrap.dedent(wrapper)


    @asynccontextmanager
    async def _get_db_session_context(self):
        """
        Internal context manager for obtaining an async database session.
        Used when the sandbox service needs a DB session independently.
        """
        # get_async_db_session() is already an async context manager, so use async with
        async with get_async_db_session() as db_session:
            yield db_session

    async def run_tool_in_sandbox(self, tool_call: MCPToolCall, tool_script_content: str) -> MCPToolResponse:
        """
        Executes a dynamically generated tool within a secure Docker container.
        The tool's output must be a JSON string printed to stdout.
        """

        # Ensure Docker is ready before proceeding
        try:
            await self.ensure_container_ready()
        except Exception as e:
            error_msg = f"Failed to initialize Docker: {e}"
            logger.error(error_msg)
            return MCPToolResponse(
                tool_name=tool_call.tool_name,
                status="error",
                output=None,
                error_message=error_msg,
                call_id=tool_call.call_id
            )

        if not self._is_initialized or not self._docker_available:
            error_msg = "Sandbox service is not ready. Docker daemon or image is unavailable."
            logger.error(error_msg)
            return MCPToolResponse(
                tool_name=tool_call.tool_name,
                status="error",
                output=None,
                error_message=error_msg,
                call_id=tool_call.call_id
            )

        container = None
        temp_dir = None

        try:
            if not tool_script_content:
                error_msg = f"Tool '{tool_call.tool_name}' received no executable script content."
                logger.error(error_msg)
                return MCPToolResponse(
                    tool_name=tool_call.tool_name,
                    status="error",
                    output=None,
                    error_message=error_msg,
                    call_id=tool_call.call_id
                )

            # --- Clean and validate tool script ---
            cleaned_script = tool_script_content.strip()
            # Remove LLM artifacts like triple backticks
            if cleaned_script.startswith("```python"):
                cleaned_script = cleaned_script[len("```python"):].strip()
            if cleaned_script.endswith("```"):
                cleaned_script = cleaned_script[:-3].strip()

            # Dedent for consistent indentation
            cleaned_script = textwrap.dedent(cleaned_script).strip()

            logger.debug(f"🧼 Cleaned tool script:\n{cleaned_script}")

            # --- Validate Python syntax before running in Docker ---
            try:
                compile(cleaned_script, filename="<tool_script>", mode="exec")
            except SyntaxError as e:
                error_msg = f"❌ Syntax error in generated tool script: {e}"
                logger.error(error_msg)
                return MCPToolResponse(
                    tool_name=tool_call.tool_name,
                    status="error",
                    output=None,
                    error_message=error_msg,
                    call_id=tool_call.call_id
                )

            # Wrap the raw function with main-execution logic
            try:
                wrapped_code = self.wrap_tool_function(cleaned_script)
            except Exception as e:
                error_msg = f"Tool wrapping failed: {e}"
                logger.error(error_msg)
                return MCPToolResponse(
                    tool_name=tool_call.tool_name,
                    status="error",
                    output=None,
                    error_message=error_msg,
                    call_id=tool_call.call_id
                )
            
            # Optional: check for syntax issues in wrapped code before writing
            try:
                compile(wrapped_code, filename="<wrapped_tool_script>", mode="exec")
            except SyntaxError as e:
                error_msg = f"Syntax error in wrapped tool script: {e}"
                logger.error(error_msg)
                return MCPToolResponse(
                    tool_name=tool_call.tool_name,
                    status="error",
                    output=None,
                    error_message=error_msg,
                    call_id=tool_call.call_id
                )

            # Create a temporary dir and write the wrapped tool
            temp_dir = tempfile.mkdtemp(prefix=f"ai_sandbox_{tool_call.tool_name}_", dir=self.sandbox_base_dir)
            tool_script_path_in_host = os.path.join(temp_dir, "tool_script.py")
            tool_script_path_in_container = "/sandbox/tool_script.py"

            with open(tool_script_path_in_host, "w") as f:
                f.write(wrapped_code)

            # Docker configuration
            docker_image = settings.SANDBOX_IMAGE
            container_name = f"sandbox_tool_{tool_call.tool_name}_{tool_call.call_id or uuid4().hex}"
            volumes = {
                temp_dir: {
                    "bind": "/sandbox",
                    "mode": "ro"  # Read-only for the container
                }
            }
            command = ["python", tool_script_path_in_container]
            env_vars = {"TOOL_ARGUMENTS": json.dumps(tool_call.tool_arguments)}

            logger.info(f"Launching container '{container_name}' for tool '{tool_call.tool_name}' (Call ID: {tool_call.call_id or 'N/A'})")

            # Run the container
            container = await asyncio.to_thread(
                self.client.containers.run,
                image=docker_image,
                name=container_name,
                command=command,
                volumes=volumes,
                environment=env_vars,
                detach=True,
                remove=False, # Keep container after run to get logs, then remove in finally
                network_mode="bridge", # bridge network
                mem_limit=f"{settings.SANDBOX_MEMORY_LIMIT_MB}m",
                cpu_shares=getattr(settings, "SANDBOX_CPU_SHARES", 256), # Default to 256 if not set
            )
            logger.debug(f"Container '{container.id}' for '{tool_call.tool_name}' started successfully.")

            # Wait for the container to finish or timeout
            result_wait = await asyncio.to_thread(container.wait, timeout=settings.SANDBOX_MAX_CONTAINER_RUNTIME_SECONDS)
            exit_code = result_wait.get('StatusCode', -1)

            # Get logs (stdout and stderr)
            stdout_bytes = await asyncio.to_thread(container.logs, stdout=True, stderr=False)
            stderr_bytes = await asyncio.to_thread(container.logs, stdout=False, stderr=True)
            stdout = stdout_bytes.decode('utf-8', errors='ignore').strip()
            stderr = stderr_bytes.decode('utf-8', errors='ignore').strip()

            logger.debug(f"Container '{container.id}' for '{tool_call.tool_name}' finished with exit code: {exit_code}")
            logger.debug(f"🪵 STDOUT for tool '{tool_call.tool_name}':\n{stdout}")
            logger.debug(f"🪵 STDERR for tool '{tool_call.tool_name}':\n{stderr}")

            # --- Parse Tool Output ---
            tool_status = "error"
            tool_output: Any = None
            tool_error_message: Optional[str] = None

            if exit_code == 0:
                try:
                    # Attempt to parse stdout as JSON
                    parsed_output = json.loads(stdout)
                    if parsed_output.get("status") == "success":
                        tool_status = "success"
                        tool_output = parsed_output.get("output")
                        logger.info(f"Tool '{tool_call.tool_name}' (ID: {tool_call.call_id}) executed successfully. Parsed Output: {tool_output}")
                    elif parsed_output.get("status") == "error":
                        tool_status = "error"
                        tool_error_message = parsed_output.get("message", "Tool reported an error via JSON but no message was provided.")
                        tool_output = parsed_output.get("output", stdout) # Include raw stdout as output for debug if available
                        logger.error(f"Tool '{tool_call.tool_name}' (ID: {tool_call.call_id}) reported error via JSON. Message: {tool_error_message}")
                    else:
                        tool_status = "error"
                        tool_error_message = f"Tool '{tool_call.tool_name}' returned invalid JSON status. Raw stdout: {stdout}"
                        tool_output = stdout # Store raw stdout if JSON status is invalid
                        logger.error(tool_error_message)
                except json.JSONDecodeError:
                    logger.warning(f"Tool '{tool_call.tool_name}' stdout is not valid JSON. Treating as successful raw output. stdout: {stdout[:500]}")
                    tool_status = "success"
                    tool_output = stdout
                    tool_error_message = None
                except Exception as e:
                    tool_status = "error"
                    tool_error_message = f"Unexpected error parsing tool '{tool_call.tool_name}' stdout: {e}. Raw stdout: {stdout[:500]}..."
                    tool_output = stdout
                    logger.error(tool_error_message)
            else:
                # Non-zero exit code indicates an error
                tool_status = "error"
                if stderr:
                    try:
                        error_json = json.loads(stderr)
                        tool_error_message = error_json.get("message", f"Tool failed with exit code {exit_code}. Stderr JSON output: {error_json}")
                        tool_output = error_json # Store the error JSON as output
                    except json.JSONDecodeError:
                        tool_error_message = f"Tool '{tool_call.tool_name}' failed with exit code {exit_code}. Stderr: {stderr[:500]}..."
                        tool_output = {"stdout": stdout, "stderr": stderr} # Provide both for context
                else:
                    tool_error_message = f"Tool '{tool_call.tool_name}' failed with exit code {exit_code}. No stderr output. Stdout: {stdout[:500]}..."
                    tool_output = {"stdout": stdout, "stderr": stderr} # Provide both for context
                logger.error(tool_error_message)

            return MCPToolResponse(
                tool_name=tool_call.tool_name,
                status=tool_status,
                output=tool_output,
                error_message=tool_error_message,
                call_id=tool_call.call_id
            )

        except ImageNotFound:
            error_msg = f"Docker image '{docker_image}' not found for tool '{tool_call.tool_name}'. Please ensure it's built/pulled."
            logger.critical(error_msg)
            return MCPToolResponse(
                tool_name=tool_call.tool_name,
                status="error",
                output=None,
                error_message=error_msg,
                call_id=tool_call.call_id
            )
        except ContainerError as e:
            stderr_decoded = e.stderr.decode('utf-8', errors='ignore').strip() if e.stderr else "N/A"
            stdout_decoded = e.stdout.decode('utf-8', errors='ignore').strip() if e.stdout else "N/A"
            error_msg = f"Container failed to execute tool '{tool_call.tool_name}': {e}. Stderr: {stderr_decoded}"
            logger.error(error_msg)
            return MCPToolResponse(
                tool_name=tool_call.tool_name,
                status="error",
                output={"detail": stderr_decoded, "stdout": stdout_decoded}, # Include stdout for more context
                error_message=error_msg,
                call_id=tool_call.call_id
            )
        except APIError as e:
            error_msg = f"Docker API error during tool '{tool_call.tool_name}' execution: {e}. Check Docker daemon status and permissions."
            logger.error(error_msg, exc_info=True)
            return MCPToolResponse(
                tool_name=tool_call.tool_name,
                status="error",
                output={"detail": str(e)},
                error_message=error_msg,
                call_id=tool_call.call_id
            )
        except asyncio.TimeoutError:
            error_msg = f"Tool '{tool_call.tool_name}' execution timed out after {settings.SANDBOX_MAX_CONTAINER_RUNTIME_SECONDS} seconds."
            logger.error(error_msg)
            if container:
                try:
                    await asyncio.to_thread(container.stop, timeout=5)
                    await asyncio.to_thread(container.remove)
                    logger.warning(f"Timed-out container '{container.id}' was force-stopped and removed.")
                except Exception as cleanup_e:
                    logger.error(f"Error stopping/removing timed-out container '{container.id}': {cleanup_e}")
            return MCPToolResponse(
                tool_name=tool_call.tool_name,
                status="error",
                output={"detail": "Execution timed out. Container likely force-stopped."},
                error_message=error_msg,
                call_id=tool_call.call_id
            )
        except Exception as e:
            error_msg = f"An unexpected error occurred during tool '{tool_call.tool_name}' execution: {e.__class__.__name__}: {e}"
            logger.exception(error_msg)
            return MCPToolResponse(
                tool_name=tool_call.tool_name,
                status="error",
                output={"detail": str(e)},
                error_message=error_msg,
                call_id=tool_call.call_id
            )
        finally:
            if container:
                try:
                    # Refresh container state before trying to stop/remove
                    await asyncio.to_thread(container.reload) 
                    if container.status == 'running':
                        logger.debug(f"Stopping running container '{container.id}' for '{tool_call.tool_name}'.")
                        await asyncio.to_thread(container.stop, timeout=5)
                    logger.debug(f"Removing container '{container.id}' for '{tool_call.tool_name}'.")
                    await asyncio.to_thread(container.remove)
                    logger.debug(f"Container '{container.id}' for '{tool_call.tool_name}' cleaned up successfully.")
                except docker.errors.NotFound:
                    logger.warning(f"Container '{container.id}' for '{tool_call.tool_name}' not found during cleanup. Possibly already removed.")
                except APIError as e:
                    logger.error(f"Docker API error during container cleanup for '{container.id}': {e}", exc_info=True)
                except Exception as e:
                    logger.error(f"Unexpected error during container cleanup for '{container.id}': {e}", exc_info=True)

            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    logger.debug(f"Temporary directory '{temp_dir}' cleaned up.")
                except Exception as e:
                    logger.error(f"Error cleaning up sandbox directory '{temp_dir}': {e}", exc_info=True)


# Singleton instance
sandbox_service = SandboxService()

# Lifespan functions (to be called in FastAPI's lifespan context manager)
async def initialize_sandbox_service():
    """Wrapper for lifespan startup event."""
    await sandbox_service.initialize()
    logger.info("Sandbox service startup complete.")

async def shutdown_sandbox_service():
    """Wrapper for lifespan shutdown event."""
    await sandbox_service.shutdown()
    logger.info("Sandbox service shutdown complete.")