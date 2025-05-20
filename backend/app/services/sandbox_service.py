import docker
import asyncio
import os
import shutil
import json
import tempfile
from contextlib import asynccontextmanager
import logging
import platform # For platform-specific Docker checks

from docker.errors import ImageNotFound, ContainerError, APIError
from typing import Dict, Any, Union

from app.schemas.tool import MCPToolCall, MCPToolResponse
from app.core.config import settings
# from app.utils.logger import logger # Assuming app_logger is the main logger you want to use here
from app.services.tool_loader import tool_loader_service
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.base import get_db as get_async_db_session # Alias to avoid naming conflict with service logger

# Get a module-specific logger for SandboxService
logger = logging.getLogger(__name__)

class SandboxService:
    _instance = None # Singleton instance holder
    client: docker.DockerClient = None
    _is_initialized: bool = False # Flag to track if initialization has completed
    _docker_available: bool = False # Flag to track Docker availability

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SandboxService, cls).__new__(cls)
            cls._instance.client = None
            cls._instance._is_initialized = False
            cls._instance._docker_available = False
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
        Initializes the Docker client, checks Docker availability,
        and ensures the sandbox base directory exists.
        This is called once during application startup (FastAPI lifespan event).
        """
        if self._is_initialized:
            logger.debug("SandboxService already initialized.")
            return

        logger.info("Initializing SandboxService...")
        try:
            self._check_docker_availability() # Check Docker before proceeding

            if not self._docker_available:
                # If Docker is not available, we stop initialization and raise.
                # The _check_docker_availability method already logs the critical error.
                raise RuntimeError("Docker daemon is not available. Sandbox service cannot start.")

            self.sandbox_base_dir = settings.SANDBOX_BASE_DIR
            os.makedirs(self.sandbox_base_dir, exist_ok=True)
            logger.info(f"Sandbox base directory ensured: {self.sandbox_base_dir}")

            self._is_initialized = True
            logger.info("✅ SandboxService initialized successfully.")

        except docker.errors.DockerException as e:
            logger.critical(f"🛑 Failed to initialize Docker client due to Docker error: {e}")
            raise
        except Exception as e:
            logger.critical(f"🛑 An unexpected error occurred during sandbox initialization: {e}")
            raise

    def _check_docker_availability(self):
        """
        Internal method to check if the Docker daemon is running and accessible.
        This is a blocking call, so it's run via asyncio.to_thread in the initialize method.
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


    async def shutdown(self):
        """
        Shuts down the Docker client and cleans up any lingering resources.
        This is called during application shutdown (FastAPI lifespan event).
        """
        if self.client:
            try:
                # Ensure client is properly closed if it was initialized
                await asyncio.to_thread(self.client.close)
                self.client = None
                self._is_initialized = False
                self._docker_available = False
                logger.info("👋 Docker client shut down and SandboxService reset.")
            except Exception as e:
                logger.error(f"Error during Docker client shutdown: {e}", exc_info=True)

    @asynccontextmanager
    async def _get_db_session_context(self):
        """
        Internal context manager for obtaining an async database session.
        Used when the sandbox service needs a DB session independently.
        """
        async for db_session in get_async_db_session():
            yield db_session

    async def run_tool_in_sandbox(self, tool_call: MCPToolCall) -> MCPToolResponse:
        """
        Executes a tool within a Docker container sandbox.
        """
        if not self._is_initialized or not self._docker_available:
            error_msg = "Sandbox service is not ready. Docker daemon or image is unavailable."
            logger.error(error_msg)
            return MCPToolResponse(
                tool_name=tool_call.tool_name,
                status="error",
                output={"error_message": error_msg}, # Use 'error_message' for consistency
                call_id=tool_call.call_id
            )

        container = None
        temp_dir = None
        tool_code = None

        try:
            # Get tool code from the database using tool_loader_service
            async with self._get_db_session_context() as db: # Use the internal context manager
                # Assuming tool_loader_service.get_tool_by_name exists and returns MCPToolDefinition
                tool_db_object = await tool_loader_service.get_tool_by_name(db, tool_call.tool_name)
                if not tool_db_object:
                    error_msg = f"Tool '{tool_call.tool_name}' not found in the tool registry for execution."
                    logger.error(error_msg)
                    return MCPToolResponse(
                        tool_name=tool_call.tool_name,
                        status="error",
                        output={"error_message": error_msg},
                        call_id=tool_call.call_id
                    )
                tool_code = tool_db_object.code
                if not tool_code:
                    error_msg = f"Tool '{tool_call.tool_name}' has no executable code defined."
                    logger.error(error_msg)
                    return MCPToolResponse(
                        tool_name=tool_call.tool_name,
                        status="error",
                        output={"error_message": error_msg},
                        call_id=tool_call.call_id
                    )
                logger.debug(f"Retrieved code for tool: {tool_call.tool_name}")

            # Create a temporary directory for the tool script and bind mount it
            # Ensure sandbox_base_dir is guaranteed to exist by initialize()
            temp_dir = tempfile.mkdtemp(prefix=f"ai_sandbox_{tool_call.tool_name}_", dir=self.sandbox_base_dir)
            tool_script_path_in_host = os.path.join(temp_dir, "tool_script.py")
            tool_script_path_in_container = "/sandbox/tool_script.py"

            with open(tool_script_path_in_host, "w") as f:
                f.write(tool_code)
            logger.debug(f"Tool script written to {tool_script_path_in_host}")

            # Prepare environment variables for the container
            tool_args_json = json.dumps(tool_call.tool_arguments)
            env_vars = {
                "TOOL_NAME": tool_call.tool_name,
                "TOOL_ARGUMENTS": tool_args_json,
                # Add any other necessary environment variables for your tools
                # E.g., API keys, if you inject them this way (use with extreme caution)
            }

            container_name = f"tool-sandbox-{tool_call.call_id or uuid4().hex}" # Use uuid4().hex for random if call_id is None
            docker_image = settings.SANDBOX_IMAGE

            volumes = {
                temp_dir: {'bind': '/sandbox', 'mode': 'rw'}
            }

            command = ["python", tool_script_path_in_container]

            logger.info(f"Launching container '{container_name}' for tool '{tool_call.tool_name}' (Call ID: {tool_call.call_id or 'N/A'})")

            # Run the container in a separate thread to not block the event loop
            container = await asyncio.to_thread(
                self.client.containers.run,
                image=docker_image,
                name=container_name,
                command=command,
                volumes=volumes,
                environment=env_vars,
                detach=True,
                remove=False, # We'll remove explicitly in finally block for more control
                network_mode="none", # Isolate container from host network
                mem_limit=f"{settings.SANDBOX_MEMORY_LIMIT_MB}m", # Apply memory limit
                # cpu_shares is deprecated, use cpu_period/cpu_quota or cpu_count/cpuset_cpus for finer control
                # For basic CPU limiting, cpu_period and cpu_quota are better.
                # e.g., cpu_quota=int(settings.SANDBOX_CPU_LIMIT_CORES * 100000), cpu_period=100000
                # Using cpu_shares as a fallback if specific settings aren't available or preferred.
                cpu_shares=settings.SANDBOX_CPU_SHARES if hasattr(settings, 'SANDBOX_CPU_SHARES') else None,
                # Example for cleaner CPU limits:
                # cpu_quota=int(settings.SANDBOX_CPU_LIMIT_CORES * 100000), # 100% of one CPU = 100000
                # cpu_period=100000,
            )
            logger.debug(f"Container '{container.id}' for '{tool_call.tool_name}' started successfully.")

            # Wait for the container to exit, with a timeout
            result = await asyncio.to_thread(container.wait, timeout=settings.SANDBOX_MAX_CONTAINER_RUNTIME_SECONDS)
            exit_code = result.get('StatusCode', -1) # Get status code, default to -1 if not found

            stdout_bytes = await asyncio.to_thread(container.logs, stdout=True, stderr=False)
            stderr_bytes = await asyncio.to_thread(container.logs, stdout=False, stderr=True)

            stdout = stdout_bytes.decode('utf-8').strip()
            stderr = stderr_bytes.decode('utf-8').strip()

            if exit_code == 0:
                logger.info(f"Tool '{tool_call.tool_name}' (ID: {tool_call.call_id}) executed successfully. Output: {stdout[:500]}...") # Log truncated output
                return MCPToolResponse(
                    tool_name=tool_call.tool_name,
                    status="success",
                    output=stdout,
                    call_id=tool_call.call_id
                )
            else:
                error_message = f"Tool '{tool_call.tool_name}' failed with exit code {exit_code}. Stderr: {stderr[:500]}..."
                logger.error(error_message)
                return MCPToolResponse(
                    tool_name=tool_call.tool_name,
                    status="error",
                    output={"error_message": error_message, "stdout": stdout, "stderr": stderr},
                    call_id=tool_call.call_id
                )

        except ImageNotFound:
            error_msg = f"Docker image '{docker_image}' not found for tool '{tool_call.tool_name}'. Please ensure it's built/pulled."
            logger.critical(error_msg) # Critical as this is a setup issue
            return MCPToolResponse(
                tool_name=tool_call.tool_name,
                status="error",
                output={"error_message": error_msg, "detail": "Docker image not found"},
                call_id=tool_call.call_id
            )
        except ContainerError as e:
            # This captures errors if the container itself couldn't start or run the command
            error_msg = f"Container failed to execute tool '{tool_call.tool_name}': {e}. Stderr: {e.stderr.decode('utf-8').strip()}"
            logger.error(error_msg)
            return MCPToolResponse(
                tool_name=tool_call.tool_name,
                status="error",
                output={"error_message": error_msg, "detail": e.stderr.decode('utf-8').strip()},
                call_id=tool_call.call_id
            )
        except APIError as e:
            error_msg = f"Docker API error during tool '{tool_call.tool_name}' execution: {e}. Check Docker daemon status and permissions."
            logger.error(error_msg, exc_info=True) # Log traceback for API errors
            return MCPToolResponse(
                tool_name=tool_call.tool_name,
                status="error",
                output={"error_message": error_msg, "detail": str(e)},
                call_id=tool_call.call_id
            )
        except asyncio.TimeoutError:
            error_msg = f"Tool '{tool_call.tool_name}' execution timed out after {settings.SANDBOX_MAX_CONTAINER_RUNTIME_SECONDS} seconds."
            logger.error(error_msg)
            # Attempt to stop and remove the container if it timed out and is still running
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
                output={"error_message": error_msg, "detail": "Execution timed out. Container likely force-stopped."},
                call_id=tool_call.call_id
            )
        except Exception as e:
            error_msg = f"An unexpected error occurred during tool '{tool_call.tool_name}' execution: {e.__class__.__name__}: {e}"
            logger.exception(error_msg) # Use exception() for full traceback logging
            return MCPToolResponse(
                tool_name=tool_call.tool_name,
                status="error",
                output={"error_message": error_msg, "detail": str(e)},
                call_id=tool_call.call_id
            )
        finally:
            # Ensure container is stopped and removed, and temp directory is cleaned
            if container:
                try:
                    # Refresh container state before trying to stop/remove
                    current_container = await asyncio.to_thread(self.client.containers.get, container.id)
                    if current_container.status == 'running':
                        logger.debug(f"Stopping running container '{container.id}' for '{tool_call.tool_name}'.")
                        await asyncio.to_thread(current_container.stop, timeout=5)
                    logger.debug(f"Removing container '{container.id}' for '{tool_call.tool_name}'.")
                    await asyncio.to_thread(current_container.remove)
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