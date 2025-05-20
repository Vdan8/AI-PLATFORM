# backend/app/services/sandbox_service.py
import docker
import asyncio
import os
import shutil
import json
import tempfile
from contextlib import asynccontextmanager

from docker.errors import ImageNotFound, ContainerError, APIError
from typing import Dict, Any, Union

from app.schemas.tool import MCPToolCall, MCPToolResponse
from app.core.config import settings
from app.utils.logger import logger
from app.services.tool_loader import tool_loader_service
from sqlalchemy.ext.asyncio import AsyncSession # CORRECTED: Use AsyncSession
from app.models.base import get_db as get_async_db_session

class SandboxService:
    _instance = None
    client: docker.DockerClient = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SandboxService, cls).__new__(cls)
            cls._instance.client = None
        return cls._instance

    def __init__(self):
        pass

    async def initialize(self):
        """
        Initializes the Docker client and ensures the sandbox base directory exists.
        This is called during application startup (FastAPI lifespan event).
        """
        if self.client is None:
            try:
                self.client = docker.from_env()
                await asyncio.to_thread(self.client.ping)
                logger.info("✅ Docker client initialized successfully")

                self.sandbox_base_dir = settings.SANDBOX_BASE_DIR
                os.makedirs(self.sandbox_base_dir, exist_ok=True)
                logger.info(f"Sandbox base directory ensured: {self.sandbox_base_dir}")

            except docker.errors.DockerException as e:
                logger.critical(f"🛑 Failed to initialize Docker client: {e}")
                raise
            except Exception as e:
                logger.critical(f"🛑 An unexpected error occurred during sandbox initialization: {e}")
                raise


    async def shutdown(self):
        """
        Shuts down the Docker client and cleans up any lingering resources.
        This is called during application shutdown (FastAPI lifespan event).
        """
        if self.client:
            try:
                await asyncio.to_thread(self.client.close)
                self.client = None
                logger.info("👋 Docker client shut down.")
            except Exception as e:
                logger.error(f"Error during Docker client shutdown: {e}")

    @asynccontextmanager
    async def get_db_session(self):
        """
        Provides an async context manager for database sessions using the project's get_db.
        """
        async for db_session in get_async_db_session():
            yield db_session

    async def run_tool_in_sandbox(self, tool_call: MCPToolCall) -> MCPToolResponse:
        """
        Executes a tool within a Docker container sandbox.
        """
        if not self.client:
            error_msg = "Sandbox service not initialized. Docker client is not available."
            logger.error(error_msg)
            return MCPToolResponse(
                tool_name=tool_call.tool_name,
                status="error",
                output={"error": error_msg},
                call_id=tool_call.call_id
            )

        container = None
        temp_dir = None
        tool_code = None

        try:
            async with self.get_db_session() as db: # db is AsyncSession here
                tool_db_object = await tool_loader_service.get_tool_by_name(db, tool_call.tool_name) # CORRECTED: await
                if not tool_db_object:
                    error_msg = f"Tool '{tool_call.tool_name}' not found in the tool registry."
                    logger.error(error_msg)
                    return MCPToolResponse(
                        tool_name=tool_call.tool_name,
                        status="error",
                        output={"error": error_msg},
                        call_id=tool_call.call_id
                    )
                tool_code = tool_db_object.code
                logger.debug(f"Retrieved code for tool: {tool_call.tool_name}")


            temp_dir = tempfile.mkdtemp(prefix=f"ai_sandbox_{tool_call.tool_name}_", dir=self.sandbox_base_dir)
            tool_script_path_in_host = os.path.join(temp_dir, "tool_script.py")
            tool_script_path_in_container = "/sandbox/tool_script.py"

            with open(tool_script_path_in_host, "w") as f:
                f.write(tool_code)
            logger.debug(f"Tool script written to {tool_script_path_in_host}")

            tool_args_json = json.dumps(tool_call.tool_arguments)
            env_vars = {
                "TOOL_NAME": tool_call.tool_name,
                "TOOL_ARGUMENTS": tool_args_json,
            }

            container_name = f"tool-sandbox-{tool_call.call_id or os.urandom(4).hex()}"
            docker_image = settings.SANDBOX_IMAGE

            volumes = {
                temp_dir: {'bind': '/sandbox', 'mode': 'rw'}
            }

            command = ["python", tool_script_path_in_container]

            logger.info(f"Launching container '{container_name}' for tool '{tool_call.tool_name}' (Call ID: {tool_call.call_id or 'N/A'})")

            container = await asyncio.to_thread(
                self.client.containers.run,
                image=docker_image,
                name=container_name,
                command=command,
                volumes=volumes,
                environment=env_vars,
                detach=True,
                remove=False,
                network_mode="none",
                mem_limit=settings.SANDBOX_MEM_LIMIT,
                cpu_shares=settings.SANDBOX_CPU_SHARES,
            )
            logger.debug(f"Container '{container.id}' started.")

            result = await asyncio.to_thread(container.wait, timeout=settings.SANDBOX_TIMEOUT_SECONDS)

            stdout_bytes = await asyncio.to_thread(container.logs, stdout=True, stderr=False)
            stderr_bytes = await asyncio.to_thread(container.logs, stdout=False, stderr=True)

            stdout = stdout_bytes.decode('utf-8').strip()
            stderr = stderr_bytes.decode('utf-8').strip()

            if result['StatusCode'] == 0:
                logger.info(f"Tool '{tool_call.tool_name}' executed successfully. Output: {stdout}")
                return MCPToolResponse(
                    tool_name=tool_call.tool_name,
                    status="success",
                    output=stdout,
                    call_id=tool_call.call_id
                )
            else:
                error_message = f"Tool '{tool_call.tool_name}' failed with exit code {result['StatusCode']}. Stderr: {stderr}"
                logger.error(error_message)
                return MCPToolResponse(
                    tool_name=tool_call.tool_name,
                    status="error",
                    output={"error": error_message, "stdout": stdout, "stderr": stderr},
                    call_id=tool_call.call_id
                )

        except ImageNotFound:
            error_msg = f"Docker image '{docker_image}' not found. Please ensure it's pulled."
            logger.error(error_msg)
            return MCPToolResponse(
                tool_name=tool_call.tool_name,
                status="error",
                output={"error": error_msg, "detail": "Docker image not found"},
                call_id=tool_call.call_id
            )
        except ContainerError as e:
            error_msg = f"Container error during tool '{tool_call.tool_name}' execution: {e}. Stderr: {e.stderr.decode('utf-8')}"
            logger.error(error_msg)
            return MCPToolResponse(
                tool_name=tool_call.tool_name,
                status="error",
                output={"error": error_msg, "detail": e.stderr.decode('utf-8')},
                call_id=tool_call.call_id
            )
        except APIError as e:
            error_msg = f"Docker API error during tool '{tool_call.tool_name}' execution: {e}"
            logger.error(error_msg)
            return MCPToolResponse(
                tool_name=tool_call.tool_name,
                status="error",
                output={"error": error_msg, "detail": str(e)},
                call_id=tool_call.call_id
            )
        except asyncio.TimeoutError:
            error_msg = f"Tool '{tool_call.tool_name}' execution timed out after {settings.SANDBOX_TIMEOUT_SECONDS} seconds."
            logger.error(error_msg)
            return MCPToolResponse(
                tool_name=tool_call.tool_name,
                status="error",
                output={"error": error_msg, "detail": "Execution timed out."},
                call_id=tool_call.call_id
            )
        except Exception as e:
            error_msg = f"An unexpected error occurred during tool '{tool_call.tool_name}' execution: {e}"
            logger.exception(error_msg)
            return MCPToolResponse(
                tool_name=tool_call.tool_name,
                status="error",
                output={"error": error_msg, "detail": str(e)},
                call_id=tool_call.call_id
            )
        finally:
            if container:
                try:
                    current_container = await asyncio.to_thread(self.client.containers.get, container.id)
                    await asyncio.to_thread(current_container.stop, timeout=5)
                    await asyncio.to_thread(current_container.remove)
                    logger.debug(f"Container '{container.id}' for '{tool_call.tool_name}' cleaned up.")
                except docker.errors.NotFound:
                    logger.warning(f"Container '{container.id}' not found during cleanup. Possibly already removed.")
                except APIError as e:
                    logger.warning(f"Failed to clean up container '{container.id}' for '{tool_call.tool_name}': {e}")
                except Exception as e:
                    logger.error(f"Unexpected error during container cleanup for '{container.id}': {e}")

            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    logger.debug(f"Temporary directory '{temp_dir}' cleaned up.")
                except FileNotFoundError:
                    logger.warning(f"Sandbox directory '{temp_dir}' not found during cleanup.")
                except Exception as e:
                    logger.error(f"Error cleaning up sandbox directory '{temp_dir}': {e}")

# Singleton instance
sandbox_service = SandboxService()

async def initialize_sandbox():
    await sandbox_service.initialize()

async def shutdown_sandbox():
    await sandbox_service.shutdown()