# app/services/sandbox_service.py
import docker
import asyncio
import os
import shutil  # For deleting directories
from typing import Dict, Any
from app.utils.logger import logger
from app.core.config import settings  # Import settings for configuration

class SandboxService:
    def __init__(self):
        self.client = None
        self.containers = {}
        self.sandbox_base_dir = settings.SANDBOX_BASE_DIR # Get base directory from settings
        # Ensure the sandbox base directory exists
        os.makedirs(self.sandbox_base_dir, exist_ok=True)

    async def initialize(self):
        """Initialize Docker client"""
        try:
            self.client = docker.from_env()
            self.client.ping()  # Check if Docker is running
            logger.info("✅ Docker client initialized successfully")
        except docker.errors.DockerException as e:
            logger.critical(f"🛑 Failed to initialize Docker client: {e}")
            raise  # Propagate the exception to fail startup

    async def execute_tool(self, tool_name: str, tool_module: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool in a sandbox container."""

        # 1. Create a unique sandbox directory for this execution
        sandbox_id = os.urandom(16).hex()  # Generate a random hex string
        sandbox_path = os.path.join(self.sandbox_base_dir, sandbox_id)
        os.makedirs(sandbox_path, exist_ok=True)

        # 2. Prepare input for the tool (e.g., save arguments to a file)
        input_path = os.path.join(sandbox_path, "input.json")
        with open(input_path, "w") as f:
            import json
            json.dump(arguments, f)

        # 3. Define container parameters (image, command, resources, volumes)
        container_name = f"tool-sandbox-{sandbox_id}"
        image_name = settings.SANDBOX_IMAGE  # Use a configurable image name
        command = ["python", "-m", tool_module, input_path]  # Execute the tool

        volumes = {
            sandbox_path: {'bind': '/sandbox', 'mode': 'rw'}  # Mount the sandbox dir
        }
        environment = {} # Add any necessary environment variables

        # Resource limits (configurable)
        mem_limit = settings.SANDBOX_MEM_LIMIT # e.g., "128m"
        cpu_shares = settings.SANDBOX_CPU_SHARES # e.g., 256

        try:
            # 4. Create and run the container
            container = self.client.containers.run(
                image=image_name,
                name=container_name,
                command=command,
                volumes=volumes,
                environment=environment,
                mem_limit=mem_limit,
                cpu_shares=cpu_shares,
                detach=True,  # Run in detached mode
            )
            self.containers[container.id] = container

            # 5. Wait for the container to finish and get the results
            result = await self._wait_for_completion(container)
            return result

        except docker.errors.ImageNotFound as e:
            logger.error(f"Docker image not found: {e}")
            raise  # Re-raise to be handled upstream
        except docker.errors.APIError as e:
            logger.error(f"Docker API error: {e}")
            raise  # Re-raise
        except Exception as e:
            logger.error(f"Error executing tool '{tool_name}': {e}")
            raise
        finally:
            # 6. Clean up the container and sandbox directory
            await self._cleanup_container(container_name)
            self._cleanup_sandbox(sandbox_path)

    async def _wait_for_completion(self, container):
        """Waits for the container to finish and retrieves the output."""
        while True:
            if container.status == 'exited':
                logs = container.logs().decode('utf-8')
                exit_code = container.wait()['StatusCode']
                if exit_code != 0:
                    logger.error(f"Tool container exited with code {exit_code}: {logs}")
                    raise Exception(f"Tool execution failed with exit code {exit_code}")
                return logs
            elif container.status == 'running':
                await asyncio.sleep(0.1)  # Check status periodically
            else:
                logger.warning(f"Unexpected container status: {container.status}")
                await asyncio.sleep(0.1)

    async def _cleanup_container(self, container_name: str):
        """Stops and removes the container."""
        try:
            container = self.client.containers.get(container_name)
            container.stop(timeout=5)
            container.remove()
            if container.id in self.containers:
                del self.containers[container.id]
        except docker.errors.NotFound:
            logger.warning(f"Container '{container_name}' not found during cleanup.")
        except docker.errors.APIError as e:
            logger.error(f"Error cleaning up container '{container_name}': {e}")
        except Exception as e:
            logger.error(f"Unexpected error during container cleanup: {e}")

    def _cleanup_sandbox(self, sandbox_path: str):
        """Removes the sandbox directory and its contents."""
        try:
            shutil.rmtree(sandbox_path)
        except FileNotFoundError:
            logger.warning(f"Sandbox directory '{sandbox_path}' not found during cleanup.")
        except Exception as e:
            logger.error(f"Error cleaning up sandbox directory '{sandbox_path}': {e}")

    async def shutdown(self):
        """Clean up resources (Docker client)."""
        if self.client:
            await asyncio.to_thread(self.client.close)  # Close in a separate thread
            self.client = None

# Singleton instance
sandbox_service = SandboxService()

# Initialize the Docker client during application startup
async def initialize_sandbox():
    await sandbox_service.initialize()

# Shutdown the Docker client during application shutdown
async def shutdown_sandbox():
    await sandbox_service.shutdown()