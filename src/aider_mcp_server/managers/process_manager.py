"""
Process Manager for multi-client server instances.

This module provides the ProcessManager class that handles the lifecycle
of server subprocesses, including spawning, termination, health monitoring,
and automated restarts.
"""

import asyncio
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from aider_mcp_server.atoms.logging.logger import get_logger
from aider_mcp_server.atoms.types.data_types import ProcessInfo
# Assuming PortPool will be available from this path
from aider_mcp_server.utils.multi_client.port_pool import PortPool

DEFAULT_HEALTH_CHECK_INTERVAL = 10  # seconds
DEFAULT_TERMINATION_TIMEOUT = 5  # seconds
MAX_RESTARTS_DEFAULT = 3
PORT_ENV_VAR_DEFAULT = "AIDER_MCP_PORT"


class ProcessManager:
    """
    Manages the lifecycle of server subprocesses.
    """

    def __init__(
        self,
        port_pool: PortPool, # Type hint for PortPool
        max_restarts: int = MAX_RESTARTS_DEFAULT,
        health_check_interval: int = DEFAULT_HEALTH_CHECK_INTERVAL,
        termination_timeout: int = DEFAULT_TERMINATION_TIMEOUT,
    ) -> None:
        """
        Initialize the ProcessManager.

        Args:
            port_pool: An instance of PortPool for managing port allocation.
            max_restarts: Maximum number of times a process can be restarted.
            health_check_interval: Interval in seconds for periodic health checks.
            termination_timeout: Timeout in seconds for graceful process termination.
        """
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self._lock = asyncio.Lock()
        self._active_processes: Dict[str, ProcessInfo] = {}
        # _process_metadata stores info not in ProcessInfo but needed for respawn
        self._process_metadata: Dict[str, Dict[str, Any]] = {}

        self.port_pool = port_pool
        self.max_restarts = max_restarts
        self.health_check_interval = health_check_interval
        self.termination_timeout = termination_timeout

        self._health_check_task: Optional[asyncio.Task[Any]] = None
        self.logger.info("ProcessManager initialized")

    async def spawn_server_process(
        self,
        client_id: str,
        workspace_path: Path,
        command: List[str],
        port_env_var: Optional[str] = PORT_ENV_VAR_DEFAULT,
        extra_env: Optional[Dict[str, str]] = None,
    ) -> ProcessInfo:
        """
        Spawn a new server subprocess.

        Args:
            client_id: The client ID associated with this process.
            workspace_path: The working directory for the process.
            command: The command and arguments to execute.
            port_env_var: Name of the environment variable to pass the allocated port.
                          If None, port is not passed as an env var.
            extra_env: Additional environment variables for the subprocess.

        Returns:
            ProcessInfo: Information about the spawned process.

        Raises:
            RuntimeError: If port acquisition or process spawning fails.
        """
        async with self._lock:
            process_id = f"proc_{uuid.uuid4().hex[:8]}"
            acquired_port = -1
            try:
                acquired_port = await self.port_pool.acquire_port()
                if acquired_port == -1: # Assuming -1 means failure from PortPool
                    raise RuntimeError("Failed to acquire a port from PortPool.")
                self.logger.info(f"Acquired port {acquired_port} for process {process_id}")

                env = os.environ.copy()
                if port_env_var:
                    env[port_env_var] = str(acquired_port)
                if extra_env:
                    env.update(extra_env)

                # Ensure workspace_path exists
                workspace_path.mkdir(parents=True, exist_ok=True)

                sub_process = await asyncio.create_subprocess_exec(
                    *command,
                    cwd=str(workspace_path),
                    env=env,
                    stdout=asyncio.subprocess.PIPE, # Capture stdout/stderr for logging/debugging
                    stderr=asyncio.subprocess.PIPE,
                )
                self.logger.info(f"Spawned process {process_id} (PID: {sub_process.pid}) on port {acquired_port} with command: {' '.join(command)}")

                process_info = ProcessInfo(
                    process_id=process_id,
                    client_id=client_id,
                    port=acquired_port,
                    workspace_path=workspace_path,
                    process=sub_process,
                    status="starting",
                    command=command,
                    restart_count=0,
                )
                self._active_processes[process_id] = process_info
                self._process_metadata[process_id] = {
                    "port_env_var": port_env_var,
                    "extra_env": extra_env.copy() if extra_env else {},
                }
                return process_info
            except Exception as e:
                self.logger.error(f"Failed to spawn server process {process_id}: {e}")
                if acquired_port != -1:
                    await self.port_pool.release_port(acquired_port)
                raise RuntimeError(f"Failed to spawn server process: {e}") from e

    async def terminate_server_process(self, process_id: str) -> None:
        """
        Terminate a server subprocess.

        Args:
            process_id: The ID of the process to terminate.

        Raises:
            ValueError: If the process ID is not found.
        """
        async with self._lock:
            process_info = self._active_processes.get(process_id)
            if not process_info:
                self.logger.warning(f"Process {process_id} not found for termination.")
                raise ValueError(f"Process {process_id} not found.")

            if process_info.status in ["stopping", "stopped", "failed"]:
                self.logger.info(f"Process {process_id} is already {process_info.status}.")
                # Ensure port is released if it was somehow left acquired
                if process_info.port != -1 and process_info.status != "stopped": # 'stopped' status implies port released
                     # Check if port is still held by this process_info before releasing
                    pass # Port release is handled in _cleanup_process_resources
                return

            await self._terminate_process_object(process_info)
            await self._cleanup_process_resources(process_id, process_info, "stopped")


    async def _terminate_process_object(self, process_info: ProcessInfo) -> None:
        """Helper to terminate the actual OS process."""
        self.logger.info(f"Terminating process {process_info.process_id} (PID: {process_info.process.pid})...")
        process_info.status = "stopping"
        
        if process_info.process.returncode is None: # Process is running
            try:
                process_info.process.terminate() # Send SIGTERM
                await asyncio.wait_for(process_info.process.wait(), timeout=self.termination_timeout)
                self.logger.info(f"Process {process_info.process_id} terminated gracefully (SIGTERM).")
            except asyncio.TimeoutError:
                self.logger.warning(
                    f"Process {process_info.process_id} did not terminate gracefully within {self.termination_timeout}s. Sending SIGKILL."
                )
                if process_info.process.returncode is None: # Check again before kill
                    process_info.process.kill()
                    await process_info.process.wait() # Ensure it's killed
                self.logger.info(f"Process {process_info.process_id} killed (SIGKILL).")
            except Exception as e: # Catch other potential errors during termination
                self.logger.error(f"Error during process termination for {process_info.process_id}: {e}")
                if process_info.process.returncode is None: # If still running after error
                    process_info.process.kill() # Ensure it's killed
                    await process_info.process.wait()

        # Log stdout/stderr if available
        # This needs to be done carefully to avoid blocking if pipes are full or process hangs
        # For simplicity, this example doesn't read stdout/stderr during termination
        # In a real scenario, you might want to drain pipes before/after termination.

    async def _cleanup_process_resources(self, process_id: str, process_info: ProcessInfo, final_status: str) -> None:
        """Helper to release port and remove process from active tracking."""
        if process_info.port != -1:
            await self.port_pool.release_port(process_info.port)
            self.logger.info(f"Released port {process_info.port} for process {process_id}")
            process_info.port = -1 # Mark port as released in ProcessInfo

        process_info.status = final_status
        # Keep failed processes in the list for inspection, but they are effectively "stopped"
        if final_status == "stopped":
            if process_id in self._active_processes: # It might have been removed by respawn logic
                del self._active_processes[process_id]
            if process_id in self._process_metadata:
                del self._process_metadata[process_id]
            self.logger.info(f"Process {process_id} removed from active tracking.")


    async def get_process_info(self, process_id: str) -> Optional[ProcessInfo]:
        """Get information about a specific process."""
        async with self._lock:
            return self._active_processes.get(process_id)

    async def list_active_processes(self) -> List[ProcessInfo]:
        """List all currently managed processes."""
        async with self._lock:
            return list(self._active_processes.values())
            
    async def get_process_by_client_id(self, client_id: str) -> Optional[ProcessInfo]:
        """Get process information for a given client ID."""
        async with self._lock:
            for proc_info in self._active_processes.values():
                if proc_info.client_id == client_id:
                    return proc_info
            return None

    async def health_check_all_processes(self) -> None:
        """Periodically check the health of all managed processes."""
        async with self._lock:
            # Create a list of items to iterate over to avoid issues with dict size changes during iteration
            processes_to_check = list(self._active_processes.items())

        for process_id, process_info in processes_to_check:
            # Re-acquire lock for modifications or sensitive reads if necessary,
            # but basic returncode check can be outside.
            # For this iteration, we'll lock per process modification.
            
            proc = process_info.process
            if proc.returncode is None:
                # Process is still running (or thinks it is)
                if process_info.status == "starting": # First successful check after starting
                    process_info.status = "running"
                process_info.last_health_check = datetime.now(timezone.utc)
                # self.logger.debug(f"Process {process_id} is healthy (PID: {proc.pid}).")
            else:
                # Process has terminated
                self.logger.warning(
                    f"Process {process_id} (PID: {proc.pid}) has terminated with code {proc.returncode}."
                    f" Current status: {process_info.status}"
                )
                # Lock before modifying shared state
                async with self._lock:
                    # Check if process still exists and status hasn't changed (e.g. by explicit termination)
                    current_process_info = self._active_processes.get(process_id)
                    if current_process_info and current_process_info.status not in ["stopping", "stopped", "failed"]:
                        await self._handle_process_completion(process_id, proc.returncode)


    async def _handle_process_completion(self, process_id: str, return_code: int) -> None:
        """
        Handle a process that has completed/terminated.
        This method assumes the lock `self._lock` is already held.
        """
        process_info = self._active_processes.get(process_id)
        if not process_info:
            self.logger.warning(f"Process {process_id} not found during completion handling.")
            return

        self.logger.info(f"Handling completion for process {process_id} with return code {return_code}.")

        # Release the port used by the terminated process
        if process_info.port != -1:
            await self.port_pool.release_port(process_info.port)
            self.logger.info(f"Released port {process_info.port} for completed process {process_id}")
            original_port = process_info.port # Keep for respawn attempt if needed
            process_info.port = -1 # Mark as released
        else: # Should not happen if port was properly managed
            original_port = -1 


        if process_info.restart_count < self.max_restarts:
            process_info.restart_count += 1
            process_info.status = "restarting"
            self.logger.info(
                f"Attempting to restart process {process_id} (attempt {process_info.restart_count}/{self.max_restarts})."
            )
            
            # Retrieve metadata for respawn
            metadata = self._process_metadata.get(process_id, {})
            port_env_var = metadata.get("port_env_var", PORT_ENV_VAR_DEFAULT)
            extra_env = metadata.get("extra_env")

            try:
                # We need a new port for the restarted process
                new_acquired_port = await self.port_pool.acquire_port()
                if new_acquired_port == -1:
                    raise RuntimeError("Failed to acquire a new port for restart.")
                self.logger.info(f"Acquired new port {new_acquired_port} for restarting process {process_id}")

                env = os.environ.copy()
                if port_env_var:
                    env[port_env_var] = str(new_acquired_port)
                if extra_env:
                    env.update(extra_env)

                new_sub_process = await asyncio.create_subprocess_exec(
                    *process_info.command, # Use original command
                    cwd=str(process_info.workspace_path),
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                self.logger.info(
                    f"Respawned process {process_id} as new PID {new_sub_process.pid} on port {new_acquired_port}."
                )
                
                # Update ProcessInfo with new process and port
                process_info.process = new_sub_process
                process_info.port = new_acquired_port # Update to the new port
                process_info.status = "starting" # Or "running" if health check is immediate
                process_info.created_at = datetime.now(timezone.utc)
                process_info.last_health_check = datetime.now(timezone.utc)
                # _active_processes already holds this process_info, just updated
            except Exception as e:
                self.logger.error(f"Failed to restart process {process_id}: {e}")
                process_info.status = "failed" # Mark as failed if restart fails
                # Ensure the newly acquired port (if any) is released if restart fails mid-way
                if new_acquired_port != -1 and new_acquired_port != original_port : # original_port already released
                    await self.port_pool.release_port(new_acquired_port)
                # If original_port was re-acquired and failed, it should be released.
                # This logic assumes new_acquired_port is different or managed correctly by PortPool.
                await self._cleanup_process_resources(process_id, process_info, "failed")
        else:
            self.logger.error(
                f"Process {process_id} reached max restart attempts ({self.max_restarts}). Marking as failed."
            )
            await self._cleanup_process_resources(process_id, process_info, "failed")


    async def start_monitoring(self) -> None:
        """Start the periodic health check task."""
        if self._health_check_task is None or self._health_check_task.done():
            self._health_check_task = asyncio.create_task(self._periodic_health_check())
            self.logger.info("Process health monitoring started.")
        else:
            self.logger.info("Process health monitoring is already running.")

    async def _periodic_health_check(self) -> None:
        """The background task that runs health checks periodically."""
        while True:
            try:
                await self.health_check_all_processes()
            except asyncio.CancelledError:
                self.logger.info("Health check task cancelled.")
                break
            except Exception as e:
                self.logger.error(f"Error in periodic health check loop: {e}", exc_info=True)
            
            try:
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError: # Handle cancellation during sleep
                self.logger.info("Health check task cancelled during sleep.")
                break


    async def shutdown(self) -> None:
        """Shutdown the ProcessManager and terminate all managed processes."""
        self.logger.info("Shutting down ProcessManager...")

        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                self.logger.info("Health check task successfully cancelled during shutdown.")
            except Exception as e: # Should not happen if task handles CancelledError
                self.logger.error(f"Error waiting for health check task to cancel: {e}")
            self._health_check_task = None
        
        async with self._lock:
            # Create a list of process IDs to avoid issues with dict size changes during iteration
            process_ids = list(self._active_processes.keys())
            
            termination_tasks = []
            for process_id in process_ids:
                process_info = self._active_processes.get(process_id)
                if process_info: # Should always be true here
                    self.logger.info(f"Initiating shutdown for process {process_id}.")
                    # Create task for termination to run them concurrently
                    termination_tasks.append(
                        asyncio.create_task(self._terminate_process_object(process_info))
                    )
            
            if termination_tasks:
                await asyncio.gather(*termination_tasks, return_exceptions=True)
                self.logger.info("All process termination tasks completed.")

            # Final cleanup of resources
            for process_id in process_ids:
                process_info = self._active_processes.get(process_id) # Get potentially updated info
                if process_info:
                     # Status might be "stopping", "stopped", or still "running" if termination failed
                    await self._cleanup_process_resources(process_id, process_info, "stopped")

            self._active_processes.clear()
            self._process_metadata.clear()

        self.logger.info("ProcessManager shutdown completed.")
