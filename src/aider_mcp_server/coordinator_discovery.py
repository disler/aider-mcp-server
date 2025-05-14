"""
Coordinator Discovery for Aider MCP Server.

This module provides mechanisms for discovering and connecting to existing
ApplicationCoordinator instances across different processes.
"""

import asyncio
import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar

from aider_mcp_server.atoms.logging import get_logger

T = TypeVar("T")

# Configure logging
logger = get_logger(__name__)

# Default discovery file location
DEFAULT_DISCOVERY_DIR = Path(tempfile.gettempdir()) / "aider_mcp_coordinator"
DEFAULT_DISCOVERY_FILE = DEFAULT_DISCOVERY_DIR / "coordinator_registry.json"

# Environment variable to override discovery file location
ENV_DISCOVERY_FILE = "AIDER_MCP_COORDINATOR_DISCOVERY_FILE"


class CoordinatorInfo:
    """Information about a running coordinator."""

    def __init__(
        self,
        coordinator_id: str,
        host: str,
        port: int,
        transport_type: str = "sse",
        start_time: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize coordinator information.

        Args:
            coordinator_id: Unique identifier for the coordinator
            host: Host where the coordinator is running
            port: Port where the coordinator is listening
            transport_type: Type of transport (e.g., "sse", "websocket")
            start_time: Time when the coordinator was started (defaults to now)
            metadata: Additional metadata about the coordinator
        """
        self.coordinator_id = coordinator_id
        self.host = host
        self.port = port
        self.transport_type = transport_type
        self.start_time = start_time or time.time()
        self.metadata = metadata or {}
        self.last_heartbeat = self.start_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "coordinator_id": self.coordinator_id,
            "host": self.host,
            "port": self.port,
            "transport_type": self.transport_type,
            "start_time": self.start_time,
            "last_heartbeat": self.last_heartbeat,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CoordinatorInfo":
        """Create from dictionary after deserialization."""
        info = cls(
            coordinator_id=data["coordinator_id"],
            host=data["host"],
            port=data["port"],
            transport_type=data["transport_type"],
            start_time=data["start_time"],
            metadata=data.get("metadata", {}),
        )
        info.last_heartbeat = data.get("last_heartbeat", info.start_time)
        return info

    def update_heartbeat(self) -> None:
        """Update the last heartbeat time."""
        self.last_heartbeat = time.time()

    def is_active(self, max_age_seconds: float = 30.0) -> bool:
        """
        Check if this coordinator is likely still active.

        Args:
            max_age_seconds: Maximum age in seconds since last heartbeat

        Returns:
            True if the coordinator is likely still active
        """
        return (time.time() - self.last_heartbeat) < max_age_seconds


class CoordinatorDiscovery:
    """
    Handles discovery and registration of ApplicationCoordinator instances.

    This class provides mechanisms for:
    1. Registering a coordinator so other processes can find it
    2. Discovering existing coordinators
    3. Maintaining the registry with periodic heartbeats
    """

    def __init__(
        self,
        discovery_file: Optional[Path] = None,
        heartbeat_interval: float = 10.0,
    ):
        """
        Initialize the coordinator discovery.

        Args:
            discovery_file: Path to the discovery file (defaults to system temp dir)
            heartbeat_interval: Interval in seconds for heartbeat updates
        """
        # Use environment variable if set, otherwise use default or provided value
        env_file = os.environ.get(ENV_DISCOVERY_FILE)
        if env_file:
            self.discovery_file = Path(env_file)
            logger.info(f"Using discovery file from environment: {self.discovery_file}")
        else:
            self.discovery_file = discovery_file or DEFAULT_DISCOVERY_FILE
            logger.info(f"Using discovery file: {self.discovery_file}")

        # Ensure the directory exists
        self.discovery_file.parent.mkdir(parents=True, exist_ok=True)

        self.heartbeat_interval = heartbeat_interval
        self._heartbeat_task: Optional[asyncio.Task[None]] = None
        self._shutdown_event = asyncio.Event()
        self._registered_coordinator: Optional[CoordinatorInfo] = None
        self._file_lock = asyncio.Lock()

    async def register_coordinator(
        self,
        host: str,
        port: int,
        transport_type: str = "sse",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Register a coordinator in the discovery file.

        Args:
            host: Host where the coordinator is running
            port: Port where the coordinator is listening
            transport_type: Type of transport (e.g., "sse", "websocket")
            metadata: Additional metadata about the coordinator

        Returns:
            The coordinator ID
        """
        coordinator_id = f"coordinator_{uuid.uuid4().hex[:8]}"
        info = CoordinatorInfo(
            coordinator_id=coordinator_id,
            host=host,
            port=port,
            transport_type=transport_type,
            metadata=metadata,
        )

        # Store the registered coordinator
        self._registered_coordinator = info

        # Write to the discovery file
        await self._update_registry(info)

        # Start the heartbeat task
        self._start_heartbeat_task()

        logger.info(
            f"Registered coordinator {coordinator_id} at {host}:{port} ({transport_type})"
        )
        return coordinator_id

    async def discover_coordinators(
        self, max_age_seconds: float = 30.0
    ) -> List[CoordinatorInfo]:
        """
        Discover active coordinators from the registry.

        Args:
            max_age_seconds: Maximum age in seconds for a coordinator to be considered active

        Returns:
            List of active coordinator information
        """
        try:
            async with self._file_lock:
                if not self.discovery_file.exists():
                    logger.debug(f"Discovery file {self.discovery_file} does not exist")
                    return []

                # Read the registry file
                registry_data = json.loads(self.discovery_file.read_text())
                coordinators = [
                    CoordinatorInfo.from_dict(data) for data in registry_data
                ]

                # Filter for active coordinators
                active_coordinators = [
                    coord for coord in coordinators if coord.is_active(max_age_seconds)
                ]

                if len(active_coordinators) < len(coordinators):
                    logger.debug(
                        f"Filtered out {len(coordinators) - len(active_coordinators)} inactive coordinators"
                    )

                return active_coordinators
        except Exception as e:
            logger.error(f"Error discovering coordinators: {e}")
            return []

    async def find_best_coordinator(self) -> Optional[CoordinatorInfo]:
        """
        Find the best coordinator to connect to.

        Currently selects the most recently started active coordinator.

        Returns:
            The best coordinator or None if none are available
        """
        coordinators = await self.discover_coordinators()
        if not coordinators:
            return None

        # Sort by start time (newest first)
        coordinators.sort(key=lambda c: c.start_time, reverse=True)
        return coordinators[0]

    async def _update_registry(self, info: Optional[CoordinatorInfo] = None) -> None:
        """
        Update the registry file with the current coordinator information.

        Args:
            info: Optional coordinator info to update or add
        """
        try:
            async with self._file_lock:
                # Read existing registry if it exists
                registry_data = []
                if self.discovery_file.exists():
                    try:
                        registry_data = json.loads(self.discovery_file.read_text())
                    except json.JSONDecodeError:
                        logger.warning(
                            f"Invalid JSON in discovery file {self.discovery_file}, starting fresh"
                        )

                # Convert to CoordinatorInfo objects
                coordinators = [
                    CoordinatorInfo.from_dict(data) for data in registry_data
                ]

                # Update the specific coordinator if provided
                if info:
                    # Update if exists, otherwise add
                    updated = False
                    for i, coord in enumerate(coordinators):
                        if coord.coordinator_id == info.coordinator_id:
                            coordinators[i] = info
                            updated = True
                            break
                    if not updated:
                        coordinators.append(info)

                # Filter out inactive coordinators
                active_coordinators = [
                    coord for coord in coordinators if coord.is_active()
                ]

                # Write back to the file
                registry_data = [coord.to_dict() for coord in active_coordinators]
                self.discovery_file.write_text(json.dumps(registry_data, indent=2))
        except Exception as e:
            logger.error(f"Error updating coordinator registry: {e}")

    async def _heartbeat_loop(self) -> None:
        """Background task to periodically update the heartbeat."""
        try:
            while not self._shutdown_event.is_set():
                if self._registered_coordinator:
                    self._registered_coordinator.update_heartbeat()
                    await self._update_registry(self._registered_coordinator)
                    logger.debug(
                        f"Updated heartbeat for coordinator {self._registered_coordinator.coordinator_id}"
                    )

                try:
                    # Wait with timeout for shutdown event
                    await asyncio.wait_for(
                        self._shutdown_event.wait(), self.heartbeat_interval
                    )
                except asyncio.TimeoutError:
                    # This is expected, just continue the loop
                    pass
        except asyncio.CancelledError:
            logger.debug("Heartbeat task cancelled")
        except Exception as e:
            logger.error(f"Error in heartbeat loop: {e}")

    def _start_heartbeat_task(self) -> None:
        """Start the background heartbeat task."""
        if self._heartbeat_task is None or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(
                self._heartbeat_loop(), name="coordinator_heartbeat"
            )
            logger.debug("Started coordinator heartbeat task")

    async def shutdown(self) -> None:
        """Shut down the discovery service and clean up."""
        logger.info("Shutting down coordinator discovery")
        self._shutdown_event.set()

        # Cancel heartbeat task
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Remove our coordinator from the registry
        if self._registered_coordinator:
            try:
                async with self._file_lock:
                    if self.discovery_file.exists():
                        registry_data = json.loads(self.discovery_file.read_text())
                        coordinators = [
                            CoordinatorInfo.from_dict(data) for data in registry_data
                        ]
                        # Remove our coordinator
                        coordinators = [
                            coord
                            for coord in coordinators
                            if coord.coordinator_id
                            != self._registered_coordinator.coordinator_id
                        ]
                        # Write back to the file
                        registry_data = [coord.to_dict() for coord in coordinators]
                        self.discovery_file.write_text(
                            json.dumps(registry_data, indent=2)
                        )
                        logger.info(
                            f"Removed coordinator {self._registered_coordinator.coordinator_id} from registry"
                        )
            except Exception as e:
                logger.error(f"Error removing coordinator from registry: {e}")

    async def __aenter__(self) -> "CoordinatorDiscovery":
        """Enter async context."""
        return self

    async def __aexit__(
        self, 
        exc_type: Optional[type[BaseException]], 
        exc_val: Optional[BaseException], 
        exc_tb: Optional[Any]
    ) -> None:
        """Exit async context and clean up."""
        await self.shutdown()