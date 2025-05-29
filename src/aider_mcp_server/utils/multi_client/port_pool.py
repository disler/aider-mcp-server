"""
Port Pool for dynamic port management and allocation.

This module provides the PortPool class for managing a pool of available
ports for multi-client HTTP server instances.
"""

import asyncio
import socket
from typing import Set

from aider_mcp_server.atoms.logging.logger import get_logger
from aider_mcp_server.atoms.utils.config_constants import (
    MULTI_CLIENT_PORT_RANGE_START,
    MULTI_CLIENT_PORT_RANGE_END,
)


class PortPool:
    """
    Manages a pool of available ports for dynamic allocation.
    
    This class handles port allocation and deallocation with proper
    thread safety and port availability checking.
    """

    def __init__(self, start_port: int = None, end_port: int = None) -> None:
        """
        Initialize the PortPool.

        Args:
            start_port: The beginning of the port range. Defaults to MULTI_CLIENT_PORT_RANGE_START.
            end_port: The end of the port range (inclusive). Defaults to MULTI_CLIENT_PORT_RANGE_END.
        """
        self.start_port = start_port if start_port is not None else MULTI_CLIENT_PORT_RANGE_START
        self.end_port = end_port if end_port is not None else MULTI_CLIENT_PORT_RANGE_END

        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self._lock = asyncio.Lock()

        if self.start_port > self.end_port:
            self.logger.warning(
                f"PortPool initialized with invalid range: start_port ({self.start_port}) > end_port ({self.end_port}). "
                "No ports will be available."
            )
            self._available_ports: Set[int] = set()
        else:
            self._available_ports: Set[int] = set(range(self.start_port, self.end_port + 1))
        
        self._allocated_ports: Set[int] = set()

        self.logger.info(
            f"PortPool initialized with range {self.start_port}-{self.end_port}. "
            f"Total manageable ports: {len(self._available_ports)}"
        )

    async def acquire_port(self) -> int:
        """
        Acquire an available port from the pool.

        Checks for system-level availability before allocating.
        Ports found to be in use by other processes are removed from the pool.

        Returns:
            An available port number.
        
        Raises:
            RuntimeError: If no ports are available in the pool or if all
                          ports in the pool are currently in use on the system.
        """
        async with self._lock:
            if not self._available_ports:
                self.logger.error("No ports left in the internal available pool.")
                raise RuntimeError("No ports available in the pool.")

            # Iterate through a list copy of available ports to allow modification of the set
            ports_to_check = list(self._available_ports)
            
            for port in ports_to_check:
                if await self.is_port_available(port):
                    self._available_ports.remove(port)
                    self._allocated_ports.add(port)
                    self.logger.info(f"Port {port} acquired.")
                    return port
                else:
                    # Port is in our _available_ports set, but system check says it's in use.
                    # This means another process is using it. Remove from our pool.
                    self.logger.warning(
                        f"Port {port} from pool is already in use on the system. "
                        "Removing from available set for this session."
                    )
                    if port in self._available_ports: # ensure it hasn't been removed by another concurrent check (though lock prevents this)
                        self._available_ports.remove(port)
            
            self.logger.error(
                "No system-available ports found in the configured range. "
                "All potentially available ports were occupied by other processes."
            )
            raise RuntimeError("No system-available ports found in the configured range.")
        
    async def release_port(self, port: int) -> None:
        """
        Release a port back to the pool.

        Args:
            port: The port number to release.
        """
        async with self._lock:
            if port in self._allocated_ports:
                self._allocated_ports.remove(port)
                # Only add back to available_ports if it's within the originally configured range
                if self.start_port <= port <= self.end_port:
                    self._available_ports.add(port)
                    self.logger.info(f"Port {port} released and returned to the pool.")
                else:
                    # This should not happen if acquire_port works correctly
                    self.logger.warning(
                        f"Port {port} was allocated but is outside the configured pool range "
                        f"({self.start_port}-{self.end_port}). Not adding back to available set."
                    )
            elif self.start_port <= port <= self.end_port:
                if port in self._available_ports:
                    self.logger.warning(
                        f"Port {port} release requested, but it was already in the available set. No action taken."
                    )
                else:
                    # Port is in range, not allocated, and not in available.
                    # This might happen if acquire_port removed it due to system conflict.
                    # Adding it back makes it available for future acquisition attempts.
                    self._available_ports.add(port)
                    self.logger.info(
                        f"Port {port} release requested. It was not allocated but is in pool range. "
                        "Added to available set."
                    )
            else:
                self.logger.warning(
                    f"Attempted to release port {port}, which is outside the configured pool range "
                    f"({self.start_port}-{self.end_port}) and was not in the allocated set."
                )
        
    async def is_port_available(self, port: int) -> bool:
        """
        Check if a port is available on the system for binding on host "127.0.0.1".

        Args:
            port: The port number to check.
        
        Returns:
            True if the port is available for binding, False otherwise.
        """
        host_to_check = "127.0.0.1"  # Standard host for local services

        if not (0 < port < 65536): # Ports are 1-65535. Port 0 has special meaning.
            self.logger.debug(f"Port {port} is outside the valid range (1-65535). Considered unavailable.")
            return False

        # This is a blocking synchronous function
        def _check_socket_binding_sync():
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    # Set SO_REUSEADDR to allow reuse of local addresses in TIME_WAIT state
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    s.bind((host_to_check, port))
                return True
            except socket.error as e:
                # Common errors: EADDRINUSE (Address already in use)
                self.logger.debug(f"Port {port} on {host_to_check} is not available for binding. Error: {e}")
                return False
            except Exception as e: # Catch any other unexpected errors during socket operations
                self.logger.warning(
                    f"Unexpected error while checking port {port} on {host_to_check} via socket: {e}"
                )
                return False

        try:
            # Run the blocking socket check in a separate thread
            return await asyncio.to_thread(_check_socket_binding_sync)
        except Exception as e: # Catch errors from asyncio.to_thread itself
            self.logger.error(
                f"Error executing port check for {port} on {host_to_check} using asyncio.to_thread: {e}"
            )
            return False
