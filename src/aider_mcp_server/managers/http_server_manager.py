"""
HTTP Server Manager for multi-client orchestration.

This module provides the HttpServerManager class that orchestrates multiple
HTTP server instances, managing client sessions and server lifecycle.
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from aider_mcp_server.atoms.logging.logger import get_logger
from aider_mcp_server.atoms.types.data_types import ClientRequest, ServerInfo, SessionInfo
from aider_mcp_server.atoms.utils.config_constants import (
    CLIENT_SESSION_TIMEOUT,
    MAX_CONCURRENT_CLIENTS,
)


class HttpServerManager:
    """
    Manages multiple HTTP server instances for multi-client support.

    This class orchestrates the creation, management, and cleanup of HTTP server
    instances, ensuring proper client isolation and session management.
    """

    def __init__(self) -> None:
        """Initialize the HTTP Server Manager."""
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")

        # Thread safety
        self._lock = asyncio.Lock()

        # Internal state tracking
        self._active_sessions: Dict[str, SessionInfo] = {}  # client_id -> SessionInfo
        self._active_servers: Dict[str, ServerInfo] = {}  # server_id -> ServerInfo
        self._client_to_server: Dict[str, str] = {}  # client_id -> server_id

        self.logger.info("HttpServerManager initialized")

    async def create_client_session(self, client_request: ClientRequest) -> SessionInfo:
        """
        Create a new client session with dedicated server instance.

        Args:
            client_request: Request containing client information

        Returns:
            SessionInfo: Information about the created session

        Raises:
            ValueError: If client already has an active session
            RuntimeError: If maximum concurrent clients exceeded
        """
        async with self._lock:
            client_id = client_request.client_id

            # Check if client already has a session
            if client_id in self._active_sessions:
                raise ValueError(f"Client {client_id} already has an active session")

            # Check concurrent client limit
            if len(self._active_sessions) >= MAX_CONCURRENT_CLIENTS:
                raise RuntimeError(f"Maximum concurrent clients ({MAX_CONCURRENT_CLIENTS}) exceeded")

            # Generate unique session and server IDs
            session_id = f"session_{uuid.uuid4().hex[:8]}"
            server_id = f"server_{uuid.uuid4().hex[:8]}"

            # Create session info
            session_info = SessionInfo(
                session_id=session_id,
                client_id=client_id,
                workspace_id=client_request.workspace_id,
                created_at=datetime.now(),
                last_activity=datetime.now(),
                status="active",
                metadata=client_request.request_data.copy(),
            )

            # Create server info (will be populated when server starts)
            server_info = ServerInfo(
                server_id=server_id,
                host="127.0.0.1",  # Default host
                port=0,  # Will be assigned by port pool
                status="starting",
                workspace_id=client_request.workspace_id,
                created_at=datetime.now(),
                active_clients=1,
            )

            # Store session and server mappings
            self._active_sessions[client_id] = session_info
            self._active_servers[server_id] = server_info
            self._client_to_server[client_id] = server_id

            self.logger.info(f"Created client session: {session_id} for client {client_id} with server {server_id}")

            return session_info

    async def destroy_client_session(self, client_id: str) -> None:
        """
        Destroy a client session and cleanup associated resources.

        Args:
            client_id: ID of the client session to destroy

        Raises:
            ValueError: If client session does not exist
        """
        async with self._lock:
            if client_id not in self._active_sessions:
                raise ValueError(f"No active session found for client {client_id}")

            # Get associated server
            server_id = self._client_to_server.get(client_id)

            # Remove session
            session_info = self._active_sessions.pop(client_id)

            # Remove client-to-server mapping
            if client_id in self._client_to_server:
                del self._client_to_server[client_id]

            # Update or remove server info
            if server_id and server_id in self._active_servers:
                server_info = self._active_servers[server_id]
                server_info.active_clients -= 1

                # If no more clients, mark server for cleanup
                if server_info.active_clients <= 0:
                    server_info.status = "stopping"
                    # TODO: Actual server cleanup will be implemented with ProcessManager
                    del self._active_servers[server_id]

            self.logger.info(f"Destroyed client session: {session_info.session_id} for client {client_id}")

    async def get_client_server_info(self, client_id: str) -> Optional[ServerInfo]:
        """
        Get server information for a specific client.

        Args:
            client_id: ID of the client

        Returns:
            ServerInfo if client has an active session, None otherwise
        """
        async with self._lock:
            server_id = self._client_to_server.get(client_id)
            if server_id:
                return self._active_servers.get(server_id)
            return None

    async def list_active_sessions(self) -> List[SessionInfo]:
        """
        List all active client sessions.

        Returns:
            List of active SessionInfo objects
        """
        async with self._lock:
            # Update last activity times for active sessions
            current_time = datetime.now()
            active_sessions = []

            for session_info in self._active_sessions.values():
                # Check if session is still within timeout
                time_since_activity = (current_time - session_info.last_activity).total_seconds()

                if time_since_activity > CLIENT_SESSION_TIMEOUT:
                    session_info.status = "idle"

                active_sessions.append(session_info)

            return active_sessions.copy()

    async def update_session_activity(self, client_id: str) -> None:
        """
        Update the last activity time for a client session.

        Args:
            client_id: ID of the client to update
        """
        async with self._lock:
            if client_id in self._active_sessions:
                self._active_sessions[client_id].last_activity = datetime.now()
                self._active_sessions[client_id].status = "active"

    async def get_session_count(self) -> int:
        """
        Get the current number of active sessions.

        Returns:
            Number of active client sessions
        """
        async with self._lock:
            return len(self._active_sessions)

    async def get_server_count(self) -> int:
        """
        Get the current number of active servers.

        Returns:
            Number of active server instances
        """
        async with self._lock:
            return len(self._active_servers)

    async def cleanup_expired_sessions(self) -> List[str]:
        """
        Clean up sessions that have exceeded the timeout.

        Returns:
            List of client IDs that were cleaned up
        """
        async with self._lock:
            current_time = datetime.now()
            expired_clients = []

            for client_id, session_info in list(self._active_sessions.items()):
                time_since_activity = (current_time - session_info.last_activity).total_seconds()

                if time_since_activity > CLIENT_SESSION_TIMEOUT:
                    expired_clients.append(client_id)

            # Clean up expired sessions
            for client_id in expired_clients:
                try:
                    await self.destroy_client_session(client_id)
                    self.logger.info(f"Cleaned up expired session for client {client_id}")
                except Exception as e:
                    self.logger.error(f"Error cleaning up session for client {client_id}: {e}")

            return expired_clients

    async def shutdown(self) -> None:
        """
        Shutdown the HTTP Server Manager and cleanup all resources.
        """
        async with self._lock:
            self.logger.info("Shutting down HttpServerManager...")

            # Get list of all active clients
            active_clients = list(self._active_sessions.keys())

            # Destroy all active sessions
            for client_id in active_clients:
                try:
                    await self.destroy_client_session(client_id)
                except Exception as e:
                    self.logger.error(f"Error destroying session for client {client_id}: {e}")

            # Clear all state
            self._active_sessions.clear()
            self._active_servers.clear()
            self._client_to_server.clear()

            self.logger.info("HttpServerManager shutdown completed")
