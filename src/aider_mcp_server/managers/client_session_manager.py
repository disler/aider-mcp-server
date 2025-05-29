"""
Client Session Manager for multi-client orchestration.

This module provides the ClientSessionManager class that manages client sessions,
their lifecycles, workspace isolation, and session timeout handling.
"""

import asyncio
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel, Field

# Assuming this file will be placed in a location like src/aider_mcp_server/managers/
# for these imports to work correctly.
from aider_mcp_server.atoms.logging.logger import get_logger
from aider_mcp_server.atoms.utils.config_constants import (
    CLIENT_SESSION_TIMEOUT,
    WORKSPACE_BASE_DIR,
)


class ClientSession(BaseModel):
    """Information about an active client session."""

    session_id: str
    client_id: str
    workspace_path: Path
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ClientSessionManager:
    """
    Manages client sessions, their lifecycles, and workspace isolation.
    """

    def __init__(self) -> None:
        """Initialize the ClientSessionManager."""
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self._lock = asyncio.Lock()
        self._active_sessions: Dict[str, ClientSession] = {}
        self.logger.info("ClientSessionManager initialized")

    async def register_client(self, client_id: str) -> ClientSession:
        """
        Register a new client and create a session.

        Args:
            client_id: The unique identifier for the client.

        Returns:
            ClientSession: Information about the created session.

        Raises:
            ValueError: If the client already has an active session.
            RuntimeError: If workspace creation fails.
        """
        async with self._lock:
            if client_id in self._active_sessions:
                self.logger.warning(f"Client {client_id} already has an active session.")
                raise ValueError(f"Client {client_id} already has an active session.")

            session_id = f"session_{uuid.uuid4().hex[:8]}"

            base_workspace_dir = Path(WORKSPACE_BASE_DIR).expanduser().resolve()
            workspace_path = base_workspace_dir / client_id

            try:
                os.makedirs(workspace_path, exist_ok=True)
                self.logger.info(f"Ensured workspace directory exists: {workspace_path}")
            except OSError as e:
                self.logger.error(f"Failed to create workspace directory {workspace_path}: {e}")
                raise RuntimeError(f"Failed to create workspace for client {client_id}") from e

            session = ClientSession(
                session_id=session_id,
                client_id=client_id,
                workspace_path=workspace_path,
            )
            self._active_sessions[client_id] = session
            self.logger.info(f"Registered client {client_id} with session {session_id}. Workspace: {workspace_path}")
            return session

    async def get_client_workspace(self, client_id: str) -> Path:
        """
        Get the workspace path for a given client.
        Updates the session's last activity time.

        Args:
            client_id: The unique identifier for the client.

        Returns:
            Path: The path to the client's workspace.

        Raises:
            ValueError: If no active session is found for the client.
        """
        async with self._lock:
            session = self._active_sessions.get(client_id)
            if not session:
                self.logger.warning(f"No active session found for client {client_id} in get_client_workspace.")
                raise ValueError(f"No active session found for client {client_id}")

            session.last_activity = datetime.now(timezone.utc)
            self.logger.debug(f"Updated last activity for client {client_id}.")
            return session.workspace_path

    async def cleanup_client_session(self, client_id: str) -> None:
        """
        Clean up and remove a client's session.

        Args:
            client_id: The unique identifier for the client.

        Raises:
            ValueError: If no active session is found for the client to clean up.
        """
        async with self._lock:
            await self._cleanup_client_session_internal(client_id)

    async def _cleanup_client_session_internal(self, client_id: str) -> None:
        """
        Internal method to clean up a client session. Assumes lock is held.
        Does not remove the workspace directory itself.
        """
        if client_id not in self._active_sessions:
            self.logger.warning(f"No active session found for client {client_id} during cleanup attempt.")
            raise ValueError(f"No active session found for client {client_id} to clean up.")

        session = self._active_sessions.pop(client_id)
        self.logger.info(f"Cleaned up session {session.session_id} for client {client_id}.")

    async def update_session_activity(self, client_id: str) -> None:
        """
        Update the last activity time for a client session.

        Args:
            client_id: ID of the client to update.
        """
        async with self._lock:
            if client_id in self._active_sessions:
                session = self._active_sessions[client_id]
                session.last_activity = datetime.now(timezone.utc)
                self.logger.debug(f"Session activity updated for client {client_id}.")
            else:
                self.logger.warning(f"Attempted to update activity for non-existent client session: {client_id}")

    async def cleanup_expired_sessions(self) -> List[str]:
        """
        Clean up sessions that have exceeded the CLIENT_SESSION_TIMEOUT.

        Returns:
            List of client IDs whose sessions were cleaned up.
        """
        async with self._lock:
            current_time = datetime.now(timezone.utc)
            expired_client_ids: List[str] = []

            for client_id, session in list(self._active_sessions.items()):
                time_since_activity = (current_time - session.last_activity).total_seconds()
                if time_since_activity > CLIENT_SESSION_TIMEOUT:
                    self.logger.info(
                        f"Session for client {client_id} expired. Last activity: {session.last_activity}."
                        f" Timeout: {CLIENT_SESSION_TIMEOUT}s"
                    )
                    expired_client_ids.append(client_id)

            for client_id in expired_client_ids:
                try:
                    await self._cleanup_client_session_internal(client_id)
                except ValueError:
                    # Session might have been cleaned up by a concurrent call or direct cleanup
                    self.logger.warning(
                        f"Session for client {client_id} was already cleaned up during expired session sweep."
                    )
                except Exception as e:
                    self.logger.error(f"Error cleaning up expired session for client {client_id}: {e}")

            if expired_client_ids:
                self.logger.info(
                    f"Attempted cleanup for {len(expired_client_ids)} expired sessions: {expired_client_ids}"
                )
            return expired_client_ids
