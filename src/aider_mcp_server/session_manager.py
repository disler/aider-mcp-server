import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, Optional, Set

from aider_mcp_server.security import Permissions

logger = logging.getLogger(__name__)


class Session:
    """
    Represents a client session with authentication and permissions.
    """

    def __init__(self, transport_id: str) -> None:
        """
        Initialize a new session.

        Args:
            transport_id: Unique identifier for the transport associated with this session
        """
        self.transport_id = transport_id
        self.creation_time = datetime.now()
        self.last_accessed_time = self.creation_time
        self.user_info: Optional[Dict[str, Any]] = None
        self.permissions: Set[Permissions] = set()
        self.custom_data: Dict[str, Any] = {}


class SessionManager:
    """
    Manages client sessions with creation times and timeouts.

    Responsibilities:
    - Creates and stores session information
    - Tracks session permissions
    - Handles session authentication state
    - Provides session cleanup functionality
    """

    def __init__(self, start_cleanup: bool = True) -> None:
        """
        Initialize the Session Manager.

        Args:
            start_cleanup: Whether to start the cleanup task (disabled for testing)
        """
        self.sessions: Dict[str, Session] = {}
        self.lock = asyncio.Lock()

        # Session timeout in seconds
        self.session_timeout = 3600  # 1 hour

        # Start a task to clean up expired sessions periodically
        if start_cleanup:
            try:
                asyncio.get_running_loop()
                asyncio.create_task(self.cleanup_expired_sessions())
            except RuntimeError:
                # No event loop running, skip cleanup task (for testing)
                pass

    async def get_or_create_session(self, transport_id: str) -> Session:
        async with self.lock:
            if transport_id in self.sessions:
                session = self.sessions[transport_id]
                session.last_accessed_time = datetime.now()
                return session
            else:
                new_session = Session(transport_id)
                self.sessions[transport_id] = new_session
                return new_session

    async def update_session(self, transport_id: str, data: Dict[str, Any]) -> Optional[Session]:
        async with self.lock:
            if transport_id in self.sessions:
                session = self.sessions[transport_id]
                session.last_accessed_time = datetime.now()
                session.custom_data.update(data)
                return session
            else:
                logger.warning(f"Session for transport '{transport_id}' not found.")
                return None

    async def remove_session(self, transport_id: str) -> None:
        async with self.lock:
            if transport_id in self.sessions:
                del self.sessions[transport_id]
            else:
                logger.warning(f"Session for transport '{transport_id}' not found.")

    async def check_permission(self, transport_id: str, required_permission: Permissions) -> bool:
        async with self.lock:
            if transport_id in self.sessions:
                session = self.sessions[transport_id]
                session.last_accessed_time = datetime.now()
                return required_permission in session.permissions
            else:
                logger.warning(f"Session for transport '{transport_id}' not found.")
                return False

    async def cleanup_expired_sessions(self, run_once: bool = False) -> None:
        """
        Clean up expired sessions periodically or once when explicitly called.

        Args:
            run_once: If True, run cleanup once and return. Used for testing.
        """
        while True:
            if not run_once:
                await asyncio.sleep(60)  # Check every minute

            current_time = datetime.now()
            expired_sessions = [
                transport_id
                for transport_id, session in self.sessions.items()
                if (current_time - session.last_accessed_time).total_seconds() > self.session_timeout
            ]

            async with self.lock:
                for transport_id in expired_sessions:
                    del self.sessions[transport_id]
                    logger.info(f"Expired session for transport '{transport_id}' removed.")

            if run_once:
                break

    async def get_all_sessions(self) -> Dict[str, Session]:
        async with self.lock:
            return self.sessions.copy()

    async def set_permissions(self, transport_id: str, permissions: Set[Permissions]) -> Optional[Session]:
        async with self.lock:
            if transport_id in self.sessions:
                session = self.sessions[transport_id]
                session.permissions = permissions
                return session
            else:
                logger.warning(f"Session for transport '{transport_id}' not found.")
                return None
