import asyncio
from datetime import datetime
from typing import Any, Dict, Optional, Set

from aider_mcp_server.atoms.logging import get_logger
from aider_mcp_server.security import Permissions

# Module-level logger can be kept for module-level concerns if any,
# but SessionManager will use its own instance.
# import logging
# logger = logging.getLogger(__name__)


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
        self.logger = get_logger(__name__)
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
                self.logger.verbose(
                    f"Accessed existing session for transport '{transport_id}'. Last accessed: {session.last_accessed_time}"
                )
                return session
            else:
                new_session = Session(transport_id)
                self.sessions[transport_id] = new_session
                self.logger.verbose(
                    f"Created new session for transport '{transport_id}'. Creation time: {new_session.creation_time}"
                )
                return new_session

    async def update_session(self, transport_id: str, data: Dict[str, Any]) -> Optional[Session]:
        async with self.lock:
            if transport_id in self.sessions:
                session = self.sessions[transport_id]
                session.last_accessed_time = datetime.now()
                session.custom_data.update(data)
                self.logger.verbose(
                    f"Updated session for transport '{transport_id}'. Last accessed: {session.last_accessed_time}, Data updated: {list(data.keys())}"
                )
                return session
            else:
                self.logger.warning(f"Session for transport '{transport_id}' not found during update.")
                return None

    async def remove_session(self, transport_id: str) -> None:
        async with self.lock:
            if transport_id in self.sessions:
                del self.sessions[transport_id]
                self.logger.verbose(f"Removed session for transport '{transport_id}'.")
            else:
                self.logger.warning(f"Session for transport '{transport_id}' not found during removal.")

    async def check_permission(self, transport_id: str, required_permission: Permissions) -> bool:
        async with self.lock:
            if transport_id in self.sessions:
                session = self.sessions[transport_id]
                session.last_accessed_time = datetime.now()
                has_perm = required_permission in session.permissions
                self.logger.verbose(
                    f"Permission check for transport '{transport_id}': Required '{required_permission.value}', Has: {has_perm}. Last accessed: {session.last_accessed_time}"
                )
                return has_perm
            else:
                self.logger.warning(f"Session for transport '{transport_id}' not found during permission check.")
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
            self.logger.verbose("Running cleanup for expired sessions.")

            current_time = datetime.now()
            # Need to acquire lock to iterate over self.sessions items safely if it can be modified elsewhere,
            # or copy items first. Let's copy keys to avoid issues if lock is held for too long.
            # However, the current logic iterates then acquires lock to delete, which is fine.
            
            # To be safe, let's get a snapshot of sessions or keys under lock if we were to do complex things before the delete lock
            # For now, the existing logic is okay as it re-checks under the lock implicitly by iterating `expired_sessions`
            # which was derived from `self.sessions.items()` before the lock for deletion.

            expired_session_ids = []
            # Iterate over a copy of items for safety if sessions could change during iteration
            # For this specific loop, it's okay as it's just building a list of IDs
            # The critical part is deleting under the lock.
            for transport_id, session in list(self.sessions.items()): # list() for a copy
                if (current_time - session.last_accessed_time).total_seconds() > self.session_timeout:
                    expired_session_ids.append(transport_id)
            
            if expired_session_ids:
                self.logger.verbose(f"Found {len(expired_session_ids)} expired sessions: {expired_session_ids}")
                async with self.lock:
                    for transport_id in expired_session_ids:
                        # Ensure session still exists before deleting, in case it was removed by another task
                        if transport_id in self.sessions:
                            del self.sessions[transport_id]
                            self.logger.info(f"Expired session for transport '{transport_id}' removed.") # Keep info for actual removal
                        else:
                            self.logger.verbose(f"Session '{transport_id}' already removed before cleanup.")
            else:
                self.logger.verbose("No expired sessions found during cleanup.")

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
                session.last_accessed_time = datetime.now() # Also update last_accessed_time
                perm_values = {p.value for p in permissions}
                self.logger.verbose(
                    f"Set permissions for transport '{transport_id}' to {perm_values}. Last accessed: {session.last_accessed_time}"
                )
                return session
            else:
                self.logger.warning(f"Session for transport '{transport_id}' not found when setting permissions.")
                return None
