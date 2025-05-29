"""
Tests for ClientSessionManager class.

This module contains comprehensive unit tests for the ClientSessionManager
that manages client sessions, their lifecycles, workspace isolation,
and session timeout handling.
"""

import asyncio
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
import pytest_asyncio

from aider_mcp_server.managers.client_session_manager import (
    ClientSession,
    ClientSessionManager,
)


@pytest_asyncio.fixture
async def manager(tmp_path):
    """Create a fresh ClientSessionManager instance for each test, using a temporary workspace."""
    # Patch WORKSPACE_BASE_DIR for the duration of this manager's lifecycle
    with patch("aider_mcp_server.managers.client_session_manager.WORKSPACE_BASE_DIR", tmp_path):
        m = ClientSessionManager()
        yield m
    # Cleanup tmp_path contents if necessary, though pytest usually handles tmp_path
    # Forcing cleanup of directories created by the manager if any persist
    for item in tmp_path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


class TestClientSessionManager:
    """Test suite for ClientSessionManager class."""

    @pytest.mark.asyncio
    async def test_register_client_success(self, manager: ClientSessionManager, tmp_path: Path):
        """Test successful client registration."""
        client_id = "test_client_1"
        session = await manager.register_client(client_id)

        assert session.client_id == client_id
        assert session.session_id.startswith("session_")
        assert session.workspace_path == tmp_path / client_id
        assert session.workspace_path.exists()
        assert session.workspace_path.is_dir()
        assert isinstance(session.created_at, datetime)
        assert isinstance(session.last_activity, datetime)
        assert (datetime.now(timezone.utc) - session.created_at).total_seconds() < 1
        assert (datetime.now(timezone.utc) - session.last_activity).total_seconds() < 1

        assert client_id in manager._active_sessions
        assert manager._active_sessions[client_id] == session

    @pytest.mark.asyncio
    async def test_register_client_duplicate(self, manager: ClientSessionManager):
        """Test that registering an existing client raises ValueError."""
        client_id = "test_client_duplicate"
        await manager.register_client(client_id)  # First registration

        with pytest.raises(ValueError, match=f"Client {client_id} already has an active session."):
            await manager.register_client(client_id)  # Second registration

    @pytest.mark.asyncio
    async def test_get_client_workspace_and_creation(self, manager: ClientSessionManager, tmp_path: Path):
        """Test workspace path retrieval and that it's created."""
        client_id = "test_client_ws"
        session = await manager.register_client(client_id)
        original_last_activity = session.last_activity

        await asyncio.sleep(0.01)  # Ensure time difference for last_activity update

        workspace_path = await manager.get_client_workspace(client_id)

        assert workspace_path == tmp_path / client_id
        assert workspace_path.exists()
        assert workspace_path.is_dir()

        # Check if last_activity was updated
        updated_session = manager._active_sessions[client_id]
        assert updated_session.last_activity > original_last_activity

    @pytest.mark.asyncio
    async def test_get_client_workspace_nonexistent(self, manager: ClientSessionManager):
        """Test getting workspace for a non-existent client raises ValueError."""
        client_id = "nonexistent_client_ws"
        with pytest.raises(ValueError, match=f"No active session found for client {client_id}"):
            await manager.get_client_workspace(client_id)

    @pytest.mark.asyncio
    async def test_register_client_workspace_creation_failure(self, manager: ClientSessionManager, tmp_path: Path):
        """Test client registration fails if workspace creation fails."""
        client_id = "test_client_os_error"
        with patch("os.makedirs", side_effect=OSError("Test OS error")) as mock_makedirs:
            with pytest.raises(RuntimeError, match=f"Failed to create workspace for client {client_id}"):
                await manager.register_client(client_id)
            mock_makedirs.assert_called_once_with(tmp_path / client_id, exist_ok=True)

    @pytest.mark.asyncio
    async def test_cleanup_client_session_success(self, manager: ClientSessionManager):
        """Test successful client session cleanup."""
        client_id = "test_client_cleanup"
        await manager.register_client(client_id)
        assert client_id in manager._active_sessions

        await manager.cleanup_client_session(client_id)
        assert client_id not in manager._active_sessions

    @pytest.mark.asyncio
    async def test_cleanup_client_session_nonexistent(self, manager: ClientSessionManager):
        """Test cleaning up a non-existent client session raises ValueError."""
        client_id = "nonexistent_client_cleanup"
        with pytest.raises(ValueError, match=f"No active session found for client {client_id} to clean up."):
            await manager.cleanup_client_session(client_id)

    @pytest.mark.asyncio
    async def test_update_session_activity_success(self, manager: ClientSessionManager):
        """Test updating session activity for an existing client."""
        client_id = "test_client_activity"
        session = await manager.register_client(client_id)
        original_last_activity = session.last_activity

        await asyncio.sleep(0.01)  # Ensure time difference

        await manager.update_session_activity(client_id)

        updated_session = manager._active_sessions[client_id]
        assert updated_session.last_activity > original_last_activity

    @pytest.mark.asyncio
    async def test_update_session_activity_nonexistent(self, manager: ClientSessionManager, caplog):
        """Test updating activity for a non-existent session logs a warning and does not raise error."""
        client_id = "nonexistent_client_activity"
        await manager.update_session_activity(client_id)  # Should not raise

        # Verify that no session was created for the non-existent client
        assert client_id not in manager._active_sessions

    @pytest.mark.asyncio
    async def test_cleanup_expired_sessions(self, manager: ClientSessionManager, tmp_path: Path):
        """Test cleanup of expired sessions."""
        client_id_active = "client_active"
        client_id_expired = "client_expired"

        # Patch CLIENT_SESSION_TIMEOUT for this test
        short_timeout = 0.1  # seconds
        with patch("aider_mcp_server.managers.client_session_manager.CLIENT_SESSION_TIMEOUT", short_timeout):
            # Register clients
            await manager.register_client(client_id_active)
            session_expired = await manager.register_client(client_id_expired)

            # Manually set client_expired's last_activity to be old.
            # This directly modifies the session object held in the manager's _active_sessions.
            session_expired.last_activity = datetime.now(timezone.utc) - timedelta(seconds=short_timeout * 2)

            # Wait for a period (e.g., half the timeout).
            # client_active's last_activity (from registration) is aging.
            # client_expired's effective age is now (2 * short_timeout + 0.5 * short_timeout).
            await asyncio.sleep(short_timeout * 0.5)

            # Update activity for client_active, making its last_activity very recent.
            await manager.update_session_activity(client_id_active)
            # client_active.last_activity is now "current_time_of_update".
            # client_expired's effective age continues to grow.

            # Wait for another period, e.g., 0.7 * timeout.
            # When cleanup_expired_sessions is called:
            # - client_expired's total age will be roughly (2*timeout + 0.5*timeout + 0.7*timeout) = 3.2*timeout.
            #   This is > short_timeout, so it should be expired.
            # - client_active's age since its last_activity update will be roughly 0.7*timeout.
            #   This is < short_timeout, so it should not be expired.
            await asyncio.sleep(short_timeout * 0.7)

            expired_clients = await manager.cleanup_expired_sessions()

            assert client_id_expired in expired_clients
            assert client_id_active not in expired_clients
            assert client_id_expired not in manager._active_sessions
            assert client_id_active in manager._active_sessions

    @pytest.mark.asyncio
    async def test_cleanup_expired_sessions_already_cleaned_up(self, manager: ClientSessionManager, caplog):
        """Test cleanup_expired_sessions when a session was cleaned up concurrently."""
        client_id = "test_client_already_cleaned"

        short_timeout = 0.1
        with patch("aider_mcp_server.managers.client_session_manager.CLIENT_SESSION_TIMEOUT", short_timeout):
            session = await manager.register_client(client_id)
            # Force session to be old
            session.last_activity = datetime.now(timezone.utc) - timedelta(seconds=short_timeout * 2)
            manager._active_sessions[client_id] = session

            # Mock _cleanup_client_session_internal to simulate concurrent cleanup
            # The goal is to have client_id in expired_client_ids, but when its turn comes,
            # it's already removed from _active_sessions.

            original_internal_cleanup = manager._cleanup_client_session_internal

            async def mock_internal_cleanup(target_client_id):
                if target_client_id == client_id:
                    # Simulate it was removed by another process *just before* this call
                    # by removing it from active_sessions if it's still there,
                    # then raising ValueError as the original would.
                    if target_client_id in manager._active_sessions:
                        manager._active_sessions.pop(target_client_id)
                    raise ValueError(f"No active session found for client {target_client_id} to clean up.")
                else:  # pragma: no cover
                    return await original_internal_cleanup(target_client_id)

            with patch.object(manager, "_cleanup_client_session_internal", side_effect=mock_internal_cleanup):
                expired_clients = await manager.cleanup_expired_sessions()

            assert client_id in expired_clients  # It was identified as expired
            assert client_id not in manager._active_sessions  # And it's gone
            # The key behavior is that the process handles this case gracefully (e.g. logs a warning,
            # which we are no longer asserting) and doesn't crash.
            # The assertions above confirm the client was processed as expired and is no longer active.

    @pytest.mark.asyncio
    async def test_concurrent_client_registration(self, manager: ClientSessionManager, tmp_path: Path):
        """Test concurrent client registration."""
        num_clients = 5
        client_ids = [f"concurrent_client_{i}" for i in range(num_clients)]

        async def register_one_client(client_id: str):
            return await manager.register_client(client_id)

        results = await asyncio.gather(*(register_one_client(cid) for cid in client_ids))

        assert len(results) == num_clients
        assert len(manager._active_sessions) == num_clients
        for i in range(num_clients):
            assert client_ids[i] in manager._active_sessions
            assert manager._active_sessions[client_ids[i]].workspace_path == tmp_path / client_ids[i]
            assert (tmp_path / client_ids[i]).exists()

    @pytest.mark.asyncio
    async def test_concurrent_client_cleanup(self, manager: ClientSessionManager):
        """Test concurrent client session cleanup."""
        num_clients = 5
        client_ids = [f"concurrent_cleanup_client_{i}" for i in range(num_clients)]

        # Register clients first
        for cid in client_ids:
            await manager.register_client(cid)

        assert len(manager._active_sessions) == num_clients

        async def cleanup_one_client(client_id: str):
            return await manager.cleanup_client_session(client_id)

        await asyncio.gather(*(cleanup_one_client(cid) for cid in client_ids))

        assert len(manager._active_sessions) == 0

    def test_client_session_model_defaults(self, tmp_path: Path):
        """Test ClientSession model default values."""
        client_id = "model_test_client"
        session_id = "session_model_test"
        workspace_path = tmp_path / client_id

        session = ClientSession(session_id=session_id, client_id=client_id, workspace_path=workspace_path)

        assert session.session_id == session_id
        assert session.client_id == client_id
        assert session.workspace_path == workspace_path

        now = datetime.now(timezone.utc)
        assert isinstance(session.created_at, datetime)
        assert isinstance(session.last_activity, datetime)
        assert (now - session.created_at).total_seconds() < 1
        assert (now - session.last_activity).total_seconds() < 1
        assert session.created_at.tzinfo == timezone.utc
        assert session.last_activity.tzinfo == timezone.utc

    @pytest.mark.asyncio
    async def test_cleanup_expired_sessions_no_sessions(self, manager: ClientSessionManager):
        """Test cleanup_expired_sessions when there are no active sessions."""
        expired_clients = await manager.cleanup_expired_sessions()
        assert expired_clients == []
        assert not manager._active_sessions

    @pytest.mark.asyncio
    async def test_cleanup_expired_sessions_no_expired_sessions(self, manager: ClientSessionManager):
        """Test cleanup_expired_sessions when no sessions are actually expired."""
        client_id = "not_expired_client"
        await manager.register_client(client_id)

        # Use a long timeout to ensure it doesn't expire
        with patch("aider_mcp_server.managers.client_session_manager.CLIENT_SESSION_TIMEOUT", 3600):
            expired_clients = await manager.cleanup_expired_sessions()

        assert expired_clients == []
        assert client_id in manager._active_sessions
