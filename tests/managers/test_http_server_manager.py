"""
Tests for HttpServerManager class.

This module contains comprehensive unit tests for the HttpServerManager
that orchestrates multiple HTTP server instances for multi-client support.
"""

import asyncio
from datetime import datetime
from unittest.mock import patch

import pytest
import pytest_asyncio

from aider_mcp_server.atoms.types.data_types import ClientRequest
from aider_mcp_server.managers.http_server_manager import HttpServerManager


class TestHttpServerManager:
    """Test suite for HttpServerManager class."""

    @pytest_asyncio.fixture
    async def manager(self):
        """Create a fresh HttpServerManager instance for each test."""
        return HttpServerManager()

    @pytest.fixture
    def client_request(self):
        """Create a sample client request."""
        return ClientRequest(
            client_id="test_client_1", workspace_id="workspace_1", request_data={"test_key": "test_value"}
        )

    @pytest.mark.asyncio
    async def test_create_client_session_success(self, manager, client_request):
        """Test successful client session creation."""
        session_info = await manager.create_client_session(client_request)

        assert session_info.client_id == "test_client_1"
        assert session_info.workspace_id == "workspace_1"
        assert session_info.status == "active"
        assert session_info.session_id.startswith("session_")
        assert "test_key" in session_info.metadata
        assert session_info.metadata["test_key"] == "test_value"

        # Verify internal state
        assert await manager.get_session_count() == 1
        assert await manager.get_server_count() == 1

    @pytest.mark.asyncio
    async def test_create_client_session_duplicate_client(self, manager, client_request):
        """Test that creating a session for an existing client raises an error."""
        # Create first session
        await manager.create_client_session(client_request)

        # Attempt to create second session for same client
        with pytest.raises(ValueError, match="already has an active session"):
            await manager.create_client_session(client_request)

    @pytest.mark.asyncio
    async def test_create_client_session_max_clients_exceeded(self, manager):
        """Test that exceeding max concurrent clients raises an error."""
        # Mock the MAX_CONCURRENT_CLIENTS to a small number for testing
        with patch("aider_mcp_server.managers.http_server_manager.MAX_CONCURRENT_CLIENTS", 2):
            # Create max number of sessions
            for i in range(2):
                request = ClientRequest(client_id=f"client_{i}")
                await manager.create_client_session(request)

            # Attempt to create one more session
            excess_request = ClientRequest(client_id="excess_client")
            with pytest.raises(RuntimeError, match="Maximum concurrent clients"):
                await manager.create_client_session(excess_request)

    @pytest.mark.asyncio
    async def test_destroy_client_session_success(self, manager, client_request):
        """Test successful client session destruction."""
        # Create session
        await manager.create_client_session(client_request)
        assert await manager.get_session_count() == 1

        # Destroy session
        await manager.destroy_client_session("test_client_1")

        # Verify session is removed
        assert await manager.get_session_count() == 0
        assert await manager.get_server_count() == 0

    @pytest.mark.asyncio
    async def test_destroy_client_session_nonexistent(self, manager):
        """Test destroying a non-existent client session raises an error."""
        with pytest.raises(ValueError, match="No active session found"):
            await manager.destroy_client_session("nonexistent_client")

    @pytest.mark.asyncio
    async def test_get_client_server_info_success(self, manager, client_request):
        """Test getting server info for an existing client."""
        # Create session
        await manager.create_client_session(client_request)

        # Get server info
        server_info = await manager.get_client_server_info("test_client_1")

        assert server_info is not None
        assert server_info.server_id.startswith("server_")
        assert server_info.host == "127.0.0.1"
        assert server_info.status == "starting"
        assert server_info.workspace_id == "workspace_1"
        assert server_info.active_clients == 1

    @pytest.mark.asyncio
    async def test_get_client_server_info_nonexistent(self, manager):
        """Test getting server info for non-existent client returns None."""
        server_info = await manager.get_client_server_info("nonexistent_client")
        assert server_info is None

    @pytest.mark.asyncio
    async def test_list_active_sessions_empty(self, manager):
        """Test listing active sessions when none exist."""
        sessions = await manager.list_active_sessions()
        assert sessions == []

    @pytest.mark.asyncio
    async def test_list_active_sessions_multiple(self, manager):
        """Test listing multiple active sessions."""
        # Create multiple sessions
        for i in range(3):
            request = ClientRequest(client_id=f"client_{i}", workspace_id=f"workspace_{i}")
            await manager.create_client_session(request)

        # List sessions
        sessions = await manager.list_active_sessions()

        assert len(sessions) == 3
        client_ids = {session.client_id for session in sessions}
        assert client_ids == {"client_0", "client_1", "client_2"}

    @pytest.mark.asyncio
    async def test_update_session_activity(self, manager, client_request):
        """Test updating session activity."""
        # Create session
        session_info = await manager.create_client_session(client_request)
        original_activity = session_info.last_activity

        # Wait a small amount to ensure time difference
        await asyncio.sleep(0.01)

        # Update activity
        await manager.update_session_activity("test_client_1")

        # Get updated session
        sessions = await manager.list_active_sessions()
        updated_session = sessions[0]

        assert updated_session.last_activity > original_activity
        assert updated_session.status == "active"

    @pytest.mark.asyncio
    async def test_update_session_activity_nonexistent(self, manager):
        """Test updating activity for non-existent session does not raise error."""
        # Should not raise an exception
        await manager.update_session_activity("nonexistent_client")

    @pytest.mark.asyncio
    async def test_cleanup_expired_sessions(self, manager):
        """Test cleanup of expired sessions."""
        # Mock a very short timeout for testing
        with patch("aider_mcp_server.managers.http_server_manager.CLIENT_SESSION_TIMEOUT", 0.1):
            # Create session
            request = ClientRequest(client_id="test_client")
            await manager.create_client_session(request)

            # Verify session exists
            assert await manager.get_session_count() == 1

            # Wait for session to expire
            await asyncio.sleep(0.2)

            # Cleanup expired sessions
            expired_clients = await manager.cleanup_expired_sessions()

            assert "test_client" in expired_clients
            assert await manager.get_session_count() == 0

    @pytest.mark.asyncio
    async def test_session_timeout_status_update(self, manager):
        """Test that sessions are marked as idle when they timeout."""
        # Mock a very short timeout for testing
        with patch("aider_mcp_server.managers.http_server_manager.CLIENT_SESSION_TIMEOUT", 0.1):
            # Create session
            request = ClientRequest(client_id="test_client")
            await manager.create_client_session(request)

            # Wait for session to timeout
            await asyncio.sleep(0.2)

            # List sessions (this should update status)
            sessions = await manager.list_active_sessions()

            assert len(sessions) == 1
            assert sessions[0].status == "idle"

    @pytest.mark.asyncio
    async def test_concurrent_session_creation(self, manager):
        """Test concurrent session creation for thread safety."""

        async def create_session(client_id):
            request = ClientRequest(client_id=client_id)
            return await manager.create_client_session(request)

        # Create multiple sessions concurrently
        tasks = [create_session(f"client_{i}") for i in range(5)]
        sessions = await asyncio.gather(*tasks)

        # Verify all sessions were created
        assert len(sessions) == 5
        assert await manager.get_session_count() == 5

        # Verify all client IDs are unique
        client_ids = {session.client_id for session in sessions}
        assert len(client_ids) == 5

    @pytest.mark.asyncio
    async def test_concurrent_session_destruction(self, manager):
        """Test concurrent session destruction for thread safety."""
        # Create multiple sessions
        client_ids = []
        for i in range(5):
            request = ClientRequest(client_id=f"client_{i}")
            await manager.create_client_session(request)
            client_ids.append(f"client_{i}")

        # Destroy sessions concurrently
        destroy_tasks = [manager.destroy_client_session(client_id) for client_id in client_ids]
        await asyncio.gather(*destroy_tasks)

        # Verify all sessions were destroyed
        assert await manager.get_session_count() == 0

    @pytest.mark.asyncio
    async def test_shutdown(self, manager):
        """Test manager shutdown cleanup."""
        # Create multiple sessions
        for i in range(3):
            request = ClientRequest(client_id=f"client_{i}")
            await manager.create_client_session(request)

        assert await manager.get_session_count() == 3

        # Shutdown manager
        await manager.shutdown()

        # Verify all sessions and servers are cleaned up
        assert await manager.get_session_count() == 0
        assert await manager.get_server_count() == 0

    @pytest.mark.asyncio
    async def test_session_info_fields(self, manager, client_request):
        """Test that SessionInfo contains all expected fields."""
        session_info = await manager.create_client_session(client_request)

        # Check all required fields are present
        assert hasattr(session_info, "session_id")
        assert hasattr(session_info, "client_id")
        assert hasattr(session_info, "workspace_id")
        assert hasattr(session_info, "created_at")
        assert hasattr(session_info, "last_activity")
        assert hasattr(session_info, "status")
        assert hasattr(session_info, "metadata")

        # Check field types
        assert isinstance(session_info.created_at, datetime)
        assert isinstance(session_info.last_activity, datetime)
        assert isinstance(session_info.metadata, dict)

    @pytest.mark.asyncio
    async def test_server_info_fields(self, manager, client_request):
        """Test that ServerInfo contains all expected fields."""
        await manager.create_client_session(client_request)
        server_info = await manager.get_client_server_info("test_client_1")

        # Check all required fields are present
        assert hasattr(server_info, "server_id")
        assert hasattr(server_info, "host")
        assert hasattr(server_info, "port")
        assert hasattr(server_info, "actual_port")
        assert hasattr(server_info, "status")
        assert hasattr(server_info, "workspace_id")
        assert hasattr(server_info, "created_at")
        assert hasattr(server_info, "active_clients")
        assert hasattr(server_info, "transport_adapter_id")

        # Check field types and values
        assert isinstance(server_info.created_at, datetime)
        assert isinstance(server_info.active_clients, int)
        assert server_info.active_clients > 0
