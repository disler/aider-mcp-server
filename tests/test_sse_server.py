from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aider_mcp_server.sse_server import serve_sse


# Test function
@pytest.mark.asyncio
async def test_serve_sse_startup_and_run():
    # Create patch context for all mocks
    with (
        patch("aider_mcp_server.sse_server.is_git_repository", return_value=(True, None)),
        patch("aider_mcp_server.sse_server.asyncio.Event") as mock_event_cls,
        patch("aider_mcp_server.sse_server.SSETransportAdapter") as mock_adapter_cls,
        patch("aider_mcp_server.sse_server.ApplicationCoordinator") as mock_coordinator_cls,
    ):
        # Set up mock event
        mock_event = MagicMock()
        mock_event.wait = AsyncMock()
        mock_event.set = MagicMock()
        mock_event.is_set = MagicMock(return_value=True)  # Make it return immediately
        mock_event_cls.return_value = mock_event

        # Set up mock adapter
        mock_adapter = MagicMock()
        mock_adapter.transport_id = "sse_test_id"
        mock_adapter.initialize = AsyncMock()
        mock_adapter_cls.return_value = mock_adapter

        # Set up mock coordinator
        mock_coordinator = MagicMock()
        mock_coordinator.register_handler = AsyncMock()
        mock_coordinator.register_transport = AsyncMock()
        mock_coordinator.__aenter__ = AsyncMock(return_value=mock_coordinator)
        mock_coordinator.__aexit__ = AsyncMock()
        mock_coordinator_cls.getInstance = AsyncMock(return_value=mock_coordinator)

        # Call the function under test
        host, port, editor_model, cwd, heartbeat = (
            "127.0.0.1",
            8888,
            "test_model",
            "/test/repo",
            20.0,
        )
        await serve_sse(host, port, editor_model, cwd, heartbeat_interval=heartbeat)

        # Verify correct methods were called
        mock_coordinator_cls.getInstance.assert_awaited_once()
        mock_coordinator.__aenter__.assert_awaited_once()
        mock_coordinator.__aexit__.assert_awaited_once()

        # Verify adapter was initialized and registered
        mock_adapter.initialize.assert_awaited_once()
        mock_coordinator.register_transport.assert_awaited_once()

        # Verify that handlers are no longer directly registered with the coordinator
        # as they are now registered through FastMCP in the SSE transport adapter
        assert mock_coordinator.register_handler.await_count == 0
