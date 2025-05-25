"""
Tests for SSE connection establishment and termination.

These tests verify that the SSE Transport Adapter properly handles
the lifecycle of connections, including initialization, listening,
and graceful shutdown.
"""

import asyncio
import socket
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import uvicorn
from starlette.applications import Starlette

from aider_mcp_server.organisms.transports.sse.sse_transport_adapter import SSETransportAdapter
from aider_mcp_server.templates.servers.sse_server import run_sse_server


@pytest.fixture
def free_port():
    """Get an available port for testing."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


@pytest.mark.asyncio
async def test_adapter_initialization():
    """Test that the adapter initializes properly."""
    # Create a mock coordinator
    mock_coordinator = AsyncMock()
    mock_coordinator.register_transport_adapter = AsyncMock()

    # Create the adapter with the mock coordinator
    adapter = SSETransportAdapter(coordinator=mock_coordinator)

    # Initialize the adapter
    await adapter.initialize()

    # Verify that the adapter's properties were initialized
    assert adapter._app is not None
    assert isinstance(adapter._app, Starlette)

    # Verify that the coordinator was called
    mock_coordinator.register_transport_adapter.assert_called_once_with(adapter)


@pytest.mark.asyncio
async def test_adapter_start_listening(free_port):
    """Test that the adapter starts listening on the specified port."""
    # Create a mock coordinator
    mock_coordinator = AsyncMock()

    # Create the adapter with the mock coordinator and the free port
    adapter = SSETransportAdapter(coordinator=mock_coordinator, host="127.0.0.1", port=free_port)

    # Initialize the adapter
    await adapter.initialize()

    # Mock uvicorn.Server.serve to avoid actually starting a server
    with patch("uvicorn.Server.serve", new_callable=AsyncMock) as mock_serve:
        # Start listening
        await adapter.start_listening()

        # Verify that a server instance was created
        assert adapter._server_instance is not None
        assert isinstance(adapter._server_instance, uvicorn.Server)

        # Verify that the server was configured with the correct host and port
        assert adapter._server_instance.config.host == "127.0.0.1"
        assert adapter._server_instance.config.port == free_port

        # Verify that the server.serve method was called
        assert adapter._server_task is not None
        mock_serve.assert_called_once()


@pytest.mark.asyncio
async def test_adapter_shutdown():
    """Test that the adapter shuts down properly."""
    # Create a mock coordinator
    mock_coordinator = AsyncMock()

    # Create the adapter with the mock coordinator
    adapter = SSETransportAdapter(coordinator=mock_coordinator)

    # Initialize the adapter
    await adapter.initialize()

    # Set up a mock server instance and task
    mock_server = MagicMock()
    mock_task = AsyncMock()
    adapter._server_instance = mock_server
    adapter._server_task = mock_task

    # Add a test connection to verify it gets cleaned up
    test_queue = asyncio.Queue()
    adapter._active_connections = {"test-connection": test_queue}

    # Mock asyncio.wait_for to avoid waiting
    with (
        patch("asyncio.wait_for", new_callable=AsyncMock) as mock_wait_for,
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        # Shut down the adapter
        await adapter.shutdown()

        # Verify that the active connections were cleared
        assert adapter._active_connections == {}

        # Verify that the server instance was shut down
        assert adapter._server_instance is None
        assert adapter._server_task is None

        # Verify that wait_for was called with the server task
        mock_wait_for.assert_called_once_with(mock_task, timeout=5.0)

        # Verify that the server's should_exit flag was set
        assert mock_server.should_exit is True


@pytest.mark.asyncio
async def test_adapter_shutdown_with_no_server():
    """Test that the adapter handles shutdown when no server is running."""
    # Create a mock coordinator
    mock_coordinator = AsyncMock()

    # Create the adapter with the mock coordinator
    adapter = SSETransportAdapter(coordinator=mock_coordinator)

    # Initialize the adapter
    await adapter.initialize()

    # Ensure there's no server instance
    adapter._server_instance = None
    adapter._server_task = None

    # Add a test connection to verify it gets cleaned up
    test_queue = asyncio.Queue()
    adapter._active_connections = {"test-connection": test_queue}

    # Mock asyncio.sleep to avoid waiting
    with patch("asyncio.sleep", new_callable=AsyncMock):
        # Shut down the adapter
        await adapter.shutdown()

        # Verify that the active connections were cleared
        assert adapter._active_connections == {}

        # Verify that no errors occurred (since there was no server to shut down)


@pytest.mark.asyncio
async def test_adapter_shutdown_with_server_timeout():
    """Test that the adapter handles server shutdown timeouts gracefully."""
    # Create a mock coordinator
    mock_coordinator = AsyncMock()

    # Create the adapter with the mock coordinator
    adapter = SSETransportAdapter(coordinator=mock_coordinator)

    # Initialize the adapter
    await adapter.initialize()

    # Set up a mock server instance and task
    mock_server = MagicMock()
    mock_task = AsyncMock()
    adapter._server_instance = mock_server
    adapter._server_task = mock_task

    # Mock asyncio.wait_for to simulate a timeout
    with (
        patch("asyncio.wait_for", side_effect=asyncio.TimeoutError),
        patch("asyncio.sleep", new_callable=AsyncMock),
        patch.object(adapter, "logger") as mock_logger,
    ):
        # Shut down the adapter
        await adapter.shutdown()

        # Verify that a warning was logged
        mock_logger.warning.assert_called_with("Server shutdown timed out")

        # Verify that the server resources were still cleaned up
        assert adapter._server_instance is None
        assert adapter._server_task is None


@pytest.mark.asyncio
async def test_adapter_shutdown_with_task_cancellation():
    """Test that the adapter handles server task cancellation during shutdown."""
    # Create a mock coordinator
    mock_coordinator = AsyncMock()

    # Create the adapter with the mock coordinator
    adapter = SSETransportAdapter(coordinator=mock_coordinator)

    # Initialize the adapter
    await adapter.initialize()

    # Set up a mock server instance and task
    mock_server = MagicMock()
    mock_task = AsyncMock()
    adapter._server_instance = mock_server
    adapter._server_task = mock_task

    # Mock asyncio.wait_for to simulate a CancelledError
    with (
        patch("asyncio.wait_for", side_effect=asyncio.CancelledError()),
        patch("asyncio.sleep", new_callable=AsyncMock),
        patch.object(adapter, "logger") as mock_logger,
    ):
        # Shut down the adapter
        await adapter.shutdown()

        # Verify that no error was logged (cancellation is expected)
        mock_logger.error.assert_not_called()

        # Verify that the server resources were still cleaned up
        assert adapter._server_instance is None
        assert adapter._server_task is None


@pytest.mark.asyncio
async def test_run_sse_server_initialization_and_shutdown():
    """Test that run_sse_server initializes and shuts down properly."""
    # Create a mock coordinator with async context manager support
    mock_coordinator = AsyncMock()
    mock_coordinator.__aenter__ = AsyncMock(return_value=mock_coordinator)
    mock_coordinator.__aexit__ = AsyncMock(return_value=None)

    # Mock SSETransportAdapter
    mock_adapter = AsyncMock()
    mock_adapter.initialize = AsyncMock()
    mock_adapter.start_listening = AsyncMock()
    mock_adapter.shutdown = AsyncMock()
    mock_adapter._server_task = None  # Ensure _wait_for_shutdown doesn't try to gather an AsyncMock

    # Set up for signal handling
    shutdown_event = asyncio.Event()
    # Set the event so the server exits immediately
    shutdown_event.set()

    # Path ApplicationCoordinator.getInstance and SSETransportAdapter constructor
    with (
        patch(
            "aider_mcp_server.templates.servers.sse_server.ApplicationCoordinator.getInstance", new_callable=AsyncMock
        ) as mock_get_instance,
        patch("aider_mcp_server.templates.servers.sse_server.SSETransportAdapter", return_value=mock_adapter),
        patch("asyncio.Event", return_value=shutdown_event),  # Standard asyncio.Event
        patch("aider_mcp_server.templates.servers.sse_server.is_git_repository", return_value=(True, "")),
        patch("asyncio.get_event_loop") as mock_get_loop,  # Standard asyncio.get_event_loop
    ):
        # Set up mock loop and signal handlers
        mock_loop = MagicMock()
        mock_loop.add_signal_handler = MagicMock()
        mock_get_loop.return_value = mock_loop

        # Mock getInstance to return the coordinator directly
        mock_get_instance.return_value = mock_coordinator

        # Call run_sse_server
        await run_sse_server(host="127.0.0.1", port=8765, current_working_dir="/mock/git/repo")

        # Verify that signal handlers were added for graceful shutdown
        assert mock_loop.add_signal_handler.call_count >= 2  # At least SIGTERM and SIGINT

        # Verify adapter methods were called in the correct order
        mock_adapter.initialize.assert_called_once()
        mock_adapter.start_listening.assert_called_once()
        mock_adapter.shutdown.assert_called_once()


@pytest.mark.asyncio
async def test_adapter_close_active_connections():
    """Test that the adapter closes active connections during shutdown."""
    # Create a mock coordinator
    mock_coordinator = AsyncMock()

    # Create the adapter with the mock coordinator
    adapter = SSETransportAdapter(coordinator=mock_coordinator)

    # Initialize the adapter
    await adapter.initialize()

    # Set up multiple test connections
    test_queues = {"connection1": asyncio.Queue(), "connection2": asyncio.Queue(), "connection3": asyncio.Queue()}
    adapter._active_connections = test_queues

    # Mock asyncio.sleep to avoid waiting
    with patch("asyncio.sleep", new_callable=AsyncMock):
        # Shut down the adapter
        await adapter.shutdown()

        # Verify that CLOSE_CONNECTION messages were put in each queue
        for queue_name, queue in test_queues.items():
            try:
                message = queue.get_nowait()
                assert message == "CLOSE_CONNECTION"
            except asyncio.QueueEmpty:
                pytest.fail(f"Queue {queue_name} did not receive a CLOSE_CONNECTION message")

        # Verify that the active connections were cleared
        assert adapter._active_connections == {}


@pytest.mark.asyncio
async def test_adapter_handle_queue_full_during_shutdown():
    """Test that the adapter handles full queues during shutdown gracefully."""
    # Create a mock coordinator
    mock_coordinator = AsyncMock()

    # Create the adapter with the mock coordinator
    adapter = SSETransportAdapter(coordinator=mock_coordinator)

    # Initialize the adapter
    await adapter.initialize()

    # Create a queue that's already full
    full_queue = asyncio.Queue(maxsize=1)
    await full_queue.put("existing-message")  # Queue is now full

    # Mock the put method to prevent blocking
    full_queue.put = AsyncMock()

    # Set up test connections including the full queue
    adapter._active_connections = {"normal-connection": asyncio.Queue(), "full-connection": full_queue}

    # Mock asyncio.sleep to avoid waiting
    with patch("asyncio.sleep", new_callable=AsyncMock), patch.object(adapter, "logger") as mock_logger:
        # Shut down the adapter
        await adapter.shutdown()

        # Verify that the active connections were cleared despite the error
        assert adapter._active_connections == {}

        # Verify that debug logging occurred during shutdown
        mock_logger.debug.assert_called()


@pytest.mark.asyncio
async def test_create_app_multiple_times():
    """Test that calling _create_app multiple times doesn't cause issues."""
    # Create a mock coordinator
    mock_coordinator = AsyncMock()

    # Create the adapter with the mock coordinator
    adapter = SSETransportAdapter(coordinator=mock_coordinator)

    # Call _create_app multiple times
    await adapter._create_app()
    first_app = adapter._app

    await adapter._create_app()
    second_app = adapter._app

    # Verify that each call created a valid app instance
    assert first_app is not None
    assert second_app is not None

    # Verify that the second call replaced the first app
    assert first_app is not second_app
