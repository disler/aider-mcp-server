"""
Tests for SSE error handling during transmission.

These tests verify that the SSE Transport Adapter properly handles
various error conditions that can occur during message transmission,
including network errors, serialization problems, and client disconnections.
"""

import asyncio
import json
import socket
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.exceptions import HTTPException

from aider_mcp_server.atoms.event_types import EventTypes
from aider_mcp_server.sse_server import run_sse_server
from aider_mcp_server.sse_transport_adapter import SSETransportAdapter


@pytest.fixture
def adapter():
    """Fixture providing a basic SSETransportAdapter instance."""
    adapter = SSETransportAdapter()
    return adapter


@pytest.mark.asyncio
async def test_send_event_connection_error(adapter):
    """Test handling connection errors when sending events."""
    # Set up a test connection with a queue that raises an exception
    mock_queue = MagicMock()
    mock_queue.put_nowait = MagicMock(side_effect=ConnectionError("Connection lost"))
    adapter._active_connections = {"test-connection": mock_queue}

    # Create a test event
    event_type = EventTypes.STATUS
    event_data = {"message": "Test message", "id": "test-id"}

    # Send the event
    with patch.object(adapter, "logger") as mock_logger:
        await adapter.send_event(event_type, event_data)

        # Verify that an error was logged
        mock_logger.error.assert_called_once()
        log_message = mock_logger.error.call_args[0][0]
        assert "Error putting event into queue" in log_message
        assert "test-connection" in log_message


@pytest.mark.asyncio
async def test_send_event_json_serialization_error(adapter):
    """Test handling JSON serialization errors when sending events."""
    # Set up a test connection
    test_queue = asyncio.Queue()
    adapter._active_connections = {"test-connection": test_queue}

    # Create a test event with an unserializable object
    event_type = EventTypes.STATUS
    event_data = {
        "message": "Test message",
        "id": "test-id",
        "unserializable": object(),  # This can't be converted to JSON
    }

    # Mock json.dumps to simulate a serialization error
    with (
        patch("json.dumps", side_effect=TypeError("Object not serializable")),
        patch.object(adapter, "logger") as mock_logger,
    ):
        # Send the event
        await adapter.send_event(event_type, event_data)

        # Verify that an error was logged
        mock_logger.error.assert_called_once()
        log_message = mock_logger.error.call_args[0][0]
        assert "JSON serialization error for event type status: Object not serializable" in log_message

        # Verify that the queue is still empty (no message was sent)
        assert test_queue.empty()


@pytest.mark.asyncio
async def test_handle_sse_request_client_disconnect():
    """Test handling client disconnection during SSE request processing."""
    # Create a mock coordinator
    mock_coordinator = AsyncMock()

    # Create the adapter with the mock coordinator
    adapter = SSETransportAdapter(coordinator=mock_coordinator)

    # Mock the MCP transport and server
    mock_mcp_transport = MagicMock()
    mock_mcp_server = MagicMock()

    # Create a context manager for connect_sse that simulates a client disconnect
    class DisconnectContextManager:
        async def __aenter__(self):
            return (MagicMock(), MagicMock())  # mock streams

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            # Simulate a client disconnect during processing
            raise ConnectionResetError("Client disconnected")

    mock_mcp_transport.connect_sse.return_value = DisconnectContextManager()
    mock_mcp_server._mcp_server = MagicMock()
    mock_mcp_server._mcp_server.run = AsyncMock()

    adapter._mcp_transport = mock_mcp_transport
    adapter._mcp_server = mock_mcp_server

    # Create a mock request
    mock_request = MagicMock()
    mock_request.scope = {"type": "http", "client": ("127.0.0.1", 12345)}
    mock_request.receive = AsyncMock()
    mock_request._send = AsyncMock()

    # Mock Response
    with patch("starlette.responses.Response") as mock_response_cls, patch.object(adapter, "logger") as mock_logger:
        mock_response = MagicMock()
        mock_response_cls.return_value = mock_response

        # Handle the request
        response = await adapter.handle_sse_request(mock_request)

        # Verify that the error was logged
        mock_logger.error.assert_called_once()
        log_message = mock_logger.error.call_args[0][0]
        assert "Error handling SSE connection" in log_message

        # Verify that a 500 error response was returned
        mock_response_cls.assert_called_once_with(status_code=500)
        assert response == mock_response


@pytest.mark.asyncio
async def test_handle_sse_request_http_exception():
    """Test handling HTTP exceptions during SSE request processing."""
    # Create a mock coordinator
    mock_coordinator = AsyncMock()

    # Create the adapter with the mock coordinator
    adapter = SSETransportAdapter(coordinator=mock_coordinator)

    # Mock the MCP transport and server
    mock_mcp_transport = MagicMock()
    mock_mcp_server = MagicMock()

    # Mock connect_sse to raise an HTTP exception
    mock_mcp_transport.connect_sse = AsyncMock(side_effect=HTTPException(status_code=403, detail="Forbidden"))

    adapter._mcp_transport = mock_mcp_transport
    adapter._mcp_server = mock_mcp_server

    # Create a mock request
    mock_request = MagicMock()
    mock_request.scope = {"type": "http", "client": ("127.0.0.1", 12345)}
    mock_request.receive = AsyncMock()
    mock_request._send = AsyncMock()

    # Mock Response
    with patch("starlette.responses.Response") as mock_response_cls, patch.object(adapter, "logger") as mock_logger:
        mock_response = MagicMock()
        mock_response_cls.return_value = mock_response

        # Handle the request
        response = await adapter.handle_sse_request(mock_request)

        # Verify that the error was logged
        mock_logger.error.assert_called_once()
        log_message = mock_logger.error.call_args[0][0]
        assert "Error handling SSE connection" in log_message

        # Verify that a 500 error response was returned
        mock_response_cls.assert_called_once_with(status_code=500)
        assert response == mock_response


@pytest.mark.asyncio
async def test_handle_sse_request_runtime_error():
    """Test handling runtime errors during SSE request processing."""
    # Create a mock coordinator
    mock_coordinator = AsyncMock()

    # Create the adapter with the mock coordinator
    adapter = SSETransportAdapter(coordinator=mock_coordinator)

    # Mock the MCP transport and server
    mock_mcp_transport = MagicMock()
    mock_mcp_server = MagicMock()

    # Create a context manager for connect_sse
    class MockContextManager:
        async def __aenter__(self):
            return (MagicMock(), MagicMock())  # mock streams

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    mock_mcp_transport.connect_sse.return_value = MockContextManager()
    mock_mcp_server._mcp_server = MagicMock()
    mock_mcp_server._mcp_server.run = AsyncMock(side_effect=RuntimeError("Test runtime error"))

    adapter._mcp_transport = mock_mcp_transport
    adapter._mcp_server = mock_mcp_server

    # Create a mock request
    mock_request = MagicMock()
    mock_request.scope = {"type": "http", "client": ("127.0.0.1", 12345)}
    mock_request.receive = AsyncMock()
    mock_request._send = AsyncMock()

    # Mock Response
    with patch("starlette.responses.Response") as mock_response_cls, patch.object(adapter, "logger") as mock_logger:
        mock_response = MagicMock()
        mock_response_cls.return_value = mock_response

        # Handle the request
        response = await adapter.handle_sse_request(mock_request)

        # Verify that the error was logged
        mock_logger.error.assert_called_once()
        log_message = mock_logger.error.call_args[0][0]
        assert "Error handling SSE connection" in log_message
        assert "Test runtime error" in log_message

        # Verify that a 500 error response was returned
        mock_response_cls.assert_called_once_with(status_code=500)
        assert response == mock_response


@pytest.mark.asyncio
async def test_run_sse_server_network_bind_error():
    """Test handling network bind errors when starting the SSE server."""
    # Mock the ApplicationCoordinator.getInstance method
    mock_coordinator = AsyncMock()

    # Mock SSETransportAdapter to simulate a network bind error
    mock_adapter = AsyncMock()
    mock_adapter.initialize = AsyncMock()
    mock_adapter.start_listening = AsyncMock(side_effect=OSError(socket.errno.EADDRINUSE, "Address already in use"))
    mock_adapter.shutdown = AsyncMock()

    # Path ApplicationCoordinator.getInstance and SSETransportAdapter constructor
    with (
        patch(
            "aider_mcp_server.sse_server.ApplicationCoordinator.getInstance", return_value=asyncio.Future()
        ) as mock_get_instance,
        patch("aider_mcp_server.sse_server.SSETransportAdapter", return_value=mock_adapter),
        patch("aider_mcp_server.sse_server.is_git_repository", return_value=(True, "")),
        patch("aider_mcp_server.sse_server.get_logger") as mock_get_logger,
    ):
        # Set up the mock logger
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Set the mock_coordinator as the result of getInstance
        mock_get_instance.return_value.set_result(mock_coordinator)

        # Call run_sse_server and expect an OSError
        with pytest.raises(OSError) as excinfo:
            await run_sse_server(host="127.0.0.1", port=8765, current_working_dir="/mock/git/repo")

        # Verify the error
        assert socket.errno.EADDRINUSE == excinfo.value.errno
        assert "Address already in use" in str(excinfo.value)

        # Verify that the adapter methods were called in the correct order
        mock_adapter.initialize.assert_called_once()
        mock_adapter.start_listening.assert_called_once()

        # The adapter should be shut down after the error
        mock_adapter.shutdown.assert_called_once()


@pytest.mark.asyncio
async def test_run_sse_server_async_error_during_setup():
    """Test handling async errors during SSE server setup."""
    # Mock the ApplicationCoordinator.getInstance method to raise an error
    # Path ApplicationCoordinator.getInstance and other components
    with (
        patch(
            "aider_mcp_server.sse_server.ApplicationCoordinator.getInstance",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Test error during coordinator instantiation"),
        ),
        patch("aider_mcp_server.sse_server.is_git_repository", return_value=(True, "")),
        patch("aider_mcp_server.sse_server.get_logger") as mock_get_logger,
    ):
        # Set up the mock logger
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Call run_sse_server and expect a RuntimeError
        with pytest.raises(RuntimeError) as excinfo:
            await run_sse_server(host="127.0.0.1", port=8765, current_working_dir="/mock/git/repo")

        # Verify the error
        assert "Test error during coordinator instantiation" in str(excinfo.value)


@pytest.mark.asyncio
async def test_sse_server_partial_initialization_error():
    """Test handling errors during partial initialization of the SSE server."""
    # Create a mock coordinator
    mock_coordinator = AsyncMock()

    # Create an adapter that fails during initialization
    adapter = SSETransportAdapter(coordinator=mock_coordinator)

    # Mock _create_app to raise an exception
    original_create_app = adapter._create_app

    async def failing_create_app():
        # Call the first part of the original method
        from mcp.server.sse import SseServerTransport

        adapter._mcp_transport = SseServerTransport("/messages/")

        # Then raise an exception
        raise RuntimeError("Failed to create Starlette app")

    adapter._create_app = failing_create_app

    # Initialize the adapter and expect an error
    with pytest.raises(RuntimeError) as excinfo:
        await adapter.initialize()

    # Verify the error
    assert "Failed to create Starlette app" in str(excinfo.value)

    # Verify that partial initialization occurred
    assert adapter._mcp_transport is not None
    assert adapter._app is None

    # Restore the original method
    adapter._create_app = original_create_app


@pytest.mark.asyncio
async def test_handle_message_fragmentation():
    """Test handling message fragmentation during transmission."""
    # Create a mock coordinator
    mock_coordinator = AsyncMock()

    # Create the adapter with the mock coordinator
    adapter = SSETransportAdapter(coordinator=mock_coordinator)

    # Set up a test connection
    test_queue = asyncio.Queue()
    adapter._active_connections = {"test-connection": test_queue}

    # Create a very large event data that might get fragmented
    event_type = EventTypes.TOOL_RESULT
    event_data = {
        "id": "large-data-id",
        "result": "x" * 50000,  # A large string that might cause fragmentation
    }

    # Send the event
    await adapter.send_event(event_type, event_data)

    # Verify that the event was put in the queue
    assert not test_queue.empty()
    message = test_queue.get_nowait()

    # Verify the message format
    assert f"event: {event_type.value}" in message

    # Extract and parse the JSON data
    data_line = [line for line in message.split("\n") if line.startswith("data:")][0]
    data_json = data_line[len("data: ") :]
    parsed_data = json.loads(data_json)

    # Verify that the large data was preserved
    assert parsed_data["id"] == "large-data-id"
    assert len(parsed_data["result"]) == 50000
