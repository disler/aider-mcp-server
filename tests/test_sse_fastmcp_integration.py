"""
Tests for SSE server integration with FastMCP.

These tests verify that the SSE Transport Adapter properly integrates with FastMCP,
including tool registration, handling MCP requests, and response formatting.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aider_mcp_server.atoms.security.context import SecurityContext
from aider_mcp_server.organisms.transports.sse.sse_transport_adapter import SSETransportAdapter


@pytest.fixture
def mock_mcp_transport():
    """Fixture providing a mock MCP SSE transport."""
    mock = MagicMock()
    mock.connect_sse = AsyncMock()

    # Create a context manager for connect_sse
    class MockContextManager:
        async def __aenter__(self):
            return (MagicMock(), MagicMock())  # mock streams

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    mock.connect_sse.return_value = MockContextManager()
    return mock


@pytest.fixture
def mock_fastmcp():
    """Fixture providing a mock FastMCP instance."""
    mock = MagicMock()
    mock.tool = MagicMock()
    mock._mcp_server = MagicMock()
    mock._mcp_server.run = AsyncMock()
    mock._mcp_server.create_initialization_options = MagicMock(return_value={})
    return mock


@pytest.mark.asyncio
async def test_adapter_initializes_fastmcp():
    """Test that the adapter initializes FastMCP when a coordinator is provided."""
    # Create a mock coordinator
    mock_coordinator = AsyncMock()

    # Create the adapter with the mock coordinator
    with patch("aider_mcp_server.organisms.transports.sse.sse_transport_adapter.FastMCP") as mock_fastmcp_class:
        mock_fastmcp = MagicMock()
        mock_fastmcp.tool = MagicMock(return_value=lambda func: func)
        mock_fastmcp_class.return_value = mock_fastmcp

        adapter = SSETransportAdapter(coordinator=mock_coordinator)

        # Initialize the adapter
        await adapter.initialize()

        # Verify that FastMCP was initialized
        mock_fastmcp_class.assert_called_once_with("aider-sse")
        assert adapter._mcp_server == mock_fastmcp
        assert adapter._fastmcp_initialized is True


@pytest.mark.asyncio
async def test_adapter_skips_fastmcp_init_without_coordinator():
    """Test that the adapter skips FastMCP initialization when no coordinator is provided."""
    # Create the adapter without a coordinator
    with patch("aider_mcp_server.organisms.transports.sse.sse_transport_adapter.FastMCP") as mock_fastmcp_class:
        adapter = SSETransportAdapter(coordinator=None)

        # Initialize the adapter
        with patch.object(adapter, "logger") as mock_logger:
            await adapter.initialize()

            # Verify that FastMCP was not initialized
            mock_fastmcp_class.assert_not_called()
            assert adapter._mcp_server is None
            assert adapter._fastmcp_initialized is False

            # Verify that a warning was logged
            mock_logger.warning.assert_any_call("No coordinator available, FastMCP will not be initialized")


@pytest.mark.asyncio
async def test_adapter_registers_tools_with_fastmcp():
    """Test that the adapter registers tools with FastMCP."""
    # Create a mock coordinator
    mock_coordinator = AsyncMock()

    # Create the adapter with the mock coordinator
    with patch("aider_mcp_server.organisms.transports.sse.sse_transport_adapter.FastMCP") as mock_fastmcp_class:
        mock_fastmcp = MagicMock()
        # Create a list to capture the decorated functions
        registered_tools = []

        # Mock the tool decorator to capture the registered functions
        def mock_tool_decorator(func):
            registered_tools.append(func.__name__)
            return func

        mock_fastmcp.tool = MagicMock(return_value=mock_tool_decorator)
        mock_fastmcp_class.return_value = mock_fastmcp

        adapter = SSETransportAdapter(coordinator=mock_coordinator)

        # Initialize the adapter
        await adapter.initialize()

        # Verify that tools were registered with FastMCP
        assert mock_fastmcp.tool.call_count >= 2
        assert "aider_ai_code" in registered_tools
        assert "list_models" in registered_tools


@pytest.mark.skip(reason="MCP transport integration test - implementation in progress")
@pytest.mark.asyncio
async def test_adapter_integrates_with_mcp_transport(mock_mcp_transport, mock_fastmcp):
    """Test that the adapter properly integrates with the MCP SSE transport."""
    # Create a mock coordinator
    mock_coordinator = AsyncMock()

    # Create the adapter with the mock coordinator
    with (
        patch("aider_mcp_server.organisms.transports.sse.sse_transport_adapter.FastMCP", return_value=mock_fastmcp),
        patch("aider_mcp_server.organisms.transports.sse.sse_transport_adapter.SseServerTransport", return_value=mock_mcp_transport),
    ):
        adapter = SSETransportAdapter(coordinator=mock_coordinator)

        # Initialize the adapter
        await adapter.initialize()

        # Verify that the MCP transport was initialized
        assert adapter._mcp_transport == mock_mcp_transport


@pytest.mark.skip(reason="Skipping due to hanging/timeout issues, needs investigation.")
@pytest.mark.asyncio
async def test_adapter_handle_sse_request_with_fastmcp(mock_mcp_transport, mock_fastmcp):
    """Test that the adapter properly handles SSE requests with FastMCP."""
    # Create a mock coordinator
    mock_coordinator = AsyncMock()

    # Create a mock request
    mock_request = MagicMock()
    mock_request.scope = {"type": "http", "client": ("127.0.0.1", 12345)}
    mock_request.receive = AsyncMock()
    mock_request._send = AsyncMock()

    # Create the adapter with the mock coordinator
    with (
        patch("aider_mcp_server.organisms.transports.sse.sse_transport_adapter.FastMCP", return_value=mock_fastmcp),
        patch("aider_mcp_server.organisms.transports.sse.sse_transport_adapter.SseServerTransport", return_value=mock_mcp_transport),
        patch("starlette.responses.Response") as mock_response_cls,
    ):
        mock_response = MagicMock()
        mock_response_cls.return_value = mock_response

        adapter = SSETransportAdapter(coordinator=mock_coordinator)

        # Initialize the adapter
        await adapter.initialize()

        # Handle an SSE request
        response = await adapter.handle_sse_request(mock_request)

        # Verify that the MCP transport's connect_sse method was called
        mock_mcp_transport.connect_sse.assert_called_once_with(
            mock_request.scope, mock_request.receive, mock_request._send
        )

        # Verify that the MCP server's run method was called
        mock_fastmcp._mcp_server.run.assert_called_once()

        # Verify that a response was returned
        assert response == mock_response


@pytest.mark.skip(reason="Skipping due to hanging/timeout issues, needs investigation.")
@pytest.mark.asyncio
async def test_adapter_handle_sse_request_without_mcp(mock_mcp_transport, mock_fastmcp):
    """Test that the adapter handles SSE requests properly when MCP is not initialized."""
    # Create a mock request
    mock_request = MagicMock()
    mock_request.scope = {"type": "http", "client": ("127.0.0.1", 12345)}
    mock_request.receive = AsyncMock()
    mock_request._send = AsyncMock()

    # Create the adapter without initializing MCP
    with patch("starlette.responses.Response") as mock_response_cls:
        mock_response = MagicMock()
        mock_response_cls.return_value = mock_response

        adapter = SSETransportAdapter()

        # Mock the logger
        with patch.object(adapter, "logger") as mock_logger:
            # Handle an SSE request
            response = await adapter.handle_sse_request(mock_request)

            # Verify that an error was logged
            mock_logger.error.assert_called_once_with("SSE transport or MCP server not initialized")

            # Verify that a 500 error response was returned
            mock_response_cls.assert_called_once_with("Server not properly initialized", status_code=500)
            assert response == mock_response


@pytest.mark.skip(reason="Skipping due to hanging/timeout issues, needs investigation.")
@pytest.mark.asyncio
async def test_adapter_handle_sse_request_with_exception(mock_mcp_transport, mock_fastmcp):
    """Test that the adapter properly handles exceptions during SSE request processing."""
    # Create a mock coordinator
    mock_coordinator = AsyncMock()

    # Create a mock request
    mock_request = MagicMock()
    mock_request.scope = {"type": "http", "client": ("127.0.0.1", 12345)}
    mock_request.receive = AsyncMock()
    mock_request._send = AsyncMock()

    # Make the MCP server's run method raise an exception
    mock_fastmcp._mcp_server.run = AsyncMock(side_effect=Exception("Test exception"))

    # Create the adapter with the mock coordinator
    with (
        patch("aider_mcp_server.organisms.transports.sse.sse_transport_adapter.FastMCP", return_value=mock_fastmcp),
        patch("aider_mcp_server.organisms.transports.sse.sse_transport_adapter.SseServerTransport", return_value=mock_mcp_transport),
        patch("starlette.responses.Response") as mock_response_cls,
    ):
        mock_response = MagicMock()
        mock_response_cls.return_value = mock_response

        adapter = SSETransportAdapter(coordinator=mock_coordinator)

        # Initialize the adapter
        await adapter.initialize()

        # Mock the logger
        with patch.object(adapter, "logger") as mock_logger:
            # Handle an SSE request
            response = await adapter.handle_sse_request(mock_request)

            # Verify that an error was logged
            mock_logger.error.assert_called_once()
            assert "Error handling SSE connection" in mock_logger.error.call_args[0][0]

            # Verify that a 500 error response was returned
            mock_response_cls.assert_called_once_with(status_code=500)
            assert response == mock_response


@pytest.mark.skip(reason="Skipping due to hanging/timeout issues, needs investigation.")
@pytest.mark.asyncio
async def test_adapter_handle_sse_request_with_cancellation(mock_mcp_transport, mock_fastmcp):
    """Test that the adapter properly handles cancellation during SSE request processing."""
    # Create a mock coordinator
    mock_coordinator = AsyncMock()

    # Create a mock request
    mock_request = MagicMock()
    mock_request.scope = {"type": "http", "client": ("127.0.0.1", 12345)}
    mock_request.receive = AsyncMock()
    mock_request._send = AsyncMock()

    # Make the MCP server's run method raise a CancelledError
    mock_fastmcp._mcp_server.run = AsyncMock(side_effect=asyncio.CancelledError())

    # Create the adapter with the mock coordinator
    with (
        patch("aider_mcp_server.organisms.transports.sse.sse_transport_adapter.FastMCP", return_value=mock_fastmcp),
        patch("aider_mcp_server.organisms.transports.sse.sse_transport_adapter.SseServerTransport", return_value=mock_mcp_transport),
        patch("starlette.responses.Response") as mock_response_cls,
    ):
        mock_response = MagicMock()
        mock_response_cls.return_value = mock_response

        adapter = SSETransportAdapter(coordinator=mock_coordinator)

        # Initialize the adapter
        await adapter.initialize()

        # Mock the logger
        with patch.object(adapter, "logger") as mock_logger:
            # Handle an SSE request
            response = await adapter.handle_sse_request(mock_request)

            # Verify that a debug message was logged
            mock_logger.debug.assert_any_call("SSE connection cancelled (client disconnect)")

            # Verify that a normal response was returned (not an error)
            mock_response_cls.assert_called_once_with()
            assert response == mock_response


@pytest.mark.asyncio
async def test_aider_ai_code_tool_integration():
    """Test the integration of the aider_ai_code tool with the process_aider_ai_code_request function."""
    # Create a mock coordinator
    mock_coordinator = AsyncMock()

    # Create the adapter with the mock coordinator
    adapter = SSETransportAdapter(
        coordinator=mock_coordinator, editor_model="test-model", current_working_dir="/test/dir"
    )

    # Mock the process_aider_ai_code_request function
    with (
        patch("aider_mcp_server.organisms.processors.handlers.process_aider_ai_code_request") as mock_process_request,
        patch("aider_mcp_server.organisms.transports.sse.sse_transport_adapter.FastMCP") as mock_fastmcp_class,
        patch.object(adapter, "logger"),
    ):
        # Configure the mock FastMCP to capture and execute the tool function
        mock_fastmcp = MagicMock()
        mock_fastmcp._tool_functions = {}  # Initialize a dictionary to store tool functions

        def mock_tool_decorator(func):
            # Store the decorated function by its name
            mock_fastmcp._tool_functions[func.__name__] = func
            return func

        mock_fastmcp.tool = MagicMock(return_value=mock_tool_decorator)
        mock_fastmcp_class.return_value = mock_fastmcp

        # Mock the result of process_aider_ai_code_request
        mock_result = MagicMock()
        mock_result.result = {"output": "Test output"}
        mock_process_request.return_value = mock_result

        # Initialize the adapter
        await adapter.initialize()

        # Get the decorated function by name
        aider_ai_code_func = mock_fastmcp._tool_functions.get("aider_ai_code")
        assert aider_ai_code_func is not None, "aider_ai_code tool not registered"

        # Call the function with test parameters
        result = await aider_ai_code_func(
            ai_coding_prompt="Test prompt",
            relative_editable_files=["test.py"],
            relative_readonly_files=["readonly.py"],
            model="test-model",
        )

        # Verify that process_aider_ai_code_request was called with the correct parameters
        mock_process_request.assert_called_once()
        call_args = mock_process_request.call_args[1]

        assert call_args["transport_id"] == "sse"
        assert call_args["params"] == {
            "ai_coding_prompt": "Test prompt",
            "relative_editable_files": ["test.py"],
            "relative_readonly_files": ["readonly.py"],
            "model": "test-model",
        }
        assert call_args["editor_model"] == "test-model"
        assert call_args["current_working_dir"] == "/test/dir"
        assert isinstance(call_args["security_context"], SecurityContext)

        # Verify that the correct result was returned
        assert result == {"output": "Test output"}


@pytest.mark.asyncio
async def test_list_models_tool_integration():
    """Test the integration of the list_models tool with the process_list_models_request function."""
    # Create a mock coordinator
    mock_coordinator = AsyncMock()

    # Create the adapter with the mock coordinator
    adapter = SSETransportAdapter(coordinator=mock_coordinator)

    # Mock the process_list_models_request function
    with (
        patch("aider_mcp_server.organisms.processors.handlers.process_list_models_request") as mock_process_request,
        patch("aider_mcp_server.organisms.transports.sse.sse_transport_adapter.FastMCP") as mock_fastmcp_class,
        patch.object(adapter, "logger"),
    ):
        # Configure the mock FastMCP to capture and execute the tool function
        mock_fastmcp = MagicMock()

        def mock_tool_decorator(func):
            # Store the decorated function
            if func.__name__ == "list_models":
                mock_fastmcp._list_models_func = func
            return func

        mock_fastmcp.tool = MagicMock(return_value=mock_tool_decorator)
        mock_fastmcp_class.return_value = mock_fastmcp

        # Mock the result of process_list_models_request
        mock_result = {"models": ["model1", "model2"]}
        mock_process_request.return_value = mock_result

        # Initialize the adapter
        await adapter.initialize()

        # Get the decorated function
        list_models_func = mock_fastmcp._list_models_func

        # Call the function with test parameters
        result = await list_models_func(substring="test")

        # Verify that process_list_models_request was called with the correct parameters
        mock_process_request.assert_called_once()
        call_args = mock_process_request.call_args[1]

        assert call_args["transport_id"] == "sse"
        assert call_args["params"] == {"substring": "test"}
        assert isinstance(call_args["security_context"], SecurityContext)

        # Verify that the correct result was returned
        assert result == ["model1", "model2"]


@pytest.mark.asyncio
async def test_tool_error_handling():
    """Test that errors in tool functions are properly handled."""
    # Create a mock coordinator
    mock_coordinator = AsyncMock()

    # Create the adapter with the mock coordinator
    adapter = SSETransportAdapter(coordinator=mock_coordinator)

    # Mock the process_aider_ai_code_request function to raise an exception
    with (
        patch(
            "aider_mcp_server.organisms.processors.handlers.process_aider_ai_code_request",
            side_effect=Exception("Test exception"),
        ),
        patch("aider_mcp_server.organisms.transports.sse.sse_transport_adapter.FastMCP") as mock_fastmcp_class,
        patch.object(adapter, "logger") as mock_logger,
    ):
        # Configure the mock FastMCP to capture and execute the tool function
        mock_fastmcp = MagicMock()
        mock_fastmcp._tool_functions = {}  # Initialize a dictionary to store tool functions

        def mock_tool_decorator(func):
            # Store the decorated function by its name
            mock_fastmcp._tool_functions[func.__name__] = func
            return func

        mock_fastmcp.tool = MagicMock(return_value=mock_tool_decorator)
        mock_fastmcp_class.return_value = mock_fastmcp

        # Initialize the adapter
        await adapter.initialize()

        # Get the decorated function by name
        aider_ai_code_func = mock_fastmcp._tool_functions.get("aider_ai_code")
        assert aider_ai_code_func is not None, "aider_ai_code tool not registered"

        # Call the function with test parameters
        result = await aider_ai_code_func(ai_coding_prompt="Test prompt", relative_editable_files=["test.py"])

        # Verify that an error was logged
        mock_logger.error.assert_called_once()
        assert "Error in aider_ai_code" in mock_logger.error.call_args[0][0]

        # Verify that an error response was returned
        assert "error" in result
        assert "Test exception" in result["error"]
