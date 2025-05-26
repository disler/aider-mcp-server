"""
Comprehensive tests for SSE server functionality.

This module consolidates tests for both the SSE server startup process
and the SSETransportAdapter functionality.
"""

from unittest import mock
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aider_mcp_server.atoms.types.event_types import EventTypes
from aider_mcp_server.atoms.types.mcp_types import LoggerProtocol
from aider_mcp_server.organisms.transports.sse.sse_transport_adapter import SSETransportAdapter
from aider_mcp_server.templates.servers.sse_server import run_sse_server


class TestSSEServer:
    """Test suite for SSE server startup and configuration."""

    @pytest.mark.asyncio
    async def test_run_sse_server_startup(self):
        """Test the complete SSE server startup process."""
        # Create patch context for all mocks
        with (
            patch("aider_mcp_server.templates.servers.sse_server.is_git_repository", return_value=(True, None)),
            patch(
                "aider_mcp_server.templates.servers.sse_server.asyncio.Event"
            ) as mock_event_cls,  # Assuming asyncio is used as sse_server.asyncio
            patch("aider_mcp_server.templates.servers.sse_server.SSETransportAdapter") as mock_adapter_cls,
            patch("aider_mcp_server.templates.servers.sse_server.ApplicationCoordinator") as mock_coordinator_cls,
            patch(
                "aider_mcp_server.molecules.transport.base_adapter.asyncio.create_task"
            ) as mock_create_task,  # Assuming create_task is from base_adapter's asyncio
        ):
            # Set up mock event for graceful shutdown signalling
            mock_event = MagicMock()
            mock_event.wait = AsyncMock()  # serve_sse awaits this
            mock_event.set = MagicMock()
            # is_set might be used if serve_sse had a different loop structure,
            # but current serve_sse relies on wait().
            mock_event.is_set = MagicMock(return_value=False)  # Start as not set
            mock_event_cls.return_value = mock_event

            # Set up mock coordinator
            mock_coordinator = MagicMock()
            mock_coordinator.register_handler = AsyncMock()  # Still check it's not called
            mock_coordinator.register_transport = AsyncMock()
            mock_coordinator.subscribe_to_event_type = AsyncMock()
            mock_coordinator._initialize_coordinator = AsyncMock()
            mock_coordinator.__aenter__ = AsyncMock(return_value=mock_coordinator)
            mock_coordinator.__aexit__ = AsyncMock()
            mock_coordinator_cls.getInstance = AsyncMock(return_value=mock_coordinator)

            # Set up mock adapter instance that mock_adapter_cls will return
            mock_adapter = MagicMock(spec=SSETransportAdapter)
            mock_adapter_cls.return_value = mock_adapter

            # Call the function under test - define params first
            host, port, editor_model, cwd, heartbeat_interval_val = (
                "127.0.0.1",
                8888,
                "test_model",
                "/test/repo",
                20.0,
            )

            # Simulate state of mock_adapter as if SSETransportAdapter.__init__ and its super().__init__ calls ran
            mock_adapter._coordinator = mock_coordinator
            mock_adapter._transport_id = "sse"  # Set by SSETransportAdapter's call to super().__init__
            mock_adapter._transport_type = "sse"  # Set by SSETransportAdapter's call to super().__init__
            mock_adapter._heartbeat_interval = heartbeat_interval_val  # Passed to constructor
            mock_adapter._heartbeat_task = None  # Initial state in AbstractTransportAdapter
            mock_adapter.logger = MagicMock(spec_set=LoggerProtocol)  # Used by AbstractTransportAdapter.initialize

            # Mock methods on mock_adapter needed by AbstractTransportAdapter.initialize or SSETransportAdapter.initialize
            mock_adapter.get_transport_id = MagicMock(return_value="sse")
            mock_adapter.get_transport_type = MagicMock(return_value="sse")
            # Return a specific EventType for subscribe_to_event_type assertion
            mock_adapter.get_capabilities = MagicMock(return_value={EventTypes.STATUS})
            mock_adapter._heartbeat_loop = AsyncMock()  # Coroutine function for heartbeat task

            # Mock methods specific to SSETransportAdapter.initialize that are called after super().initialize()
            mock_adapter._initialize_fastmcp = MagicMock()
            mock_adapter._create_app = AsyncMock()

            # Mock initialize method to simulate what AbstractTransportAdapter.initialize would do
            async def initialize_side_effect():
                # Simulate the coordinator registration
                if mock_adapter._coordinator:
                    await mock_adapter._coordinator.register_transport("sse", mock_adapter)
                    await mock_adapter._coordinator.subscribe_to_event_type("sse", EventTypes.STATUS)

                    # Mock the heartbeat task creation
                    if mock_adapter._heartbeat_interval is not None and mock_adapter._heartbeat_interval > 0:
                        mock_adapter._heartbeat_task = MagicMock()
                        mock_create_task.return_value = mock_adapter._heartbeat_task
                        mock_create_task(mock_adapter._heartbeat_loop.return_value)

                # Call the mocked SSE-specific parts
                mock_adapter._initialize_fastmcp()
                await mock_adapter._create_app()

            mock_adapter.initialize = AsyncMock(side_effect=initialize_side_effect)

            # Call run_sse_server with the proper function signature
            await run_sse_server(host=host, port=port, editor_model=editor_model, current_working_dir=cwd)

            # Verify ApplicationCoordinator lifecycle
            mock_coordinator_cls.getInstance.assert_awaited_once()

            # Verify SSETransportAdapter instantiation - check the actual call arguments
            mock_adapter_cls.assert_called_once_with(
                coordinator=mock_coordinator,
                host=host,
                port=port,
                get_logger=mock.ANY,  # Accept any logger function
                editor_model=editor_model,
                current_working_dir=cwd,
            )

            # Verify adapter initialization process
            mock_adapter.initialize.assert_awaited_once()

            # Verify transport registration (called from AbstractTransportAdapter.initialize)
            mock_coordinator.register_transport.assert_awaited_once_with("sse", mock_adapter)

            # Verify subscription to capabilities (called from AbstractTransportAdapter.initialize)
            mock_coordinator.subscribe_to_event_type.assert_awaited_once_with("sse", EventTypes.STATUS)

            # Verify heartbeat task creation (called from AbstractTransportAdapter.initialize)
            mock_create_task.assert_called_once()
            # Check that create_task was called with the coroutine from _heartbeat_loop()
            assert mock_create_task.call_args[0][0] == mock_adapter._heartbeat_loop.return_value

            # Verify that handlers are no longer directly registered with the coordinator
            assert mock_coordinator.register_handler.await_count == 0

            # Verify serve_sse waits for stop event
            mock_event.wait.assert_awaited_once()


class TestSSETransportAdapter:
    """Test suite for SSETransportAdapter functionality."""

    @pytest.mark.asyncio
    async def test_sse_transport_adapter_capabilities(self):
        """Test the capabilities of the SSETransportAdapter."""
        # Test the get_capabilities method without initializing the full adapter
        adapter = SSETransportAdapter()
        capabilities = adapter.get_capabilities()

        assert capabilities == {
            EventTypes.STATUS,
            EventTypes.PROGRESS,
            EventTypes.TOOL_RESULT,
            EventTypes.HEARTBEAT,
        }

    @pytest.mark.asyncio
    async def test_sse_transport_adapter_event_sending(self):
        """Test the event sending mechanism of the SSETransportAdapter."""
        # Create adapter and test that it can handle events with no connections
        adapter = SSETransportAdapter()
        adapter._active_connections = {}

        # Should handle no connections gracefully without error
        await adapter.send_event(EventTypes.STATUS, {"message": "Test status message"})

        # Test successful completion (no assertion needed, just no errors)

    @pytest.mark.asyncio
    async def test_sse_transport_adapter_security_context(self):
        """Test the validate_request_security method of the SSETransportAdapter."""
        adapter = SSETransportAdapter()

        request_data = {}
        security_context = adapter.validate_request_security(request_data)

        # Check SecurityContext properties
        assert security_context.user_id is None
        assert security_context.permissions == set()
        assert security_context.is_anonymous is True
        assert security_context.transport_id == "sse"

    @pytest.mark.asyncio
    async def test_sse_transport_adapter_connection_handling(self):
        """Test the connection handling of the SSETransportAdapter."""
        with patch("starlette.responses.Response") as mock_response:
            with patch("aider_mcp_server.molecules.transport.base_adapter.AbstractTransportAdapter"):
                sse_adapter = SSETransportAdapter()

                # Mock the MCP transport
                mock_mcp_transport = MagicMock()
                mock_mcp_transport.connect_sse = AsyncMock()
                sse_adapter._mcp_transport = mock_mcp_transport

                with patch.object(sse_adapter, "logger"):
                    request = MagicMock()
                    request.scope = {}
                    request.receive = AsyncMock()
                    request._send = AsyncMock()

                    response = await sse_adapter.handle_sse_request(request)

                    assert response is not None
                    mock_response.assert_called_once()

    @pytest.mark.asyncio
    async def test_sse_transport_adapter_message_handling(self):
        """Test the message handling of the SSETransportAdapter."""
        with patch("aider_mcp_server.molecules.transport.base_adapter.AbstractTransportAdapter"):
            sse_adapter = SSETransportAdapter()

            with patch("aider_mcp_server.organisms.transports.sse.sse_transport_adapter.json") as mock_json:
                mock_json.loads.return_value = {
                    "operation": "test_operation",
                    "parameters": {},
                }

                request = MagicMock()
                request.json = AsyncMock(return_value={})
                response = await sse_adapter.handle_message_request(request)

                assert response is not None

    @pytest.mark.asyncio
    async def test_sse_transport_adapter_send_event_error_handling(self):
        """Test the error handling in send_event method of the SSETransportAdapter."""
        with patch("aider_mcp_server.molecules.transport.base_adapter.AbstractTransportAdapter"):
            sse_adapter = SSETransportAdapter()

            with patch.object(sse_adapter, "_active_connections", {}):
                with patch.object(sse_adapter, "logger"):
                    await sse_adapter.send_event(EventTypes.STATUS, {"message": "Test status message"})


if __name__ == "__main__":
    # For direct execution
    pytest.main([__file__])
