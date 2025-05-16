"""
Comprehensive tests for the SSE server functionality.

These tests focus on basic functionality and error handling scenarios.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aider_mcp_server.atoms.event_types import EventTypes
from aider_mcp_server.sse_transport_adapter import SSETransportAdapter


@pytest.mark.asyncio
async def test_sse_transport_adapter_capabilities():
    """
    Test the capabilities of the SSETransportAdapter.
    """
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
async def test_sse_transport_adapter_event_sending():
    """
    Test the event sending mechanism of the SSETransportAdapter.
    """
    # Create adapter and test that it can handle events with no connections
    adapter = SSETransportAdapter()
    adapter._active_connections = {}

    # Should handle no connections gracefully without error
    await adapter.send_event(EventTypes.STATUS, {"message": "Test status message"})

    # Test successful completion (no assertion needed, just no errors)


@pytest.mark.asyncio
async def test_sse_transport_adapter_security_context():
    """
    Test the validate_request_security method of the SSETransportAdapter.
    """
    adapter = SSETransportAdapter()

    request_data = {}
    security_context = adapter.validate_request_security(request_data)

    # Check SecurityContext properties
    assert security_context.user_id is None
    assert security_context.permissions == set()
    assert security_context.is_anonymous is True
    assert security_context.transport_id == "sse"


@pytest.mark.asyncio
async def test_sse_transport_adapter_connection_handling():
    """
    Test the connection handling of the SSETransportAdapter.
    """
    with patch("starlette.responses.Response") as mock_response:
        with patch("aider_mcp_server.transport_adapter.AbstractTransportAdapter"):
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
async def test_sse_transport_adapter_message_handling():
    """
    Test the message handling of the SSETransportAdapter.
    """
    with patch("aider_mcp_server.transport_adapter.AbstractTransportAdapter"):
        sse_adapter = SSETransportAdapter()

        with patch("aider_mcp_server.sse_transport_adapter.json") as mock_json:
            mock_json.loads.return_value = {
                "operation": "test_operation",
                "parameters": {},
            }

            request = MagicMock()
            request.json = AsyncMock(return_value={})
            response = await sse_adapter.handle_message_request(request)

            assert response is not None


@pytest.mark.asyncio
async def test_sse_transport_adapter_send_event_error_handling():
    """
    Test the error handling in send_event method of the SSETransportAdapter.
    """
    with patch("aider_mcp_server.transport_adapter.AbstractTransportAdapter"):
        sse_adapter = SSETransportAdapter()

        with patch.object(sse_adapter, "_active_connections", {}):
            with patch.object(sse_adapter, "logger"):
                await sse_adapter.send_event(EventTypes.STATUS, {"message": "Test status message"})
