"""
Comprehensive tests for the SSE server functionality.

These tests focus on basic functionality and error handling scenarios.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aider_mcp_server.atoms.event_types import EventTypes
from aider_mcp_server.sse_transport_adapter import SSETransportAdapter


@pytest.mark.asyncio
async def test_sse_transport_adapter_initialization():
    """
    Test the initialization of the SSETransportAdapter.
    """
    with patch("aider_mcp_server.transport_adapter.AbstractTransportAdapter"):
        sse_adapter = SSETransportAdapter()

        assert sse_adapter.transport_id.startswith("sse_")
        assert sse_adapter.get_capabilities() == {
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
    with patch("aider_mcp_server.transport_adapter.AbstractTransportAdapter"):
        sse_adapter = SSETransportAdapter()

        with patch.object(sse_adapter, "_active_connections", {}):
            with patch.object(sse_adapter, "logger"):
                await sse_adapter.send_event(
                    EventTypes.STATUS, {"message": "Test status message"}
                )


@pytest.mark.asyncio
async def test_sse_transport_adapter_connection_handling():
    """
    Test the connection handling of the SSETransportAdapter.
    """
    with patch("sse_starlette.sse.EventSourceResponse"):
        with patch("aider_mcp_server.transport_adapter.AbstractTransportAdapter"):
            sse_adapter = SSETransportAdapter()

            with patch.object(sse_adapter, "logger"):
                request = MagicMock()
                response = await sse_adapter.handle_sse_request(request)

                assert response is not None


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
async def test_sse_transport_adapter_validate_request_security():
    """
    Test the validate_request_security method of the SSETransportAdapter.
    """
    with patch("aider_mcp_server.transport_adapter.AbstractTransportAdapter"):
        sse_adapter = SSETransportAdapter()

        from aider_mcp_server.security import ANONYMOUS_SECURITY_CONTEXT

        request_data = {}
        security_context = sse_adapter.validate_request_security(request_data)

        assert security_context == ANONYMOUS_SECURITY_CONTEXT


@pytest.mark.asyncio
async def test_sse_transport_adapter_start_listening():
    """
    Test the start_listening method of the SSETransportAdapter.
    """
    with patch("aider_mcp_server.transport_adapter.AbstractTransportAdapter"):
        sse_adapter = SSETransportAdapter()

        with patch.object(sse_adapter, "logger"):
            await sse_adapter.start_listening()

            sse_adapter.logger.debug.assert_called_once_with(
                f"SSE adapter {sse_adapter.transport_id} start_listening called (no-op)"
            )


@pytest.mark.asyncio
async def test_sse_transport_adapter_send_event_error_handling():
    """
    Test the error handling in send_event method of the SSETransportAdapter.
    """
    with patch("aider_mcp_server.transport_adapter.AbstractTransportAdapter"):
        sse_adapter = SSETransportAdapter()

        with patch.object(sse_adapter, "_active_connections", {}):
            with patch.object(sse_adapter, "logger"):
                await sse_adapter.send_event(
                    EventTypes.STATUS, {"message": "Test status message"}
                )
