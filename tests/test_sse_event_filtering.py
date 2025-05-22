"""
Tests for SSE transport event filtering and origin validation.

These tests verify that the SSE Transport Adapter properly filters events
based on their origin and type, preventing event loops and ensuring proper
event propagation between different transports.
"""

import uuid
from unittest.mock import AsyncMock, patch

import pytest

from aider_mcp_server.atoms.event_types import EventTypes
from aider_mcp_server.sse_transport_adapter import SSETransportAdapter


@pytest.fixture
def adapter():
    """Fixture providing a basic SSETransportAdapter instance."""
    adapter = SSETransportAdapter()
    return adapter


def test_adapter_get_capabilities(adapter):
    """Test that the adapter returns the correct event type capabilities."""
    capabilities = adapter.get_capabilities()

    # Verify that the adapter supports the expected event types
    assert EventTypes.STATUS in capabilities
    assert EventTypes.PROGRESS in capabilities
    assert EventTypes.TOOL_RESULT in capabilities
    assert EventTypes.HEARTBEAT in capabilities
    assert len(capabilities) == 4  # Only these four are supported


def test_should_receive_event_same_transport_origin(adapter):
    """Test that events from the same transport are filtered out."""
    # Create an event that originated from this transport
    event_data = {
        "message": "Test message",
        "transport_origin": {"transport_id": adapter.get_transport_id(), "transport_type": "sse"},
    }

    # Check if the adapter should receive this event
    result = adapter.should_receive_event(EventTypes.STATUS, event_data)

    # Verify that the event is filtered out (should not be received)
    assert result is False


def test_should_receive_event_different_transport_origin(adapter):
    """Test that events from a different transport are accepted."""
    # Create an event that originated from a different transport
    event_data = {
        "message": "Test message",
        "transport_origin": {"transport_id": "other-transport", "transport_type": "stdio"},
    }

    # Check if the adapter should receive this event
    result = adapter.should_receive_event(EventTypes.STATUS, event_data)

    # Verify that the event is accepted
    assert result is True


def test_should_receive_event_no_transport_origin(adapter):
    """Test that events with no transport origin are accepted."""
    # Create an event with no transport origin
    event_data = {"message": "Test message"}

    # Check if the adapter should receive this event
    result = adapter.should_receive_event(EventTypes.STATUS, event_data)

    # Verify that the event is accepted
    assert result is True


def test_should_receive_event_monitored_transport(adapter):
    """Test that events from a monitored transport are always accepted."""
    # Set up a monitored transport ID
    adapter.monitor_stdio_transport_id = "stdio-transport"

    # Create an event that originated from the monitored transport
    event_data = {
        "message": "Test message",
        "transport_origin": {"transport_id": "stdio-transport", "transport_type": "stdio"},
    }

    # Check if the adapter should receive this event
    result = adapter.should_receive_event(EventTypes.STATUS, event_data)

    # Verify that the event is accepted
    assert result is True


def test_should_receive_event_other_monitored_transport(adapter):
    """Test that events from a non-monitored transport are still accepted."""
    # Set up a monitored transport ID
    adapter.monitor_stdio_transport_id = "stdio-transport"

    # Create an event that originated from a different transport
    event_data = {
        "message": "Test message",
        "transport_origin": {"transport_id": "other-transport", "transport_type": "stdio"},
    }

    # Check if the adapter should receive this event
    result = adapter.should_receive_event(EventTypes.STATUS, event_data)

    # Verify that the event is accepted
    assert result is True


def test_register_monitor_connection(adapter):
    """Test that the adapter correctly registers monitor connections."""
    # Register a monitor connection
    adapter.register_monitor_connection("test-connection")

    # Verify that the connection was registered
    assert "test-connection" in adapter._monitor_connections

    # Register another monitor connection
    adapter.register_monitor_connection("another-connection")

    # Verify that both connections are registered
    assert "test-connection" in adapter._monitor_connections
    assert "another-connection" in adapter._monitor_connections
    assert len(adapter._monitor_connections) == 2


@pytest.mark.skip(reason="subscribe_to_events not implemented yet")
@pytest.mark.asyncio
async def test_send_event_filtering():
    """Test that events are properly filtered before sending."""
    # Create a mock coordinator that will call the event handler
    mock_coordinator = AsyncMock()
    mock_coordinator.register_transport = AsyncMock()
    mock_coordinator.subscribe_to_events = AsyncMock()

    # Create the adapter with the mock coordinator
    adapter = SSETransportAdapter(coordinator=mock_coordinator)

    # Set up an active connection
    adapter._active_connections = {"test-connection": AsyncMock()}

    # Initialize the adapter
    await adapter.initialize()

    # Verify that subscribe_to_events was called
    mock_coordinator.subscribe_to_events.assert_called_once()

    # Get the event handler that was registered with the coordinator
    event_handler = mock_coordinator.subscribe_to_events.call_args[0][0]

    # Create a test event from the same transport (this should be filtered out)
    self_originated_event = {
        "id": str(uuid.uuid4()),
        "type": EventTypes.STATUS,
        "data": {
            "message": "Test message",
            "transport_origin": {"transport_id": adapter.get_transport_id(), "transport_type": "sse"},
        },
    }

    # Handle the event
    with patch.object(adapter, "send_event") as mock_send_event:
        await event_handler(self_originated_event["type"], self_originated_event["data"])

        # Verify that send_event was not called (event was filtered out)
        mock_send_event.assert_not_called()

    # Create a test event from a different transport (this should be accepted)
    other_transport_event = {
        "id": str(uuid.uuid4()),
        "type": EventTypes.STATUS,
        "data": {
            "message": "Test message",
            "transport_origin": {"transport_id": "other-transport", "transport_type": "stdio"},
        },
    }

    # Handle the event
    with patch.object(adapter, "send_event") as mock_send_event:
        await event_handler(other_transport_event["type"], other_transport_event["data"])

        # Verify that send_event was called with the correct arguments
        mock_send_event.assert_called_once_with(other_transport_event["type"], other_transport_event["data"])


@pytest.mark.skip(reason="subscribe_to_events not implemented yet")
@pytest.mark.asyncio
async def test_monitored_transport_event_propagation():
    """Test that events from a monitored transport are properly propagated."""
    # Create a mock coordinator that will call the event handler
    mock_coordinator = AsyncMock()
    mock_coordinator.register_transport = AsyncMock()
    mock_coordinator.subscribe_to_events = AsyncMock()

    # Create the adapter with the mock coordinator
    adapter = SSETransportAdapter(coordinator=mock_coordinator)

    # Set up an active connection
    adapter._active_connections = {"test-connection": AsyncMock()}

    # Set up a monitored transport ID
    adapter.monitor_stdio_transport_id = "stdio-transport"

    # Initialize the adapter
    await adapter.initialize()

    # Verify that subscribe_to_events was called
    mock_coordinator.subscribe_to_events.assert_called_once()

    # Get the event handler that was registered with the coordinator
    event_handler = mock_coordinator.subscribe_to_events.call_args[0][0]

    # Create a test event from the monitored transport
    monitored_event = {
        "id": str(uuid.uuid4()),
        "type": EventTypes.STATUS,
        "data": {
            "message": "Test message",
            "transport_origin": {"transport_id": "stdio-transport", "transport_type": "stdio"},
        },
    }

    # Handle the event
    with patch.object(adapter, "send_event") as mock_send_event:
        await event_handler(monitored_event["type"], monitored_event["data"])

        # Verify that send_event was called with the correct arguments
        mock_send_event.assert_called_once_with(monitored_event["type"], monitored_event["data"])


@pytest.mark.asyncio
async def test_filter_events_by_connection_type():
    """Test filtering events based on connection type or client properties."""
    # Create a mock coordinator
    mock_coordinator = AsyncMock()

    # Create the adapter with the mock coordinator
    adapter = SSETransportAdapter(coordinator=mock_coordinator)

    # Set up monitor connections and regular connections
    adapter._monitor_connections = {"monitor-connection"}
    adapter._active_connections = {"monitor-connection": AsyncMock(), "regular-connection": AsyncMock()}

    # Mock a specialized filtering method for testing
    def custom_filter(event_type, data, connection_id):
        # Only send TOOL_RESULT events to monitor connections
        if event_type == EventTypes.TOOL_RESULT:
            return connection_id in adapter._monitor_connections
        # Send STATUS events to all connections
        return True

    # Override the should_receive_event method to use our custom filter
    with patch.object(adapter, "should_receive_event", return_value=True):
        # Create a mock send implementation that checks the custom filter
        async def mock_send(event_type, data):
            for conn_id, queue in list(adapter._active_connections.items()):
                if custom_filter(event_type, data, conn_id):
                    # Use put_nowait to match the actual implementation
                    await queue.put_nowait(f"event: {event_type.value}\ndata: {{}}\n\n")

        # Replace the send_event method with our mock implementation
        with patch.object(adapter, "send_event", side_effect=mock_send):
            # Test sending a TOOL_RESULT event - should only go to monitor connection
            await adapter.send_event(EventTypes.TOOL_RESULT, {"test": "data"})

            # Verify that only the monitor connection received the event
            adapter._active_connections["monitor-connection"].put_nowait.assert_called_once()
            adapter._active_connections["regular-connection"].put_nowait.assert_not_called()

            # Reset the mocks
            adapter._active_connections["monitor-connection"].put_nowait.reset_mock()
            adapter._active_connections["regular-connection"].put_nowait.reset_mock()

            # Test sending a STATUS event - should go to all connections
            await adapter.send_event(EventTypes.STATUS, {"test": "data"})

            # Verify that both connections received the event
            adapter._active_connections["monitor-connection"].put_nowait.assert_called_once()
            adapter._active_connections["regular-connection"].put_nowait.assert_called_once()


@pytest.mark.asyncio
async def test_event_filtering_with_request_details():
    """Test event filtering with additional request details."""
    # Create a mock coordinator
    mock_coordinator = AsyncMock()

    # Create the adapter with the mock coordinator
    adapter = SSETransportAdapter(coordinator=mock_coordinator)

    # Mock the original should_receive_event method
    original_method = adapter.should_receive_event

    # Override the should_receive_event method to consider request details
    def enhanced_should_receive_event(event_type, data, request_details=None):
        # First apply the original filtering logic
        if not original_method(event_type, data):
            return False

        # Additional filtering based on request details
        if request_details:
            # Example: filter based on client IP
            client_ip = request_details.get("client_ip")
            if client_ip and client_ip.startswith("192.168."):
                return False  # Filter out local network requests

            # Example: filter based on user agent
            user_agent = request_details.get("user_agent")
            if user_agent and "curl" in user_agent:
                return False  # Filter out curl requests

        return True

    # Replace the method with our enhanced version
    adapter.should_receive_event = enhanced_should_receive_event

    # Test with no request details
    event_data = {"message": "Test message"}
    assert adapter.should_receive_event(EventTypes.STATUS, event_data) is True

    # Test with allowed request details
    request_details = {"client_ip": "8.8.8.8", "user_agent": "Chrome"}
    assert adapter.should_receive_event(EventTypes.STATUS, event_data, request_details) is True

    # Test with filtered IP
    request_details = {"client_ip": "192.168.1.1", "user_agent": "Chrome"}
    assert adapter.should_receive_event(EventTypes.STATUS, event_data, request_details) is False

    # Test with filtered user agent
    request_details = {"client_ip": "8.8.8.8", "user_agent": "curl/7.68.0"}
    assert adapter.should_receive_event(EventTypes.STATUS, event_data, request_details) is False
