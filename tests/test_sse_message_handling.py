"""
Tests for SSE message sending and receiving.

These tests verify that the SSE Transport Adapter properly handles
sending messages to clients and processing incoming data, including
formatting, queuing, and event typing.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aider_mcp_server.atoms.event_types import EventTypes
from aider_mcp_server.sse_transport_adapter import SSETransportAdapter


@pytest.fixture
def adapter():
    """Fixture providing a basic SSETransportAdapter instance."""
    adapter = SSETransportAdapter()
    return adapter


@pytest.mark.asyncio
async def test_send_event_basic(adapter):
    """Test that the adapter can send a basic event to a client."""
    # Set up a test connection
    test_queue = asyncio.Queue()
    adapter._active_connections = {"test-connection": test_queue}
    
    # Create a test event
    event_type = EventTypes.STATUS
    event_data = {"message": "Test message", "id": "test-id"}
    
    # Send the event
    await adapter.send_event(event_type, event_data)
    
    # Verify that the event was put in the queue
    assert not test_queue.empty()
    message = test_queue.get_nowait()
    
    # Verify the message format
    assert f"event: {event_type.value}" in message
    assert f"data: {json.dumps(event_data)}" in message


@pytest.mark.asyncio
async def test_send_event_multiple_clients(adapter):
    """Test that the adapter can send an event to multiple clients."""
    # Set up multiple test connections
    test_queues = {
        "connection1": asyncio.Queue(),
        "connection2": asyncio.Queue(),
        "connection3": asyncio.Queue()
    }
    adapter._active_connections = test_queues
    
    # Create a test event
    event_type = EventTypes.STATUS
    event_data = {"message": "Test message", "id": "test-id"}
    
    # Send the event
    await adapter.send_event(event_type, event_data)
    
    # Verify that the event was put in each queue
    for queue_name, queue in test_queues.items():
        assert not queue.empty(), f"Queue {queue_name} is empty"
        message = queue.get_nowait()
        
        # Verify the message format
        assert f"event: {event_type.value}" in message
        assert f"data: {json.dumps(event_data)}" in message


@pytest.mark.asyncio
async def test_send_event_no_clients(adapter):
    """Test that the adapter handles sending an event when there are no clients."""
    # Ensure there are no active connections
    adapter._active_connections = {}
    
    # Create a test event
    event_type = EventTypes.STATUS
    event_data = {"message": "Test message", "id": "test-id"}
    
    # Send the event - should not raise an exception
    try:
        await adapter.send_event(event_type, event_data)
    except Exception as e:
        pytest.fail(f"send_event raised an exception: {e}")


@pytest.mark.asyncio
async def test_send_event_queue_full():
    """Test that the adapter handles sending an event to a full queue."""
    # Create the adapter
    adapter = SSETransportAdapter()
    
    # Create a queue with a small max size
    test_queue = asyncio.Queue(maxsize=1)
    # Fill the queue
    await test_queue.put("existing-message")
    
    # Set up the test connection with the full queue
    adapter._active_connections = {"test-connection": test_queue}
    
    # Create a test event
    event_type = EventTypes.STATUS
    event_data = {"message": "Test message", "id": "test-id"}
    
    # Send the event
    with patch.object(adapter, "logger") as mock_logger:
        await adapter.send_event(event_type, event_data)
        
        # Verify that a warning was logged
        mock_logger.warning.assert_called_once()
        log_message = mock_logger.warning.call_args[0][0]
        assert "Queue full" in log_message
        assert "test-connection" in log_message
    
    # Verify that the original message is still in the queue
    assert not test_queue.empty()
    message = test_queue.get_nowait()
    assert message == "existing-message"


@pytest.mark.asyncio
async def test_send_event_removed_connection(adapter):
    """Test that the adapter handles sending an event to a removed connection."""
    # Set up a connection that will be removed
    adapter._active_connections = {"test-connection": asyncio.Queue()}
    
    # Create a test event
    event_type = EventTypes.STATUS
    event_data = {"message": "Test message", "id": "test-id"}
    
    # Mock the connections dict to simulate concurrent removal
    original_get = adapter._active_connections.get
    call_count = 0
    
    def mock_get(key, default=None):
        nonlocal call_count
        call_count += 1
        if call_count > 1:  # Return the queue for the first call, then None
            return None
        return original_get(key, default)
    
    adapter._active_connections.get = mock_get
    
    # Send the event
    with patch.object(adapter, "logger") as mock_logger:
        await adapter.send_event(event_type, event_data)
        
        # Verify that a debug message was logged
        mock_logger.debug.assert_called_once()
        log_message = mock_logger.debug.call_args[0][0]
        assert "Connection test-connection removed" in log_message


@pytest.mark.asyncio
async def test_send_event_queue_error(adapter):
    """Test that the adapter handles errors when putting events in queues."""
    # Set up a test connection
    mock_queue = MagicMock()
    mock_queue.put_nowait = MagicMock(side_effect=Exception("Test exception"))
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
async def test_send_different_event_types(adapter):
    """Test that the adapter can send different types of events."""
    # Set up a test connection
    test_queue = asyncio.Queue()
    adapter._active_connections = {"test-connection": test_queue}
    
    # Define test events for different event types
    test_events = [
        (EventTypes.STATUS, {"message": "Status message", "id": "status-id"}),
        (EventTypes.PROGRESS, {"progress": 0.5, "id": "progress-id"}),
        (EventTypes.TOOL_RESULT, {"result": "Tool result", "id": "tool-id"}),
        (EventTypes.HEARTBEAT, {"timestamp": 123456789, "id": "heartbeat-id"})
    ]
    
    # Send each event
    for event_type, event_data in test_events:
        await adapter.send_event(event_type, event_data)
        
        # Verify that the event was put in the queue
        assert not test_queue.empty()
        message = test_queue.get_nowait()
        
        # Verify the message format
        assert f"event: {event_type.value}" in message
        assert f"data: {json.dumps(event_data)}" in message


@pytest.mark.asyncio
async def test_send_event_with_complex_data(adapter):
    """Test that the adapter can send events with complex nested data structures."""
    # Set up a test connection
    test_queue = asyncio.Queue()
    adapter._active_connections = {"test-connection": test_queue}
    
    # Create a test event with a complex data structure
    event_type = EventTypes.TOOL_RESULT
    event_data = {
        "id": "complex-id",
        "result": {
            "name": "Test Tool",
            "status": "success",
            "details": {
                "runtime": 1.23,
                "outputs": [
                    {"type": "text", "content": "Output 1"},
                    {"type": "code", "content": "print('Hello World')"},
                    {"type": "file", "path": "/tmp/test.txt"}
                ],
                "metadata": {
                    "version": "1.0.0",
                    "timestamp": 123456789
                }
            }
        }
    }
    
    # Send the event
    await adapter.send_event(event_type, event_data)
    
    # Verify that the event was put in the queue
    assert not test_queue.empty()
    message = test_queue.get_nowait()
    
    # Verify the message format
    assert f"event: {event_type.value}" in message
    
    # Extract the JSON data and parse it
    data_line = [line for line in message.split("\n") if line.startswith("data:")][0]
    data_json = data_line[len("data: "):]
    parsed_data = json.loads(data_json)
    
    # Verify that the complex data structure was preserved
    assert parsed_data["id"] == "complex-id"
    assert parsed_data["result"]["name"] == "Test Tool"
    assert parsed_data["result"]["status"] == "success"
    assert parsed_data["result"]["details"]["runtime"] == 1.23
    assert len(parsed_data["result"]["details"]["outputs"]) == 3
    assert parsed_data["result"]["details"]["metadata"]["version"] == "1.0.0"


@pytest.mark.asyncio
async def test_handle_message_request(adapter):
    """Test that the adapter handles incoming message requests."""
    # Set up a mock request
    mock_request = MagicMock()
    
    # Handle the message request
    result = await adapter.handle_message_request(mock_request)
    
    # Verify that an error response was returned (SSE is unidirectional)
    assert "error" in result
    assert "SSE transport does not support incoming messages" in result["error"]


@pytest.mark.asyncio
async def test_logging_during_send_event(adapter):
    """Test that the adapter logs appropriate information during send_event."""
    # Set up a test connection
    test_queue = asyncio.Queue()
    adapter._active_connections = {"test-connection": test_queue}
    
    # Create a test event
    event_type = EventTypes.PROGRESS
    event_data = {"progress": 0.5, "id": "progress-id"}
    
    # Send the event
    with patch.object(adapter, "logger") as mock_logger:
        await adapter.send_event(event_type, event_data)
        
        # Verify that a debug log was made for progress events
        mock_logger.debug.assert_called_once()
        log_message = mock_logger.debug.call_args[0][0]
        assert "Broadcasting progress event" in log_message
        assert str(event_data) in log_message


@pytest.mark.asyncio
async def test_sse_message_format(adapter):
    """Test that the adapter formats SSE messages correctly."""
    # Set up a test connection
    test_queue = asyncio.Queue()
    adapter._active_connections = {"test-connection": test_queue}
    
    # Create a test event
    event_type = EventTypes.STATUS
    event_data = {"message": "Test message", "id": "test-id"}
    
    # Send the event
    await adapter.send_event(event_type, event_data)
    
    # Get the message from the queue
    message = test_queue.get_nowait()
    
    # Verify the SSE format
    lines = message.split("\n")
    assert len(lines) >= 3
    assert lines[0] == f"event: {event_type.value}"
    assert lines[1].startswith("data: ")
    assert lines[2] == ""  # Empty line after data
    assert lines[3] == ""  # Empty line terminating the event
    
    # Parse the data JSON
    data_json = lines[1][len("data: "):]
    parsed_data = json.loads(data_json)
    assert parsed_data == event_data


@pytest.mark.asyncio
async def test_message_ordering(adapter):
    """Test that the adapter preserves message order in the queue."""
    # Set up a test connection
    test_queue = asyncio.Queue()
    adapter._active_connections = {"test-connection": test_queue}
    
    # Create multiple test events
    events = [
        (EventTypes.STATUS, {"message": "Status 1", "id": "status-1"}),
        (EventTypes.STATUS, {"message": "Status 2", "id": "status-2"}),
        (EventTypes.PROGRESS, {"progress": 0.3, "id": "progress-1"}),
        (EventTypes.PROGRESS, {"progress": 0.6, "id": "progress-2"}),
        (EventTypes.TOOL_RESULT, {"result": "Result", "id": "tool-1"})
    ]
    
    # Send all events
    for event_type, event_data in events:
        await adapter.send_event(event_type, event_data)
    
    # Verify that the events were put in the queue in the correct order
    received_events = []
    while not test_queue.empty():
        message = test_queue.get_nowait()
        
        # Extract the event type and data
        event_line = [line for line in message.split("\n") if line.startswith("event:")][0]
        data_line = [line for line in message.split("\n") if line.startswith("data:")][0]
        
        event_type_value = event_line[len("event: "):]
        data_json = data_line[len("data: "):]
        parsed_data = json.loads(data_json)
        
        received_events.append((event_type_value, parsed_data["id"]))
    
    # Verify the order
    expected_order = [
        (events[0][0].value, events[0][1]["id"]),
        (events[1][0].value, events[1][1]["id"]),
        (events[2][0].value, events[2][1]["id"]),
        (events[3][0].value, events[3][1]["id"]),
        (events[4][0].value, events[4][1]["id"])
    ]
    assert received_events == expected_order