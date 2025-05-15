"""Tests for stdio-SSE coordination using the coordinator discovery mechanism."""

import asyncio
import io
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest
import pytest_asyncio

from aider_mcp_server.sse_transport_adapter import SSETransportAdapter
from aider_mcp_server.stdio_transport_adapter import StdioTransportAdapter
from aider_mcp_server.transport_coordinator import ApplicationCoordinator


@pytest_asyncio.fixture
async def coordinator_with_discovery():
    """Create a coordinator with discovery enabled for testing."""
    # Create a temporary file for the coordinator registry
    with tempfile.TemporaryDirectory() as temp_dir:
        discovery_file = Path(temp_dir) / "test_coordinator_registry.json"

        # Get singleton coordinator instance and reset it for testing
        coordinator = await ApplicationCoordinator.getInstance()
        ApplicationCoordinator._instance = None
        ApplicationCoordinator._initialized = False

        # Create a fresh instance with discovery enabled
        coordinator = await ApplicationCoordinator.getInstance()
        await coordinator._initialize_coordinator(
            host="localhost",
            port=8000,
            register_in_discovery=True,
            discovery_file=discovery_file,
        )

        # Return values
        yield (coordinator, discovery_file)

        # Clean up
        await coordinator.shutdown()


@pytest.fixture
def mock_input_output():
    """Create mock input and output streams for testing."""
    input_stream = io.StringIO()
    output_stream = io.StringIO()
    return input_stream, output_stream


class MockQueue:
    """Mock SSE client queue for testing."""

    def __init__(self):
        self.messages: List[Dict[str, Any]] = []

    async def put(self, message):
        """Add a message to the queue."""
        self.messages.append(message)


class MockSSETransportAdapter(SSETransportAdapter):
    """Mock SSE transport adapter for testing."""

    def __init__(self):
        # Skip initialization to avoid actual web server
        self.transport_id = "sse_test"
        self.transport_type = "sse"
        self._coordinator = None
        self.logger = MagicMock()
        self._client_queues = {}
        self.monitor_stdio_transport_id = None
        self.sent_events = []  # Initialize empty list for sent events

    async def send_event(self, event_type, data):
        """Mock send_event to capture events."""
        self.sent_events.append((event_type, data))

    def should_receive_event(self, event_type, data, request_details=None):
        """Determine if this adapter should receive the event.
        For testing, always return True to receive all events.
        """
        return True

    def get_capabilities(self):
        """Return all event types as capabilities."""
        from aider_mcp_server.atoms.event_types import EventTypes

        return {
            EventTypes.STATUS,
            EventTypes.PROGRESS,
            EventTypes.TOOL_RESULT,
            EventTypes.HEARTBEAT,
        }


@pytest.mark.asyncio
async def test_stdio_registration_with_coordinator(
    coordinator_with_discovery, mock_input_output
):
    """Test registering a stdio transport with an existing coordinator."""
    coordinator, discovery_file = coordinator_with_discovery
    input_stream, output_stream = mock_input_output

    # Create stdio transport with discovery
    stdio_adapter = StdioTransportAdapter(
        coordinator=coordinator,
        input_stream=input_stream,
        output_stream=output_stream,
        discovery_file=discovery_file,
    )

    # Initialize and verify registration
    await stdio_adapter.initialize()

    # Check that stdio is registered with coordinator
    assert stdio_adapter.transport_id in coordinator._transports

    # Clean up
    await stdio_adapter.shutdown()


@pytest.mark.asyncio
async def test_stdio_find_and_connect(coordinator_with_discovery, mock_input_output):
    """Test finding and connecting to an existing coordinator."""
    coordinator, discovery_file = coordinator_with_discovery
    input_stream, output_stream = mock_input_output

    # Use the find_and_connect factory method
    stdio_adapter = await StdioTransportAdapter.find_and_connect(
        discovery_file=discovery_file,
        input_stream=input_stream,
        output_stream=output_stream,
    )

    # Verify adapter was created and connected
    assert stdio_adapter is not None
    assert stdio_adapter._coordinator is not None

    # Check that it's registered with the coordinator
    assert stdio_adapter.transport_id in coordinator._transports

    # Clean up
    await stdio_adapter.shutdown()


@pytest.mark.asyncio
async def test_stdio_to_sse_event_forwarding(
    coordinator_with_discovery, mock_input_output
):
    """Test that events from stdio are properly forwarded to SSE."""
    coordinator, discovery_file = coordinator_with_discovery
    input_stream, output_stream = mock_input_output

    # Create mock SSE adapter
    sse_adapter = MockSSETransportAdapter()

    # Set up monitor for stdio
    sse_adapter._coordinator = coordinator
    await coordinator.register_transport(sse_adapter.transport_id, sse_adapter)

    # Create stdio adapter
    stdio_adapter = StdioTransportAdapter(
        coordinator=coordinator,
        input_stream=input_stream,
        output_stream=output_stream,
    )
    await stdio_adapter.initialize()

    # Configure SSE to monitor stdio
    sse_adapter.monitor_stdio_transport_id = stdio_adapter.transport_id

    # Send a test event through coordinator from stdio
    test_data = {
        "request_id": "test_request_1",
        "message": "Test message",
        "details": {"test": True},
    }

    # Simulate event from stdio
    from aider_mcp_server.atoms.event_types import EventTypes

    await coordinator._send_event_to_transports(
        event_type=EventTypes.STATUS,
        data=test_data,
        originating_transport_id=stdio_adapter.transport_id,
    )

    # Give some time for the event to be processed
    await asyncio.sleep(0.1)

    # Check that the event was received by SSE
    assert len(sse_adapter.sent_events) > 0

    # Check event data was forwarded correctly
    for _event_type, data in sse_adapter.sent_events:
        if data.get("request_id") == "test_request_1":
            assert data.get("message") == "Test message"
            break
    else:
        pytest.fail("Event not forwarded to SSE adapter")

    # Clean up
    await stdio_adapter.shutdown()
    await coordinator.unregister_transport(sse_adapter.transport_id)


@pytest.mark.asyncio
async def test_health_check_monitoring(coordinator_with_discovery, mock_input_output):
    """Test that SSE can monitor the health status of a stdio transport."""
    coordinator, discovery_file = coordinator_with_discovery
    input_stream, output_stream = mock_input_output

    # Create mock SSE adapter
    sse_adapter = MockSSETransportAdapter()
    sse_adapter._client_queues = {"test_client": MockQueue()}

    # Register SSE adapter
    sse_adapter._coordinator = coordinator
    await coordinator.register_transport(sse_adapter.transport_id, sse_adapter)

    # Create stdio adapter
    stdio_adapter = StdioTransportAdapter(
        coordinator=coordinator,
        input_stream=input_stream,
        output_stream=output_stream,
    )
    await stdio_adapter.initialize()

    # Configure SSE to monitor stdio
    sse_adapter.monitor_stdio_transport_id = stdio_adapter.transport_id

    # Simulate heartbeat event
    from aider_mcp_server.atoms.event_types import EventTypes

    await coordinator.broadcast_event(
        event_type=EventTypes.HEARTBEAT,
        data={
            "transport_id": stdio_adapter.transport_id,
            "timestamp": 12345,
        },
    )

    # Give some time for the event to be processed
    await asyncio.sleep(0.1)

    # Verify SSE received the heartbeat
    assert len(sse_adapter.sent_events) > 0

    # Check for heartbeat event
    for event_type, data in sse_adapter.sent_events:
        if event_type == EventTypes.HEARTBEAT:
            assert data.get("transport_id") == stdio_adapter.transport_id
            break
    else:
        pytest.fail("Heartbeat event not forwarded to SSE adapter")

    # Clean up
    await stdio_adapter.shutdown()
    await coordinator.unregister_transport(sse_adapter.transport_id)


@pytest.mark.asyncio
async def test_coordinator_discovery_integration(mock_input_output):
    """Test the full integration of coordinator discovery with stdio and SSE."""
    input_stream, output_stream = mock_input_output

    # Create a temporary file for discovery
    with tempfile.TemporaryDirectory() as temp_dir:
        discovery_file = Path(temp_dir) / "integration_test_registry.json"

        # Clear any existing coordinator instance
        ApplicationCoordinator._instance = None
        ApplicationCoordinator._initialized = False

        # 1. First, create and register an SSE coordinator
        sse_coordinator = await ApplicationCoordinator.getInstance()
        await sse_coordinator._initialize_coordinator(
            host="127.0.0.1",
            port=8001,
            register_in_discovery=True,
            discovery_file=discovery_file,
        )

        # 2. Create SSE adapter (mock for testing)
        sse_adapter = MockSSETransportAdapter()
        sse_adapter._coordinator = sse_coordinator
        await sse_coordinator.register_transport(sse_adapter.transport_id, sse_adapter)

        # 3. Now create a stdio adapter that discovers and connects to the coordinator
        stdio_adapter = await StdioTransportAdapter.find_and_connect(
            discovery_file=discovery_file,
            input_stream=input_stream,
            output_stream=output_stream,
        )

        # Verify stdio adapter was created and found the coordinator
        assert stdio_adapter is not None
        assert stdio_adapter._coordinator is not None
        assert stdio_adapter.transport_id in sse_coordinator._transports

        # Configure SSE to monitor stdio
        sse_adapter.monitor_stdio_transport_id = stdio_adapter.transport_id

        # 4. Test communication from stdio to SSE
        # Write a message to stdin
        tool_request = {
            "request_id": "test_integration_1",
            "name": "test_tool",
            "parameters": {"param1": "value1"},
        }
        input_stream.write(json.dumps(tool_request) + "\n")
        input_stream.seek(0)  # Reset position for reading

        # Start listening on stdio
        await stdio_adapter.start_listening()

        # Give some time for message processing
        await asyncio.sleep(0.1)

        # Simulate an event from stdio
        await sse_coordinator._send_event_to_transports(
            event_type=sse_coordinator.atoms.event_types.EventTypes.STATUS,
            data={"request_id": "test_integration_1", "status": "received"},
            originating_transport_id=stdio_adapter.transport_id,
        )

        # Check that event was received by SSE
        assert len(sse_adapter.sent_events) > 0

        # Clean up
        await stdio_adapter.shutdown()
        await sse_coordinator.shutdown()
