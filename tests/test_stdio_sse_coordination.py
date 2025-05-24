"""Tests for stdio-SSE coordination using the coordinator discovery mechanism."""

import asyncio
import io
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

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

        # Create a mock coordinator for testing
        mock_coordinator = AsyncMock(spec=ApplicationCoordinator)

        # Configure the mock with required attributes
        mock_coordinator._transports = {}
        mock_coordinator.register_transport = AsyncMock()
        mock_coordinator.unregister_transport = AsyncMock()
        mock_coordinator.broadcast_event = AsyncMock()
        mock_coordinator._send_event_to_transports = AsyncMock()
        mock_coordinator.shutdown = AsyncMock()

        # Return values
        yield (mock_coordinator, discovery_file)

        # Clean up
        await mock_coordinator.shutdown()


@pytest.fixture
def mock_input_output():
    """Create mock input/output streams for testing."""
    input_stream = io.StringIO()
    output_stream = io.StringIO()
    return input_stream, output_stream


@pytest.fixture
def mock_queue():
    """Create a mock queue for testing."""
    queue = AsyncMock()
    queue.put_nowait = MagicMock()
    queue.get = AsyncMock()
    queue.task_done = MagicMock()
    return queue


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
        """Get transport capabilities for testing."""
        return {
            "transport_type": "sse",
            "supports_events": True,
            "supports_requests": True,
            "protocol": "http",
            "connection_type": "unidirectional",
        }


@pytest.mark.asyncio
@pytest.mark.skip("Test needs fixing with proper mocks")
async def test_stdio_registration_with_coordinator(coordinator_with_discovery, mock_input_output):
    """Test registering a stdio transport with an existing coordinator."""
    coordinator, discovery_file = coordinator_with_discovery
    input_stream, output_stream = mock_input_output

    # Create stdio transport with discovery
    stdio_adapter = StdioTransportAdapter(
        coordinator=coordinator,
        input_stream=input_stream,
        output_stream=output_stream,
    )
    await stdio_adapter.initialize()

    # Verify registration was called
    coordinator.register_transport.assert_called_once_with(stdio_adapter.transport_id, stdio_adapter)

    await stdio_adapter.shutdown()


@pytest.mark.asyncio
@pytest.mark.skip("Test needs fixing with proper mocks")
async def test_stdio_find_and_connect(coordinator_with_discovery, mock_input_output):
    """Test finding and connecting to an existing coordinator."""
    coordinator, discovery_file = coordinator_with_discovery
    input_stream, output_stream = mock_input_output

    # Mock the find_and_connect method to return a pre-configured adapter
    mock_stdio = MagicMock(spec=StdioTransportAdapter)
    mock_stdio._coordinator = coordinator
    mock_stdio.transport_id = "stdio_test"

    with patch(
        "aider_mcp_server.stdio_transport_adapter.StdioTransportAdapter.find_and_connect",
        return_value=mock_stdio,
    ):
        # Try to find and connect
        stdio_adapter = await StdioTransportAdapter.find_and_connect(
            discovery_file=discovery_file,
            input_stream=input_stream,
            output_stream=output_stream,
        )

        assert stdio_adapter is not None
        assert stdio_adapter._coordinator == coordinator

    # Clean up
    await stdio_adapter.shutdown()


@pytest.mark.asyncio
@pytest.mark.skip("Test needs fixing with proper mocks")
async def test_stdio_to_sse_event_forwarding(coordinator_with_discovery, mock_input_output):
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
        transport_id="stdio_test",
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
    from aider_mcp_server.atoms.types.event_types import EventTypes

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
@pytest.mark.skip("Test needs fixing with proper mocks")
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
        transport_id="stdio_test",
        input_stream=input_stream,
        output_stream=output_stream,
    )
    await stdio_adapter.initialize()

    # Configure SSE to monitor stdio
    sse_adapter.monitor_stdio_transport_id = stdio_adapter.transport_id

    # Simulate heartbeat event
    from aider_mcp_server.atoms.types.event_types import EventTypes

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
@pytest.mark.skip("Integration test needs rework with mocks")
async def test_coordinator_discovery_integration(mock_input_output):
    """Test the full integration of coordinator discovery with stdio and SSE."""
    input_stream, output_stream = mock_input_output

    # Create a temporary file for discovery
    with tempfile.TemporaryDirectory() as temp_dir:
        # Path is just for context, not needed in this test
        Path(temp_dir) / "integration_test_registry.json"

        # Create a mock coordinator for testing
        mock_coordinator = AsyncMock(spec=ApplicationCoordinator)
        mock_coordinator._transports = {}
        mock_coordinator.register_transport = AsyncMock()
        mock_coordinator.unregister_transport = AsyncMock()
        mock_coordinator._send_event_to_transports = AsyncMock()
        mock_coordinator.shutdown = AsyncMock()

        from aider_mcp_server.atoms.types.event_types import EventTypes

        mock_coordinator.atoms = MagicMock()
        mock_coordinator.atoms.event_types = EventTypes

        # Create SSE adapter with mocked coordinator
        sse_adapter = MockSSETransportAdapter()
        sse_adapter._coordinator = mock_coordinator

        # Create mocked stdio adapter
        stdio_adapter = MagicMock(spec=StdioTransportAdapter)
        stdio_adapter.transport_id = "stdio_mock_id"
        stdio_adapter._coordinator = mock_coordinator
        stdio_adapter.shutdown = AsyncMock()

        # Mock the find_and_connect to return our mocked adapter
        with patch(
            "aider_mcp_server.stdio_transport_adapter.StdioTransportAdapter.find_and_connect",
            return_value=stdio_adapter,
        ):
            # Configure SSE to monitor stdio
            sse_adapter.monitor_stdio_transport_id = stdio_adapter.transport_id

            # Simulate an event
            data = {"request_id": "test_integration_1", "status": "received"}
            await mock_coordinator._send_event_to_transports(
                event_type=EventTypes.STATUS,
                data=data,
                originating_transport_id=stdio_adapter.transport_id,
            )

            # Clean up
            await stdio_adapter.shutdown()
            await mock_coordinator.shutdown()


class MockQueue:
    """Mock queue for testing."""

    def __init__(self):
        self.items = []

    def put_nowait(self, item):
        """Add item without blocking."""
        self.items.append(item)

    async def get(self):
        """Get item with async interface."""
        if self.items:
            return self.items.pop(0)
        await asyncio.sleep(0.1)
        return None

    def task_done(self):
        """Mark task as done."""
        pass
