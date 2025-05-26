"""
Comprehensive tests for stdio-SSE coordination functionality.

This module consolidates tests for cross-transport event relay, coordination,
and discovery mechanisms between STDIO and SSE transports.
"""

from io import StringIO
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from aider_mcp_server.atoms.types.event_types import EventTypes
from aider_mcp_server.molecules.transport.discovery import CoordinatorInfo
from ...pages.application.coordinator import ApplicationCoordinator
from aider_mcp_server.organisms.transports.sse.sse_transport_adapter import SSETransportAdapter
from aider_mcp_server.organisms.transports.stdio.stdio_transport_adapter import (
    AIDER_EVENT_TYPES_TO_RELAY,
    StdioTransportAdapter,
)


@pytest_asyncio.fixture
async def mock_coordinator():
    """Create a mock ApplicationCoordinator."""
    coordinator = AsyncMock(spec=ApplicationCoordinator)
    coordinator.subscribe_to_event_type = AsyncMock()
    coordinator.initialize = AsyncMock()
    coordinator.register_transport = AsyncMock()
    coordinator.register_transport_adapter = AsyncMock()
    coordinator.unregister_transport = AsyncMock()
    coordinator.broadcast_event = AsyncMock()
    coordinator._send_event_to_transports = AsyncMock()
    coordinator.shutdown = AsyncMock()
    coordinator._transports = {}
    return coordinator


@pytest_asyncio.fixture
async def mock_streaming_coordinator():
    """Create a mock streaming coordinator for discovery."""
    return CoordinatorInfo(
        coordinator_id="streaming_coord_1",
        host="localhost",
        port=8080,
        transport_type="sse",
        streaming_capabilities={
            "sse_endpoints": {
                "aider_events": "/events/aider",
                "error_events": "/events/errors",
                "progress_events": "/events/progress",
                "health_check": "/health",
            },
            "supported_event_types": ["aider", "errors", "progress"],
        },
    )


@pytest_asyncio.fixture
async def mock_discovery(mock_streaming_coordinator):
    """Create a mock CoordinatorDiscovery that returns streaming coordinators."""
    discovery = AsyncMock()
    discovery.find_streaming_coordinators = AsyncMock(return_value=[mock_streaming_coordinator])
    discovery.shutdown = AsyncMock()
    return discovery


@pytest_asyncio.fixture
async def stdio_adapter(tmp_path, mock_coordinator):
    """Create a StdioTransportAdapter for testing."""
    discovery_file = tmp_path / "test_discovery.json"
    input_stream = StringIO()
    output_stream = StringIO()

    adapter = StdioTransportAdapter(
        coordinator=mock_coordinator,
        input_stream=input_stream,
        output_stream=output_stream,
        discovery_file=discovery_file,
    )

    yield adapter

    # Cleanup
    try:
        await adapter.shutdown()
    except Exception:  # noqa: S110
        # Ignore shutdown errors in test cleanup - this is intentional for test teardown
        pass


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

    def get_capabilities(self):
        """Get transport capabilities for testing."""
        return {
            "transport_type": "sse",
            "supports_events": True,
            "supports_requests": True,
            "protocol": "http",
            "connection_type": "unidirectional",
        }


class TestCrossTransportEventRelay:
    """Test suite for cross-transport event relay functionality."""

    @pytest.mark.asyncio
    async def test_aider_event_subscription(self, stdio_adapter, mock_coordinator):
        """Test that STDIO adapter subscribes to AIDER events on local coordinator."""
        stdio_adapter._coordinator = mock_coordinator

        await stdio_adapter._subscribe_to_aider_events()

        # Verify subscription to all AIDER event types
        assert mock_coordinator.subscribe_to_event_type.call_count == len(AIDER_EVENT_TYPES_TO_RELAY)

        # Check that all expected event types were subscribed to
        subscribed_events = set()
        for call in mock_coordinator.subscribe_to_event_type.call_args_list:
            args, kwargs = call
            subscribed_events.add(args[1])  # Second argument is event_type

        assert subscribed_events == AIDER_EVENT_TYPES_TO_RELAY


class TestStdioSSECoordination:
    """Test suite for stdio-SSE coordination and integration."""

    @pytest.mark.asyncio
    async def test_stdio_transport_creation(self, mock_coordinator):
        """Test creating a stdio transport adapter."""
        input_stream = StringIO()
        output_stream = StringIO()

        # Create stdio transport
        stdio_adapter = StdioTransportAdapter(
            coordinator=mock_coordinator,
            input_stream=input_stream,
            output_stream=output_stream,
        )

        # Verify basic properties
        assert stdio_adapter._coordinator == mock_coordinator
        assert stdio_adapter._input == input_stream
        assert stdio_adapter._output == output_stream
        assert stdio_adapter.transport_id.startswith("stdio_")

    @pytest.mark.asyncio
    async def test_event_relay_functionality(self, mock_coordinator):
        """Test that AIDER events are properly handled."""
        input_stream = StringIO()
        output_stream = StringIO()

        # Create stdio adapter
        stdio_adapter = StdioTransportAdapter(
            coordinator=mock_coordinator,
            input_stream=input_stream,
            output_stream=output_stream,
        )

        # Test that AIDER event types are properly defined
        assert EventTypes.AIDER_SESSION_STARTED in AIDER_EVENT_TYPES_TO_RELAY
        assert EventTypes.AIDER_RATE_LIMIT_DETECTED in AIDER_EVENT_TYPES_TO_RELAY

        # Test event sending without actual network calls
        await stdio_adapter.send_event(EventTypes.STATUS, {"message": "test"})

        # Should complete without error (no actual assertions since we're not testing real behavior)


class TestStdioTransportAdapterInitialization:
    """Test suite for STDIO transport adapter initialization with discovery."""

    @pytest.mark.asyncio
    async def test_initialization_with_discovery_file(self, tmp_path):
        """Test adapter initialization with discovery file."""
        discovery_file = tmp_path / "test_discovery.json"

        adapter = StdioTransportAdapter(
            input_stream=StringIO(),
            output_stream=StringIO(),
            discovery_file=discovery_file,
        )

        # Verify discovery was initialized
        assert adapter._discovery is not None
        assert adapter._discovery_file == discovery_file

    @pytest.mark.asyncio
    async def test_initialization_without_discovery_file(self):
        """Test adapter initialization without discovery file."""
        adapter = StdioTransportAdapter(
            input_stream=StringIO(),
            output_stream=StringIO(),
        )

        # Verify discovery was not initialized
        assert adapter._discovery is None
        assert adapter._discovery_file is None

        # Verify basic adapter properties
        assert adapter.transport_id.startswith("stdio_")
        assert adapter._input is not None
        assert adapter._output is not None


if __name__ == "__main__":
    # For direct execution
    pytest.main([__file__])
