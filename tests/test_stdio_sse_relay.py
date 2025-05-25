"""
Tests for Phase 2.3 Cross-Transport Event Relay functionality.

Tests the STDIO transport adapter's ability to discover streaming coordinators
and relay AIDER events to SSE endpoints.
"""

import asyncio
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
import pytest_asyncio

from aider_mcp_server.atoms.types.event_types import EventTypes
from aider_mcp_server.molecules.transport.discovery import CoordinatorInfo
from aider_mcp_server.organisms.transports.stdio.stdio_transport_adapter import (
    AIDER_EVENT_TYPES_TO_RELAY,
    StdioTransportAdapter,
)


@pytest_asyncio.fixture
async def mock_coordinator():
    """Create a mock ApplicationCoordinator."""
    coordinator = AsyncMock()
    coordinator.subscribe_to_event_type = AsyncMock()
    coordinator.initialize = AsyncMock()
    coordinator.register_transport_adapter = AsyncMock()
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
    except Exception:
        pass


class TestCrossTransportEventRelay:
    """Test suite for cross-transport event relay functionality."""

    @pytest.mark.asyncio
    async def test_streaming_coordinator_discovery(self, stdio_adapter, mock_discovery, mock_streaming_coordinator):
        """Test that STDIO adapter discovers streaming coordinators."""
        with patch.object(stdio_adapter, "_discovery", mock_discovery):
            await stdio_adapter._auto_discover_coordinator()

            # Verify discovery was called
            mock_discovery.find_streaming_coordinators.assert_called_once()

            # Check that streaming coordinators were stored
            assert len(stdio_adapter._streaming_coordinators) == 0  # No discovery file was actually set up
            
            # Manually set up the discovery and test
            stdio_adapter._discovery = mock_discovery
            await stdio_adapter._auto_discover_coordinator()
            
            # After setting discovery, should find streaming coordinators
            assert stdio_adapter._streaming_coordinators == [mock_streaming_coordinator]

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

    @pytest.mark.asyncio
    async def test_aider_event_relay_to_sse(self, stdio_adapter, mock_streaming_coordinator):
        """Test that AIDER events are relayed to discovered SSE endpoints."""
        # Set up adapter with streaming coordinator
        stdio_adapter._streaming_coordinators = [mock_streaming_coordinator]

        # Mock aiohttp ClientSession
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.raise_for_status = MagicMock()
        mock_session.post.return_value.__aenter__.return_value = mock_response
        stdio_adapter._client_session = mock_session

        # Test sending an AIDER event
        test_event = EventTypes.AIDER_RATE_LIMIT_DETECTED
        test_data = {"provider": "openai", "attempt": 1}

        await stdio_adapter.send_event(test_event, test_data)

        # Verify HTTP POST was made to SSE endpoint
        mock_session.post.assert_called_once()
        call_args = mock_session.post.call_args
        assert call_args[0][0] == "http://localhost:8080/events/aider"  # URL
        assert call_args[1]["json"]["event"] == test_event.value
        assert call_args[1]["json"]["data"] == test_data

    @pytest.mark.asyncio
    async def test_non_aider_event_not_relayed(self, stdio_adapter, mock_streaming_coordinator):
        """Test that non-AIDER events are not relayed to SSE endpoints."""
        # Set up adapter with streaming coordinator
        stdio_adapter._streaming_coordinators = [mock_streaming_coordinator]

        # Mock aiohttp ClientSession
        mock_session = AsyncMock()
        stdio_adapter._client_session = mock_session

        # Test sending a non-AIDER event
        test_event = EventTypes.STATUS
        test_data = {"status": "ready"}

        await stdio_adapter.send_event(test_event, test_data)

        # Verify no HTTP POST was made (since it's not an AIDER event)
        mock_session.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_http_client_session_lifecycle(self, stdio_adapter, mock_streaming_coordinator, mock_discovery):
        """Test aiohttp ClientSession creation and cleanup."""
        stdio_adapter._discovery = mock_discovery
        stdio_adapter._streaming_coordinators = [mock_streaming_coordinator]

        # Initialize adapter - should create client session
        await stdio_adapter.initialize()

        # Verify client session was created
        assert stdio_adapter._client_session is not None
        assert isinstance(stdio_adapter._client_session, aiohttp.ClientSession)

        # Shutdown adapter - should close client session
        await stdio_adapter.shutdown()

        # Verify client session was closed and cleared
        assert stdio_adapter._client_session is None

    @pytest.mark.asyncio
    async def test_relay_failure_handling(self, stdio_adapter, mock_streaming_coordinator):
        """Test graceful handling when streaming endpoints are unavailable."""
        # Set up adapter with streaming coordinator
        stdio_adapter._streaming_coordinators = [mock_streaming_coordinator]

        # Mock aiohttp ClientSession with connection error
        mock_session = AsyncMock()
        mock_session.post.side_effect = aiohttp.ClientConnectorError(connection_key=None, os_error=None)
        stdio_adapter._client_session = mock_session

        # Test sending an AIDER event (should not raise exception)
        test_event = EventTypes.AIDER_SESSION_STARTED
        test_data = {"session_id": "test_session"}

        # This should complete without raising an exception
        await stdio_adapter.send_event(test_event, test_data)

        # Verify HTTP POST was attempted
        mock_session.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_streaming_coordinators_found(self, stdio_adapter, mock_discovery):
        """Test behavior when no streaming coordinators are discovered."""
        # Mock discovery to return empty list
        mock_discovery.find_streaming_coordinators.return_value = []
        stdio_adapter._discovery = mock_discovery

        await stdio_adapter._auto_discover_coordinator()

        # Verify no streaming coordinators were found
        assert len(stdio_adapter._streaming_coordinators) == 0

        # Test sending AIDER event (should complete without error)
        test_event = EventTypes.AIDER_RATE_LIMIT_DETECTED
        test_data = {"provider": "openai"}

        await stdio_adapter.send_event(test_event, test_data)

        # Should complete successfully even with no streaming coordinators

    @pytest.mark.asyncio
    async def test_streaming_coordinator_without_aider_endpoint(self, stdio_adapter):
        """Test handling of streaming coordinator without aider_events endpoint."""
        # Create streaming coordinator without aider_events endpoint
        coord_without_aider = CoordinatorInfo(
            coordinator_id="coord_no_aider",
            host="localhost",
            port=8081,
            transport_type="sse",
            streaming_capabilities={
                "sse_endpoints": {
                    "health_check": "/health",
                },
                "supported_event_types": ["health"],
            },
        )

        stdio_adapter._streaming_coordinators = [coord_without_aider]

        # Mock aiohttp ClientSession
        mock_session = AsyncMock()
        stdio_adapter._client_session = mock_session

        # Test sending an AIDER event
        test_event = EventTypes.AIDER_RATE_LIMIT_DETECTED
        test_data = {"provider": "openai"}

        await stdio_adapter.send_event(test_event, test_data)

        # Verify no HTTP POST was made (coordinator has no aider_events endpoint)
        mock_session.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_multiple_streaming_coordinators(self, stdio_adapter):
        """Test relaying events to multiple streaming coordinators."""
        # Create multiple streaming coordinators
        coord1 = CoordinatorInfo(
            coordinator_id="coord_1",
            host="localhost",
            port=8080,
            streaming_capabilities={"sse_endpoints": {"aider_events": "/events/aider"}},
        )
        coord2 = CoordinatorInfo(
            coordinator_id="coord_2",
            host="localhost",
            port=8081,
            streaming_capabilities={"sse_endpoints": {"aider_events": "/api/aider"}},
        )

        stdio_adapter._streaming_coordinators = [coord1, coord2]

        # Mock aiohttp ClientSession
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.raise_for_status = MagicMock()
        mock_session.post.return_value.__aenter__.return_value = mock_response
        stdio_adapter._client_session = mock_session

        # Test sending an AIDER event
        test_event = EventTypes.AIDER_SESSION_PROGRESS
        test_data = {"progress": 50}

        await stdio_adapter.send_event(test_event, test_data)

        # Give asyncio tasks time to complete
        await asyncio.sleep(0.1)

        # Verify HTTP POST was made to both coordinators
        assert mock_session.post.call_count == 2

        # Check URLs
        call_urls = [call[0][0] for call in mock_session.post.call_args_list]
        assert "http://localhost:8080/events/aider" in call_urls
        assert "http://localhost:8081/api/aider" in call_urls


class TestStdioTransportAdapterInitialization:
    """Test suite for STDIO transport adapter initialization with discovery."""

    @pytest.mark.asyncio
    async def test_initialization_with_discovery_file(self, tmp_path):
        """Test adapter initialization with discovery file."""
        discovery_file = tmp_path / "test_discovery.json"
        
        with patch("aider_mcp_server.organisms.transports.stdio.stdio_transport_adapter.ApplicationCoordinator.getInstance") as mock_get_instance:
            mock_coordinator = AsyncMock()
            mock_get_instance.return_value = mock_coordinator

            adapter = StdioTransportAdapter(
                input_stream=StringIO(),
                output_stream=StringIO(),
                discovery_file=discovery_file,
            )

            # Verify discovery was initialized
            assert adapter._discovery is not None
            assert adapter._discovery_file == discovery_file

            # Initialize adapter
            await adapter.initialize()

            # Cleanup
            await adapter.shutdown()

    @pytest.mark.asyncio
    async def test_initialization_without_discovery_file(self):
        """Test adapter initialization without discovery file."""
        with patch("aider_mcp_server.organisms.transports.stdio.stdio_transport_adapter.ApplicationCoordinator.getInstance") as mock_get_instance:
            mock_coordinator = AsyncMock()
            mock_get_instance.return_value = mock_coordinator

            adapter = StdioTransportAdapter(
                input_stream=StringIO(),
                output_stream=StringIO(),
            )

            # Verify discovery was not initialized
            assert adapter._discovery is None
            assert adapter._discovery_file is None

            # Initialize adapter
            await adapter.initialize()

            # Should still work without discovery
            assert adapter._coordinator is not None

            # Cleanup
            await adapter.shutdown()