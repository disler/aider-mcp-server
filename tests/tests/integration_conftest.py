"""Shared fixtures for integration tests."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from aider_mcp_server.atoms.types.event_types import EventTypes
from aider_mcp_server.atoms.types.internal_types import InternalEvent
from aider_mcp_server.organisms.discovery.transport_discovery import DiscoveryService
from aider_mcp_server.organisms.transports.sse.sse_transport_adapter import SSETransportAdapter
from aider_mcp_server.organisms.transports.stdio.stdio_transport_adapter import StdioTransportAdapter
from aider_mcp_server.pages.application.coordinator import ApplicationCoordinator


@pytest.fixture
def mock_coordinator():
    """Provide a standard AsyncMock for ApplicationCoordinator."""
    coordinator = AsyncMock(spec=ApplicationCoordinator)
    coordinator.register_transport = AsyncMock()
    coordinator.register_transport_adapter = AsyncMock()
    coordinator.unregister_transport = AsyncMock()
    coordinator.broadcast_event = AsyncMock()
    coordinator.subscribe_to_event_type = AsyncMock()
    coordinator.process_request = AsyncMock()
    coordinator.initialize = AsyncMock()
    coordinator.shutdown = AsyncMock()
    coordinator._send_event_to_transports = AsyncMock()
    coordinator._event_coordinator = AsyncMock()
    coordinator._event_coordinator.publish_event = AsyncMock()
    coordinator._event_coordinator.subscribe = AsyncMock()
    return coordinator


@pytest_asyncio.fixture
async def clean_coordinator():
    """Provide a real ApplicationCoordinator instance with singleton cleanup."""
    # Reset singleton state before test
    ApplicationCoordinator._instance = None
    ApplicationCoordinator._initialized = False

    coordinator = ApplicationCoordinator()
    await coordinator.initialize()

    yield coordinator

    # Clean up after test
    await coordinator.shutdown()

    # Reset singleton state after test
    ApplicationCoordinator._instance = None
    ApplicationCoordinator._initialized = False


@pytest.fixture
def mock_coordinator_with_transports(mock_coordinator):
    """Provide a mock coordinator with a _transports dictionary."""
    mock_coordinator._transports = {}
    return mock_coordinator


@pytest.fixture
def mock_sse_adapter():
    """Provide a standard AsyncMock for SSETransportAdapter."""
    adapter = AsyncMock(spec=SSETransportAdapter)
    adapter.transport_id = "mock_sse_adapter_id"
    adapter.transport_type = "sse"
    adapter.initialize = AsyncMock()
    adapter.shutdown = AsyncMock()
    adapter.send_event = AsyncMock()
    adapter.get_capabilities = MagicMock(return_value={
        "transport_type": "sse",
        "supports_events": True,
        "supports_requests": True,
        "protocol": "http",
        "connection_type": "unidirectional",
        "event_types": list(EventTypes), # Example: subscribe to all
    })
    adapter.get_transport_id = MagicMock(return_value=adapter.transport_id)
    adapter.should_receive_event = MagicMock(return_value=True)
    return adapter


@pytest.fixture
def mock_stdio_adapter():
    """Provide a standard AsyncMock for StdioTransportAdapter."""
    adapter = AsyncMock(spec=StdioTransportAdapter)
    adapter.transport_id = "mock_stdio_adapter_id"
    adapter.transport_type = "stdio"
    adapter.initialize = AsyncMock()
    adapter.shutdown = AsyncMock()
    adapter.send_event = AsyncMock()
    adapter._coordinator = None # Can be set by tests if needed
    adapter._discovery = AsyncMock()
    return adapter


@pytest.fixture
def mock_sse_request():
    """Provide a standard MagicMock for an SSE request."""
    request = MagicMock()
    request.scope = {"type": "http", "method": "GET", "path": "/events"}
    request.receive = AsyncMock()
    request.send = AsyncMock()
    return request


@pytest.fixture
def mock_aider_request():
    """Provide a standard dictionary mock for an AIDER tool request."""
    return {
        "type": "aider_tool_request",
        "tool_name": "code_with_aider",
        "parameters": {
            "ai_coding_prompt": "Test prompt",
            "relative_editable_files": ["test.py"],
            "relative_readonly_files": [],
            "model": "test-model",
            "working_dir": "/tmp/test_project",
        },
        "request_id": "aider_req_123",
    }


@pytest.fixture
def mock_discovery_service():
    """Provide an AsyncMock for DiscoveryService."""
    service = AsyncMock(spec=DiscoveryService)
    service.find_streaming_coordinators = AsyncMock(return_value=[])
    service.notify_transport_available = AsyncMock()
    service.get_transport_info = AsyncMock(return_value=None)
    service.get_available_transports = AsyncMock(return_value={})
    service.check_coordinator_available = AsyncMock(return_value=True)
    service.set_coordinator = MagicMock()
    service.shutdown = AsyncMock()
    return service


@pytest.fixture
def sample_internal_event():
    """Provide a sample InternalEvent for testing."""
    return InternalEvent(
        event_type=EventTypes.STATUS,
        data={"message": "Sample event data", "value": 123},
        metadata={"source": "test_fixture"},
        timestamp=asyncio.get_event_loop().time(),
    )
