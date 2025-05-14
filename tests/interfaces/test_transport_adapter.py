"""
Tests for the TransportAdapter interface and base class.

This module tests the standardized interfaces for transport adapters.
"""

import uuid
from typing import Set
from unittest.mock import MagicMock

import pytest

from aider_mcp_server.atoms.event_types import EventTypes
from aider_mcp_server.interfaces.transport_adapter import (
    ITransportAdapter,
    TransportAdapterBase,
)
from aider_mcp_server.mcp_types import EventData, RequestParameters
from aider_mcp_server.security import SecurityContext


class ConcreteTransportAdapter(TransportAdapterBase):
    """Concrete implementation of the abstract base class for testing."""

    def __init__(self, transport_id: str, transport_type: str = "test"):
        self.transport_id = transport_id
        self.transport_type = transport_type
        self.sent_events = []

    async def send_event(self, event_type: EventTypes, data: EventData) -> None:
        """Record sent events for testing."""
        self.sent_events.append((event_type, data))

    def validate_request_security(
        self, request_data: RequestParameters
    ) -> SecurityContext:
        """Simple security validation for testing."""
        # Create a mock security context for testing
        mock_context = MagicMock(spec=SecurityContext)
        return mock_context

    def get_capabilities(self) -> Set[EventTypes]:
        """Get the capabilities of this transport."""
        return {
            EventTypes.STATUS,
            EventTypes.PROGRESS,
            EventTypes.TOOL_RESULT,
            EventTypes.HEARTBEAT,
        }


@pytest.fixture
def transport_adapter():
    """Create a concrete transport adapter for testing."""
    adapter_id = f"test-{uuid.uuid4()}"
    return ConcreteTransportAdapter(adapter_id)


class TestTransportAdapterInterface:
    """Tests for the TransportAdapter interface and implementations."""

    def test_adapter_implements_protocol(self, transport_adapter):
        """Test that our concrete adapter implements the ITransportAdapter protocol."""
        assert isinstance(transport_adapter, ITransportAdapter)

    def test_adapter_basic_properties(self, transport_adapter):
        """Test that the adapter has the required properties."""
        assert transport_adapter.transport_id.startswith("test-")
        assert transport_adapter.transport_type == "test"

    def test_get_capabilities(self, transport_adapter):
        """Test the default capabilities implementation."""
        capabilities = transport_adapter.get_capabilities()
        assert isinstance(capabilities, set)
        assert len(capabilities) > 0
        # All items should be EventTypes
        for item in capabilities:
            assert isinstance(item, EventTypes)

    def test_should_receive_event_default(self, transport_adapter):
        """Test that the default should_receive_event implementation returns True."""
        event_type = EventTypes.STATUS
        data = {"test": "data"}
        assert transport_adapter.should_receive_event(event_type, data)
        # With request details
        assert transport_adapter.should_receive_event(
            event_type, data, {"request_id": "123"}
        )

    @pytest.mark.asyncio
    async def test_start_listening_default(self, transport_adapter):
        """Test that start_listening is a no-op by default."""
        # Should not raise any exceptions
        await transport_adapter.start_listening()


class TestTransportAdapterImplementation:
    """Tests for the concrete implementation of TransportAdapter."""

    @pytest.mark.asyncio
    async def test_send_event(self, transport_adapter):
        """Test sending events."""
        event_type = EventTypes.STATUS
        data = {"message": "test"}
        await transport_adapter.send_event(event_type, data)
        assert len(transport_adapter.sent_events) == 1
        assert transport_adapter.sent_events[0][0] == event_type
        assert transport_adapter.sent_events[0][1] == data

    def test_validate_request_security(self, transport_adapter):
        """Test security validation."""
        request_data = {"auth_token": "test_token"}
        result = transport_adapter.validate_request_security(request_data)
        assert isinstance(result, SecurityContext)


# Additional tests for the framework's usage of the interface would be created separately
# in test_transport_coordinator.py or similar files.
