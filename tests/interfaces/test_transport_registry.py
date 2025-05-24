"""
Tests for the transport adapter registry.
"""

from typing import Any, Dict, Optional, Set
from unittest.mock import MagicMock

import pytest
import pytest_asyncio

from aider_mcp_server.atoms.types.event_types import EventTypes
from aider_mcp_server.interfaces.transport_adapter import (
    TransportAdapterBase,
)
from aider_mcp_server.interfaces.transport_registry import TransportAdapterRegistry


# Define a mock adapter for testing
class MockTransportAdapter(TransportAdapterBase):
    """Mock transport adapter for testing."""

    def __init__(self, transport_id: str = "mock_id", transport_type: str = "mock", **kwargs: Any):
        super().__init__(transport_id=transport_id, transport_type=transport_type, **kwargs)

    @classmethod
    def get_default_capabilities(cls) -> Set[EventTypes]:
        """Get default capabilities for testing."""
        return {EventTypes.STATUS, EventTypes.PROGRESS}

    async def initialize(self) -> None:
        """Initialize the mock adapter."""
        pass

    async def shutdown(self) -> None:
        """Shutdown the mock adapter."""
        pass

    async def send_event(self, event_type: EventTypes, data: Dict[str, Any]) -> None:
        """Send an event via the mock adapter."""
        pass

    def get_capabilities(self) -> Set[EventTypes]:
        """Get the mock adapter's capabilities."""
        return {EventTypes.STATUS, EventTypes.PROGRESS, EventTypes.TOOL_RESULT}

    def validate_request_security(self, request_data: Dict[str, Any]) -> Any:
        """Mock security validation."""
        return MagicMock()

    async def start_listening(self) -> None:
        """Start the mock adapter's listener."""
        pass

    def should_receive_event(
        self,
        event_type: EventTypes,
        data: Dict[str, Any],
        request_details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Check if the mock adapter should receive an event."""
        return True


@pytest_asyncio.fixture
async def registry():
    """Fixture for creating and initializing a clean registry for each test."""
    # Reset the singleton state for each test
    TransportAdapterRegistry._instance = None
    TransportAdapterRegistry._adapter_classes = {}
    TransportAdapterRegistry._adapter_cache = {}
    TransportAdapterRegistry._initialized = False

    # Get a fresh instance
    registry = await TransportAdapterRegistry.get_instance()
    return registry


@pytest.mark.asyncio
async def test_registry_singleton(registry):
    """Test that the registry uses the singleton pattern."""
    # Get a second instance - should be the same object
    registry2 = await TransportAdapterRegistry.get_instance()
    assert registry is registry2


@pytest.mark.asyncio
async def test_register_adapter_class(registry):
    """Test registering an adapter class."""
    # Register a mock adapter
    registry.register_adapter_class("mock", MockTransportAdapter)

    # Verify it's registered
    assert "mock" in registry.list_adapter_types()
    assert registry.get_adapter_class("mock") is MockTransportAdapter


@pytest.mark.asyncio
async def test_register_invalid_adapter_class(registry):
    """Test that registering an invalid adapter class raises an error."""

    # Try to register a class that doesn't implement ITransportAdapter
    class InvalidAdapter:
        pass

    with pytest.raises(TypeError):
        registry.register_adapter_class("invalid", InvalidAdapter)


@pytest.mark.asyncio
async def test_unregister_adapter_class(registry):
    """Test unregistering an adapter class."""
    # Register and then unregister
    registry.register_adapter_class("mock", MockTransportAdapter)
    registry.unregister_adapter_class("mock")

    # Verify it's gone
    assert "mock" not in registry.list_adapter_types()
    assert registry.get_adapter_class("mock") is None


@pytest.mark.asyncio
async def test_get_adapter_capabilities(registry):
    """Test getting adapter capabilities without instantiation."""
    # Register the mock adapter
    registry.register_adapter_class("mock", MockTransportAdapter)

    # Get capabilities
    capabilities = registry.get_adapter_capabilities("mock")
    assert capabilities is not None
    assert EventTypes.STATUS in capabilities
    assert EventTypes.PROGRESS in capabilities


@pytest.mark.asyncio
async def test_create_adapter(registry):
    """Test creating an adapter instance."""
    # Register the mock adapter
    registry.register_adapter_class("mock", MockTransportAdapter)

    # Create an instance
    adapter = await registry.create_adapter("mock", transport_id="test_id")
    assert adapter is not None
    assert adapter.get_transport_id() == "test_id"
    assert adapter.get_transport_type() == "mock"

    # Verify it's cached
    cached_adapter = registry.get_cached_adapter("mock", "test_id")
    assert cached_adapter is adapter


@pytest.mark.asyncio
async def test_create_adapter_with_kwargs(registry):
    """Test creating an adapter with additional kwargs."""

    # Create a mock adapter class that accepts extra kwargs
    class KwargsAdapter(MockTransportAdapter):
        def __init__(
            self,
            transport_id: str = "kwargs_id",
            transport_type: str = "kwargs",
            extra_param: str = "default",
            **kwargs: Any,
        ):
            super().__init__(transport_id=transport_id, transport_type=transport_type, **kwargs)
            self.extra_param = extra_param

    # Register the adapter
    registry.register_adapter_class("kwargs", KwargsAdapter)

    # Create with extra param
    adapter = await registry.create_adapter("kwargs", extra_param="custom")
    assert adapter is not None
    assert adapter.extra_param == "custom"


@pytest.mark.asyncio
async def test_create_nonexistent_adapter(registry):
    """Test attempting to create an adapter that doesn't exist."""
    adapter = await registry.create_adapter("nonexistent")
    assert adapter is None


@pytest.mark.asyncio
async def test_remove_cached_adapter(registry):
    """Test removing a cached adapter."""
    # Register and create
    registry.register_adapter_class("mock", MockTransportAdapter)
    adapter = await registry.create_adapter("mock", transport_id="test_id")

    # Verify it's cached and then remove
    assert registry.get_cached_adapter("mock", "test_id") is adapter
    registry.remove_cached_adapter("mock", "test_id")

    # Verify it's gone
    assert registry.get_cached_adapter("mock", "test_id") is None


@pytest.mark.asyncio
async def test_initialize_discovers_built_in_adapters(registry):
    """Test that initialize discovers built-in adapters."""
    # Verify built-in adapters are found
    assert "sse" in registry.list_adapter_types()
    assert "stdio" in registry.list_adapter_types()


@pytest.mark.asyncio
async def test_adapter_creation_error_handling():
    """Test error handling during adapter creation."""
    registry = await TransportAdapterRegistry.get_instance()

    # Create an adapter class that raises an exception during initialization
    class ErrorAdapter(MockTransportAdapter):
        def __init__(self, **kwargs: Any):
            raise ValueError("Simulated initialization error")

    # Register the error-prone adapter
    registry.register_adapter_class("error", ErrorAdapter)

    # Try to create it - should return None
    adapter = await registry.create_adapter("error")
    assert adapter is None
