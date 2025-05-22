import asyncio
from typing import Any, Dict, Optional, Set
from unittest.mock import AsyncMock, MagicMock

import pytest

from aider_mcp_server.atoms.event_types import EventTypes
from aider_mcp_server.atoms.internal_types import InternalEvent
from aider_mcp_server.event_coordinator import EventCoordinator
from aider_mcp_server.event_system import EventSystem
from aider_mcp_server.interfaces.event_handler import IEventHandler
from aider_mcp_server.interfaces.transport_adapter import ITransportAdapter
from aider_mcp_server.mcp_types import LoggerProtocol


# --- Mock Implementations ---
class MockEventHandler(IEventHandler):
    def __init__(self, name: str = "MockEventHandler"):
        self.name = name
        self.handle_event_mock = AsyncMock(return_value=None)
        self.handled_events_log = []

    async def handle_event(self, event: InternalEvent) -> None:
        self.handled_events_log.append(event)
        # Propagate the call to the mock for assertion tracking and side effects
        await self.handle_event_mock(event)

    def __repr__(self) -> str:
        return f"<{self.name} instance at {hex(id(self))}>"


class MockTransportAdapter(ITransportAdapter):
    def __init__(self, transport_id: str = "mock_transport_1"):
        self._transport_id = transport_id
        self.get_transport_id_mock = MagicMock(return_value=self._transport_id)
        # Stub other required methods
        self.initialize_mock = AsyncMock()
        self.shutdown_mock = AsyncMock()
        self.send_event_mock = AsyncMock()
        self.get_capabilities_mock = MagicMock(return_value=set())
        self.is_active_mock = MagicMock(return_value=True)
        self.get_transport_type_mock = MagicMock(return_value="mock_type")

    def get_transport_id(self) -> str:
        return self.get_transport_id_mock()

    async def initialize(self) -> None:
        await self.initialize_mock()

    async def shutdown(self) -> None:
        await self.shutdown_mock()

    async def send_event(self, event_type: EventTypes, data: Dict[str, Any]) -> None:
        await self.send_event_mock(event_type, data)

    def get_capabilities(self) -> Set[EventTypes]:
        return self.get_capabilities_mock()

    def is_active(self) -> bool:
        return self.is_active_mock()

    def get_transport_type(self) -> str:
        return self.get_transport_type_mock()


# --- Fixtures ---
@pytest.fixture
def mock_logger_factory():
    """Fixture for a mock logger factory."""
    factory = MagicMock()
    mock_log = MagicMock(spec=LoggerProtocol)
    mock_log.info = MagicMock()
    mock_log.debug = MagicMock()
    mock_log.warning = MagicMock()
    mock_log.error = MagicMock()
    mock_log.exception = MagicMock() # For exc_info=True in error logging
    factory.return_value = mock_log
    return factory


@pytest.fixture
def mock_event_system():
    """Fixture for a mock EventSystem."""
    mock_es = AsyncMock(spec=EventSystem)
    # Ensure async methods are AsyncMocks
    mock_es.subscribe = AsyncMock()
    mock_es.unsubscribe = AsyncMock()
    mock_es.broadcast = AsyncMock()
    return mock_es


@pytest.fixture
def event_coordinator(mock_logger_factory: MagicMock, mock_event_system: AsyncMock):
    """Fixture for an EventCoordinator instance."""
    coordinator = EventCoordinator(mock_logger_factory, mock_event_system)
    return coordinator


# --- Helper Functions ---
def create_internal_event(
    event_type: EventTypes, data: Optional[Dict[str, Any]] = None, metadata: Optional[Dict[str, Any]] = None
) -> InternalEvent:
    """Helper to create InternalEvent instances for tests."""
    return InternalEvent(event_type=event_type, data=data or {}, metadata=metadata)


# --- Test Classes ---
@pytest.mark.asyncio
class TestEventCoordinatorLifecycle:
    async def test_initialization(self, event_coordinator: EventCoordinator, mock_logger_factory: MagicMock, mock_event_system: AsyncMock):
        logger = mock_logger_factory.return_value
        mock_logger_factory.assert_called_once_with("aider_mcp_server.event_coordinator")
        logger.info.assert_called_once_with("EventCoordinator initialized.")
        assert event_coordinator._event_system is mock_event_system
        assert event_coordinator._handlers == {}
        assert event_coordinator._transport_adapters == {}
        assert isinstance(event_coordinator._lock, asyncio.Lock)

    async def test_startup(self, event_coordinator: EventCoordinator):
        logger = event_coordinator._logger
        await event_coordinator.startup()
        logger.info.assert_any_call("EventCoordinator starting up...") # Use any_call if other logs exist
        logger.info.assert_any_call("EventCoordinator started.")

    async def test_shutdown(self, event_coordinator: EventCoordinator):
        logger = event_coordinator._logger
        # Add some handlers and adapters to check they are cleared
        handler = MockEventHandler()
        adapter = MockTransportAdapter("adapter1")
        event_coordinator._handlers[EventTypes.STATUS] = [handler]
        event_coordinator._transport_adapters["adapter1"] = adapter

        await event_coordinator.shutdown()

        logger.info.assert_any_call("EventCoordinator shutting down...")
        logger.info.assert_any_call("EventCoordinator shut down.")
        assert event_coordinator._handlers == {}
        assert event_coordinator._transport_adapters == {}


@pytest.mark.asyncio
class TestEventCoordinatorSubscription:
    async def test_subscribe_new_handler(self, event_coordinator: EventCoordinator):
        logger = event_coordinator._logger
        handler = MockEventHandler()
        event_type = EventTypes.STATUS

        await event_coordinator.subscribe(event_type, handler)

        assert event_type in event_coordinator._handlers
        assert handler in event_coordinator._handlers[event_type]
        logger.debug.assert_called_with(
            f"Handler '{type(handler).__name__}' subscribed to event type '{event_type.value}'"
        )

    async def test_subscribe_same_handler_multiple_times(self, event_coordinator: EventCoordinator):
        handler = MockEventHandler()
        event_type = EventTypes.PROGRESS
        await event_coordinator.subscribe(event_type, handler)
        await event_coordinator.subscribe(event_type, handler) # Subscribe again

        assert len(event_coordinator._handlers[event_type]) == 1
        event_coordinator._logger.debug.assert_any_call( # Second call
            f"Handler '{type(handler).__name__}' already subscribed to event type '{event_type.value}'"
        )

    async def test_unsubscribe_handler(self, event_coordinator: EventCoordinator):
        logger = event_coordinator._logger
        handler = MockEventHandler()
        event_type = EventTypes.TOOL_RESULT
        await event_coordinator.subscribe(event_type, handler)

        await event_coordinator.unsubscribe(event_type, handler)

        assert event_type not in event_coordinator._handlers # Assuming it's the only handler
        logger.debug.assert_any_call(
            f"Handler '{type(handler).__name__}' unsubscribed from event type '{event_type.value}'"
        )
        logger.debug.assert_any_call( # Check for removal of event type key
            f"Event type '{event_type.value}' removed as it has no more internal handlers."
        )

    async def test_unsubscribe_handler_leaves_other_handlers(self, event_coordinator: EventCoordinator):
        handler1 = MockEventHandler("Handler1")
        handler2 = MockEventHandler("Handler2")
        event_type = EventTypes.STATUS
        await event_coordinator.subscribe(event_type, handler1)
        await event_coordinator.subscribe(event_type, handler2)

        await event_coordinator.unsubscribe(event_type, handler1)

        assert handler1 not in event_coordinator._handlers[event_type]
        assert handler2 in event_coordinator._handlers[event_type]
        event_coordinator._logger.debug.assert_any_call(
            f"Handler '{type(handler1).__name__}' unsubscribed from event type '{event_type.value}'"
        )

    async def test_unsubscribe_non_existent_handler(self, event_coordinator: EventCoordinator):
        logger = event_coordinator._logger
        handler = MockEventHandler()
        non_existent_handler = MockEventHandler("NonExistent")
        event_type = EventTypes.STATUS
        await event_coordinator.subscribe(event_type, handler)

        await event_coordinator.unsubscribe(event_type, non_existent_handler)
        logger.debug.assert_called_with(
            f"Handler '{type(non_existent_handler).__name__}' not found for event type '{event_type.value}', or event type not registered for internal handling."
        )
        assert handler in event_coordinator._handlers[event_type] # Original handler still there

    async def test_unsubscribe_handler_from_non_existent_event_type(self, event_coordinator: EventCoordinator):
        logger = event_coordinator._logger
        handler = MockEventHandler()
        event_type = EventTypes.HEARTBEAT # Assume not subscribed to

        await event_coordinator.unsubscribe(event_type, handler)
        logger.debug.assert_called_with(
            f"Handler '{type(handler).__name__}' not found for event type '{event_type.value}', or event type not registered for internal handling."
        )


@pytest.mark.asyncio
class TestEventCoordinatorPublishing:
    async def test_publish_event_to_single_handler(self, event_coordinator: EventCoordinator, mock_event_system: AsyncMock):
        handler = MockEventHandler()
        event_type = EventTypes.STATUS
        event_data = {"key": "value"}
        event = create_internal_event(event_type, data=event_data)
        await event_coordinator.subscribe(event_type, handler)

        await event_coordinator.publish_event(event)

        handler.handle_event_mock.assert_awaited_once_with(event)
        assert event in handler.handled_events_log
        mock_event_system.broadcast.assert_awaited_once_with(event_type.value, event_data)

    async def test_publish_event_to_multiple_handlers(self, event_coordinator: EventCoordinator, mock_event_system: AsyncMock):
        handler1 = MockEventHandler("Handler1")
        handler2 = MockEventHandler("Handler2")
        event_type = EventTypes.PROGRESS
        event = create_internal_event(event_type)
        await event_coordinator.subscribe(event_type, handler1)
        await event_coordinator.subscribe(event_type, handler2)

        await event_coordinator.publish_event(event)

        handler1.handle_event_mock.assert_awaited_once_with(event)
        handler2.handle_event_mock.assert_awaited_once_with(event)
        mock_event_system.broadcast.assert_awaited_once_with(event_type.value, event.data)

    async def test_publish_event_no_internal_handlers(self, event_coordinator: EventCoordinator, mock_event_system: AsyncMock):
        logger = event_coordinator._logger
        event_type = EventTypes.TOOL_RESULT
        event = create_internal_event(event_type)

        await event_coordinator.publish_event(event)

        logger.debug.assert_any_call(f"No internal handlers for event type '{event_type.value}'")
        mock_event_system.broadcast.assert_awaited_once_with(event_type.value, event.data)

    async def test_publish_event_with_metadata(self, event_coordinator: EventCoordinator, mock_event_system: AsyncMock):
        handler = MockEventHandler()
        event_type = EventTypes.STATUS
        event_data = {"info": "data"}
        metadata = {"source": "test"}
        event = create_internal_event(event_type, data=event_data, metadata=metadata)
        await event_coordinator.subscribe(event_type, handler)

        await event_coordinator.publish_event(event)

        handler.handle_event_mock.assert_awaited_once_with(event)
        assert handler.handled_events_log[0].metadata == metadata
        mock_event_system.broadcast.assert_awaited_once_with(event_type.value, event_data)

    async def test_publish_event_handler_exception(self, event_coordinator: EventCoordinator, mock_event_system: AsyncMock):
        logger = event_coordinator._logger
        failing_handler = MockEventHandler("FailingHandler")
        failing_handler.handle_event_mock.side_effect = RuntimeError("Handler failed")
        successful_handler = MockEventHandler("SuccessfulHandler")
        event_type = EventTypes.STATUS
        event = create_internal_event(event_type)

        await event_coordinator.subscribe(event_type, failing_handler)
        await event_coordinator.subscribe(event_type, successful_handler)

        await event_coordinator.publish_event(event)

        failing_handler.handle_event_mock.assert_awaited_once_with(event)
        successful_handler.handle_event_mock.assert_awaited_once_with(event) # Still called
        logger.error.assert_called_once_with(
            f"Error in internal handler '{type(failing_handler).__name__}' for event '{event_type.value}': Handler failed",
            exc_info=True
        )
        mock_event_system.broadcast.assert_awaited_once_with(event_type.value, event.data) # Broadcast still happens

    async def test_publish_event_event_system_broadcast_exception(self, event_coordinator: EventCoordinator, mock_event_system: AsyncMock):
        logger = event_coordinator._logger
        event_type = EventTypes.HEARTBEAT
        event = create_internal_event(event_type)
        mock_event_system.broadcast.side_effect = RuntimeError("Broadcast failed")

        await event_coordinator.publish_event(event)

        logger.error.assert_called_once_with(
            f"Error broadcasting event '{event_type.value}' via EventSystem: Broadcast failed",
            exc_info=True
        )


@pytest.mark.asyncio
class TestEventCoordinatorTransportManagement:
    async def test_register_transport_adapter(self, event_coordinator: EventCoordinator):
        logger = event_coordinator._logger
        adapter = MockTransportAdapter(transport_id="adapter_x")

        await event_coordinator.register_transport_adapter(adapter)

        assert "adapter_x" in event_coordinator._transport_adapters
        assert event_coordinator._transport_adapters["adapter_x"] is adapter
        logger.info.assert_called_with("Transport adapter adapter_x registered.")
        adapter.get_transport_id_mock.assert_called()

    async def test_register_same_transport_adapter_id_overwrites(self, event_coordinator: EventCoordinator):
        logger = event_coordinator._logger
        adapter1 = MockTransportAdapter(transport_id="adapter_y")
        adapter2 = MockTransportAdapter(transport_id="adapter_y") # Same ID

        await event_coordinator.register_transport_adapter(adapter1)
        await event_coordinator.register_transport_adapter(adapter2)

        assert event_coordinator._transport_adapters["adapter_y"] is adapter2 # Overwritten
        logger.warning.assert_called_with(
            "Transport adapter adapter_y already registered. Overwriting."
        )

    async def test_unregister_transport_adapter(self, event_coordinator: EventCoordinator):
        logger = event_coordinator._logger
        adapter = MockTransportAdapter(transport_id="adapter_z")
        await event_coordinator.register_transport_adapter(adapter)

        await event_coordinator.unregister_transport_adapter("adapter_z")

        assert "adapter_z" not in event_coordinator._transport_adapters
        logger.info.assert_called_with("Transport adapter adapter_z unregistered.")

    async def test_unregister_non_existent_transport_adapter(self, event_coordinator: EventCoordinator):
        logger = event_coordinator._logger
        await event_coordinator.unregister_transport_adapter("non_existent_adapter")
        logger.warning.assert_called_with(
            "Attempted to unregister non-existent transport adapter non_existent_adapter."
        )


@pytest.mark.asyncio
class TestEventCoordinatorConcurrency:
    async def test_concurrent_subscribe_unsubscribe(self, event_coordinator: EventCoordinator):
        event_type = EventTypes.STATUS
        num_handlers = 5
        handlers = [MockEventHandler(f"Handler{i}") for i in range(num_handlers)]

        subscribe_tasks = [event_coordinator.subscribe(event_type, h) for h in handlers]
        await asyncio.gather(*subscribe_tasks)

        assert len(event_coordinator._handlers[event_type]) == num_handlers

        unsubscribe_tasks = [event_coordinator.unsubscribe(event_type, h) for h in handlers[:num_handlers//2]]
        await asyncio.gather(*unsubscribe_tasks)

        assert len(event_coordinator._handlers[event_type]) == num_handlers - (num_handlers // 2)

        # Unsubscribe the rest
        unsubscribe_rest_tasks = [event_coordinator.unsubscribe(event_type, h) for h in handlers[num_handlers//2:]]
        await asyncio.gather(*unsubscribe_rest_tasks)
        assert event_type not in event_coordinator._handlers


    async def test_concurrent_publish_event(self, event_coordinator: EventCoordinator, mock_event_system: AsyncMock):
        handler = MockEventHandler()
        event_type = EventTypes.PROGRESS
        await event_coordinator.subscribe(event_type, handler)

        num_events = 10
        events = [create_internal_event(event_type, data={"count": i}) for i in range(num_events)]

        publish_tasks = [event_coordinator.publish_event(e) for e in events]
        await asyncio.gather(*publish_tasks)

        assert handler.handle_event_mock.await_count == num_events
        assert mock_event_system.broadcast.await_count == num_events
        # Check if all events were logged by handler (order might not be guaranteed)
        assert len(handler.handled_events_log) == num_events
        logged_event_data = sorted([e.data["count"] for e in handler.handled_events_log])
        expected_event_data = sorted([e.data["count"] for e in events])
        assert logged_event_data == expected_event_data


    async def test_concurrent_register_unregister_adapter(self, event_coordinator: EventCoordinator):
        num_adapters = 5
        adapter_ids = [f"concurrent_adapter_{i}" for i in range(num_adapters)]
        adapters = [MockTransportAdapter(tid) for tid in adapter_ids]

        register_tasks = [event_coordinator.register_transport_adapter(a) for a in adapters]
        await asyncio.gather(*register_tasks)

        assert len(event_coordinator._transport_adapters) == num_adapters

        unregister_tasks = [event_coordinator.unregister_transport_adapter(aid) for aid in adapter_ids[:num_adapters//2]]
        await asyncio.gather(*unregister_tasks)

        assert len(event_coordinator._transport_adapters) == num_adapters - (num_adapters // 2)

        # Unregister the rest
        unregister_rest_tasks = [event_coordinator.unregister_transport_adapter(aid) for aid in adapter_ids[num_adapters//2:]]
        await asyncio.gather(*unregister_rest_tasks)
        assert not event_coordinator._transport_adapters
