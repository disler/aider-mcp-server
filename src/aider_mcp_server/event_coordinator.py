"""
Event Coordinator for handling event publishing and subscription functionality.
Implements the IEventCoordinator interface and uses an EventSystem for low-level
event broadcasting to external transports. It manages internal event handlers
and can also manage transport adapter registrations.
"""

import asyncio
from typing import Dict, List, Optional, Any, Union

from aider_mcp_server.atoms.event_types import EventTypes
from aider_mcp_server.atoms.internal_types import InternalEvent
# TODO: Remove EventMediator import and Union type hint after refactoring is complete
from aider_mcp_server.event_mediator import EventMediator
from aider_mcp_server.event_system import EventSystem
from aider_mcp_server.interfaces.event_coordinator import IEventCoordinator
from aider_mcp_server.interfaces.event_handler import IEventHandler
from aider_mcp_server.interfaces.transport_adapter import ITransportAdapter
from aider_mcp_server.mcp_types import LoggerFactory, LoggerProtocol


class EventCoordinator(IEventCoordinator):
    _logger: LoggerProtocol
    _event_system: EventSystem
    _handlers: Dict[EventTypes, List[IEventHandler]]
    _transport_adapters: Dict[str, ITransportAdapter]  # transport_id -> adapter
    _lock: asyncio.Lock

    # TODO: Remove EventMediator from Union and related logic after refactoring component initialization.
    def __init__(self, logger_factory: LoggerFactory, event_system_or_mediator: Union[EventSystem, EventMediator]):
        self._logger = logger_factory(__name__)
        self._handlers = {}
        self._transport_adapters = {}
        self._lock = asyncio.Lock()

        if isinstance(event_system_or_mediator, EventMediator):
            self._logger.warning(
                "EventCoordinator initialized with EventMediator. This is deprecated and will be removed. "
                "Please update to pass EventSystem directly."
            )
            # Extract the EventSystem instance from the EventMediator
            if hasattr(event_system_or_mediator, "_event_system") and isinstance(
                getattr(event_system_or_mediator, "_event_system"), EventSystem
            ):
                self._event_system = getattr(event_system_or_mediator, "_event_system")
            else:
                # This case should ideally not happen if EventMediator is correctly structured
                self._logger.error(
                    "Failed to extract EventSystem from EventMediator. EventMediator instance lacks a valid '_event_system' attribute."
                )
                raise TypeError(
                    "EventMediator instance passed to EventCoordinator does not have a valid '_event_system' attribute of type EventSystem."
                )
        elif isinstance(event_system_or_mediator, EventSystem):
            self._event_system = event_system_or_mediator
        else:
            # This case handles unexpected types (should not happen with proper typing)
            self._logger.error(  # type: ignore[unreachable]
                f"EventCoordinator initialized with an invalid type for event_system_or_mediator: {type(event_system_or_mediator)}"
            )
            raise TypeError(
                "EventCoordinator must be initialized with an EventSystem or EventMediator instance."
            )

        self._logger.info("EventCoordinator initialized.")

    async def startup(self) -> None:
        """Initializes and starts the event coordinator."""
        self._logger.info("EventCoordinator starting up...")
        # Placeholder for any specific startup logic.
        # For example, could involve initialising or verifying EventSystem state if needed.
        await asyncio.sleep(0)  # Ensures async context if no other awaitables initially.
        self._logger.info("EventCoordinator started.")

    async def shutdown(self) -> None:
        """Shuts down the event coordinator and cleans up resources."""
        self._logger.info("EventCoordinator shutting down...")
        async with self._lock:
            self._handlers.clear()
            # Note: This does not call shutdown on adapters themselves.
            # Lifecycle management of adapters is assumed to be external.
            self._transport_adapters.clear()
        self._logger.info("EventCoordinator shut down.")

    # --- Transport Adapter Management (Not part of IEventCoordinator interface) ---
    async def register_transport_adapter(self, adapter: ITransportAdapter) -> None:
        """Registers a transport adapter with the coordinator."""
        async with self._lock:
            transport_id = adapter.get_transport_id()
            if transport_id in self._transport_adapters:
                self._logger.warning(
                    f"Transport adapter {transport_id} already registered. Overwriting."
                )
            self._transport_adapters[transport_id] = adapter
            self._logger.info(f"Transport adapter {transport_id} registered.")

    async def unregister_transport_adapter(self, transport_id: str) -> None:
        """Unregisters a transport adapter from the coordinator."""
        async with self._lock:
            if transport_id in self._transport_adapters:
                del self._transport_adapters[transport_id]
                self._logger.info(f"Transport adapter {transport_id} unregistered.")
            else:
                self._logger.warning(
                    f"Attempted to unregister non-existent transport adapter {transport_id}."
                )

    # --- IEventCoordinator Implementation ---
    async def subscribe(self, event_type: EventTypes, handler: IEventHandler) -> None:
        """Subscribes an event handler to a specific event type."""
        async with self._lock:
            if event_type not in self._handlers:
                self._handlers[event_type] = []

            if handler not in self._handlers[event_type]:
                self._handlers[event_type].append(handler)
                self._logger.debug(
                    f"Handler '{type(handler).__name__}' subscribed to event type '{event_type.value}'"
                )
            else:
                self._logger.debug(
                    f"Handler '{type(handler).__name__}' already subscribed to event type '{event_type.value}'"
                )

    async def unsubscribe(self, event_type: EventTypes, handler: IEventHandler) -> None:
        """Unsubscribes an event handler from a specific event type."""
        async with self._lock:
            if event_type in self._handlers and handler in self._handlers[event_type]:
                self._handlers[event_type].remove(handler)
                self._logger.debug(
                    f"Handler '{type(handler).__name__}' unsubscribed from event type '{event_type.value}'"
                )
                if not self._handlers[event_type]:  # Remove event type if no handlers left
                    del self._handlers[event_type]
                    self._logger.debug(
                        f"Event type '{event_type.value}' removed as it has no more internal handlers."
                    )
            else:
                self._logger.debug(
                    f"Handler '{type(handler).__name__}' not found for event type '{event_type.value}', or event type not registered for internal handling."
                )

    async def publish_event(self, event: InternalEvent) -> None:
        """
        Publishes an internal event to subscribed IEventHandlers and
        broadcasts its data via the EventSystem to external transports.
        """
        self._logger.debug(
            f"Publishing event: Type='{event.event_type.value}', Data='{event.data}', Metadata='{event.metadata}'"
        )

        # 1. Notify internal handlers
        handlers_to_notify: List[IEventHandler] = []
        async with self._lock:  # Protects read access to self._handlers
            if event.event_type in self._handlers:
                # Create a copy for safe iteration, in case a handler tries to unsubscribe
                handlers_to_notify = list(self._handlers[event.event_type])

        if handlers_to_notify:
            self._logger.debug(
                f"Notifying {len(handlers_to_notify)} internal handler(s) for '{event.event_type.value}'"
            )
            for handler in handlers_to_notify:
                try:
                    # Handler might return a new event. For now, we don't chain publish them here.
                    # This could be extended by collecting returned events.
                    await handler.handle_event(event)
                except Exception as e:
                    self._logger.error(
                        f"Error in internal handler '{type(handler).__name__}' for event '{event.event_type.value}': {e}",
                        exc_info=True,
                    )
        else:
            self._logger.debug(f"No internal handlers for event type '{event.event_type.value}'")

        # 2. Broadcast event data via EventSystem for external transports
        # Transport adapters are expected to subscribe to the EventSystem
        # using string-based event types (event.event_type.value).
        # Transport-specific filtering (should_receive_event) and capability checks
        # should be handled within the transport adapters' EventSystem callbacks.
        self._logger.debug(
            f"Broadcasting event '{event.event_type.value}' with data via EventSystem."
        )
        try:
            # Bridge: Use enum's value (string) for EventSystem
            await self._event_system.broadcast(event.event_type.value, event.data)
        except Exception as e:
            self._logger.error(
                f"Error broadcasting event '{event.event_type.value}' via EventSystem: {e}",
                exc_info=True,
            )

    # --- Backward Compatibility Shims (Temporary) ---
    # TODO: Remove these methods after refactoring is complete and all callers are updated.

    async def broadcast_event(
        self,
        event_type: EventTypes,
        data: Dict[str, Any],
        exclude_transport_id: Optional[str] = None,
        test_mode: bool = False,
    ) -> None:
        """
        [DEPRECATED] Broadcasts an event to all relevant transports via publish_event.
        The `exclude_transport_id` parameter is not fully supported in the new model.
        """
        self._logger.warning(
            "DEPRECATED: EventCoordinator.broadcast_event is deprecated and will be removed. "
            "Use publish_event with an InternalEvent object instead."
        )
        if exclude_transport_id:
            self._logger.warning(
                f"broadcast_event called with exclude_transport_id='{exclude_transport_id}'. "
                "This exclusion is not directly supported by the new publish_event mechanism "
                "and may not behave as expected."
            )
        if test_mode:
            self._logger.info(
                f"broadcast_event called with test_mode=True for event_type='{event_type.value}'. "
                "Passing test_mode in metadata."
            )

        metadata = {"deprecated_test_mode": test_mode}
        internal_event = InternalEvent(event_type=event_type, data=data, metadata=metadata)
        await self.publish_event(internal_event)

    async def send_event_to_transport(
        self,
        transport_id: str,
        event_type: EventTypes,
        data: Dict[str, Any],
        test_mode: bool = False,
    ) -> None:
        """
        [DEPRECATED] Sends an event directly to a specific transport adapter.
        """
        self._logger.warning(
            "DEPRECATED: EventCoordinator.send_event_to_transport is deprecated and will be removed. "
            "Direct transport communication should be handled differently or via EventSystem capabilities."
        )
        if test_mode:
            self._logger.warning(
                f"send_event_to_transport called with test_mode=True for transport_id='{transport_id}', "
                f"event_type='{event_type.value}'. The 'test_mode' parameter is not used by ITransportAdapter.send_event."
            )

        adapter = None
        async with self._lock:  # Protects read access to self._transport_adapters
            adapter = self._transport_adapters.get(transport_id)

        if adapter:
            try:
                # Note: ITransportAdapter.send_event does not accept test_mode.
                # If test_mode behavior is critical, adapter logic would need to change.
                await adapter.send_event(event_type, data)
                self._logger.debug(
                    f"Sent event '{event_type.value}' to transport '{transport_id}'."
                )
            except Exception as e:
                self._logger.error(
                    f"Error sending event '{event_type.value}' to transport '{transport_id}': {e}",
                    exc_info=True,
                )
        else:
            self._logger.warning(
                f"Transport adapter '{transport_id}' not found for send_event_to_transport."
            )

    async def subscribe_to_event_type(
        self, transport_id: str, event_type: EventTypes
    ) -> None:
        """
        [DEPRECATED] Stub for transport-specific event subscription.
        TODO: Implement proper transport-level subscription management if needed, or remove.
        """
        self._logger.warning(
            f"DEPRECATED: EventCoordinator.subscribe_to_event_type('{transport_id}', '{event_type.value}') "
            "is a stub and currently non-functional. It will be properly implemented or removed."
        )
        # Placeholder: Actual implementation would involve tracking subscriptions per transport.

    async def unsubscribe_from_event_type(
        self, transport_id: str, event_type: EventTypes
    ) -> None:
        """
        [DEPRECATED] Stub for transport-specific event unsubscription.
        TODO: Implement proper transport-level unsubscription management if needed, or remove.
        """
        self._logger.warning(
            f"DEPRECATED: EventCoordinator.unsubscribe_from_event_type('{transport_id}', '{event_type.value}') "
            "is a stub and currently non-functional. It will be properly implemented or removed."
        )
        # Placeholder: Actual implementation would involve modifying transport subscriptions.

    async def is_subscribed(self, transport_id: str, event_type: EventTypes) -> bool:
        """
        [DEPRECATED] Stub for checking transport-specific event subscription.
        TODO: Implement proper transport-level subscription checking if needed, or remove.
        """
        self._logger.warning(
            f"DEPRECATED: EventCoordinator.is_subscribed('{transport_id}', '{event_type.value}') "
            "is a stub and currently returns False. It will be properly implemented or removed."
        )
        # Placeholder: Actual implementation would check transport subscriptions.
        return False
