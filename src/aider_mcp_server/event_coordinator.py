"""
Event Coordinator for handling event publishing and subscription functionality.
It acts as a client to the EventMediator to dispatch events externally
and can participate in internal event handling if needed.
"""

from typing import Any, Dict, Optional, Set

from aider_mcp_server.atoms.event_types import EventTypes

# EventSystem is no longer a direct dependency here
# TransportAdapterRegistry is no longer a direct dependency here
from aider_mcp_server.event_mediator import EventMediator
from aider_mcp_server.event_participant import IEventParticipant  # Using interface directly
from aider_mcp_server.mcp_types import LoggerFactory, LoggerProtocol


class EventCoordinator(IEventParticipant):  # Implement IEventParticipant
    _logger: LoggerProtocol
    _mediator: EventMediator

    def __init__(
        self,
        logger_factory: LoggerFactory,
        event_mediator: EventMediator,  # Receives EventMediator
    ) -> None:
        # Removed transport_registry from constructor
        self._logger = logger_factory(__name__)
        self._mediator = event_mediator
        # self._event_system = EventSystem(transport_registry) # Removed

        # EventCoordinator might register itself if it needs to handle internal events.
        # For now, its primary role is to use the mediator to interact with EventSystem.
        # If it needs to handle events, it would call:
        # asyncio.create_task(self._mediator.register_participant(self))
        # For this refactor, let's assume it doesn't handle internal events itself,
        # but this can be added later by defining get_handled_events and handle_event.
        self._logger.verbose("EventCoordinator initialized, using EventMediator.")

    # --- IEventParticipant implementation ---
    def get_participant_name(self) -> str:
        return "EventCoordinator"

    def get_handled_events(self) -> Set[EventTypes]:
        """
        Defines which event types this EventCoordinator instance itself handles
        from the internal mediator bus. By default, none.
        """
        self._logger.debug(f"{self.get_participant_name()} currently handles no internal events directly.")
        return set()

    async def handle_event(
        self, event_type: EventTypes, data: Dict[str, Any], originator: Optional[IEventParticipant]
    ) -> None:
        """
        Handles events if EventCoordinator is subscribed to them via the mediator.
        By default, this implementation does nothing.
        """
        originator_name = originator.get_participant_name() if originator else "N/A"
        self._logger.debug(
            f"{self.get_participant_name()} received internal event {event_type.value} from {originator_name}, but has no specific handler. Data: {data}"
        )
        # If EventCoordinator were to handle specific internal events, logic would go here.
        pass

    # --- Methods for external event dispatch via Mediator ---

    async def subscribe_to_event_type(self, transport_id: str, event_type: EventTypes) -> None:
        self._logger.verbose(
            f"EventCoordinator: Requesting mediator to subscribe transport {transport_id} to event type {event_type.value}"
        )
        await self._mediator.subscribe_to_event_type_externally(transport_id, event_type)

    async def unsubscribe_from_event_type(self, transport_id: str, event_type: EventTypes) -> None:
        self._logger.verbose(
            f"EventCoordinator: Requesting mediator to unsubscribe transport {transport_id} from event type {event_type.value}"
        )
        await self._mediator.unsubscribe_from_event_type_externally(transport_id, event_type)

    async def update_transport_capabilities(self, transport_id: str, capabilities: Set[EventTypes]) -> None:
        self._logger.verbose(
            f"EventCoordinator: Requesting mediator to update capabilities for transport {transport_id}: {[c.value for c in capabilities]}"
        )
        await self._mediator.update_transport_capabilities_externally(transport_id, capabilities)

    async def update_transport_subscriptions(self, transport_id: str, subscriptions: Set[EventTypes]) -> None:
        self._logger.verbose(
            f"EventCoordinator: Requesting mediator to update subscriptions for transport {transport_id}: {[s.value for s in subscriptions]}"
        )
        await self._mediator.update_transport_subscriptions_externally(transport_id, subscriptions)

    async def broadcast_event(
        self,
        event_type: EventTypes,
        data: Dict[str, Any],
        exclude_transport_id: Optional[str] = None,
        test_mode: bool = False,
    ) -> None:
        self._logger.verbose(
            f"EventCoordinator: Requesting mediator to broadcast event {event_type.value} externally. Data: {data}, Exclude: {exclude_transport_id}, TestMode: {test_mode}"
        )
        await self._mediator.broadcast_event_externally(event_type, data, exclude_transport_id, test_mode=test_mode)

    async def send_event_to_transport(
        self,
        transport_id: str,
        event_type: EventTypes,
        data: Dict[str, Any],
        test_mode: bool = False,
    ) -> None:
        self._logger.verbose(
            f"EventCoordinator: Requesting mediator to send event {event_type.value} to transport {transport_id} externally. Data: {data}, TestMode: {test_mode}"
        )
        await self._mediator.send_event_to_transport_externally(transport_id, event_type, data, test_mode=test_mode)

    async def is_subscribed(self, transport_id: str, event_type: EventTypes) -> bool:
        self._logger.verbose(
            f"EventCoordinator: Requesting mediator to check if transport {transport_id} is subscribed to event type {event_type.value}"
        )
        subscribed = await self._mediator.is_subscribed_externally(transport_id, event_type)
        self._logger.verbose(
            f"EventCoordinator: Mediator reports transport {transport_id} subscribed to {event_type.value}: {subscribed}"
        )
        return subscribed
