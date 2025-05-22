import asyncio
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

from aider_mcp_server.atoms.event_types import EventTypes
from aider_mcp_server.event_participant import IEventParticipant

# MODIFIED: Import EventSystem from event_system.py
from aider_mcp_server.event_system import EventSystem
from aider_mcp_server.mcp_types import LoggerFactory, LoggerProtocol


class EventMediator:
    """
    Central hub for event communication between IEventParticipant components.
    It also provides a facade for interacting with the EventSystem for external event dispatch.
    """

    _logger: LoggerProtocol
    # MODIFIED: Update type hint
    _event_system: EventSystem
    _internal_handlers: Dict[EventTypes, List[IEventParticipant]]
    _participants: Set[IEventParticipant]  # To keep track of registered participants

    # MODIFIED: Update constructor parameter type hint
    def __init__(self, logger_factory: LoggerFactory, event_system: EventSystem):
        self._logger = logger_factory(__name__)
        self._event_system = event_system
        self._internal_handlers = defaultdict(list)
        self._participants = set()
        self._logger.verbose("EventMediator initialized.")

    async def register_participant(self, participant: IEventParticipant) -> None:
        """Registers an event participant with the mediator."""
        if participant in self._participants:
            self._logger.warning(f"Participant {participant.get_participant_name()} already registered. Skipping.")
            return

        self._participants.add(participant)
        handled_event_types = participant.get_handled_events()
        self._logger.verbose(
            f"Registering participant: {participant.get_participant_name()} for event types: {[et.value for et in handled_event_types]}"
        )
        for event_type in handled_event_types:
            if participant not in self._internal_handlers[event_type]:
                self._internal_handlers[event_type].append(participant)
            else:
                self._logger.debug(
                    f"Participant {participant.get_participant_name()} already in handler list for {event_type.value}"
                )

    async def unregister_participant(self, participant: IEventParticipant) -> None:
        """Unregisters an event participant from the mediator."""
        if participant not in self._participants:
            self._logger.warning(
                f"Participant {participant.get_participant_name()} not registered. Skipping unregistration."
            )
            return

        self._logger.verbose(f"Unregistering participant: {participant.get_participant_name()}")
        self._participants.remove(participant)
        for event_type in list(self._internal_handlers.keys()):  # Iterate over a copy of keys for safe modification
            if participant in self._internal_handlers[event_type]:
                self._internal_handlers[event_type].remove(participant)
                if not self._internal_handlers[event_type]:  # Cleanup if no handlers left
                    del self._internal_handlers[event_type]

    async def emit_internal_event(
        self,
        event_type: EventTypes,
        data: Dict[str, Any],
        originator: Optional[IEventParticipant] = None,
    ) -> None:
        """
        Emits an event to internal participants subscribed to this event type.

        Args:
            event_type: The type of the event.
            data: The event data.
            originator: The participant that originally emitted the event (to avoid self-notification if needed).
        """
        originator_name = originator.get_participant_name() if originator else "None"
        self._logger.verbose(
            f"Emitting internal event: {event_type.value}, Data: {data}, Originator: {originator_name}"
        )

        tasks = []
        if event_type in self._internal_handlers:
            # Create a stable list of handlers for this event type before iterating
            current_handlers = list(self._internal_handlers[event_type])
            for handler in current_handlers:
                if handler is not originator:  # Avoid sending event back to originator
                    self._logger.debug(
                        f"Queueing internal event {event_type.value} for handler: {handler.get_participant_name()}"
                    )
                    tasks.append(handler.handle_event(event_type, data, originator))
                else:
                    self._logger.debug(
                        f"Skipping sending internal event {event_type.value} to originator: {handler.get_participant_name()}"
                    )

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # Re-fetch handlers for logging in case of concurrent modification (though less likely with list copy)
            current_handlers_for_logging = [
                h for h in self._internal_handlers.get(event_type, []) if h is not originator
            ]
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    if i < len(current_handlers_for_logging):
                        handler_for_task = current_handlers_for_logging[i]
                        self._logger.error(
                            f"Error handling internal event {event_type.value} by {handler_for_task.get_participant_name()}: {result}",
                            exc_info=result,
                        )
                    else:  # Should not happen if task list matches handler list
                        self._logger.error(
                            f"Error handling internal event {event_type.value} by an unknown handler (index out of bounds): {result}",
                            exc_info=result,
                        )
        elif not self._internal_handlers.get(event_type):
            self._logger.debug(f"No internal handlers registered for event type: {event_type.value}")

    # --- Methods to facade EventSystem for external communication ---

    async def broadcast_event_externally(
        self,
        event_type: EventTypes,
        data: Dict[str, Any],
        exclude_transport_id: Optional[str] = None,  # Not supported by simple EventSystem
        test_mode: bool = False,  # Not supported by simple EventSystem
    ) -> None:
        if exclude_transport_id is not None:
            self._logger.warning(
                f"Mediator: 'exclude_transport_id' parameter is not supported by the current EventSystem when broadcasting event {event_type.value}. It will be ignored."
            )
        if test_mode:
            self._logger.warning(
                f"Mediator: 'test_mode' parameter is not supported by the current EventSystem when broadcasting event {event_type.value}. It will be ignored."
            )

        self._logger.verbose(
            f"Mediator: Broadcasting event {event_type.value} externally via EventSystem. Data: {data}"
        )
        # Use the simple EventSystem's broadcast method
        await self._event_system.broadcast(event_type.value, data)
