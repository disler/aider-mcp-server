from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional, Set

from aider_mcp_server.atoms.types.event_types import EventTypes
from aider_mcp_server.atoms.types.mcp_types import LoggerFactory, LoggerProtocol

if TYPE_CHECKING:
    from aider_mcp_server.molecules.events.event_mediator import EventMediator


class IEventParticipant(ABC):
    """
    Interface for components that can participate in event handling via an EventMediator.
    """

    @abstractmethod
    async def handle_event(
        self, event_type: EventTypes, data: Dict[str, Any], originator: Optional["IEventParticipant"]
    ) -> None:
        """
        Handle an incoming event.

        Args:
            event_type: The type of the event.
            data: The event data.
            originator: The participant that originally emitted the event, if any.
        """
        pass

    @abstractmethod
    def get_handled_events(self) -> Set[EventTypes]:
        """
        Get the set of event types that this participant handles.

        Returns:
            A set of EventTypes.
        """
        pass

    @abstractmethod
    def get_participant_name(self) -> str:
        """
        Get a unique name for this participant, used for logging and identification.

        Returns:
            A string name for the participant.
        """
        pass


class EventParticipantBase(IEventParticipant):
    """
    Abstract base class for event participants, providing common functionality.
    """

    _mediator: "EventMediator"
    _logger: LoggerProtocol
    _participant_name: str

    def __init__(
        self,
        participant_name: str,
        mediator: "EventMediator",
        logger_factory: LoggerFactory,
    ):
        self._participant_name = participant_name
        self._logger = logger_factory(f"{__name__}.{self.__class__.__name__}[{participant_name}]")
        self._mediator = mediator
        self._logger.verbose(f"Initializing EventParticipantBase: {self.get_participant_name()}")

    def get_participant_name(self) -> str:
        return self._participant_name

    async def register_with_mediator(self) -> None:
        """Registers this participant with the mediator."""
        self._logger.verbose(
            f"Participant {self.get_participant_name()} registering with mediator for events: {[et.value for et in self.get_handled_events()]}"
        )
        await self._mediator.register_participant(self)

    async def unregister_from_mediator(self) -> None:
        """Unregisters this participant from the mediator."""
        self._logger.verbose(f"Participant {self.get_participant_name()} unregistering from mediator.")
        await self._mediator.unregister_participant(self)

    async def emit_event_via_mediator(self, event_type: EventTypes, data: Dict[str, Any]) -> None:
        """
        Emits an event through the mediator for internal component communication.

        Args:
            event_type: The type of the event.
            data: The event data.
        """
        self._logger.verbose(
            f"Participant {self.get_participant_name()} emitting internal event via mediator: {event_type.value} - {data}"
        )
        await self._mediator.emit_internal_event(event_type, data, originator=self)

    # Methods to be implemented by subclasses
    @abstractmethod
    async def handle_event(
        self, event_type: EventTypes, data: Dict[str, Any], originator: Optional["IEventParticipant"]
    ) -> None:
        pass

    @abstractmethod
    def get_handled_events(self) -> Set[EventTypes]:
        pass
