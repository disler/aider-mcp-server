from typing import Any, Dict, Optional, Set
from unittest.mock import AsyncMock, MagicMock

import pytest

from aider_mcp_server.atoms.types.event_types import EventTypes
from aider_mcp_server.molecules.events.event_mediator import EventMediator
from aider_mcp_server.molecules.events.event_participant import (
    EventParticipantBase,
    IEventParticipant,
)


def test_ievent_participant_abstract_methods():
    """Test that IEventParticipant enforces abstract methods."""

    class IncompleteBoth(IEventParticipant):
        def get_participant_name(self) -> str:
            return "incomplete_both"

        # Missing handle_event and get_handled_events

    with pytest.raises(TypeError) as excinfo_both:
        IncompleteBoth()
    assert "Can't instantiate abstract class IncompleteBoth" in str(excinfo_both.value)
    assert "get_handled_events" in str(excinfo_both.value)
    assert "handle_event" in str(excinfo_both.value)

    class IncompleteHandleEvent(IEventParticipant):
        def get_participant_name(self) -> str:
            return "incomplete_handle_event"

        def get_handled_events(self) -> Set[EventTypes]:
            return {EventTypes.STATUS}

        # Missing handle_event

    with pytest.raises(TypeError) as excinfo_handle:
        IncompleteHandleEvent()
    assert "Can't instantiate abstract class IncompleteHandleEvent" in str(excinfo_handle.value)
    assert "handle_event" in str(excinfo_handle.value)
    assert "get_handled_events" not in str(excinfo_handle.value)  # Make sure only missing one is listed

    class IncompleteGetHandledEvents(IEventParticipant):
        def get_participant_name(self) -> str:
            return "incomplete_get_handled_events"

        async def handle_event(
            self, event_type: EventTypes, data: Dict[str, Any], originator: Optional[IEventParticipant]
        ) -> None:
            pass

        # Missing get_handled_events

    with pytest.raises(TypeError) as excinfo_get_handled:
        IncompleteGetHandledEvents()
    assert "Can't instantiate abstract class IncompleteGetHandledEvents" in str(excinfo_get_handled.value)
    assert "get_handled_events" in str(excinfo_get_handled.value)
    assert "handle_event" not in str(excinfo_get_handled.value)  # Make sure only missing one is listed

    # A class that implements all abstract methods should be instantiable
    class CompleteParticipant(IEventParticipant):
        async def handle_event(
            self, event_type: EventTypes, data: Dict[str, Any], originator: Optional[IEventParticipant]
        ) -> None:
            pass

        def get_handled_events(self) -> Set[EventTypes]:
            return set()

        def get_participant_name(self) -> str:
            return "complete"

    try:
        CompleteParticipant()  # Should not raise
    except TypeError:
        pytest.fail("CompleteParticipant should be instantiable but raised TypeError")


@pytest.fixture
def mock_logger_factory():
    """Fixture for a mock logger factory."""
    mock_factory = MagicMock()
    mock_logger = MagicMock()
    mock_logger.verbose = MagicMock()
    mock_factory.return_value = mock_logger
    return mock_factory


@pytest.fixture
def mock_mediator():
    """Fixture for a mock EventMediator."""
    return AsyncMock(spec=EventMediator)


class ConcreteParticipant(EventParticipantBase):
    """A concrete implementation of EventParticipantBase for testing."""

    def __init__(
        self, participant_name: str, mediator: EventMediator, logger_factory: MagicMock, handled_events: Set[EventTypes]
    ):
        super().__init__(participant_name, mediator, logger_factory)
        self._handled_events = handled_events
        self.handled_event_calls = []  # To track calls to handle_event

    async def handle_event(
        self, event_type: EventTypes, data: Dict[str, Any], originator: Optional[IEventParticipant]
    ) -> None:
        self.handled_event_calls.append((event_type, data, originator))
        self._logger.info(f"ConcreteParticipant {self.get_participant_name()} handled {event_type.value}")

    def get_handled_events(self) -> Set[EventTypes]:
        return self._handled_events


@pytest.fixture
def concrete_participant(mock_mediator, mock_logger_factory):
    """Fixture for a ConcreteParticipant instance."""
    return ConcreteParticipant(
        participant_name="test_participant",
        mediator=mock_mediator,
        logger_factory=mock_logger_factory,
        handled_events={EventTypes.STATUS, EventTypes.PROGRESS},
    )


def test_participant_initialization(concrete_participant, mock_mediator, mock_logger_factory):
    """Test EventParticipantBase initialization."""
    assert concrete_participant.get_participant_name() == "test_participant"
    assert concrete_participant._mediator is mock_mediator
    mock_logger_factory.assert_called_once_with(
        "aider_mcp_server.molecules.events.event_participant.ConcreteParticipant[test_participant]"
    )
    concrete_participant._logger.verbose.assert_called_with("Initializing EventParticipantBase: test_participant")


@pytest.mark.asyncio
async def test_register_with_mediator(concrete_participant, mock_mediator):
    """Test register_with_mediator calls mediator's register_participant."""
    await concrete_participant.register_with_mediator()
    mock_mediator.register_participant.assert_awaited_once_with(concrete_participant)
    # Check that the verbose call was made with correct participant name
    # The order of event types in the list might vary, so we check the calls differently
    assert any(
        call.args[0].startswith("Participant test_participant registering with mediator for events:")
        for call in concrete_participant._logger.verbose.call_args_list
    )


@pytest.mark.asyncio
async def test_unregister_from_mediator(concrete_participant, mock_mediator):
    """Test unregister_from_mediator calls mediator's unregister_participant."""
    await concrete_participant.unregister_from_mediator()
    mock_mediator.unregister_participant.assert_awaited_once_with(concrete_participant)
    concrete_participant._logger.verbose.assert_any_call("Participant test_participant unregistering from mediator.")


@pytest.mark.asyncio
async def test_emit_event_via_mediator(concrete_participant, mock_mediator):
    """Test emit_event_via_mediator calls mediator's emit_internal_event."""
    event_type = EventTypes.TOOL_RESULT
    event_data = {"result": "success"}
    await concrete_participant.emit_event_via_mediator(event_type, event_data)
    mock_mediator.emit_internal_event.assert_awaited_once_with(event_type, event_data, originator=concrete_participant)
    concrete_participant._logger.verbose.assert_any_call(
        f"Participant test_participant emitting internal event via mediator: {event_type.value} - {event_data}"
    )


def test_get_handled_events(concrete_participant):
    """Test get_handled_events returns the correct set."""
    assert concrete_participant.get_handled_events() == {EventTypes.STATUS, EventTypes.PROGRESS}


@pytest.mark.asyncio
async def test_handle_event_implementation(concrete_participant):
    """Test the concrete handle_event implementation."""
    mock_originator = AsyncMock(spec=IEventParticipant)
    event_type = EventTypes.STATUS
    data = {"info": "testing"}

    await concrete_participant.handle_event(event_type, data, mock_originator)

    assert len(concrete_participant.handled_event_calls) == 1
    assert concrete_participant.handled_event_calls[0] == (event_type, data, mock_originator)
    concrete_participant._logger.info.assert_called_once_with(
        f"ConcreteParticipant test_participant handled {event_type.value}"
    )
