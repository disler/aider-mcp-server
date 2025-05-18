from unittest.mock import AsyncMock, MagicMock

import pytest

from aider_mcp_server.atoms.event_types import EventTypes
from aider_mcp_server.event_mediator import EventMediator
from aider_mcp_server.event_participant import IEventParticipant
from aider_mcp_server.event_system import EventSystem


@pytest.fixture
def mock_logger_factory():
    """Fixture for a mock logger factory."""
    mock_factory = MagicMock()
    mock_factory.return_value = MagicMock()  # This is the logger instance
    # Add verbose attribute to the logger instance
    mock_factory.return_value.verbose = MagicMock()
    mock_factory.return_value.debug = MagicMock()
    mock_factory.return_value.info = MagicMock()
    mock_factory.return_value.warning = MagicMock()
    mock_factory.return_value.error = MagicMock()
    return mock_factory


@pytest.fixture
def mock_event_system():
    """Fixture for a mock EventSystem."""
    return AsyncMock(spec=EventSystem)


@pytest.fixture
def event_mediator(mock_logger_factory, mock_event_system):
    """Fixture for an EventMediator instance."""
    return EventMediator(logger_factory=mock_logger_factory, event_system=mock_event_system)


@pytest.fixture
def mock_participant_factory():
    """Factory fixture to create mock IEventParticipant instances."""

    def _create_mock_participant(name: str, handled_events: set[EventTypes]):
        participant = AsyncMock(spec=IEventParticipant)
        participant.get_participant_name = MagicMock(return_value=name)
        participant.get_handled_events = MagicMock(return_value=handled_events)
        participant.handle_event = AsyncMock()
        return participant

    return _create_mock_participant


@pytest.mark.asyncio
async def test_register_participant(event_mediator, mock_participant_factory):
    """Test participant registration."""
    participant1 = mock_participant_factory("p1", {EventTypes.STATUS, EventTypes.PROGRESS})
    await event_mediator.register_participant(participant1)

    assert participant1 in event_mediator._participants
    assert participant1 in event_mediator._internal_handlers[EventTypes.STATUS]
    assert participant1 in event_mediator._internal_handlers[EventTypes.PROGRESS]
    # Check that the verbose call was made with the correct participant name
    # The order of event types in the list might vary, so we check the calls differently
    assert any(
        call.args[0].startswith("Registering participant: p1 for event types:")
        for call in event_mediator._logger.verbose.call_args_list
    )


@pytest.mark.asyncio
async def test_register_duplicate_participant(event_mediator, mock_participant_factory, mock_logger_factory):
    """Test registering the same participant twice."""
    participant1 = mock_participant_factory("p1", {EventTypes.STATUS})
    await event_mediator.register_participant(participant1)
    await event_mediator.register_participant(participant1)  # Register again

    assert len(event_mediator._internal_handlers[EventTypes.STATUS]) == 1
    mock_logger_factory.return_value.warning.assert_called_once_with("Participant p1 already registered. Skipping.")


@pytest.mark.asyncio
async def test_unregister_participant(event_mediator, mock_participant_factory):
    """Test participant unregistration."""
    participant1 = mock_participant_factory("p1", {EventTypes.STATUS, EventTypes.PROGRESS})
    await event_mediator.register_participant(participant1)
    await event_mediator.unregister_participant(participant1)

    assert participant1 not in event_mediator._participants
    assert EventTypes.STATUS not in event_mediator._internal_handlers  # Cleaned up
    assert EventTypes.PROGRESS not in event_mediator._internal_handlers  # Cleaned up
    event_mediator._logger.verbose.assert_any_call("Unregistering participant: p1")


@pytest.mark.asyncio
async def test_unregister_nonexistent_participant(event_mediator, mock_participant_factory, mock_logger_factory):
    """Test unregistering a participant that was not registered."""
    participant1 = mock_participant_factory("p1", {EventTypes.STATUS})
    # Do not register participant1
    await event_mediator.unregister_participant(participant1)

    mock_logger_factory.return_value.warning.assert_called_once_with(
        "Participant p1 not registered. Skipping unregistration."
    )


@pytest.mark.asyncio
async def test_emit_internal_event_single_handler(event_mediator, mock_participant_factory):
    """Test emitting an event to a single handler."""
    handler1 = mock_participant_factory("handler1", {EventTypes.STATUS})
    await event_mediator.register_participant(handler1)

    event_data = {"key": "value"}
    await event_mediator.emit_internal_event(EventTypes.STATUS, event_data)

    handler1.handle_event.assert_awaited_once_with(EventTypes.STATUS, event_data, None)


@pytest.mark.asyncio
async def test_emit_internal_event_multiple_handlers(event_mediator, mock_participant_factory):
    """Test emitting an event to multiple handlers."""
    handler1 = mock_participant_factory("handler1", {EventTypes.STATUS})
    handler2 = mock_participant_factory("handler2", {EventTypes.STATUS})
    handler_other_event = mock_participant_factory("handler_other", {EventTypes.PROGRESS})

    await event_mediator.register_participant(handler1)
    await event_mediator.register_participant(handler2)
    await event_mediator.register_participant(handler_other_event)

    event_data = {"key": "value"}
    await event_mediator.emit_internal_event(EventTypes.STATUS, event_data)

    handler1.handle_event.assert_awaited_once_with(EventTypes.STATUS, event_data, None)
    handler2.handle_event.assert_awaited_once_with(EventTypes.STATUS, event_data, None)
    handler_other_event.handle_event.assert_not_awaited()


@pytest.mark.asyncio
async def test_emit_internal_event_no_handlers(event_mediator, mock_logger_factory):
    """Test emitting an event with no registered handlers for that type."""
    event_data = {"key": "value"}
    await event_mediator.emit_internal_event(EventTypes.HEARTBEAT, event_data)
    # Check for debug log, no error should occur
    mock_logger_factory.return_value.debug.assert_any_call("No internal handlers registered for event type: heartbeat")


@pytest.mark.asyncio
async def test_emit_internal_event_originator_not_notified(event_mediator, mock_participant_factory):
    """Test that the originator of an event does not receive it back."""
    originator = mock_participant_factory("originator", {EventTypes.STATUS})
    handler1 = mock_participant_factory("handler1", {EventTypes.STATUS})

    await event_mediator.register_participant(originator)
    await event_mediator.register_participant(handler1)

    event_data = {"key": "value"}
    await event_mediator.emit_internal_event(EventTypes.STATUS, event_data, originator=originator)

    originator.handle_event.assert_not_awaited()
    handler1.handle_event.assert_awaited_once_with(EventTypes.STATUS, event_data, originator)


@pytest.mark.asyncio
async def test_emit_internal_event_handler_exception(event_mediator, mock_participant_factory, mock_logger_factory):
    """Test error handling when a participant's handle_event raises an exception."""
    handler_fails = mock_participant_factory("handler_fails", {EventTypes.STATUS})
    handler_succeeds = mock_participant_factory("handler_succeeds", {EventTypes.STATUS})

    exception_to_raise = ValueError("Test Exception")
    handler_fails.handle_event.side_effect = exception_to_raise

    await event_mediator.register_participant(handler_fails)
    await event_mediator.register_participant(handler_succeeds)

    event_data = {"key": "value"}
    await event_mediator.emit_internal_event(EventTypes.STATUS, event_data)

    handler_fails.handle_event.assert_awaited_once_with(EventTypes.STATUS, event_data, None)
    handler_succeeds.handle_event.assert_awaited_once_with(EventTypes.STATUS, event_data, None)

    mock_logger_factory.return_value.error.assert_called_once_with(
        f"Error handling internal event {EventTypes.STATUS.value} by handler_fails: {exception_to_raise}",
        exc_info=exception_to_raise,
    )


@pytest.mark.asyncio
async def test_subscribe_to_event_type_externally(event_mediator, mock_event_system):
    """Test subscribe_to_event_type_externally calls EventSystem."""
    transport_id = "transport1"
    event_type = EventTypes.STATUS
    await event_mediator.subscribe_to_event_type_externally(transport_id, event_type)
    mock_event_system.subscribe_to_event_type.assert_awaited_once_with(transport_id, event_type)


@pytest.mark.asyncio
async def test_unsubscribe_from_event_type_externally(event_mediator, mock_event_system):
    """Test unsubscribe_from_event_type_externally calls EventSystem."""
    transport_id = "transport1"
    event_type = EventTypes.STATUS
    await event_mediator.unsubscribe_from_event_type_externally(transport_id, event_type)
    mock_event_system.unsubscribe_from_event_type.assert_awaited_once_with(transport_id, event_type)


@pytest.mark.asyncio
async def test_update_transport_capabilities_externally(event_mediator, mock_event_system):
    """Test update_transport_capabilities_externally calls EventSystem."""
    transport_id = "transport1"
    capabilities = {EventTypes.STATUS, EventTypes.PROGRESS}
    await event_mediator.update_transport_capabilities_externally(transport_id, capabilities)
    mock_event_system.update_transport_capabilities.assert_awaited_once_with(transport_id, capabilities)


@pytest.mark.asyncio
async def test_update_transport_subscriptions_externally(event_mediator, mock_event_system):
    """Test update_transport_subscriptions_externally calls EventSystem."""
    transport_id = "transport1"
    subscriptions = {EventTypes.TOOL_RESULT}
    await event_mediator.update_transport_subscriptions_externally(transport_id, subscriptions)
    mock_event_system.update_transport_subscriptions.assert_awaited_once_with(transport_id, subscriptions)


@pytest.mark.asyncio
async def test_broadcast_event_externally(event_mediator, mock_event_system):
    """Test broadcast_event_externally calls EventSystem."""
    event_type = EventTypes.HEARTBEAT
    data = {"status": "alive"}
    exclude_transport_id = "transport_excluded"
    test_mode = True
    await event_mediator.broadcast_event_externally(event_type, data, exclude_transport_id, test_mode)
    mock_event_system.broadcast_event.assert_awaited_once_with(
        event_type, data, exclude_transport_id, test_mode=test_mode
    )


@pytest.mark.asyncio
async def test_send_event_to_transport_externally(event_mediator, mock_event_system):
    """Test send_event_to_transport_externally calls EventSystem."""
    transport_id = "transport_target"
    event_type = EventTypes.STATUS
    data = {"detail": "specific"}
    test_mode = False
    await event_mediator.send_event_to_transport_externally(transport_id, event_type, data, test_mode)
    mock_event_system.send_event_to_transport.assert_awaited_once_with(
        transport_id, event_type, data, test_mode=test_mode
    )


@pytest.mark.asyncio
async def test_is_subscribed_externally(event_mediator, mock_event_system):
    """Test is_subscribed_externally calls EventSystem and returns its result."""
    transport_id = "transport1"
    event_type = EventTypes.PROGRESS
    mock_event_system.is_subscribed.return_value = True  # Mock the return value

    result = await event_mediator.is_subscribed_externally(transport_id, event_type)

    mock_event_system.is_subscribed.assert_awaited_once_with(transport_id, event_type)
    assert result is True

    mock_event_system.is_subscribed.return_value = False
    result = await event_mediator.is_subscribed_externally(transport_id, event_type)
    assert result is False
