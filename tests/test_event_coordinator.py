from unittest.mock import AsyncMock, MagicMock

import pytest

from aider_mcp_server.atoms.event_types import EventTypes
from aider_mcp_server.event_coordinator import EventCoordinator
from aider_mcp_server.event_mediator import EventMediator  # Import EventMediator
from aider_mcp_server.event_participant import IEventParticipant


@pytest.fixture
def mock_logger_factory():
    """Fixture for a mock logger factory."""
    mock_factory = MagicMock()
    mock_logger = MagicMock()
    mock_logger.verbose = MagicMock()
    mock_logger.debug = MagicMock()
    mock_factory.return_value = mock_logger
    return mock_factory


@pytest.fixture
def mock_event_mediator():
    """Fixture for a mock EventMediator."""
    return AsyncMock(spec=EventMediator)


@pytest.fixture
def event_coordinator(mock_logger_factory, mock_event_mediator):
    """Fixture for an EventCoordinator instance, now using EventMediator."""
    coordinator = EventCoordinator(logger_factory=mock_logger_factory, event_mediator=mock_event_mediator)
    return coordinator


def test_initialization_with_mediator(event_coordinator, mock_logger_factory, mock_event_mediator):
    """Test EventCoordinator initialization with EventMediator."""
    assert event_coordinator._logger is mock_logger_factory.return_value
    assert event_coordinator._mediator is mock_event_mediator
    mock_logger_factory.assert_called_once_with("aider_mcp_server.event_coordinator")
    event_coordinator._logger.verbose.assert_called_once_with("EventCoordinator initialized, using EventMediator.")


def test_get_participant_name(event_coordinator):
    """Test get_participant_name returns the correct name."""
    assert event_coordinator.get_participant_name() == "EventCoordinator"


def test_get_handled_events(event_coordinator):
    """Test get_handled_events returns an empty set by default."""
    assert event_coordinator.get_handled_events() == set()
    event_coordinator._logger.debug.assert_called_once_with(
        "EventCoordinator currently handles no internal events directly."
    )


@pytest.mark.asyncio
async def test_handle_event_default_behavior(event_coordinator):
    """Test the default handle_event behavior (logs and does nothing else)."""
    mock_originator = AsyncMock(spec=IEventParticipant)
    mock_originator.get_participant_name.return_value = "MockOriginator"
    event_type = EventTypes.STATUS
    data = {"info": "some event"}

    await event_coordinator.handle_event(event_type, data, mock_originator)
    event_coordinator._logger.debug.assert_any_call(  # Use any_call due to other debug logs
        f"EventCoordinator received internal event {event_type.value} from MockOriginator, but has no specific handler. Data: {data}"
    )


@pytest.mark.asyncio
async def test_subscribe_to_event_type(event_coordinator, mock_event_mediator):
    """Test subscribe_to_event_type delegates to mediator."""
    transport_id = "transport1"
    event_type = EventTypes.STATUS
    await event_coordinator.subscribe_to_event_type(transport_id, event_type)
    mock_event_mediator.subscribe_to_event_type_externally.assert_awaited_once_with(transport_id, event_type)


@pytest.mark.asyncio
async def test_unsubscribe_from_event_type(event_coordinator, mock_event_mediator):
    """Test unsubscribe_from_event_type delegates to mediator."""
    transport_id = "transport1"
    event_type = EventTypes.STATUS
    await event_coordinator.unsubscribe_from_event_type(transport_id, event_type)
    mock_event_mediator.unsubscribe_from_event_type_externally.assert_awaited_once_with(transport_id, event_type)


@pytest.mark.asyncio
async def test_update_transport_capabilities(event_coordinator, mock_event_mediator):
    """Test update_transport_capabilities delegates to mediator."""
    transport_id = "transport1"
    capabilities = {EventTypes.PROGRESS, EventTypes.TOOL_RESULT}
    await event_coordinator.update_transport_capabilities(transport_id, capabilities)
    mock_event_mediator.update_transport_capabilities_externally.assert_awaited_once_with(transport_id, capabilities)


@pytest.mark.asyncio
async def test_update_transport_subscriptions(event_coordinator, mock_event_mediator):
    """Test update_transport_subscriptions delegates to mediator."""
    transport_id = "transport1"
    subscriptions = {EventTypes.HEARTBEAT}
    await event_coordinator.update_transport_subscriptions(transport_id, subscriptions)
    mock_event_mediator.update_transport_subscriptions_externally.assert_awaited_once_with(transport_id, subscriptions)


@pytest.mark.asyncio
async def test_broadcast_event(event_coordinator, mock_event_mediator):
    """Test broadcast_event delegates to mediator."""
    event_type = EventTypes.STATUS
    data = {"message": "global update"}
    exclude_transport_id = "transport_excluded"
    test_mode = True

    await event_coordinator.broadcast_event(event_type, data, exclude_transport_id, test_mode)
    mock_event_mediator.broadcast_event_externally.assert_awaited_once_with(
        event_type, data, exclude_transport_id, test_mode=test_mode
    )


@pytest.mark.asyncio
async def test_send_event_to_transport(event_coordinator, mock_event_mediator):
    """Test send_event_to_transport delegates to mediator."""
    transport_id = "transport_target"
    event_type = EventTypes.TOOL_RESULT
    data = {"result": "success"}
    test_mode = False

    await event_coordinator.send_event_to_transport(transport_id, event_type, data, test_mode)
    mock_event_mediator.send_event_to_transport_externally.assert_awaited_once_with(
        transport_id, event_type, data, test_mode=test_mode
    )


@pytest.mark.asyncio
async def test_is_subscribed(event_coordinator, mock_event_mediator):
    """Test is_subscribed delegates to mediator and returns its result."""
    transport_id = "transport1"
    event_type = EventTypes.PROGRESS
    mock_event_mediator.is_subscribed_externally.return_value = True  # Mock mediator's response

    result = await event_coordinator.is_subscribed(transport_id, event_type)

    mock_event_mediator.is_subscribed_externally.assert_awaited_once_with(transport_id, event_type)
    assert result is True
    event_coordinator._logger.verbose.assert_any_call(  # Check for verbose logging
        f"EventCoordinator: Mediator reports transport {transport_id} subscribed to {event_type.value}: True"
    )

    mock_event_mediator.is_subscribed_externally.return_value = False
    result = await event_coordinator.is_subscribed(transport_id, event_type)
    assert result is False
    event_coordinator._logger.verbose.assert_any_call(
        f"EventCoordinator: Mediator reports transport {transport_id} subscribed to {event_type.value}: False"
    )
