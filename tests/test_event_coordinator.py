import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aider_mcp_server.event_coordinator import EventCoordinator
from aider_mcp_server.atoms.event_types import EventTypes
from aider_mcp_server.transport_registry import TransportRegistry
from aider_mcp_server.interfaces.transport_adapter import ITransportAdapter


@pytest.fixture
def event_coordinator():
    transport_registry = MagicMock(spec=TransportRegistry)
    logger_factory = MagicMock()
    return EventCoordinator(transport_registry, logger_factory)


@pytest.fixture
def mock_transport_adapter():
    return AsyncMock(spec=ITransportAdapter)


@pytest.mark.asyncio
async def test_subscribe_to_event_type(event_coordinator, mock_transport_adapter):
    # Test subscribing to an event type
    await event_coordinator.subscribe_to_event_type("transport_id", EventTypes.STATUS)
    assert "transport_id" in event_coordinator._transport_subscriptions
    assert EventTypes.STATUS in event_coordinator._transport_subscriptions["transport_id"]


@pytest.mark.asyncio
async def test_unsubscribe_from_event_type(event_coordinator, mock_transport_adapter):
    # Test unsubscribing from an event type
    await event_coordinator.subscribe_to_event_type("transport_id", EventTypes.STATUS)
    await event_coordinator.unsubscribe_from_event_type("transport_id", EventTypes.STATUS)
    assert "transport_id" in event_coordinator._transport_subscriptions
    assert EventTypes.STATUS not in event_coordinator._transport_subscriptions["transport_id"]


@pytest.mark.asyncio
async def test_update_transport_capabilities(event_coordinator, mock_transport_adapter):
    # Test updating transport capabilities
    await event_coordinator.update_transport_capabilities("transport_id", {EventTypes.STATUS})
    assert "transport_id" in event_coordinator._transport_capabilities
    assert EventTypes.STATUS in event_coordinator._transport_capabilities["transport_id"]


@pytest.mark.asyncio
async def test_update_transport_subscriptions(event_coordinator, mock_transport_adapter):
    # Test updating transport subscriptions
    await event_coordinator.update_transport_subscriptions("transport_id", {EventTypes.STATUS})
    assert "transport_id" in event_coordinator._transport_subscriptions
    assert EventTypes.STATUS in event_coordinator._transport_subscriptions["transport_id"]


@pytest.mark.asyncio
async def test_broadcast_event(event_coordinator, mock_transport_adapter):
    # Test broadcasting an event
    await event_coordinator.subscribe_to_event_type("transport_id", EventTypes.STATUS)
    event_coordinator._transport_registry.get_transport.return_value = mock_transport_adapter
    await event_coordinator.broadcast_event(EventTypes.STATUS, {"data": "test"})
    mock_transport_adapter.send_event.assert_awaited_once_with(EventTypes.STATUS, {"data": "test"})


@pytest.mark.asyncio
async def test_send_event_to_transport(event_coordinator, mock_transport_adapter):
    # Test sending an event to a specific transport
    event_coordinator._transport_registry.get_transport.return_value = mock_transport_adapter
    await event_coordinator.send_event_to_transport("transport_id", EventTypes.STATUS, {"data": "test"})
    mock_transport_adapter.send_event.assert_awaited_once_with(EventTypes.STATUS, {"data": "test"})


@pytest.mark.asyncio
async def test_is_subscribed(event_coordinator, mock_transport_adapter):
    # Test checking if a transport is subscribed to an event type
    await event_coordinator.subscribe_to_event_type("transport_id", EventTypes.STATUS)
    assert await event_coordinator.is_subscribed("transport_id", EventTypes.STATUS)
    assert not await event_coordinator.is_subscribed("other_transport_id", EventTypes.STATUS)
