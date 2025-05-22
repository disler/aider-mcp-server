import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from typing import Dict, Any

from aider_mcp_server.event_system import EventSystem

# Path to the logger instance in the event_system module
MOCK_LOGGER_PATH = "aider_mcp_server.event_system.logger"


@pytest.fixture
def event_system():
    """Provides a new EventSystem instance for each test."""
    # Patch the logger for the EventSystem instance to avoid actual logging during tests
    # and allow for assertions on logger calls if needed by specific tests.
    # The EventSystem initializes its logger upon __init__.
    with patch(MOCK_LOGGER_PATH, new_callable=MagicMock) as mock_logger:
        system = EventSystem()
        system.logger = mock_logger # Attach mock_logger for easy access in tests if needed
        yield system


@pytest.fixture
def mock_callback_factory():
    """Factory to create AsyncMock callbacks."""
    def _create_mock_callback(name: str = "mock_callback") -> AsyncMock:
        # AsyncMock is awaitable and can be used as an EventCallback
        callback = AsyncMock()
        callback.__name__ = name  # For better logging/debugging by EventSystem
        return callback
    return _create_mock_callback


@pytest.mark.asyncio
async def test_subscribe_single_callback(event_system: EventSystem, mock_callback_factory):
    """Test subscribing a single callback."""
    callback = mock_callback_factory("cb1")
    event_type = "test_event"

    await event_system.subscribe(event_type, callback)

    assert event_type in event_system._subscribers
    assert callback in event_system._subscribers[event_type]
    event_system.logger.debug.assert_any_call(f"Callback cb1 subscribed to event type '{event_type}'")


@pytest.mark.asyncio
async def test_subscribe_multiple_callbacks_same_event(event_system: EventSystem, mock_callback_factory):
    """Test subscribing multiple different callbacks to the same event type."""
    cb1 = mock_callback_factory("cb1")
    cb2 = mock_callback_factory("cb2")
    event_type = "test_event"

    await event_system.subscribe(event_type, cb1)
    await event_system.subscribe(event_type, cb2)

    assert len(event_system._subscribers[event_type]) == 2
    assert cb1 in event_system._subscribers[event_type]
    assert cb2 in event_system._subscribers[event_type]


@pytest.mark.asyncio
async def test_subscribe_duplicate_callback(event_system: EventSystem, mock_callback_factory):
    """Test that subscribing the same callback instance multiple times results in one subscription."""
    callback = mock_callback_factory("cb_dup")
    event_type = "event_dup"

    await event_system.subscribe(event_type, callback)
    await event_system.subscribe(event_type, callback)  # Attempt duplicate subscription

    assert len(event_system._subscribers[event_type]) == 1
    event_system.logger.debug.assert_any_call(f"Callback cb_dup subscribed to event type '{event_type}'")
    event_system.logger.debug.assert_any_call(f"Callback cb_dup already subscribed to event type '{event_type}'")


@pytest.mark.asyncio
async def test_unsubscribe_single_callback(event_system: EventSystem, mock_callback_factory):
    """Test unsubscribing a single callback, ensuring event type is removed if empty."""
    callback = mock_callback_factory("cb_unsub")
    event_type = "event_unsub"

    await event_system.subscribe(event_type, callback)
    assert event_type in event_system._subscribers

    await event_system.unsubscribe(event_type, callback)

    assert event_type not in event_system._subscribers  # Event type should be removed
    event_system.logger.debug.assert_any_call(f"Callback cb_unsub unsubscribed from event type '{event_type}'")
    event_system.logger.debug.assert_any_call(f"Event type '{event_type}' removed as it has no more subscribers.")


@pytest.mark.asyncio
async def test_unsubscribe_one_of_many_callbacks(event_system: EventSystem, mock_callback_factory):
    """Test unsubscribing one of multiple callbacks for an event type."""
    cb1 = mock_callback_factory("cb1")
    cb2 = mock_callback_factory("cb2")
    event_type = "event_multi_unsub"

    await event_system.subscribe(event_type, cb1)
    await event_system.subscribe(event_type, cb2)
    assert len(event_system._subscribers[event_type]) == 2

    await event_system.unsubscribe(event_type, cb1)

    assert event_type in event_system._subscribers
    assert len(event_system._subscribers[event_type]) == 1
    assert cb1 not in event_system._subscribers[event_type]
    assert cb2 in event_system._subscribers[event_type]
    event_system.logger.debug.assert_any_call(f"Callback cb1 unsubscribed from event type '{event_type}'")


@pytest.mark.asyncio
async def test_unsubscribe_non_existent_callback(event_system: EventSystem, mock_callback_factory):
    """Test unsubscribing a callback that was not subscribed to an event."""
    cb1 = mock_callback_factory("cb1_exists")
    cb_non_existent = mock_callback_factory("cb_ghost")
    event_type = "event_ghost_cb"

    await event_system.subscribe(event_type, cb1)
    initial_subs_count = len(event_system._subscribers[event_type])

    await event_system.unsubscribe(event_type, cb_non_existent)

    assert len(event_system._subscribers[event_type]) == initial_subs_count # No change
    event_system.logger.debug.assert_any_call(
        f"Callback cb_ghost not found for event type '{event_type}', or event type not registered."
    )


@pytest.mark.asyncio
async def test_unsubscribe_from_non_existent_event_type(event_system: EventSystem, mock_callback_factory):
    """Test unsubscribing from an event type that has no subscriptions."""
    callback = mock_callback_factory("cb_lost_event")
    event_type = "event_never_existed"

    await event_system.unsubscribe(event_type, callback)
    # Ensure no error and appropriate log
    event_system.logger.debug.assert_any_call(
        f"Callback cb_lost_event not found for event type '{event_type}', or event type not registered."
    )
    assert event_type not in event_system._subscribers


@pytest.mark.asyncio
async def test_broadcast_single_subscriber(event_system: EventSystem, mock_callback_factory):
    """Test broadcasting an event to a single subscribed callback."""
    callback = mock_callback_factory("cb_broadcast_single")
    event_type = "event_b_single"
    event_data = {"data": "payload_single"}

    await event_system.subscribe(event_type, callback)
    await event_system.broadcast(event_type, event_data)

    callback.assert_awaited_once_with(event_data)


@pytest.mark.asyncio
async def test_broadcast_multiple_subscribers_same_event(event_system: EventSystem, mock_callback_factory):
    """Test broadcasting to multiple subscribers of the same event."""
    cb1 = mock_callback_factory("cb_b_multi1")
    cb2 = mock_callback_factory("cb_b_multi2")
    event_type = "event_b_multi"
    event_data = {"data": "payload_multi"}

    await event_system.subscribe(event_type, cb1)
    await event_system.subscribe(event_type, cb2)
    await event_system.broadcast(event_type, event_data)

    cb1.assert_awaited_once_with(event_data)
    cb2.assert_awaited_once_with(event_data)


@pytest.mark.asyncio
async def test_broadcast_correct_event_type_only(event_system: EventSystem, mock_callback_factory):
    """Test that events are only delivered to callbacks subscribed to that specific event type."""
    cb_event1 = mock_callback_factory("cb_event1")
    cb_event2 = mock_callback_factory("cb_event2")
    event_type1 = "event_one"
    event_type2 = "event_two"
    event_data1 = {"type": 1}

    await event_system.subscribe(event_type1, cb_event1)
    await event_system.subscribe(event_type2, cb_event2)
    await event_system.broadcast(event_type1, event_data1)

    cb_event1.assert_awaited_once_with(event_data1)
    cb_event2.assert_not_awaited()


@pytest.mark.asyncio
async def test_broadcast_to_unregistered_event_type(event_system: EventSystem):
    """Test broadcasting to an event type that has no subscribers and was never registered."""
    event_type = "event_unheard_of"
    event_data = {"data": "lost_data"}

    await event_system.broadcast(event_type, event_data)
    # Ensure no error and appropriate log
    event_system.logger.debug.assert_any_call(f"No subscribers for event type '{event_type}'. Event data: {event_data}")


@pytest.mark.asyncio
async def test_broadcast_with_empty_event_data(event_system: EventSystem, mock_callback_factory):
    """Test broadcasting with empty event data."""
    callback = mock_callback_factory("cb_empty_data")
    event_type = "event_empty"
    event_data: Dict[str, Any] = {}

    await event_system.subscribe(event_type, callback)
    await event_system.broadcast(event_type, event_data)

    callback.assert_awaited_once_with(event_data)


@pytest.mark.asyncio
async def test_broadcast_callback_exception_does_not_affect_others(event_system: EventSystem, mock_callback_factory):
    """Test that if one callback raises an exception, other callbacks are still processed."""
    cb_fails = mock_callback_factory("cb_fails")
    cb_succeeds1 = mock_callback_factory("cb_succeeds1")
    cb_succeeds2 = mock_callback_factory("cb_succeeds2")
    event_type = "event_robust"
    event_data = {"resilience": "test"}
    test_exception = RuntimeError("Callback failed intentionally")

    cb_fails.side_effect = test_exception

    await event_system.subscribe(event_type, cb_succeeds1)
    await event_system.subscribe(event_type, cb_fails) # Order might matter for some test setups
    await event_system.subscribe(event_type, cb_succeeds2)

    await event_system.broadcast(event_type, event_data)

    cb_succeeds1.assert_awaited_once_with(event_data)
    cb_fails.assert_awaited_once_with(event_data)
    cb_succeeds2.assert_awaited_once_with(event_data)

    event_system.logger.error.assert_called_once_with(
        f"Error in event callback {cb_fails.__name__} for event type '{event_type}': {test_exception}",
        exc_info=True
    )


@pytest.mark.asyncio
async def test_concurrent_broadcasts(event_system: EventSystem, mock_callback_factory):
    """Test multiple broadcasts happening concurrently."""
    cb_eventA_1 = mock_callback_factory("cb_eventA_1")
    cb_eventA_2 = mock_callback_factory("cb_eventA_2")
    cb_eventB_1 = mock_callback_factory("cb_eventB_1")

    event_A = "eventA"
    data_A = {"id": "A"}
    event_B = "eventB"
    data_B = {"id": "B"}

    await event_system.subscribe(event_A, cb_eventA_1)
    await event_system.subscribe(event_A, cb_eventA_2)
    await event_system.subscribe(event_B, cb_eventB_1)

    # Use asyncio.gather to run broadcasts concurrently
    await asyncio.gather(
        event_system.broadcast(event_A, data_A),
        event_system.broadcast(event_B, data_B),
        event_system.broadcast(event_A, data_A) # Broadcast event_A again
    )

    assert cb_eventA_1.await_count == 2
    cb_eventA_1.assert_any_await(data_A)
    assert cb_eventA_2.await_count == 2
    cb_eventA_2.assert_any_await(data_A)
    cb_eventB_1.assert_awaited_once_with(data_B)


@pytest.mark.asyncio
async def test_subscribe_during_broadcast(event_system: EventSystem, mock_callback_factory):
    """Test subscribing a new callback while a broadcast is notionally in progress."""
    event_type = "event_dynamic_sub"
    event_data = {"key": "value"}

    cb1_done_event = asyncio.Event()
    cb2 = mock_callback_factory("cb2_subscribed_late")

    async def cb1(data: Dict[str, Any]):
        # cb1 is called, and during its execution, cb2 is subscribed.
        # cb2 should not be called in *this* broadcast.
        await event_system.subscribe(event_type, cb2)
        cb1_done_event.set() # Signal cb1 has run and cb2 is subscribed

    # Use a real async def for cb1 to control its behavior
    cb1_mock_wrapper = mock_callback_factory("cb1_wrapper")
    cb1_mock_wrapper.side_effect = cb1

    await event_system.subscribe(event_type, cb1_mock_wrapper)
    
    # Initial broadcast
    await event_system.broadcast(event_type, event_data)

    cb1_mock_wrapper.assert_awaited_once_with(event_data)
    cb2.assert_not_awaited() # cb2 should not have been called in the first broadcast

    # Wait for cb1 to complete its logic (including subscribing cb2)
    await cb1_done_event.wait()

    # Second broadcast, cb2 should now be called
    await event_system.broadcast(event_type, event_data)
    cb2.assert_awaited_once_with(event_data) # Called in the second broadcast
    assert cb1_mock_wrapper.await_count == 2 # cb1 called again


@pytest.mark.asyncio
async def test_unsubscribe_during_broadcast(event_system: EventSystem, mock_callback_factory):
    """Test unsubscribing a callback (or another) while a broadcast is notionally in progress."""
    event_type = "event_dynamic_unsub"
    event_data = {"key": "value"}

    cb1_done_event = asyncio.Event()
    cb2_unsubscribed = mock_callback_factory("cb2_unsubscribed_during")
    cb3_remains = mock_callback_factory("cb3_remains")

    async def cb1(data: Dict[str, Any]):
        # cb1 is called. During its execution, cb2 is unsubscribed.
        # cb2 might still be called in *this* broadcast if it was in the copied list.
        await event_system.unsubscribe(event_type, cb2_unsubscribed)
        cb1_done_event.set()

    cb1_mock_wrapper = mock_callback_factory("cb1_wrapper_unsub")
    cb1_mock_wrapper.side_effect = cb1

    await event_system.subscribe(event_type, cb1_mock_wrapper)
    await event_system.subscribe(event_type, cb2_unsubscribed)
    await event_system.subscribe(event_type, cb3_remains)

    # Initial broadcast
    await event_system.broadcast(event_type, event_data)

    cb1_mock_wrapper.assert_awaited_once_with(event_data)
    # cb2_unsubscribed might be called or not in this first broadcast,
    # depending on its position relative to cb1_mock_wrapper in the internal list
    # and the exact timing of the unsubscribe. The key is its behavior in *future* broadcasts.
    # For this test, we'll check its call count after all broadcasts.
    cb3_remains.assert_awaited_once_with(event_data) # cb3 should always be called

    await cb1_done_event.wait() # Ensure cb1's unsubscribe action has completed

    # Store call counts before second broadcast
    cb1_call_count_before_2nd = cb1_mock_wrapper.await_count
    cb2_call_count_before_2nd = cb2_unsubscribed.await_count
    cb3_call_count_before_2nd = cb3_remains.await_count

    # Second broadcast
    await event_system.broadcast(event_type, event_data)

    # cb1_mock_wrapper and cb3_remains should be called again
    assert cb1_mock_wrapper.await_count == cb1_call_count_before_2nd + 1
    assert cb3_remains.await_count == cb3_call_count_before_2nd + 1

    # cb2_unsubscribed should NOT be called again, so its call count remains the same
    assert cb2_unsubscribed.await_count == cb2_call_count_before_2nd
    event_system.logger.debug.assert_any_call(f"Callback {cb2_unsubscribed.__name__} unsubscribed from event type '{event_type}'")


@pytest.mark.asyncio
async def test_unsubscribe_self_during_broadcast(event_system: EventSystem, mock_callback_factory):
    """Test a callback that unsubscribes itself during its execution."""
    event_type = "event_self_unsub"
    event_data = {"action": "disappear"}

    # Need a reference to the callback itself to pass to unsubscribe
    # Using a list to pass the reference into the async def due to Python's scoping
    cb_self_unsub_ref = [None]

    async def self_unsubscribing_cb(data: Dict[str, Any]):
        nonlocal cb_self_unsub_ref # To access the list
        if cb_self_unsub_ref[0]:
            await event_system.unsubscribe(event_type, cb_self_unsub_ref[0])

    # Create the mock, then assign it to the reference
    # The mock will wrap the actual self_unsubscribing_cb logic
    actual_cb_mock = AsyncMock(side_effect=self_unsubscribing_cb)
    actual_cb_mock.__name__ = "cb_self_unsub"
    cb_self_unsub_ref[0] = actual_cb_mock


    await event_system.subscribe(event_type, actual_cb_mock)

    # First broadcast: cb_self_unsub should be called and unsubscribe itself
    await event_system.broadcast(event_type, event_data)
    actual_cb_mock.assert_awaited_once_with(event_data)

    # Second broadcast: cb_self_unsub should NOT be called
    await event_system.broadcast(event_type, event_data)
    actual_cb_mock.assert_awaited_once_with(event_data) # Still only once from the first call
    event_system.logger.debug.assert_any_call(f"Callback {actual_cb_mock.__name__} unsubscribed from event type '{event_type}'")
    event_system.logger.debug.assert_any_call(f"Event type '{event_type}' removed as it has no more subscribers.")
