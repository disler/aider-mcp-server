import asyncio
import unittest
from unittest.mock import AsyncMock, patch

# Assuming EventSystem is in a file named event_system.py in the parent directory's src folder
# Adjust the import path if necessary based on the actual project structure
try:
    from aider_mcp_server.event_system import EventSystem
except ImportError:
    # Fallback import for different project structures
    import os
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
    from aider_mcp_server.event_system import EventSystem

    sys.path.pop(0)


# Mock the logger used by EventSystem
@patch("aider_mcp_server.event_system.logger")
class TestEventSystemBasics(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        """Set up a new EventSystem instance before each test."""
        self.event_system = EventSystem()

    async def test_subscribe(self, mock_logger):
        """Test that subscribing a callback adds it to the subscribers list."""
        event_type = "user_created"
        callback = AsyncMock()

        await self.event_system.subscribe(event_type, callback)

        self.assertIn(event_type, self.event_system._subscribers)
        self.assertIn(callback, self.event_system._subscribers[event_type])
        mock_logger.debug.assert_called_with(
            f"Callback {getattr(callback, '__name__', str(callback))} subscribed to event type '{event_type}'"
        )

    async def test_unsubscribe(self, mock_logger):
        """Test that unsubscribing a callback removes it."""
        event_type = "user_deleted"
        callback1 = AsyncMock()
        callback2 = AsyncMock()

        await self.event_system.subscribe(event_type, callback1)
        await self.event_system.subscribe(event_type, callback2)

        self.assertIn(callback1, self.event_system._subscribers[event_type])
        self.assertIn(callback2, self.event_system._subscribers[event_type])

        await self.event_system.unsubscribe(event_type, callback1)

        self.assertNotIn(callback1, self.event_system._subscribers[event_type])
        self.assertIn(callback2, self.event_system._subscribers[event_type])
        mock_logger.debug.assert_called_with(
            f"Callback {getattr(callback1, '__name__', str(callback1))} unsubscribed from event type '{event_type}'"
        )

        # Test removing the last subscriber removes the event type key
        await self.event_system.unsubscribe(event_type, callback2)
        self.assertNotIn(event_type, self.event_system._subscribers)
        mock_logger.debug.assert_called_with(f"Event type '{event_type}' removed as it has no more subscribers.")

    async def test_broadcast_basic(self, mock_logger):
        """Test that broadcasting an event calls the subscribed callbacks with correct data."""
        event_type = "data_updated"
        event_data = {"id": 123, "status": "processed"}
        callback1 = AsyncMock()
        callback2 = AsyncMock()

        await self.event_system.subscribe(event_type, callback1)
        await self.event_system.subscribe(event_type, callback2)

        await self.event_system.broadcast(event_type, event_data)

        callback1.assert_called_once_with(event_data)
        callback2.assert_called_once_with(event_data)
        mock_logger.debug.assert_called_with(f"Broadcasting event '{event_type}' to 2 subscribers. Data: {event_data}")


@patch("aider_mcp_server.event_system.logger")
class TestEventSystemErrorHandling(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        """Set up a new EventSystem instance before each test."""
        self.event_system = EventSystem()

    async def test_callback_error_isolation(self, mock_logger):
        """Test that an error in one callback does not prevent others from being called."""
        event_type = "task_failed"
        event_data = {"task_id": "abc"}

        # Callback that raises an exception
        error_callback = AsyncMock(side_effect=RuntimeError("Simulated callback error"))
        # Callback that should still be called
        successful_callback = AsyncMock()

        await self.event_system.subscribe(event_type, error_callback)
        await self.event_system.subscribe(event_type, successful_callback)

        await self.event_system.broadcast(event_type, event_data)

        # The successful callback should still have been called
        successful_callback.assert_called_once_with(event_data)
        # The error callback should have been called once (and raised)
        error_callback.assert_called_once_with(event_data)
        # Check that an error was logged for the failing callback
        mock_logger.error.assert_called_once()
        self.assertIn("Error in event callback", mock_logger.error.call_args[0][0])

    async def test_callback_error_logging(self, mock_logger):
        """Test that errors in callbacks are logged correctly."""
        event_type = "service_unavailable"
        event_data = {"service": "db"}

        error_callback = AsyncMock(side_effect=ValueError("Invalid service config"))

        await self.event_system.subscribe(event_type, error_callback)
        await self.event_system.broadcast(event_type, event_data)

        error_callback.assert_called_once_with(event_data)

        # Check that logger.error was called with the correct message and exc_info=True
        mock_logger.error.assert_called_once()
        call_args, call_kwargs = mock_logger.error.call_args
        self.assertIn("Error in event callback", call_args[0])
        self.assertIn(event_type, call_args[0])
        # Check for the specific error message content
        self.assertIn("Invalid service config", call_args[0])
        self.assertTrue(call_kwargs.get("exc_info"))


@patch("aider_mcp_server.event_system.logger")
class TestEventSystemConcurrency(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        """Set up a new EventSystem instance before each test."""
        self.event_system = EventSystem()

    async def test_concurrent_broadcasts(self, mock_logger):
        """Test that multiple broadcasts can happen concurrently without interference."""
        event_type_a = "event_a"
        event_type_b = "event_b"
        data_a = {"value": 1}
        data_b = {"value": 2}

        callback_a1 = AsyncMock()
        callback_a2 = AsyncMock()
        callback_b1 = AsyncMock()

        await self.event_system.subscribe(event_type_a, callback_a1)
        await self.event_system.subscribe(event_type_a, callback_a2)
        await self.event_system.subscribe(event_type_b, callback_b1)

        # Broadcast events concurrently
        await asyncio.gather(
            self.event_system.broadcast(event_type_a, data_a), self.event_system.broadcast(event_type_b, data_b)
        )

        # Check that callbacks for event_a were called with data_a
        callback_a1.assert_called_once_with(data_a)
        callback_a2.assert_called_once_with(data_a)

        # Check that callbacks for event_b were called with data_b
        callback_b1.assert_called_once_with(data_b)

        # Check logger calls (order might vary due to concurrency)
        debug_calls = [args[0] for args, kwargs in mock_logger.debug.call_args_list]
        self.assertTrue(any(f"Broadcasting event '{event_type_a}'" in call for call in debug_calls))
        self.assertTrue(any(f"Broadcasting event '{event_type_b}'" in call for call in debug_calls))

    async def test_subscribe_during_broadcast(self, mock_logger):
        """Test subscribing during a broadcast."""
        event_type = "concurrent_subscribe"
        event_data_1 = {"step": 1}
        event_data_2 = {"step": 2}

        # Callback that will subscribe another callback during its execution
        async def subscribing_callback(data):
            await self.event_system.subscribe(event_type, new_callback)
            # Simulate some work
            await asyncio.sleep(0.01)

        initial_callback = AsyncMock(side_effect=subscribing_callback)
        new_callback = AsyncMock()

        await self.event_system.subscribe(event_type, initial_callback)

        # First broadcast: initial_callback runs and subscribes new_callback
        await self.event_system.broadcast(event_type, event_data_1)

        # Check that initial_callback was called, but new_callback was NOT called by the first broadcast
        initial_callback.assert_called_once_with(event_data_1)
        new_callback.assert_not_called()

        # Second broadcast: new_callback should now be called
        await self.event_system.broadcast(event_type, event_data_2)

        # Check that both callbacks are called by the second broadcast
        # initial_callback might be called again depending on test setup, but new_callback must be called
        new_callback.assert_called_once_with(event_data_2)
        # Check initial_callback was called at least once (by the first broadcast)
        self.assertGreaterEqual(initial_callback.call_count, 1)

        # Check logger calls
        debug_calls = [args[0] for args, kwargs in mock_logger.debug.call_args_list]
        self.assertTrue(
            any(
                f"Callback {getattr(new_callback, '__name__', str(new_callback))} subscribed to event type '{event_type}'"
                in call
                for call in debug_calls
            )
        )
        self.assertTrue(
            any(f"Broadcasting event '{event_type}' to 1 subscribers" in call for call in debug_calls)
        )  # First broadcast
        self.assertTrue(
            any(f"Broadcasting event '{event_type}' to 2 subscribers" in call for call in debug_calls)
        )  # Second broadcast


@patch("aider_mcp_server.event_system.logger")
class TestEventSystemEdgeCases(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        """Set up a new EventSystem instance before each test."""
        self.event_system = EventSystem()

    async def test_duplicate_subscription(self, mock_logger):
        """Test that subscribing the same callback multiple times only adds it once."""
        event_type = "duplicate_test"
        callback = AsyncMock()

        await self.event_system.subscribe(event_type, callback)
        await self.event_system.subscribe(event_type, callback)  # Subscribe again

        self.assertIn(event_type, self.event_system._subscribers)
        self.assertEqual(len(self.event_system._subscribers[event_type]), 1)
        self.assertIn(callback, self.event_system._subscribers[event_type])

        # Check logger calls - should indicate the second subscription was a duplicate
        debug_calls = [args[0] for args, kwargs in mock_logger.debug.call_args_list]
        self.assertTrue(
            any(
                f"Callback {getattr(callback, '__name__', str(callback))} subscribed to event type '{event_type}'"
                in call
                for call in debug_calls
            )
        )
        self.assertTrue(
            any(
                f"Callback {getattr(callback, '__name__', str(callback))} already subscribed to event type '{event_type}'"
                in call
                for call in debug_calls
            )
        )

    async def test_unsubscribe_nonexistent(self, mock_logger):
        """Test unsubscribing a callback that isn't subscribed or for a non-existent event type."""
        event_type_existing = "existing_event"
        event_type_nonexistent = "nonexistent_event"
        callback_existing = AsyncMock()
        callback_nonexistent = AsyncMock()

        await self.event_system.subscribe(event_type_existing, callback_existing)

        # Try to unsubscribe a callback not subscribed to the existing event
        await self.event_system.unsubscribe(event_type_existing, callback_nonexistent)
        self.assertIn(
            callback_existing, self.event_system._subscribers[event_type_existing]
        )  # Ensure existing callback is still there
        mock_logger.debug.assert_called_with(
            f"Callback {getattr(callback_nonexistent, '__name__', str(callback_nonexistent))} not found for event type '{event_type_existing}', or event type not registered."
        )
        mock_logger.debug.reset_mock()  # Reset mock for the next check

        # Try to unsubscribe from a non-existent event type
        await self.event_system.unsubscribe(event_type_nonexistent, callback_nonexistent)
        self.assertNotIn(event_type_nonexistent, self.event_system._subscribers)
        mock_logger.debug.assert_called_with(
            f"Callback {getattr(callback_nonexistent, '__name__', str(callback_nonexistent))} not found for event type '{event_type_nonexistent}', or event type not registered."
        )

    async def test_broadcast_no_subscribers(self, mock_logger):
        """Test broadcasting an event type with no registered subscribers."""
        event_type = "empty_event"
        event_data = {"status": "done"}

        # Ensure the event type is not in subscribers
        self.assertNotIn(event_type, self.event_system._subscribers)

        # Broadcast the event
        await self.event_system.broadcast(event_type, event_data)

        # Check that no errors occurred and a debug message was logged
        mock_logger.debug.assert_called_once_with(
            f"No subscribers for event type '{event_type}'. Event data: {event_data}"
        )
        # Ensure no error was logged
        mock_logger.error.assert_not_called()


if __name__ == "__main__":
    unittest.main()
