import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, call, patch

from aider_mcp_server.templates.initialization.initialization_sequence import InitializationSequence

# Mock get_logger before it's used by the module under test
mock_logger_instance = MagicMock()
mock_get_logger_global = MagicMock(return_value=mock_logger_instance)


class TestInitializationSequence(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Reset logger mock calls for each test
        mock_logger_instance.reset_mock()

        # Patch get_logger for the InitializationSequence module
        self.get_logger_patcher = patch("aider_mcp_server.templates.initialization.initialization_sequence.get_logger", mock_get_logger_global)
        self.mock_get_logger = self.get_logger_patcher.start()
        self.mock_get_logger.reset_mock()  # Ensure clean state for the get_logger function mock
        self.addCleanup(self.get_logger_patcher.stop)

        # Patch ApplicationCoordinator
        self.app_coordinator_patcher = patch(
            "aider_mcp_server.templates.initialization.initialization_sequence.ApplicationCoordinator", autospec=True
        )
        self.mock_app_coordinator_cls = self.app_coordinator_patcher.start()
        self.addCleanup(self.app_coordinator_patcher.stop)

        # Instance returned by mocked ApplicationCoordinator class
        self.mock_coordinator_instance = self.mock_app_coordinator_cls.return_value
        self.mock_coordinator_instance.initialize = AsyncMock()
        self.mock_coordinator_instance.register_transport = AsyncMock()
        self.mock_coordinator_instance.shutdown = AsyncMock()

        # Patch asyncio.wait_for to allow fine-grained control over it
        self.wait_for_patcher = patch("aider_mcp_server.templates.initialization.initialization_sequence.asyncio.wait_for")
        self.mock_wait_for = self.wait_for_patcher.start()

        # By default, make wait_for actually await the coroutine
        async def async_wait_for(coro, timeout):
            return await coro

        self.mock_wait_for.side_effect = async_wait_for
        self.addCleanup(self.wait_for_patcher.stop)

        # Instantiate the class under test
        self.init_seq = InitializationSequence()

    def tearDown(self):
        # Ensure all patchers are stopped
        patch.stopall()

    def test_constructor(self):
        self.mock_app_coordinator_cls.assert_called_once()
        self.assertFalse(self.init_seq._initialized)
        self.assertIsInstance(self.init_seq._initialization_lock, asyncio.Lock)
        self.mock_get_logger.assert_called_once_with("aider_mcp_server.initialization_sequence")
        self.assertIs(self.init_seq._logger, mock_logger_instance)

    async def test_initialize_successful_no_transports(self):
        await self.init_seq.initialize(transport_configs=None, timeout=10.0)

        self.mock_wait_for.assert_called_once()
        self.assertEqual(self.mock_wait_for.call_args.kwargs["timeout"], 10.0)
        # Check that the coroutine passed to wait_for was from coordinator.initialize()
        # This is implicitly tested by coordinator.initialize being awaited.
        self.mock_coordinator_instance.initialize.assert_awaited_once()
        self.mock_coordinator_instance.register_transport.assert_not_called()
        self.assertTrue(self.init_seq._initialized)
        mock_logger_instance.info.assert_any_call("Starting InitializationSequence initialization...")
        mock_logger_instance.debug.assert_any_call("Initializing ApplicationCoordinator with timeout 10.0s...")
        mock_logger_instance.info.assert_any_call("ApplicationCoordinator initialized successfully.")
        mock_logger_instance.info.assert_any_call("No transport configurations provided.")
        mock_logger_instance.info.assert_any_call("InitializationSequence initialization completed successfully.")

    async def test_initialize_successful_with_transports(self):
        transport_configs = [
            {"name": "transport1", "key": "value1"},
            {"name": "transport2", "port": 1234},
        ]
        await self.init_seq.initialize(transport_configs=transport_configs, timeout=15.0)

        self.mock_wait_for.assert_called_once()
        self.assertEqual(self.mock_wait_for.call_args.kwargs["timeout"], 15.0)
        self.mock_coordinator_instance.initialize.assert_awaited_once()

        self.mock_coordinator_instance.register_transport.assert_has_awaits(
            [
                call("transport1", key="value1"),
                call("transport2", port=1234),
            ]
        )
        self.assertTrue(self.init_seq._initialized)
        mock_logger_instance.info.assert_any_call(f"Registering {len(transport_configs)} transport(s)...")
        mock_logger_instance.debug.assert_any_call("Registering transport 'transport1' with config: {'key': 'value1'}")
        mock_logger_instance.info.assert_any_call("Transport 'transport1' registered successfully.")
        mock_logger_instance.debug.assert_any_call("Registering transport 'transport2' with config: {'port': 1234}")
        mock_logger_instance.info.assert_any_call("Transport 'transport2' registered successfully.")

    async def test_initialize_already_initialized(self):
        self.init_seq._initialized = True  # Simulate already initialized
        await self.init_seq.initialize()

        mock_logger_instance.info.assert_called_once_with("InitializationSequence already initialized.")
        self.mock_coordinator_instance.initialize.assert_not_called()
        self.mock_coordinator_instance.register_transport.assert_not_called()

    async def test_initialize_transport_config_missing_name(self):
        transport_configs = [{"key": "value"}]  # Missing 'name'
        await self.init_seq.initialize(transport_configs=transport_configs)

        self.mock_coordinator_instance.register_transport.assert_not_called()
        mock_logger_instance.error.assert_called_once_with("Transport configuration missing 'name' field. Skipping.")
        self.assertTrue(self.init_seq._initialized)  # Should still complete initialization

    async def test_initialize_coordinator_init_fails(self):
        self.mock_coordinator_instance.initialize.side_effect = Exception("Coordinator init boom!")

        # mock_wait_for should propagate the exception from the coro by awaiting it
        async def await_coro_side_effect(coro, timeout):
            return await coro

        self.mock_wait_for.side_effect = await_coro_side_effect

        with self.assertRaisesRegex(RuntimeError, "InitializationSequence sequence failed"):
            await self.init_seq.initialize()

        self.mock_coordinator_instance.initialize.assert_awaited_once()
        self.mock_coordinator_instance.shutdown.assert_awaited_once()  # Attempt cleanup
        self.assertFalse(self.init_seq._initialized)
        mock_logger_instance.error.assert_any_call(
            "InitializationSequence initialization failed: Coordinator init boom!", exc_info=True
        )
        mock_logger_instance.info.assert_any_call("Attempting cleanup after failed initialization...")

    async def test_initialize_coordinator_init_timeout(self):
        self.mock_wait_for.side_effect = asyncio.TimeoutError("Coordinator init timeout!")

        with self.assertRaisesRegex(RuntimeError, "InitializationSequence sequence timed out"):
            await self.init_seq.initialize(timeout=0.1)

        # initialize() is called to produce the coroutine passed to wait_for.
        self.mock_coordinator_instance.initialize.assert_called_once()
        # We check that wait_for was called with the correct timeout.
        self.mock_wait_for.assert_called_once()
        self.assertEqual(self.mock_wait_for.call_args.kwargs["timeout"], 0.1)

        self.mock_coordinator_instance.shutdown.assert_awaited_once()  # Attempt cleanup
        self.assertFalse(self.init_seq._initialized)
        mock_logger_instance.error.assert_any_call(
            "InitializationSequence initialization timed out after 0.1 seconds.", exc_info=True
        )

    async def test_initialize_transport_registration_fails(self):
        transport_configs = [{"name": "faulty_transport", "config": "bad"}]
        self.mock_coordinator_instance.register_transport.side_effect = Exception("Transport boom!")

        with self.assertRaisesRegex(RuntimeError, "InitializationSequence sequence failed"):
            await self.init_seq.initialize(transport_configs=transport_configs)

        self.mock_coordinator_instance.register_transport.assert_awaited_once_with("faulty_transport", config="bad")
        self.mock_coordinator_instance.shutdown.assert_awaited_once()  # Attempt cleanup
        self.assertFalse(self.init_seq._initialized)
        mock_logger_instance.error.assert_any_call(
            "Failed to initialize transport 'faulty_transport': Transport boom!", exc_info=True
        )

    async def test_initialize_cleanup_on_failure_itself_fails(self):
        self.mock_coordinator_instance.initialize.side_effect = Exception("Initial failure")
        self.mock_coordinator_instance.shutdown.side_effect = Exception("Cleanup failure")

        # ensure initialize coro runs and propagates its exception
        async def await_coro_side_effect(coro, timeout):
            return await coro

        self.mock_wait_for.side_effect = await_coro_side_effect

        with self.assertRaisesRegex(RuntimeError, "InitializationSequence sequence failed") as cm:
            await self.init_seq.initialize()

        self.assertEqual(str(cm.exception.__cause__), "Initial failure")

        self.mock_coordinator_instance.initialize.assert_awaited_once()
        self.mock_coordinator_instance.shutdown.assert_awaited_once()
        self.assertFalse(self.init_seq._initialized)
        mock_logger_instance.error.assert_any_call(
            "InitializationSequence initialization failed: Initial failure", exc_info=True
        )
        mock_logger_instance.error.assert_any_call(
            "Cleanup after failed initialization also failed: Cleanup failure", exc_info=True
        )

    async def test_shutdown_successful(self):
        # First, initialize
        await self.init_seq.initialize()
        self.assertTrue(self.init_seq._initialized)
        mock_logger_instance.reset_mock()  # Reset logs after init

        await self.init_seq.shutdown()

        self.mock_coordinator_instance.shutdown.assert_awaited_once()
        self.assertFalse(self.init_seq._initialized)
        mock_logger_instance.info.assert_any_call("Starting InitializationSequence shutdown...")
        mock_logger_instance.info.assert_any_call("InitializationSequence shutdown completed successfully.")

    async def test_shutdown_not_initialized(self):
        self.assertFalse(self.init_seq._initialized)  # Pre-condition
        await self.init_seq.shutdown()

        self.mock_coordinator_instance.shutdown.assert_not_called()
        mock_logger_instance.info.assert_called_once_with(
            "InitializationSequence not initialized or already shut down."
        )
        self.assertFalse(self.init_seq._initialized)

    async def test_shutdown_called_twice(self):
        # Initialize and shutdown once
        await self.init_seq.initialize()
        await self.init_seq.shutdown()
        self.assertFalse(self.init_seq._initialized)
        self.mock_coordinator_instance.shutdown.assert_awaited_once()  # From first shutdown
        mock_logger_instance.reset_mock()

        # Call shutdown again
        await self.init_seq.shutdown()
        self.mock_coordinator_instance.shutdown.assert_awaited_once()  # Should still be 1 (not called again)
        mock_logger_instance.info.assert_called_once_with(
            "InitializationSequence not initialized or already shut down."
        )
        self.assertFalse(self.init_seq._initialized)

    async def test_shutdown_coordinator_shutdown_fails(self):
        # Initialize first
        await self.init_seq.initialize()
        self.assertTrue(self.init_seq._initialized)
        mock_logger_instance.reset_mock()

        self.mock_coordinator_instance.shutdown.side_effect = Exception("Coordinator shutdown boom!")
        with self.assertRaisesRegex(RuntimeError, "InitializationSequence shutdown sequence failed"):
            await self.init_seq.shutdown()

        self.mock_coordinator_instance.shutdown.assert_awaited_once()
        self.assertFalse(self.init_seq._initialized)  # Should be set to False even on failure
        mock_logger_instance.error.assert_called_once_with(
            "InitializationSequence shutdown failed: Coordinator shutdown boom!", exc_info=True
        )

    async def test_concurrent_initialize_calls(self):
        # Test that the lock prevents re-entrant initialization
        # Make the first call to initialize hang inside the lock
        # then try to call it again.

        original_init_coro = self.mock_coordinator_instance.initialize
        init_call_count = 0

        async def slow_init(*args, **kwargs):
            nonlocal init_call_count
            init_call_count += 1
            if init_call_count == 1:  # First call
                # Simulate work that takes time, allowing other tasks to run
                await asyncio.sleep(0.02)
            return await original_init_coro(*args, **kwargs)

        self.mock_coordinator_instance.initialize = AsyncMock(side_effect=slow_init)

        # Ensure wait_for calls our slow_init and awaits it
        async def await_coro_side_effect(coro, timeout):
            return await coro

        self.mock_wait_for.side_effect = await_coro_side_effect

        task1_done = asyncio.Event()

        async def initialize_and_set_event(init_seq_instance):
            await init_seq_instance.initialize()
            task1_done.set()

        # Start the first initialization, it will be slow
        task1 = asyncio.create_task(initialize_and_set_event(self.init_seq))
        await asyncio.sleep(0.01)  # Give task1 a chance to acquire the lock

        # Try to initialize again while the first one is "in progress"
        # This call should see _initialized as False initially, but then wait for the lock
        # or, if the first one completes fast enough, it will see _initialized as True.
        # The current implementation logs "already initialized" if the first one finishes before
        # the second one acquires the lock and checks self._initialized.
        # If the second call acquires the lock *after* the first one released it and set _initialized = True,
        # then it will log "already initialized".
        # If the second call attempts to acquire the lock *while* the first one holds it, it will block.
        # Once the first call releases, the second call will acquire, then see _initialized = True.

        # To test the lock properly, we'd want the second call to attempt to enter the critical section
        # while the first is still inside.
        # The current "already initialized" check is outside the lock for the fast path.
        # Let's assume the first call completes and sets _initialized = True.

        # The second call will hit the `if self._initialized:` check.
        await self.init_seq.initialize()  # Second call

        await task1  # Ensure task1 completes

        # The coordinator's initialize method should only be called once by the first task.
        self.mock_coordinator_instance.initialize.assert_awaited_once()
        self.assertTrue(self.init_seq._initialized)

        # The first call completes initialization.
        # The second call should log "InitializationSequence already initialized."
        # Check logs for the "already initialized" message from the second call.
        # The first call logs its success.
        logs = [c[0][0] for c in mock_logger_instance.info.call_args_list]
        self.assertIn("InitializationSequence initialization completed successfully.", logs)
        self.assertIn("InitializationSequence already initialized.", logs)


if __name__ == "__main__":
    unittest.main()
