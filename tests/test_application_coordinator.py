import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from aider_mcp_server.interfaces.transport_adapter import ITransportAdapter
from aider_mcp_server.organisms.registries.handler_registry import RequestHandler
from aider_mcp_server.pages.application.coordinator import ApplicationCoordinator

# Mock get_logger before it's used by the module
mock_logger_instance = MagicMock()
mock_get_logger = MagicMock(return_value=mock_logger_instance)

# Keep a reference to the original get_logger if needed for other tests,
# but for ApplicationCoordinator, we want it mocked from the start.
# original_get_logger = get_logger # This line is problematic if module is reloaded.

# It's better to patch 'aider_mcp_server.pages.application.coordinator.get_logger'
# and other modules if they are instantiated by ApplicationCoordinator.


class TestApplicationCoordinator(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Reset singleton for each test
        ApplicationCoordinator._instance = None
        ApplicationCoordinator._initialized = False

        # Patch get_logger for the ApplicationCoordinator module and its children components
        self.get_logger_patcher_ac = patch("aider_mcp_server.pages.application.coordinator.get_logger", mock_get_logger)
        self.get_logger_patcher_es = patch(
            "aider_mcp_server.molecules.events.event_system.get_logger_func", mock_get_logger
        )  # EventSystem uses get_logger_func
        # EventCoordinator takes logger_factory parameter, no direct import to patch for its own logger.
        # ApplicationCoordinator passes the mocked get_logger to EventCoordinator.
        self.get_logger_patcher_rp = patch(
            "aider_mcp_server.organisms.processors.request_processor.get_logger", mock_get_logger
        )
        self.get_logger_patcher_tar = patch(
            "aider_mcp_server.organisms.registries.transport_adapter_registry.get_logger_func", mock_get_logger
        )  # TAR uses get_logger_func
        self.get_logger_patcher_hr = patch(
            "aider_mcp_server.organisms.registries.handler_registry.get_logger", mock_get_logger
        )

        self.mock_ac_logger = self.get_logger_patcher_ac.start()
        self.mock_es_logger = self.get_logger_patcher_es.start()
        # No separate mock_ec_logger as EventCoordinator uses the factory passed to it.
        self.mock_rp_logger = self.get_logger_patcher_rp.start()
        self.mock_tar_logger = self.get_logger_patcher_tar.start()
        self.mock_hr_logger = self.get_logger_patcher_hr.start()

        self.addCleanup(self.get_logger_patcher_ac.stop)
        self.addCleanup(self.get_logger_patcher_es.stop)
        # No cleanup for a separate ec_logger_patcher.
        self.addCleanup(self.get_logger_patcher_rp.stop)
        self.addCleanup(self.get_logger_patcher_tar.stop)
        self.addCleanup(self.get_logger_patcher_hr.stop)

        # Mock component classes
        self.event_system_patcher = patch("aider_mcp_server.pages.application.coordinator.EventSystem", autospec=True)
        self.event_coordinator_patcher = patch(
            "aider_mcp_server.pages.application.coordinator.EventCoordinator", autospec=True
        )
        self.request_processor_patcher = patch(
            "aider_mcp_server.pages.application.coordinator.RequestProcessor", autospec=True
        )
        self.transport_registry_patcher = patch(
            "aider_mcp_server.pages.application.coordinator.TransportAdapterRegistry", autospec=True
        )
        self.handler_registry_patcher = patch(
            "aider_mcp_server.pages.application.coordinator.HandlerRegistry", autospec=True
        )

        self.mock_event_system_cls = self.event_system_patcher.start()
        self.mock_event_coordinator_cls = self.event_coordinator_patcher.start()
        self.mock_request_processor_cls = self.request_processor_patcher.start()
        self.mock_transport_registry_cls = self.transport_registry_patcher.start()
        self.mock_handler_registry_cls = self.handler_registry_patcher.start()

        self.addCleanup(self.event_system_patcher.stop)
        self.addCleanup(self.event_coordinator_patcher.stop)
        self.addCleanup(self.request_processor_patcher.stop)
        self.addCleanup(self.transport_registry_patcher.stop)
        self.addCleanup(self.handler_registry_patcher.stop)

        # Instances returned by mocked classes
        self.mock_event_system = self.mock_event_system_cls.return_value
        self.mock_event_coordinator = self.mock_event_coordinator_cls.return_value
        self.mock_request_processor = self.mock_request_processor_cls.return_value
        self.mock_transport_registry = self.mock_transport_registry_cls.return_value
        self.mock_handler_registry = self.mock_handler_registry_cls.return_value

        # Ensure AsyncMock for async methods of component instances
        self.mock_transport_registry.discover_adapters = AsyncMock()
        self.mock_transport_registry.initialize_adapter = AsyncMock(return_value=MagicMock(spec=ITransportAdapter))
        self.mock_transport_registry.shutdown_all = AsyncMock()

        self.mock_event_coordinator.register_transport_adapter = AsyncMock()  # As per AppCoordinator's usage
        self.mock_event_coordinator.broadcast_event = AsyncMock()

        self.mock_request_processor.process_request = AsyncMock(return_value={"success": True})
        self.mock_request_processor.register_handler = MagicMock()  # It's synchronous

        self.mock_handler_registry.get_supported_request_types = MagicMock(return_value=[])
        self.mock_handler_registry.get_handler = MagicMock(return_value=None)
        self.mock_handler_registry.register_handler = MagicMock()  # Synchronous
        self.mock_handler_registry.register_handler_class = MagicMock()  # Synchronous

    def tearDown(self):
        # Explicitly stop patchers if not using addCleanup or if issues arise
        patch.stopall()
        # Reset singleton again to be absolutely sure for next test
        ApplicationCoordinator._instance = None
        ApplicationCoordinator._initialized = False

    def test_singleton_pattern(self):
        coord1 = ApplicationCoordinator()
        coord2 = ApplicationCoordinator()
        self.assertIs(coord1, coord2)
        self.assertTrue(ApplicationCoordinator._initialized)

    def test_init_components(self):
        # Test that __init__ (called via __new__) instantiates components
        coordinator = ApplicationCoordinator()
        self.mock_event_system_cls.assert_called_once()
        # Per Task 4 spec, EC is called with logger_factory and EventSystem
        self.mock_event_coordinator_cls.assert_called_once_with(mock_get_logger, self.mock_event_system)
        self.mock_request_processor_cls.assert_called_once()
        self.mock_transport_registry_cls.assert_called_once()
        self.mock_handler_registry_cls.assert_called_once()
        self.assertIsNotNone(coordinator._initialization_lock)
        self.assertTrue(ApplicationCoordinator._initialized)
        mock_logger_instance.info.assert_any_call("ApplicationCoordinator components instantiated.")

    async def test_initialize_method(self):
        coordinator = ApplicationCoordinator()

        mock_handler_func = AsyncMock()
        self.mock_handler_registry.get_supported_request_types.return_value = ["test_type"]
        self.mock_handler_registry.get_handler.return_value = mock_handler_func

        await coordinator.initialize()

        self.mock_transport_registry.discover_adapters.assert_called_once_with(package_name="transports")
        self.mock_handler_registry.get_supported_request_types.assert_called_once()
        self.mock_handler_registry.get_handler.assert_called_once_with("test_type")
        self.mock_request_processor.register_handler.assert_called_once_with("test_type", mock_handler_func)
        mock_logger_instance.info.assert_any_call("ApplicationCoordinator initialization complete.")

    async def test_register_transport(self):
        coordinator = ApplicationCoordinator()
        mock_transport_instance = AsyncMock(spec=ITransportAdapter)
        self.mock_transport_registry.initialize_adapter.return_value = mock_transport_instance

        kwargs = {"config_key": "config_value"}
        returned_transport = await coordinator.register_transport("test_transport", **kwargs)

        # ApplicationCoordinator calls initialize_adapter with (transport_name, coordinator_instance, config_dict)
        # The **kwargs from register_transport are not directly passed as **kwargs to initialize_adapter.
        # Config is hardcoded to {} in ApplicationCoordinator's call.
        self.mock_transport_registry.initialize_adapter.assert_called_once_with("test_transport", coordinator, {})
        # Per Task 9 spec, EC.register_transport is called. Using register_transport_adapter due to provided EC.
        self.mock_event_coordinator.register_transport_adapter.assert_called_once_with(mock_transport_instance)
        self.assertIs(returned_transport, mock_transport_instance)

    async def test_register_transport_failure(self):
        coordinator = ApplicationCoordinator()
        self.mock_transport_registry.initialize_adapter.return_value = None  # Simulate failure

        returned_transport = await coordinator.register_transport("test_transport_fail")
        self.assertIsNone(returned_transport)
        self.mock_event_coordinator.register_transport_adapter.assert_not_called()

    def test_register_handler(self):
        coordinator = ApplicationCoordinator()
        mock_handler: RequestHandler = AsyncMock()

        coordinator.register_handler("test_op", mock_handler)

        self.mock_handler_registry.register_handler.assert_called_once_with("test_op", mock_handler)
        self.mock_request_processor.register_handler.assert_called_once_with("test_op", mock_handler)

    def test_register_handler_class(self):
        coordinator = ApplicationCoordinator()

        class DummyHandlerClass:
            async def handle_op1(self, request):
                return {}

            async def handle_op2(self, request):
                return {}

        mock_op1_handler = DummyHandlerClass().handle_op1

        # Simulate HandlerRegistry's behavior after register_handler_class
        self.mock_handler_registry.get_supported_request_types.return_value = ["op1"]
        self.mock_handler_registry.get_handler.side_effect = (
            lambda request_type: mock_op1_handler if request_type == "op1" else None
        )

        coordinator.register_handler_class(DummyHandlerClass)

        self.mock_handler_registry.register_handler_class.assert_called_once_with(DummyHandlerClass)
        self.mock_handler_registry.get_supported_request_types.assert_called()  # Called multiple times
        self.mock_handler_registry.get_handler.assert_any_call("op1")
        self.mock_request_processor.register_handler.assert_any_call("op1", mock_op1_handler)

    async def test_process_request(self):
        coordinator = ApplicationCoordinator()
        request_dict = {"type": "test_req"}
        expected_response = {"success": True, "data": "mocked_response"}
        self.mock_request_processor.process_request.return_value = expected_response

        response = await coordinator.process_request(request_dict)

        self.mock_request_processor.process_request.assert_called_once_with(request_dict)
        self.assertEqual(response, expected_response)

    async def test_broadcast_event(self):
        coordinator = ApplicationCoordinator()
        event_type_str = "my_event_type"
        event_data_dict = {"key": "value"}
        client_id_str = "client123"

        await coordinator.broadcast_event(event_type_str, event_data_dict, client_id_str)

        # EventCoordinator.broadcast_event doesn't support client_id, so it's called without it
        self.mock_event_coordinator.broadcast_event.assert_called_once_with(
            event_type_str, event_data_dict
        )

    async def test_shutdown(self):
        # Ensure __init__ runs to set _initialized = True
        coordinator = ApplicationCoordinator()
        self.assertTrue(ApplicationCoordinator._initialized)  # Should be true after __init__

        await coordinator.shutdown()

        self.mock_transport_registry.shutdown_all.assert_called_once()
        self.assertFalse(ApplicationCoordinator._initialized)
        mock_logger_instance.info.assert_any_call("ApplicationCoordinator shutdown complete.")

    async def test_shutdown_when_tar_lacks_shutdown_all(self):
        coordinator = ApplicationCoordinator()
        del self.mock_transport_registry.shutdown_all  # Simulate missing method

        await coordinator.shutdown()
        mock_logger_instance.warning.assert_any_call("TransportAdapterRegistry does not have shutdown_all method.")
        self.assertFalse(ApplicationCoordinator._initialized)


if __name__ == "__main__":
    unittest.main()
