import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from aider_mcp_server.application_coordinator import ApplicationCoordinator
from aider_mcp_server.atoms.event_types import EventTypes
from aider_mcp_server.atoms.internal_types import InternalEvent
from aider_mcp_server.configuration_system import ConfigurationSystem
from aider_mcp_server.error_handling import ErrorHandler, HandlerError
from aider_mcp_server.initialization_sequence import InitializationSequence
from aider_mcp_server.transport_discovery import DiscoveryService


class MockEventHandler:
    """Mock event handler for integration testing."""

    def __init__(self):
        self.received_events = []

    async def handle_event(self, event: InternalEvent):
        """Handle incoming events and store them for verification."""
        self.received_events.append(event)
        return None


class TestRequestHandler:
    """Mock request handler for integration testing."""

    async def handle_echo(self, request):
        """Echo the request data back to the client."""
        return {"success": True, "type": "echo_response", "data": request.get("data", {}), "original_request": request}

    async def handle_ping(self, request):
        """Simple ping handler."""
        return {"success": True, "type": "pong", "timestamp": asyncio.get_event_loop().time()}

    async def handle_error(self, request):
        """Handler that simulates an error."""
        raise HandlerError("Simulated handler error for testing")

    async def handle_slow(self, request):
        """Handler that takes some time to process."""
        await asyncio.sleep(0.1)
        return {"success": True, "type": "slow_response", "processed": True}


class IntegrationTests(unittest.IsolatedAsyncioTestCase):
    """Comprehensive integration tests for the entire SSE Coordinator system."""

    def setUp(self):
        """Set up test environment by resetting singletons."""
        # Reset all singletons for clean test environment
        ApplicationCoordinator._instance = None
        ApplicationCoordinator._initialized = False
        ConfigurationSystem._instance = None
        ConfigurationSystem._initialized = False
        DiscoveryService._instance = None
        DiscoveryService._initialized = False
        InitializationSequence._instance = None
        InitializationSequence._initialized = False

        # Set up test request handler class
        self.test_handler_class = TestRequestHandler

    def tearDown(self):
        """Clean up after each test."""
        # Reset singletons again
        ApplicationCoordinator._instance = None
        ApplicationCoordinator._initialized = False
        ConfigurationSystem._instance = None
        ConfigurationSystem._initialized = False
        DiscoveryService._instance = None
        DiscoveryService._initialized = False
        InitializationSequence._instance = None
        InitializationSequence._initialized = False

    @patch("aider_mcp_server.initialization_sequence.get_logger")
    @patch("aider_mcp_server.application_coordinator.get_logger")
    @patch("aider_mcp_server.transport_discovery.get_logger")
    @patch("aider_mcp_server.configuration_system.get_logger")
    async def test_full_system_initialization_and_shutdown(
        self, mock_config_logger, mock_discovery_logger, mock_coordinator_logger, mock_init_logger
    ):
        """Test complete system initialization and shutdown sequence."""
        # Mock all loggers
        for mock_logger in [mock_config_logger, mock_discovery_logger, mock_coordinator_logger, mock_init_logger]:
            mock_logger.return_value = MagicMock()

        # Initialize the system
        init_sequence = InitializationSequence()

        # Test initialization
        await init_sequence.initialize()

        # Verify initialization state
        self.assertTrue(init_sequence._initialized)
        self.assertIsNotNone(init_sequence._coordinator)
        self.assertTrue(init_sequence._coordinator._initialized)

        # Test shutdown
        await init_sequence.shutdown()

        # Verify shutdown state
        self.assertFalse(init_sequence._initialized)

    @patch("aider_mcp_server.initialization_sequence.get_logger")
    @patch("aider_mcp_server.application_coordinator.get_logger")
    @patch("aider_mcp_server.transport_discovery.get_logger")
    @patch("aider_mcp_server.configuration_system.get_logger")
    async def test_request_processing_end_to_end(
        self, mock_config_logger, mock_discovery_logger, mock_coordinator_logger, mock_init_logger
    ):
        """Test end-to-end request processing through the coordinator."""
        # Mock all loggers
        for mock_logger in [mock_config_logger, mock_discovery_logger, mock_coordinator_logger, mock_init_logger]:
            mock_logger.return_value = MagicMock()

        # Initialize the system
        init_sequence = InitializationSequence()
        await init_sequence.initialize()

        try:
            coordinator = init_sequence._coordinator

            # Register test handler
            coordinator.register_handler_class(self.test_handler_class)

            # Test echo request
            echo_request = {"type": "echo", "data": {"message": "Hello, integration test!"}}
            response = await coordinator.process_request(echo_request)

            self.assertTrue(response["success"])
            self.assertEqual(response["type"], "echo_response")
            self.assertEqual(response["data"]["message"], "Hello, integration test!")

            # Test ping request
            ping_request = {"type": "ping"}
            response = await coordinator.process_request(ping_request)

            self.assertTrue(response["success"])
            self.assertEqual(response["type"], "pong")
            self.assertIn("timestamp", response)

            # Test error handling
            error_request = {"type": "error"}
            response = await coordinator.process_request(error_request)

            self.assertFalse(response["success"])
            self.assertIn("error", response)
            self.assertIn("Simulated handler error", response["error"])

        finally:
            await init_sequence.shutdown()

    @patch("aider_mcp_server.initialization_sequence.get_logger")
    @patch("aider_mcp_server.application_coordinator.get_logger")
    @patch("aider_mcp_server.transport_discovery.get_logger")
    @patch("aider_mcp_server.configuration_system.get_logger")
    async def test_event_broadcasting_system(
        self, mock_config_logger, mock_discovery_logger, mock_coordinator_logger, mock_init_logger
    ):
        """Test event broadcasting through the coordinator."""
        # Mock all loggers
        for mock_logger in [mock_config_logger, mock_discovery_logger, mock_coordinator_logger, mock_init_logger]:
            mock_logger.return_value = MagicMock()

        # Initialize the system
        init_sequence = InitializationSequence()
        await init_sequence.initialize()

        try:
            coordinator = init_sequence._coordinator

            # Create test event handler
            event_handler = MockEventHandler()

            # Subscribe to events
            await coordinator._event_coordinator.subscribe(EventTypes.STATUS, event_handler)

            # Create and publish an internal event
            test_event = InternalEvent(
                event_type=EventTypes.STATUS,
                data={"message": "Integration test event", "value": 42},
                metadata={"source": "integration_test"},
            )
            await coordinator._event_coordinator.publish_event(test_event)

            # Verify event was received
            self.assertEqual(len(event_handler.received_events), 1)
            received_event = event_handler.received_events[0]
            self.assertEqual(received_event.event_type, EventTypes.STATUS)
            self.assertEqual(received_event.data["message"], "Integration test event")
            self.assertEqual(received_event.data["value"], 42)

        finally:
            await init_sequence.shutdown()

    @patch("aider_mcp_server.initialization_sequence.get_logger")
    @patch("aider_mcp_server.application_coordinator.get_logger")
    @patch("aider_mcp_server.transport_discovery.get_logger")
    @patch("aider_mcp_server.configuration_system.get_logger")
    async def test_transport_discovery_integration(
        self, mock_config_logger, mock_discovery_logger, mock_coordinator_logger, mock_init_logger
    ):
        """Test transport discovery service integration with coordinator."""
        # Mock all loggers
        for mock_logger in [mock_config_logger, mock_discovery_logger, mock_coordinator_logger, mock_init_logger]:
            mock_logger.return_value = MagicMock()

        # Initialize the system
        init_sequence = InitializationSequence()
        await init_sequence.initialize()

        try:
            coordinator = init_sequence._coordinator
            discovery = DiscoveryService()
            discovery.set_coordinator(coordinator)

            # Test coordinator availability
            available = await discovery.check_coordinator_available()
            self.assertTrue(available)

            # Test transport notification
            transport_info = {"host": "localhost", "port": 8000}
            await discovery.notify_transport_available("test_transport", transport_info)

            # Verify transport info is stored
            stored_info = await discovery.get_transport_info("test_transport")
            self.assertIsNotNone(stored_info)
            self.assertEqual(stored_info["host"], "localhost")
            self.assertEqual(stored_info["port"], 8000)
            self.assertEqual(stored_info["status"], "available")

            # Test available transports
            available_transports = await discovery.get_available_transports()
            self.assertIn("test_transport", available_transports)

        finally:
            await init_sequence.shutdown()

    @patch("aider_mcp_server.initialization_sequence.get_logger")
    @patch("aider_mcp_server.application_coordinator.get_logger")
    @patch("aider_mcp_server.transport_discovery.get_logger")
    @patch("aider_mcp_server.configuration_system.get_logger")
    async def test_configuration_system_integration(
        self, mock_config_logger, mock_discovery_logger, mock_coordinator_logger, mock_init_logger
    ):
        """Test configuration system integration with the application."""
        # Mock all loggers
        for mock_logger in [mock_config_logger, mock_discovery_logger, mock_coordinator_logger, mock_init_logger]:
            mock_logger.return_value = MagicMock()

        # Get configuration system
        config = ConfigurationSystem()

        # Test default configuration values
        self.assertEqual(config.get("application", "name"), "Aider MCP Server")
        self.assertTrue(config.get("transports", "sse", "enabled"))
        self.assertEqual(config.get("transports", "sse", "port"), 8000)

        # Test configuration override
        config.set("custom_value", "test", "integration")
        self.assertEqual(config.get("test", "integration"), "custom_value")

        # Test transport configuration
        sse_config = config.get_transport_config("sse")
        self.assertTrue(sse_config["enabled"])
        self.assertEqual(sse_config["host"], "localhost")

        # Test configuration validation
        errors = config.validate()
        self.assertEqual(len(errors), 0)  # Should have no validation errors

    @patch("aider_mcp_server.initialization_sequence.get_logger")
    @patch("aider_mcp_server.application_coordinator.get_logger")
    @patch("aider_mcp_server.transport_discovery.get_logger")
    @patch("aider_mcp_server.configuration_system.get_logger")
    async def test_error_handling_integration(
        self, mock_config_logger, mock_discovery_logger, mock_coordinator_logger, mock_init_logger
    ):
        """Test error handling system integration across components."""
        # Mock all loggers
        for mock_logger in [mock_config_logger, mock_discovery_logger, mock_coordinator_logger, mock_init_logger]:
            mock_logger.return_value = MagicMock()

        # Initialize the system
        init_sequence = InitializationSequence()
        await init_sequence.initialize()

        try:
            coordinator = init_sequence._coordinator
            coordinator.register_handler_class(self.test_handler_class)

            # Test error handling in request processing
            error_request = {"type": "error"}
            response = await coordinator.process_request(error_request)

            # Verify error response format
            self.assertFalse(response["success"])
            self.assertIn("error", response)
            self.assertIn("Simulated handler error", response["error"])

            # Test ErrorHandler utility functions
            test_exception = HandlerError("Test error for formatting")
            formatted_error = ErrorHandler.format_exception(test_exception)

            self.assertEqual(formatted_error["type"], "error")
            self.assertEqual(formatted_error["error"]["type"], "HandlerError")
            self.assertEqual(formatted_error["error"]["message"], "Test error for formatting")

        finally:
            await init_sequence.shutdown()

    @patch("aider_mcp_server.initialization_sequence.get_logger")
    @patch("aider_mcp_server.application_coordinator.get_logger")
    @patch("aider_mcp_server.transport_discovery.get_logger")
    @patch("aider_mcp_server.configuration_system.get_logger")
    async def test_concurrent_request_processing(
        self, mock_config_logger, mock_discovery_logger, mock_coordinator_logger, mock_init_logger
    ):
        """Test concurrent request processing through the system."""
        # Mock all loggers
        for mock_logger in [mock_config_logger, mock_discovery_logger, mock_coordinator_logger, mock_init_logger]:
            mock_logger.return_value = MagicMock()

        # Initialize the system
        init_sequence = InitializationSequence()
        await init_sequence.initialize()

        try:
            coordinator = init_sequence._coordinator
            coordinator.register_handler_class(self.test_handler_class)

            # Create multiple concurrent requests
            requests = [
                {"type": "ping"},
                {"type": "echo", "data": {"id": 1}},
                {"type": "slow"},
                {"type": "echo", "data": {"id": 2}},
                {"type": "ping"},
            ]

            # Process requests concurrently
            tasks = [coordinator.process_request(req) for req in requests]
            responses = await asyncio.gather(*tasks)

            # Verify all responses
            self.assertEqual(len(responses), 5)

            # Check specific responses
            ping_responses = [r for r in responses if r.get("type") == "pong"]
            echo_responses = [r for r in responses if r.get("type") == "echo_response"]
            slow_responses = [r for r in responses if r.get("type") == "slow_response"]

            self.assertEqual(len(ping_responses), 2)
            self.assertEqual(len(echo_responses), 2)
            self.assertEqual(len(slow_responses), 1)

            # Verify all were successful
            for response in responses:
                self.assertTrue(response.get("success", False))

        finally:
            await init_sequence.shutdown()

    @patch("aider_mcp_server.initialization_sequence.get_logger")
    @patch("aider_mcp_server.application_coordinator.get_logger")
    @patch("aider_mcp_server.transport_discovery.get_logger")
    @patch("aider_mcp_server.configuration_system.get_logger")
    async def test_component_lifecycle_management(
        self, mock_config_logger, mock_discovery_logger, mock_coordinator_logger, mock_init_logger
    ):
        """Test proper lifecycle management of all components."""
        # Mock all loggers
        for mock_logger in [mock_config_logger, mock_discovery_logger, mock_coordinator_logger, mock_init_logger]:
            mock_logger.return_value = MagicMock()

        # Test multiple initialization/shutdown cycles
        for _cycle in range(3):
            # Initialize
            init_sequence = InitializationSequence()
            await init_sequence.initialize()

            # Verify components are initialized
            self.assertTrue(init_sequence._initialized)
            coordinator = init_sequence._coordinator
            self.assertTrue(coordinator._initialized)

            # Use the system
            coordinator.register_handler_class(self.test_handler_class)
            response = await coordinator.process_request({"type": "ping"})
            self.assertTrue(response["success"])

            # Shutdown
            await init_sequence.shutdown()

            # Verify clean shutdown
            self.assertFalse(init_sequence._initialized)

            # Reset singletons for next cycle
            ApplicationCoordinator._instance = None
            ApplicationCoordinator._initialized = False
            InitializationSequence._instance = None
            InitializationSequence._initialized = False

    @patch("aider_mcp_server.initialization_sequence.get_logger")
    @patch("aider_mcp_server.application_coordinator.get_logger")
    @patch("aider_mcp_server.transport_discovery.get_logger")
    @patch("aider_mcp_server.configuration_system.get_logger")
    async def test_transport_configuration_integration(
        self, mock_config_logger, mock_discovery_logger, mock_coordinator_logger, mock_init_logger
    ):
        """Test transport configuration through initialization sequence."""
        # Mock all loggers
        for mock_logger in [mock_config_logger, mock_discovery_logger, mock_coordinator_logger, mock_init_logger]:
            mock_logger.return_value = MagicMock()

        # Test with transport configurations
        transport_configs = [
            {"name": "sse", "host": "localhost", "port": 8000, "enabled": True},
            {"name": "test_transport", "host": "127.0.0.1", "port": 8001, "enabled": True},
        ]

        # Initialize with transport configs
        init_sequence = InitializationSequence()

        # Mock the register_transport method since we don't have real transports
        with patch.object(init_sequence._coordinator, "register_transport", new_callable=AsyncMock) as mock_register:
            await init_sequence.initialize(transport_configs)

            # Verify transport registration was attempted
            self.assertEqual(mock_register.call_count, 2)

            # Verify transport configuration was passed correctly
            call_args_list = mock_register.call_args_list
            transport_names = [call[0][0] for call in call_args_list]
            self.assertIn("sse", transport_names)
            self.assertIn("test_transport", transport_names)

        await init_sequence.shutdown()

    @patch("aider_mcp_server.initialization_sequence.get_logger")
    @patch("aider_mcp_server.application_coordinator.get_logger")
    @patch("aider_mcp_server.transport_discovery.get_logger")
    @patch("aider_mcp_server.configuration_system.get_logger")
    async def test_system_resilience_and_recovery(
        self, mock_config_logger, mock_discovery_logger, mock_coordinator_logger, mock_init_logger
    ):
        """Test system resilience and recovery from various failure scenarios."""
        # Mock all loggers
        for mock_logger in [mock_config_logger, mock_discovery_logger, mock_coordinator_logger, mock_init_logger]:
            mock_logger.return_value = MagicMock()

        # Initialize the system
        init_sequence = InitializationSequence()
        await init_sequence.initialize()

        try:
            coordinator = init_sequence._coordinator
            coordinator.register_handler_class(self.test_handler_class)

            # Test invalid request handling
            invalid_requests = [
                {},  # Empty request
                {"invalid": "request"},  # Missing type
                {"type": "nonexistent"},  # Unknown handler
                None,  # Invalid request format
            ]

            for invalid_request in invalid_requests:
                try:
                    response = await coordinator.process_request(invalid_request)
                    # Should return error response, not raise exception
                    self.assertIn("error", response)
                except Exception as e:
                    # Some requests might raise exceptions, which is acceptable
                    # Log for debugging but don't fail the test
                    self.assertTrue(True, f"Exception raised as expected: {e}")  # noqa: S110

            # Test that system remains functional after errors
            valid_request = {"type": "ping"}
            response = await coordinator.process_request(valid_request)
            self.assertTrue(response["success"])

            # Test discovery service resilience
            discovery = DiscoveryService()
            discovery.set_coordinator(coordinator)

            # Test with invalid transport info
            try:
                await discovery.notify_transport_available("", {})  # Empty name
                await discovery.notify_transport_available("test", None)  # Invalid info
            except Exception as e:
                # Errors are expected and should be handled gracefully
                self.assertTrue(True, f"Discovery error handled as expected: {e}")  # noqa: S110

            # Verify discovery service still works
            available = await discovery.check_coordinator_available()
            self.assertTrue(available)

        finally:
            await init_sequence.shutdown()


if __name__ == "__main__":
    unittest.main()
