import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from aider_mcp_server.application_coordinator import ApplicationCoordinator
from aider_mcp_server.organisms.discovery.transport_discovery import DiscoveryService, get_discovery_service


class TestTransportDiscovery(unittest.IsolatedAsyncioTestCase):
    """Comprehensive test suite for the Transport Discovery Service."""

    def setUp(self):
        """Set up test environment by resetting the singleton."""
        # Reset the singleton for each test
        DiscoveryService._instance = None
        DiscoveryService._initialized = False

        # Mock the coordinator
        self.mock_coordinator = MagicMock(spec=ApplicationCoordinator)
        self.mock_coordinator._initialized = True
        self.mock_coordinator.broadcast_event = AsyncMock()
        self.mock_coordinator.register_transport = AsyncMock()

        # Mock transport registry
        self.mock_registry = MagicMock()
        self.mock_registry._active_adapters = {"sse": MagicMock(), "stdio": MagicMock()}
        self.mock_coordinator._transport_registry = self.mock_registry

    def tearDown(self):
        """Clean up after each test."""
        # Reset the singleton again
        DiscoveryService._instance = None
        DiscoveryService._initialized = False

    @patch("aider_mcp_server.transport_discovery.get_logger")
    def test_singleton_pattern(self, mock_get_logger):
        """Test that DiscoveryService follows singleton pattern."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Create two instances
        service1 = DiscoveryService()
        service2 = DiscoveryService()

        # Should be the same instance
        self.assertIs(service1, service2)

        # Global getter should return the same instance
        service3 = get_discovery_service()
        self.assertIs(service1, service3)

    @patch("aider_mcp_server.transport_discovery.get_logger")
    def test_set_coordinator(self, mock_get_logger):
        """Test setting the ApplicationCoordinator."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        service = DiscoveryService()
        service.set_coordinator(self.mock_coordinator)

        self.assertEqual(service._coordinator, self.mock_coordinator)
        mock_logger.info.assert_called_with("ApplicationCoordinator set for discovery service")

    @patch("aider_mcp_server.transport_discovery.get_logger")
    async def test_register_discovery_callback(self, mock_get_logger):
        """Test registering discovery callbacks."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        service = DiscoveryService()

        # Create a mock callback
        mock_callback = AsyncMock()

        # Register callback
        callback_id = await service.register_discovery_callback(mock_callback)

        # Should return a UUID-like string
        self.assertIsInstance(callback_id, str)
        self.assertEqual(len(callback_id), 36)  # UUID length

        # Should be stored in registered callbacks
        self.assertIn(callback_id, service._registered_callbacks)
        self.assertEqual(service._registered_callbacks[callback_id], mock_callback)

    @patch("aider_mcp_server.transport_discovery.get_logger")
    async def test_unregister_discovery_callback(self, mock_get_logger):
        """Test unregistering discovery callbacks."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        service = DiscoveryService()

        # Register a callback first
        mock_callback = AsyncMock()
        callback_id = await service.register_discovery_callback(mock_callback)

        # Verify it's registered
        self.assertIn(callback_id, service._registered_callbacks)

        # Unregister the callback
        await service.unregister_discovery_callback(callback_id)

        # Should be removed from registered callbacks
        self.assertNotIn(callback_id, service._registered_callbacks)

        # Unregistering non-existent callback should not raise error
        await service.unregister_discovery_callback("non-existent-id")
        mock_logger.warning.assert_called()

    @patch("aider_mcp_server.transport_discovery.get_logger")
    async def test_notify_transport_available(self, mock_get_logger):
        """Test notifying when a transport becomes available."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        service = DiscoveryService()
        service.set_coordinator(self.mock_coordinator)

        # Register a callback
        mock_callback = AsyncMock()
        await service.register_discovery_callback(mock_callback)

        # Notify transport available
        transport_name = "test_transport"
        transport_info = {"host": "localhost", "port": 8000}

        await service.notify_transport_available(transport_name, transport_info)

        # Should store transport info
        stored_info = service._transport_info[transport_name]
        self.assertEqual(stored_info["host"], "localhost")
        self.assertEqual(stored_info["port"], 8000)
        self.assertEqual(stored_info["status"], "available")
        self.assertIn("discovered_at", stored_info)

        # Should broadcast event to coordinator
        self.mock_coordinator.broadcast_event.assert_called_once_with(
            "transport_available", {"name": transport_name, "info": transport_info}
        )

        # Should call registered callback
        mock_callback.assert_called_once()

    @patch("aider_mcp_server.transport_discovery.get_logger")
    async def test_notify_transport_unavailable(self, mock_get_logger):
        """Test notifying when a transport becomes unavailable."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        service = DiscoveryService()
        service.set_coordinator(self.mock_coordinator)

        # First make transport available
        transport_name = "test_transport"
        await service.notify_transport_available(transport_name, {"host": "localhost"})

        # Then make it unavailable
        await service.notify_transport_unavailable(transport_name)

        # Should update transport status
        stored_info = service._transport_info[transport_name]
        self.assertEqual(stored_info["status"], "unavailable")
        self.assertIn("disconnected_at", stored_info)

        # Should broadcast event to coordinator
        self.mock_coordinator.broadcast_event.assert_called_with("transport_unavailable", {"name": transport_name})

    @patch("aider_mcp_server.transport_discovery.get_logger")
    async def test_check_coordinator_available(self, mock_get_logger):
        """Test checking coordinator availability."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        service = DiscoveryService()

        # No coordinator set
        available = await service.check_coordinator_available()
        self.assertFalse(available)

        # Set uninitialized coordinator
        mock_coord = MagicMock()
        mock_coord._initialized = False
        service.set_coordinator(mock_coord)

        available = await service.check_coordinator_available()
        self.assertFalse(available)

        # Set initialized coordinator
        mock_coord._initialized = True
        available = await service.check_coordinator_available()
        self.assertTrue(available)

    @patch("aider_mcp_server.transport_discovery.get_logger")
    async def test_get_available_transports(self, mock_get_logger):
        """Test getting available transports."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        service = DiscoveryService()

        # Add some transports
        await service.notify_transport_available("sse", {"host": "localhost", "port": 8000})
        await service.notify_transport_available("stdio", {"enabled": True})
        await service.notify_transport_available("http", {"host": "localhost", "port": 8001})

        # Make one unavailable
        await service.notify_transport_unavailable("http")

        # Get available transports
        available = await service.get_available_transports()

        # Should only include available transports
        self.assertEqual(len(available), 2)
        self.assertIn("sse", available)
        self.assertIn("stdio", available)
        self.assertNotIn("http", available)

        # Check transport info structure
        sse_info = available["sse"]
        self.assertEqual(sse_info["host"], "localhost")
        self.assertEqual(sse_info["port"], 8000)
        self.assertEqual(sse_info["status"], "available")

    @patch("aider_mcp_server.transport_discovery.get_logger")
    async def test_get_transport_info(self, mock_get_logger):
        """Test getting specific transport information."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        service = DiscoveryService()

        # Add a transport
        transport_name = "test_transport"
        transport_config = {"host": "localhost", "port": 8000}
        await service.notify_transport_available(transport_name, transport_config)

        # Get transport info
        info = await service.get_transport_info(transport_name)

        self.assertIsNotNone(info)
        self.assertEqual(info["host"], "localhost")
        self.assertEqual(info["port"], 8000)
        self.assertEqual(info["status"], "available")

        # Get non-existent transport info
        info = await service.get_transport_info("non_existent")
        self.assertIsNone(info)

    @patch("aider_mcp_server.transport_discovery.get_logger")
    async def test_discover_transports(self, mock_get_logger):
        """Test discovering transports from coordinator."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        service = DiscoveryService()

        # No coordinator available
        transports = await service.discover_transports()
        self.assertEqual(transports, [])

        # Set coordinator
        service.set_coordinator(self.mock_coordinator)

        # Discover transports
        transports = await service.discover_transports()

        # Should return active adapters from registry
        self.assertEqual(set(transports), {"sse", "stdio"})

    @patch("aider_mcp_server.transport_discovery.get_logger")
    async def test_register_transport_with_coordinator(self, mock_get_logger):
        """Test registering transport with coordinator."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        service = DiscoveryService()

        # No coordinator available
        result = await service.register_transport_with_coordinator("test", host="localhost")
        self.assertFalse(result)

        # Set coordinator
        service.set_coordinator(self.mock_coordinator)

        # Register transport
        result = await service.register_transport_with_coordinator("test_transport", host="localhost", port=8000)

        self.assertTrue(result)

        # Should call coordinator register_transport
        self.mock_coordinator.register_transport.assert_called_once_with("test_transport", host="localhost", port=8000)

        # Should update transport info
        info = await service.get_transport_info("test_transport")
        self.assertIsNotNone(info)
        self.assertEqual(info["host"], "localhost")
        self.assertEqual(info["port"], 8000)

    @patch("aider_mcp_server.transport_discovery.get_logger")
    async def test_callback_error_handling(self, mock_get_logger):
        """Test error handling in discovery callbacks."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        service = DiscoveryService()
        service.set_coordinator(self.mock_coordinator)

        # Register callbacks - one that works, one that raises error
        good_callback = AsyncMock()
        bad_callback = AsyncMock(side_effect=Exception("Callback error"))

        await service.register_discovery_callback(good_callback)
        await service.register_discovery_callback(bad_callback)

        # Notify transport available - should not raise error
        await service.notify_transport_available("test", {"host": "localhost"})

        # Both callbacks should have been called
        good_callback.assert_called_once()
        bad_callback.assert_called_once()

        # Error should be logged but not propagated
        mock_logger.error.assert_called()

    @patch("aider_mcp_server.transport_discovery.get_logger")
    async def test_shutdown(self, mock_get_logger):
        """Test discovery service shutdown."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        service = DiscoveryService()
        service.set_coordinator(self.mock_coordinator)

        # Add some callbacks and transports
        await service.register_discovery_callback(AsyncMock())
        await service.register_discovery_callback(AsyncMock())
        await service.notify_transport_available("sse", {"host": "localhost"})
        await service.notify_transport_available("stdio", {"enabled": True})

        # Verify they exist
        self.assertEqual(len(service._registered_callbacks), 2)
        self.assertEqual(len(service._transport_info), 2)

        # Shutdown
        await service.shutdown()

        # Should clear everything
        self.assertEqual(len(service._registered_callbacks), 0)
        self.assertEqual(len(service._transport_info), 0)
        self.assertIsNone(service._coordinator)

        mock_logger.info.assert_called()

    @patch("aider_mcp_server.transport_discovery.get_logger")
    async def test_error_propagation(self, mock_get_logger):
        """Test that appropriate errors are raised as TransportError."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        service = DiscoveryService()

        # Test error in notify_transport_available with invalid coordinator
        mock_bad_coordinator = MagicMock()
        mock_bad_coordinator._initialized = True
        mock_bad_coordinator.broadcast_event = AsyncMock(side_effect=Exception("Broadcast error"))
        service.set_coordinator(mock_bad_coordinator)

        # Should not raise error (error is caught and logged)
        await service.notify_transport_available("test", {"host": "localhost"})

        # Verify error was logged
        mock_logger.error.assert_called()


if __name__ == "__main__":
    unittest.main()
