"""
Tests for Enhanced Transport Adapter Registry with MCP 2025-03-26 support.
"""

import unittest
import warnings
from unittest.mock import AsyncMock, MagicMock, patch

from aider_mcp_server.atoms.types.mcp_types import RequestParameters
from aider_mcp_server.atoms.security.context import SecurityContext
from aider_mcp_server.molecules.transport.base_adapter import AbstractTransportAdapter
from aider_mcp_server.transport_adapter_registry_enhanced import (
    EnhancedTransportAdapterRegistry,
    TransportCapabilities,
    TransportProtocolVersion,
    TransportStatus,
)


class MockHTTPStreamableTransport(AbstractTransportAdapter):
    """Mock HTTP Streamable Transport (recommended)."""

    TRANSPORT_TYPE_NAME = "http_streamable"
    MCP_PROTOCOL_VERSION = "2025-03-26"
    SUPPORTS_AUTHORIZATION = True

    def __init__(self, coordinator=None, **kwargs):
        self.coordinator = coordinator
        self._transport_id = "mock-http-streamable"

    async def initialize(self):
        pass

    async def shutdown(self):
        pass

    def get_transport_id(self):
        return self._transport_id

    def get_transport_type(self):
        return self.TRANSPORT_TYPE_NAME

    async def start_listening(self):
        pass

    async def send_event(self, event_type, data):
        pass

    def get_capabilities(self):
        return set()

    def should_receive_event(self, event_type, data, request_details=None):
        return True

    def validate_request_security(self, request_data: RequestParameters) -> SecurityContext:
        return SecurityContext(authenticated=True, user_id="mock_http_user")


class MockModernizedSSETransport(AbstractTransportAdapter):
    """Mock Modernized SSE Transport (deprecated but compliant)."""

    TRANSPORT_TYPE_NAME = "sse_modernized"
    MCP_PROTOCOL_VERSION = "2025-03-26"
    SUPPORTS_AUTHORIZATION = True
    DEPRECATED = True

    def __init__(self, coordinator=None, **kwargs):
        self.coordinator = coordinator
        self._transport_id = "mock-sse-modernized"

    async def initialize(self):
        pass

    async def shutdown(self):
        pass

    def get_transport_id(self):
        return self._transport_id

    def get_transport_type(self):
        return self.TRANSPORT_TYPE_NAME

    async def start_listening(self):
        pass

    async def send_event(self, event_type, data):
        pass

    def get_capabilities(self):
        return set()

    def should_receive_event(self, event_type, data, request_details=None):
        return True

    def validate_request_security(self, request_data: RequestParameters) -> SecurityContext:
        return SecurityContext(authenticated=True, user_id="mock_sse_modern_user")


class MockLegacySSETransport(AbstractTransportAdapter):
    """Mock Legacy SSE Transport (deprecated)."""

    TRANSPORT_TYPE_NAME = "sse"

    def __init__(self, coordinator=None, **kwargs):
        self.coordinator = coordinator
        self._transport_id = "mock-sse-legacy"

    async def initialize(self):
        pass

    async def shutdown(self):
        pass

    def get_transport_id(self):
        return self._transport_id

    def get_transport_type(self):
        return self.TRANSPORT_TYPE_NAME

    async def start_listening(self):
        pass

    async def send_event(self, event_type, data):
        pass

    def get_capabilities(self):
        return set()

    def should_receive_event(self, event_type, data, request_details=None):
        return True

    def validate_request_security(self, request_data: RequestParameters) -> SecurityContext:
        return SecurityContext(authenticated=False, user_id="mock_legacy_user")


class MockFailingTransport(AbstractTransportAdapter):
    """Mock transport that fails during initialization."""

    TRANSPORT_TYPE_NAME = "failing"

    def __init__(self, coordinator=None, **kwargs):
        self.coordinator = coordinator
        self._transport_id = "mock-failing"

    async def initialize(self):
        raise Exception("Initialization failed")

    async def shutdown(self):
        pass

    def get_transport_id(self):
        return self._transport_id

    def get_transport_type(self):
        return self.TRANSPORT_TYPE_NAME

    async def start_listening(self):
        pass

    async def send_event(self, event_type, data):
        pass

    def get_capabilities(self):
        return set()

    def should_receive_event(self, event_type, data, request_details=None):
        return True

    def validate_request_security(self, request_data: RequestParameters) -> SecurityContext:
        return SecurityContext(authenticated=False, user_id="mock_failing_user")


class TestEnhancedTransportAdapterRegistry(unittest.IsolatedAsyncioTestCase):
    """Test suite for the enhanced transport adapter registry."""

    def setUp(self):
        """Set up test environment."""
        self.registry = EnhancedTransportAdapterRegistry()

    async def asyncTearDown(self):
        """Clean up after tests."""
        await self.registry.shutdown_all()

    @patch("importlib.import_module")
    @patch("pkgutil.iter_modules")
    def test_discover_adapters_success(self, mock_iter_modules, mock_import_module):
        """Test successful adapter discovery."""
        # Mock package discovery
        mock_package = MagicMock()
        mock_package.__name__ = "test_package"
        mock_package.__path__ = ["/fake/path"]
        mock_import_module.return_value = mock_package

        # Mock module discovery
        mock_iter_modules.return_value = [
            (None, "test_package.http_transport", False),
            (None, "test_package.sse_transport", False),
        ]

        # Mock module with transport classes
        mock_http_module = MagicMock()
        mock_sse_module = MagicMock()

        import inspect

        with patch.object(inspect, "getmembers") as mock_getmembers:

            def side_effect(module, predicate=None):
                if module == mock_http_module:
                    return [("MockHTTPStreamableTransport", MockHTTPStreamableTransport)]
                elif module == mock_sse_module:
                    return [("MockLegacySSETransport", MockLegacySSETransport)]
                return []

            mock_getmembers.side_effect = side_effect
            mock_import_module.side_effect = lambda name: {
                "test_package": mock_package,
                "test_package.http_transport": mock_http_module,
                "test_package.sse_transport": mock_sse_module,
            }.get(name, mock_package)

            # Run discovery
            self.registry.discover_adapters("test_package")

            # Verify discovery results
            self.assertIn("http_streamable", self.registry._transport_metadata)
            self.assertIn("sse", self.registry._transport_metadata)

            # Check HTTP Streamable transport metadata
            http_meta = self.registry._transport_metadata["http_streamable"]
            self.assertEqual(http_meta.status, TransportStatus.RECOMMENDED)
            self.assertEqual(http_meta.capabilities.protocol_version, TransportProtocolVersion.MCP_2025_03_26)
            self.assertTrue(http_meta.capabilities.supports_authorization)
            self.assertFalse(http_meta.capabilities.deprecated)

            # Check Legacy SSE transport metadata
            sse_meta = self.registry._transport_metadata["sse"]
            self.assertEqual(sse_meta.status, TransportStatus.DEPRECATED)
            self.assertEqual(sse_meta.capabilities.protocol_version, TransportProtocolVersion.MCP_2024_11_05)
            self.assertFalse(sse_meta.capabilities.supports_authorization)
            self.assertTrue(sse_meta.capabilities.deprecated)

    async def test_transport_prioritization(self):
        """Test that transports are properly prioritized."""
        # Manually add transport metadata for testing
        http_meta = self.registry._transport_metadata["http_streamable"] = MagicMock()
        http_meta.transport_type = "http_streamable"
        http_meta.priority = 100
        http_meta.capabilities = MagicMock()
        http_meta.capabilities.protocol_version = TransportProtocolVersion.MCP_2025_03_26
        http_meta.capabilities.deprecated = False

        sse_meta = self.registry._transport_metadata["sse"] = MagicMock()
        sse_meta.transport_type = "sse"
        sse_meta.priority = 25
        sse_meta.capabilities = MagicMock()
        sse_meta.capabilities.protocol_version = TransportProtocolVersion.MCP_2024_11_05
        sse_meta.capabilities.deprecated = True

        # Get recommended order
        recommended = await self.registry.get_recommended_transport_types()

        # HTTP Streamable should come first
        self.assertEqual(recommended[0], "http_streamable")
        self.assertEqual(recommended[1], "sse")

    async def test_transport_recommendations(self):
        """Test transport recommendation generation."""
        # Add mock metadata
        self.registry._transport_metadata = {
            "http_streamable": MagicMock(
                transport_type="http_streamable",
                adapter_class=MockHTTPStreamableTransport,
                status=TransportStatus.RECOMMENDED,
                capabilities=TransportCapabilities(
                    protocol_version=TransportProtocolVersion.MCP_2025_03_26,
                    supports_authorization=True,
                    supports_bidirectional=True,
                    supports_resumability=True,
                    supports_batching=True,
                    deprecated=False,
                ),
            ),
            "sse": MagicMock(
                transport_type="sse",
                adapter_class=MockLegacySSETransport,
                status=TransportStatus.DEPRECATED,
                capabilities=TransportCapabilities(
                    protocol_version=TransportProtocolVersion.MCP_2024_11_05,
                    supports_authorization=False,
                    supports_bidirectional=False,
                    supports_resumability=False,
                    supports_batching=False,
                    deprecated=True,
                    deprecation_message="Legacy transport - migrate to HTTP Streamable",
                ),
            ),
        }

        recommendations = await self.registry.get_transport_recommendations()

        # Verify structure
        self.assertIn("recommended", recommendations)
        self.assertIn("deprecated", recommendations)
        self.assertIn("rationale", recommendations)

        # Verify HTTP Streamable is recommended
        self.assertEqual(len(recommendations["recommended"]), 1)
        self.assertEqual(recommendations["recommended"][0]["type"], "http_streamable")

        # Verify SSE is deprecated
        self.assertEqual(len(recommendations["deprecated"]), 1)
        self.assertEqual(recommendations["deprecated"][0]["type"], "sse")

    async def test_initialize_adapter_success(self):
        """Test successful adapter initialization."""
        # Add mock metadata
        self.registry._transport_metadata["http_streamable"] = MagicMock(
            transport_type="http_streamable",
            adapter_class=MockHTTPStreamableTransport,
            status=TransportStatus.RECOMMENDED,
            capabilities=MagicMock(deprecated=False),
        )

        mock_coordinator = MagicMock()

        # Initialize adapter
        adapter = await self.registry.initialize_adapter("http_streamable", mock_coordinator)

        # Verify initialization
        self.assertIsNotNone(adapter)
        self.assertIsInstance(adapter, MockHTTPStreamableTransport)
        self.assertEqual(adapter.coordinator, mock_coordinator)

        # Verify adapter is registered
        self.assertIn("mock-http-streamable", self.registry._adapter_instances)

    async def test_initialize_adapter_deprecated_warning(self):
        """Test that deprecated transports issue warnings."""
        # Add mock metadata for deprecated transport
        self.registry._transport_metadata["sse"] = MagicMock(
            transport_type="sse",
            adapter_class=MockLegacySSETransport,
            status=TransportStatus.DEPRECATED,
            capabilities=MagicMock(deprecated=True, deprecation_message="Legacy SSE - migrate to HTTP Streamable"),
        )

        mock_coordinator = MagicMock()

        # Test that deprecation warning is issued
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            adapter = await self.registry.initialize_adapter("sse", mock_coordinator)

            # Verify warning was issued
            self.assertGreater(len(w), 0)
            warning_found = any("deprecated" in str(warning.message).lower() for warning in w)
            self.assertTrue(warning_found)

            # Verify adapter was still created
            self.assertIsNotNone(adapter)

    async def test_initialize_adapter_failure(self):
        """Test adapter initialization failure handling."""
        # Add mock metadata for failing transport
        self.registry._transport_metadata["failing"] = MagicMock(
            transport_type="failing",
            adapter_class=MockFailingTransport,
            status=TransportStatus.SUPPORTED,
            capabilities=MagicMock(deprecated=False),
        )

        mock_coordinator = MagicMock()

        # Initialize failing adapter
        adapter = await self.registry.initialize_adapter("failing", mock_coordinator)

        # Verify initialization failed
        self.assertIsNone(adapter)

        # Verify failure was tracked
        health = await self.registry.get_health_status()
        self.assertGreater(health["statistics"]["initialization_failures"], 0)

    async def test_initialize_unknown_transport(self):
        """Test initialization of unknown transport type."""
        mock_coordinator = MagicMock()

        adapter = await self.registry.initialize_adapter("unknown", mock_coordinator)

        self.assertIsNone(adapter)

    def test_analyze_transport_capabilities(self):
        """Test transport capability analysis."""
        # Test HTTP Streamable
        http_caps = self.registry._analyze_transport_capabilities(MockHTTPStreamableTransport)
        self.assertEqual(http_caps.protocol_version, TransportProtocolVersion.MCP_2025_03_26)
        self.assertTrue(http_caps.supports_authorization)
        self.assertTrue(http_caps.supports_bidirectional)
        self.assertFalse(http_caps.deprecated)

        # Test Modernized SSE
        sse_modern_caps = self.registry._analyze_transport_capabilities(MockModernizedSSETransport)
        self.assertEqual(sse_modern_caps.protocol_version, TransportProtocolVersion.MCP_2025_03_26)
        self.assertTrue(sse_modern_caps.supports_authorization)
        self.assertFalse(sse_modern_caps.supports_bidirectional)
        self.assertTrue(sse_modern_caps.deprecated)

        # Test Legacy SSE
        sse_legacy_caps = self.registry._analyze_transport_capabilities(MockLegacySSETransport)
        self.assertEqual(sse_legacy_caps.protocol_version, TransportProtocolVersion.MCP_2024_11_05)
        self.assertFalse(sse_legacy_caps.supports_authorization)
        self.assertTrue(sse_legacy_caps.deprecated)

    def test_determine_transport_status(self):
        """Test transport status determination logic."""
        # Recommended transport (latest protocol + features)
        recommended_caps = TransportCapabilities(
            protocol_version=TransportProtocolVersion.MCP_2025_03_26,
            supports_authorization=True,
            supports_bidirectional=True,
            supports_resumability=True,
            supports_batching=True,
            deprecated=False,
        )
        self.assertEqual(self.registry._determine_transport_status(recommended_caps), TransportStatus.RECOMMENDED)

        # Deprecated transport
        deprecated_caps = TransportCapabilities(
            protocol_version=TransportProtocolVersion.MCP_2025_03_26,
            supports_authorization=True,
            supports_bidirectional=False,
            supports_resumability=False,
            supports_batching=True,
            deprecated=True,
        )
        self.assertEqual(self.registry._determine_transport_status(deprecated_caps), TransportStatus.DEPRECATED)

    async def test_get_transport_capabilities(self):
        """Test retrieving transport capabilities."""
        # Add mock metadata
        mock_capabilities = TransportCapabilities(
            protocol_version=TransportProtocolVersion.MCP_2025_03_26,
            supports_authorization=True,
            supports_bidirectional=True,
            supports_resumability=True,
            supports_batching=True,
            deprecated=False,
        )

        self.registry._transport_metadata["http_streamable"] = MagicMock(capabilities=mock_capabilities)

        # Get capabilities
        capabilities = await self.registry.get_transport_capabilities("http_streamable")
        self.assertEqual(capabilities, mock_capabilities)

        # Test unknown transport
        unknown_caps = await self.registry.get_transport_capabilities("unknown")
        self.assertIsNone(unknown_caps)

    async def test_health_status(self):
        """Test health status reporting."""
        # Add some mock metadata
        self.registry._transport_metadata = {
            "http_streamable": MagicMock(status=TransportStatus.RECOMMENDED),
            "sse": MagicMock(status=TransportStatus.DEPRECATED),
        }
        self.registry._adapter_instances = {"adapter1": MagicMock()}

        health = await self.registry.get_health_status()

        # Verify health structure
        self.assertEqual(health["registry_status"], "healthy")
        self.assertEqual(health["active_adapters"], 1)
        self.assertEqual(health["discovered_transports"], 2)
        self.assertIn("statistics", health)
        self.assertIn("transport_breakdown", health)

        # Verify transport breakdown
        breakdown = health["transport_breakdown"]
        self.assertEqual(breakdown[TransportStatus.RECOMMENDED.value], 1)
        self.assertEqual(breakdown[TransportStatus.DEPRECATED.value], 1)

    async def test_get_adapter(self):
        """Test adapter retrieval."""
        # Add mock adapter
        mock_adapter = MagicMock()
        self.registry._adapter_instances["test-adapter"] = mock_adapter

        # Test successful retrieval
        adapter = self.registry.get_adapter("test-adapter")
        self.assertEqual(adapter, mock_adapter)

        # Test unknown adapter
        unknown_adapter = self.registry.get_adapter("unknown")
        self.assertIsNone(unknown_adapter)

    async def test_shutdown_all(self):
        """Test shutdown functionality."""
        # Add mock adapters
        mock_adapter1 = MagicMock()
        mock_adapter1.get_transport_id.return_value = "adapter1"
        mock_adapter1.shutdown = AsyncMock()

        mock_adapter2 = MagicMock()
        mock_adapter2.get_transport_id.return_value = "adapter2"
        mock_adapter2.shutdown = AsyncMock()

        self.registry._adapter_instances = {"adapter1": mock_adapter1, "adapter2": mock_adapter2}
        self.registry._transport_metadata = {"type1": MagicMock()}

        # Shutdown all
        await self.registry.shutdown_all()

        # Verify adapters were shutdown
        mock_adapter1.shutdown.assert_called_once()
        mock_adapter2.shutdown.assert_called_once()

        # Verify cleanup
        self.assertEqual(len(self.registry._adapter_instances), 0)
        self.assertEqual(len(self.registry._transport_metadata), 0)


class TestEnhancedRegistryIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests for the enhanced transport registry."""

    async def test_full_workflow(self):
        """Test complete workflow from discovery to shutdown."""
        registry = EnhancedTransportAdapterRegistry()

        try:
            # Manually register transports for testing
            registry._transport_metadata = {
                "http_streamable": MagicMock(
                    transport_type="http_streamable",
                    adapter_class=MockHTTPStreamableTransport,
                    status=TransportStatus.RECOMMENDED,
                    priority=100,
                    capabilities=MagicMock(protocol_version=TransportProtocolVersion.MCP_2025_03_26, deprecated=False),
                ),
                "sse": MagicMock(
                    transport_type="sse",
                    adapter_class=MockLegacySSETransport,
                    status=TransportStatus.DEPRECATED,
                    priority=25,
                    capabilities=MagicMock(
                        protocol_version=TransportProtocolVersion.MCP_2024_11_05,
                        deprecated=True,
                        deprecation_message="Legacy transport",
                    ),
                ),
            }

            # Test recommendations
            recommendations = await registry.get_transport_recommendations()
            self.assertGreater(len(recommendations["recommended"]), 0)
            self.assertGreater(len(recommendations["deprecated"]), 0)

            # Test adapter initialization
            mock_coordinator = MagicMock()
            http_adapter = await registry.initialize_adapter("http_streamable", mock_coordinator)
            self.assertIsNotNone(http_adapter)

            # Test health status
            health = await registry.get_health_status()
            self.assertEqual(health["registry_status"], "healthy")
            self.assertEqual(health["active_adapters"], 1)

        finally:
            await registry.shutdown_all()


if __name__ == "__main__":
    unittest.main()
