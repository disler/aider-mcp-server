"""
Tests for SSE monitoring endpoints - Phase 2.1 implementation.

Tests the real-time event streaming endpoints for AIDER coordinator events.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from aider_mcp_server.pages.application.coordinator import ApplicationCoordinator
from aider_mcp_server.pages.application.app import (
    _broadcast_to_sse_clients,
    broadcast_event_to_sse_clients,
    create_app,
    get_sse_connection_stats,
)


@pytest_asyncio.fixture
async def mock_coordinator():
    """Create a mock ApplicationCoordinator for testing."""
    coordinator = AsyncMock()  # Remove spec to allow setting magic methods
    coordinator.is_shutting_down = False  # Property, not method
    coordinator.broadcast_event = AsyncMock()
    coordinator.register_transport = AsyncMock()  # Use register_transport as in app.py
    coordinator.subscribe_to_event_type = AsyncMock()  # Use subscribe_to_event_type as in app.py
    coordinator._initialize_coordinator = AsyncMock()
    coordinator.__aenter__ = AsyncMock(return_value=coordinator)
    coordinator.__aexit__ = AsyncMock()
    # Mock the getInstance method if it's used to get the coordinator
    with patch(
        "aider_mcp_server.organisms.coordinators.transport_coordinator.ApplicationCoordinator.getInstance",
        new_callable=AsyncMock,
    ) as mock_get_instance:
        mock_get_instance.return_value = coordinator
        yield coordinator


@pytest_asyncio.fixture
async def test_app(mock_coordinator):
    """Create a test FastAPI app with SSE monitoring endpoints using create_app."""
    # Use dummy values for parameters required by create_app
    editor_model = "test_model"
    current_working_dir = "/tmp/test_cwd"
    heartbeat_interval = 1.0  # Use a short interval for testing

    # Patch the global adapter variable to control its state during the test
    import aider_mcp_server.pages.application.app as app_module

    original_adapter = app_module._adapter
    original_event_listener = app_module._coordinator_event_listener
    original_event_connections = app_module._event_connections

    # Patch the TransportAdapterRegistry to return a mock adapter
    with patch(
        "aider_mcp_server.pages.application.app.TransportAdapterRegistry.get_instance", new_callable=AsyncMock
    ) as mock_registry_get_instance:
        mock_registry = AsyncMock()
        mock_registry_get_instance.return_value = mock_registry

        # Create a mock adapter that mimics the behavior expected by create_app
        mock_adapter = MagicMock()
        mock_adapter.get_transport_id.return_value = "test_sse"
        mock_adapter._coordinator = mock_coordinator  # Ensure the mock coordinator is linked
        mock_adapter._heartbeat_interval = heartbeat_interval  # Match the interval passed
        mock_adapter._heartbeat_task = None  # Simulate initial state
        mock_adapter.logger = MagicMock()  # Mock logger
        mock_adapter.initialize = AsyncMock()  # Mock the initialize method

        # Configure the mock registry to return our mock adapter
        mock_registry.create_adapter = AsyncMock(return_value=mock_adapter)

        # Patch _start_coordinator_event_listener to prevent it from creating a real task
        with patch("aider_mcp_server.pages.application.app._start_coordinator_event_listener", new_callable=AsyncMock):
            # Call the actual create_app function
            app = await create_app(
                coordinator=mock_coordinator,
                editor_model=editor_model,
                current_working_dir=current_working_dir,
                heartbeat_interval=heartbeat_interval,
            )

            # The create_app function sets the global _adapter
            assert app_module._adapter is mock_adapter

            yield app

    # Cleanup: Restore original global state
    app_module._adapter = original_adapter
    app_module._coordinator_event_listener = original_event_listener
    app_module._event_connections = original_event_connections


class TestSSEMonitoringEndpoints:
    """Test suite for SSE monitoring endpoints."""

    @pytest.mark.asyncio
    async def test_health_endpoint(self, test_app):
        """Test the health endpoint returns correct status."""
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "active_sse_connections" in data
        assert "coordinator_status" in data

        # Check connection stats structure
        connections = data["active_sse_connections"]
        assert "total" in connections
        assert "aider_events" in connections
        assert "error_events" in connections
        assert "progress_events" in connections

    @pytest.mark.asyncio
    @patch("aider_mcp_server.pages.application.app._generate_sse_events")
    async def test_sse_endpoints_exist(self, mock_generate_sse_events, test_app):
        """Test that all SSE monitoring endpoints are accessible."""

        # Mock the generator to yield a minimal event and then stop, preventing the test from hanging
        async def mock_generator(*args, **kwargs):
            # Yield a minimal valid SSE event format
            yield "data: {}\n\n"
            # The generator stops after yielding one event, allowing the stream to close

        mock_generate_sse_events.side_effect = mock_generator

        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            # Test all SSE endpoints return successful responses (not 404)
            endpoints = ["/events/aider", "/events/errors", "/events/progress"]

            for endpoint in endpoints:
                # Use client.stream to establish the connection and check status
                # The mocked generator will ensure this completes quickly
                async with client.stream("GET", endpoint) as response:
                    assert response.status_code == 200
                    # The stream should close automatically after the mocked generator finishes
                    # No need for explicit aclose() here as the generator is short-lived

    @pytest.mark.asyncio
    async def test_broadcast_to_sse_clients(self):
        """Test the SSE client broadcasting functionality."""
        # Test event routing logic
        test_events = [
            ("aider.rate_limit_detected", {"provider": "openai", "attempt": 1}),
            ("aider.session_started", {"session_id": "test", "files": ["test.py"]}),
            ("aider.error_occurred", {"error": "test error", "context": "test"}),
        ]

        for event_type, event_data in test_events:
            # This should not raise any exceptions
            await _broadcast_to_sse_clients(event_type, event_data)

    @pytest.mark.asyncio
    async def test_event_filtering_logic(self):
        """Test that events are routed to correct endpoint types."""
        # Mock the global event connections
        import aider_mcp_server.pages.application.app as app_module

        original_connections = app_module._event_connections

        # Set up test connections with mock queues
        test_connections = {
            "aider": {"client1": AsyncMock()},
            "errors": {"client2": AsyncMock()},
            "progress": {"client3": AsyncMock()},
        }
        app_module._event_connections = test_connections

        try:
            # Test rate limit event (should go to aider and errors)
            await _broadcast_to_sse_clients("aider.rate_limit_detected", {"test": "data"})

            # Verify the event was sent to correct queues
            test_connections["aider"]["client1"].put_nowait.assert_called_once()
            test_connections["errors"]["client2"].put_nowait.assert_called_once()
            test_connections["progress"]["client3"].put_nowait.assert_not_called()

            # Reset mocks
            for endpoint_type in test_connections:
                for client_queue in test_connections[endpoint_type].values():
                    client_queue.reset_mock()

            # Test progress event (should go to aider and progress)
            await _broadcast_to_sse_clients("aider.session_progress", {"progress": 50})

            test_connections["aider"]["client1"].put_nowait.assert_called_once()
            test_connections["errors"]["client2"].put_nowait.assert_not_called()
            test_connections["progress"]["client3"].put_nowait.assert_called_once()

        finally:
            # Restore original connections
            app_module._event_connections = original_connections

    def test_sse_connection_stats(self):
        """Test SSE connection statistics function."""
        # Mock the global event connections
        import aider_mcp_server.pages.application.app as app_module

        original_connections = app_module._event_connections

        # Set up test connections
        test_connections = {
            "aider": {"client1": MagicMock(), "client2": MagicMock()},
            "errors": {"client3": MagicMock()},
            "progress": {},
        }
        app_module._event_connections = test_connections

        try:
            stats = get_sse_connection_stats()

            assert stats["total"] == 3
            assert stats["aider_events"] == 2
            assert stats["error_events"] == 1
            assert stats["progress_events"] == 0

        finally:
            # Restore original connections
            app_module._event_connections = original_connections

    @pytest.mark.asyncio
    async def test_public_broadcast_function(self):
        """Test the public broadcast function for coordinator integration."""
        # This should not raise any exceptions
        await broadcast_event_to_sse_clients("aider.test_event", {"test": "data"})

    @pytest.mark.asyncio
    async def test_sse_event_format(self):
        """Test that SSE events have the correct format."""
        import aider_mcp_server.pages.application.app as app_module

        original_connections = app_module._event_connections

        # Capture events sent to mock queue
        captured_events = []

        class MockQueue:
            def put_nowait(self, event):
                captured_events.append(event)

        test_connections = {
            "aider": {"client1": MockQueue()},
            "errors": {},
            "progress": {},
        }
        app_module._event_connections = test_connections

        try:
            test_data = {"provider": "openai", "attempt": 1}
            await _broadcast_to_sse_clients("aider.rate_limit_detected", test_data)

            assert len(captured_events) == 1
            event = captured_events[0]

            # Check SSE event structure
            assert "type" in event
            assert "data" in event
            assert "id" in event

            assert event["type"] == "aider.rate_limit_detected"
            assert "provider" in event["data"]
            assert "sse_timestamp" in event["data"]
            assert "correlation_id" in event["data"]

        finally:
            app_module._event_connections = original_connections


class TestSSEEndpointIntegration:
    """Integration tests for SSE endpoint behavior."""

    @pytest.mark.skip(reason="Skipping complex SSE header test for now.")
    @pytest.mark.asyncio
    @patch("sse_starlette.sse.EventSourceResponse")  # Patch the EventSourceResponse class
    async def test_sse_endpoint_headers(self, MockEventSourceResponse, test_app):
        """Test that SSE endpoints return correct headers without full streaming."""

        # Define a mock instance that simulates the ASGI response interface
        class MockEventSourceResponseInstance:
            def __init__(self, generator, headers=None, **kwargs):
                # Store headers passed during instantiation by _create_sse_monitoring_endpoint
                # EventSourceResponse adds Content-Type: text/event-stream by default
                # and merges user-provided headers. We simulate the final headers sent.
                self._headers_from_app = headers if headers is not None else {}
                self.status_code = 200  # EventSourceResponse typically returns 200 on success
                # We don't need to store or run the generator for this test

            async def __call__(self, scope, receive, send):
                # Combine headers passed from the app with the expected SSE Content-Type
                response_headers = {
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    # X-Client-ID is generated by _create_sse_monitoring_endpoint and passed in headers
                    **self._headers_from_app,
                }
                # Headers must be bytes tuples: [(b'header-name', b'header-value')]
                encoded_headers = [(k.encode(), v.encode()) for k, v in response_headers.items()]
                await send(
                    {
                        "type": "http.response.start",
                        "status": self.status_code,
                        "headers": encoded_headers,
                    }
                )
                # Simulate sending body (empty for headers check)
                await send(
                    {
                        "type": "http.response.body",
                        "body": b"",
                        "more_body": False,
                    }
                )
                # No need to run the generator or stream events

        # Configure the mock EventSourceResponse class to return instances of our mock
        MockEventSourceResponse.side_effect = MockEventSourceResponseInstance

        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            endpoints = ["/events/aider", "/events/errors", "/events/progress"]

            for endpoint in endpoints:
                # Use client.get to get the response headers
                response = await client.get(endpoint)

                # Check status code from the mock response
                assert response.status_code == 200

                # Check headers from the mock response
                headers = {k.lower(): v.lower() for k, v in response.headers.items()}

                # Check content-type starts with expected value (may have charset)
                assert "content-type" in headers
                assert headers["content-type"].startswith("text/event-stream")

                # Check other headers exactly
                assert "cache-control" in headers
                assert headers["cache-control"] == "no-cache"
                assert "connection" in headers
                assert headers["connection"] == "keep-alive"
                # Check for X-Client-ID header presence.
                # The actual UUID is generated by _create_sse_monitoring_endpoint.
                assert "x-client-id" in headers
                # We cannot predict the UUID, so we just check for presence.

    @pytest.mark.asyncio
    async def test_coordinator_integration_setup(self, test_app, mock_coordinator):
        """Test that SSE endpoints integrate with coordinator properly."""
        # This test verifies the setup doesn't crash
        # Real integration would require running coordinator events

        # Verify mock coordinator is accessible through the app
        import aider_mcp_server.pages.application.app as app_module

        assert app_module._adapter is not None
        assert app_module._adapter._coordinator == mock_coordinator
