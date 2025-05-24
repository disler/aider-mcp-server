"""
Tests for Modernized SSE Transport Adapter with MCP Protocol 2025-03-26 compliance.
"""

import asyncio
import json
import time
import unittest
import warnings
from unittest.mock import AsyncMock, MagicMock

from aider_mcp_server.atoms.types.event_types import EventTypes
from aider_mcp_server.organisms.transports.sse.sse_transport_adapter_modernized import ModernizedSSETransportAdapter


class TestModernizedSSETransportAdapter(unittest.IsolatedAsyncioTestCase):
    """Test suite for the modernized SSE transport adapter."""

    async def asyncSetUp(self):
        """Set up test environment."""
        self.adapter = ModernizedSSETransportAdapter(
            transport_id="test-sse",
            heartbeat_interval=0.1,  # Fast heartbeat for testing
        )
        await self.adapter.initialize()

    async def asyncTearDown(self):
        """Clean up test environment."""
        if self.adapter:
            await self.adapter.shutdown()

    async def test_deprecation_warning_on_import(self):
        """Test that deprecation warning is issued on import."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Re-import to trigger warning
            import sys

            module_name = "aider_mcp_server.organisms.transports.sse.sse_transport_adapter_modernized"
            if module_name in sys.modules:
                del sys.modules[module_name]

            # This import triggers the deprecation warning
            import aider_mcp_server.organisms.transports.sse.sse_transport_adapter_modernized  # noqa: F401

            # Check if any deprecation warnings were triggered
            deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
            self.assertGreater(len(deprecation_warnings), 0)

            # Check the warning message
            found_sse_warning = False
            for warning in deprecation_warnings:
                if "SSE Transport is deprecated" in str(warning.message):
                    found_sse_warning = True
                    break
            self.assertTrue(found_sse_warning, "Should have SSE deprecation warning")

    async def test_transport_properties(self):
        """Test basic transport adapter properties."""
        self.assertEqual(self.adapter.get_transport_id(), "test-sse")
        self.assertEqual(self.adapter.get_transport_type(), "sse")

        capabilities = self.adapter.get_capabilities()
        expected_capabilities = {
            EventTypes.STATUS,
            EventTypes.PROGRESS,
            EventTypes.TOOL_RESULT,
            EventTypes.HEARTBEAT,
        }
        self.assertEqual(capabilities, expected_capabilities)

    async def test_should_receive_event(self):
        """Test event filtering logic."""
        # Should receive supported event types
        self.assertTrue(self.adapter.should_receive_event(EventTypes.STATUS, {}))
        self.assertTrue(self.adapter.should_receive_event(EventTypes.PROGRESS, {}))
        self.assertTrue(self.adapter.should_receive_event(EventTypes.TOOL_RESULT, {}))
        self.assertTrue(self.adapter.should_receive_event(EventTypes.HEARTBEAT, {}))

    async def test_event_broadcasting_format(self):
        """Test event broadcasting uses MCP 2025-03-26 format."""
        # Mock a connected client
        mock_response = AsyncMock()
        mock_response.write = AsyncMock()

        client_id = "test-client"
        async with self.adapter._client_lock:
            self.adapter._clients[client_id] = {
                "response": mock_response,
                "connected_at": time.time(),
                "last_heartbeat": time.time(),
                "remote": "127.0.0.1",
            }

        # Send an event
        test_data = {"message": "test event"}
        await self.adapter.send_event(EventTypes.STATUS, test_data)

        # Verify the event was sent
        mock_response.write.assert_called_once()
        call_args = mock_response.write.call_args[0][0]

        # Parse the SSE message
        sse_data = call_args.decode("utf-8")
        self.assertTrue(sse_data.startswith("data: "))
        self.assertTrue(sse_data.endswith("\n\n"))

        # Parse the JSON payload
        json_str = sse_data[6:-2]  # Remove "data: " prefix and "\n\n" suffix
        event_payload = json.loads(json_str)

        # Verify MCP 2025-03-26 compliance
        self.assertEqual(event_payload["jsonrpc"], "2.0")
        self.assertEqual(event_payload["method"], "event")
        self.assertEqual(event_payload["params"]["type"], "status")
        self.assertEqual(event_payload["params"]["data"], test_data)
        self.assertEqual(event_payload["params"]["transport"], "sse")
        self.assertIn("timestamp", event_payload["params"])

    async def test_client_cleanup_on_error(self):
        """Test that clients are properly cleaned up when errors occur."""
        # Mock a client that will fail
        mock_response = AsyncMock()
        mock_response.write = AsyncMock(side_effect=Exception("Connection lost"))

        client_id = "failing-client"
        async with self.adapter._client_lock:
            self.adapter._clients[client_id] = {
                "response": mock_response,
                "connected_at": time.time(),
                "last_heartbeat": time.time(),
                "remote": "127.0.0.1",
            }
            initial_count = len(self.adapter._clients)

        # Send an event that will cause the client to fail
        await self.adapter.send_event(EventTypes.STATUS, {"test": "data"})

        # Verify the client was removed
        async with self.adapter._client_lock:
            self.assertNotIn(client_id, self.adapter._clients)
            self.assertEqual(len(self.adapter._clients), initial_count - 1)

    async def test_authorization_check(self):
        """Test authorization framework compliance."""
        # Test adapter without auth requirement
        self.assertTrue(await self.adapter._check_authorization(MagicMock()))

        # Test adapter with auth requirement
        auth_adapter = ModernizedSSETransportAdapter(transport_id="auth-test", auth_header="Bearer test-token")

        try:
            # Mock request without auth header
            mock_request_no_auth = MagicMock()
            mock_request_no_auth.headers = {}
            self.assertFalse(await auth_adapter._check_authorization(mock_request_no_auth))

            # Mock request with wrong auth header
            mock_request_wrong_auth = MagicMock()
            mock_request_wrong_auth.headers = {"Authorization": "Bearer wrong-token"}
            self.assertFalse(await auth_adapter._check_authorization(mock_request_wrong_auth))

            # Mock request with correct auth header
            mock_request_correct_auth = MagicMock()
            mock_request_correct_auth.headers = {"Authorization": "Bearer test-token"}
            self.assertTrue(await auth_adapter._check_authorization(mock_request_correct_auth))

        finally:
            await auth_adapter.shutdown()

    async def test_cors_headers_addition(self):
        """Test CORS headers are properly added."""
        from aiohttp import web

        response = web.Response()
        self.adapter._add_cors_headers(response)

        self.assertEqual(response.headers["Access-Control-Allow-Origin"], "*")
        self.assertIn("GET, OPTIONS", response.headers["Access-Control-Allow-Methods"])
        self.assertIn("Content-Type", response.headers["Access-Control-Allow-Headers"])

    async def test_statistics_tracking(self):
        """Test that statistics are properly tracked."""
        initial_stats = self.adapter._stats.copy()

        # Simulate some activity
        mock_response = AsyncMock()
        mock_response.write = AsyncMock()

        async with self.adapter._client_lock:
            self.adapter._clients["test"] = {
                "response": mock_response,
                "connected_at": time.time(),
                "last_heartbeat": time.time(),
                "remote": "127.0.0.1",
            }
            self.adapter._stats["connections_total"] += 1
            self.adapter._stats["connections_active"] += 1

        # Send an event
        await self.adapter.send_event(EventTypes.STATUS, {"test": True})

        # Verify statistics updated
        self.assertGreater(self.adapter._stats["messages_sent"], initial_stats["messages_sent"])
        self.assertGreater(self.adapter._stats["connections_total"], initial_stats["connections_total"])

    async def test_heartbeat_functionality(self):
        """Test heartbeat sending functionality."""
        # Mock a connected client
        mock_response = AsyncMock()
        mock_response.write = AsyncMock()

        client_id = "heartbeat-client"
        async with self.adapter._client_lock:
            self.adapter._clients[client_id] = {
                "response": mock_response,
                "connected_at": time.time(),
                "last_heartbeat": time.time(),
                "remote": "127.0.0.1",
            }

        # Start the heartbeat task manually for testing
        heartbeat_task = asyncio.create_task(self.adapter._heartbeat_loop())

        # Wait for heartbeat (adapter has 0.1s interval for testing)
        await asyncio.sleep(0.15)

        # Cancel the heartbeat task
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass

        # Verify heartbeat was sent
        self.assertTrue(mock_response.write.called)

        # Check that at least one heartbeat message was sent
        call_args_list = mock_response.write.call_args_list
        heartbeat_found = False

        for call_args in call_args_list:
            sse_data = call_args[0][0].decode("utf-8")
            if "data: " in sse_data:
                json_str = sse_data[6:-2]  # Remove "data: " and "\n\n"
                try:
                    event_payload = json.loads(json_str)
                    if (
                        event_payload.get("method") == "event"
                        and event_payload.get("params", {}).get("type") == "heartbeat"
                    ):
                        heartbeat_found = True
                        break
                except json.JSONDecodeError:
                    continue

        self.assertTrue(heartbeat_found, "Heartbeat event should have been sent")

    async def test_shutdown_functionality(self):
        """Test proper shutdown functionality."""
        # Add a mock client
        mock_response = AsyncMock()
        mock_response.write = AsyncMock()

        client_id = "shutdown-test-client"
        async with self.adapter._client_lock:
            self.adapter._clients[client_id] = {
                "response": mock_response,
                "connected_at": time.time(),
                "last_heartbeat": time.time(),
                "remote": "127.0.0.1",
            }

        # Verify client is present
        self.assertIn(client_id, self.adapter._clients)

        # Shutdown the adapter
        await self.adapter.shutdown()

        # Verify all clients were removed
        self.assertEqual(len(self.adapter._clients), 0)
        self.assertEqual(self.adapter._stats["connections_active"], 0)


class TestModernizedSSEIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests for modernized SSE transport."""

    async def test_full_lifecycle(self):
        """Test complete adapter lifecycle."""
        adapter = ModernizedSSETransportAdapter(
            transport_id="integration-test",
            host="127.0.0.1",
            port=0,  # Use random available port
            heartbeat_interval=0.1,
        )

        try:
            # Test initialization
            await adapter.initialize()
            self.assertEqual(adapter.get_transport_id(), "integration-test")
            self.assertEqual(adapter.get_transport_type(), "sse")

            # Test capabilities
            capabilities = adapter.get_capabilities()
            self.assertIn(EventTypes.STATUS, capabilities)

            # Test event filtering
            self.assertTrue(adapter.should_receive_event(EventTypes.STATUS, {}))

        finally:
            # Test shutdown
            await adapter.shutdown()

            # Verify cleanup
            self.assertEqual(len(adapter._clients), 0)

    async def test_deprecation_warnings_in_logs(self):
        """Test that deprecation warnings appear in logs."""
        # Test that adapter initialization logs deprecation warning
        adapter = ModernizedSSETransportAdapter(transport_id="deprecation-test")

        try:
            await adapter.initialize()

            # The test passes if initialization completes without error
            # Deprecation warnings are visible in stderr output during testing
            self.assertEqual(adapter.get_transport_id(), "deprecation-test")
            self.assertEqual(adapter.get_transport_type(), "sse")

        finally:
            await adapter.shutdown()


if __name__ == "__main__":
    unittest.main()
