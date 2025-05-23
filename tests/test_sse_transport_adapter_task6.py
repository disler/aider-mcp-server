import asyncio
import json
from unittest import mock

from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase


from aider_mcp_server.atoms.event_types import EventTypes
from aider_mcp_server.security import SecurityContext
from aider_mcp_server.sse_transport_adapter_task6 import SSETransportAdapter

# Helper function to read SSE events
async def read_sse_event_from_stream(stream_reader) -> dict | None:
    """
    Reads a single SSE event from the stream.
    Handles 'data: json_payload\n\n' and 'event: name\ndata: payload\n\n'.
    """
    data_lines = []
    event_field_value = None

    while True:
        line_bytes = await stream_reader.readline()
        if not line_bytes:  # EOF
            if data_lines or event_field_value: # Partial event before EOF
                # This case might indicate an improperly terminated stream
                # For this helper, we'll return what we have or None
                return {"error": "EOF_IN_EVENT"} if data_lines else None
            return None
        
        line = line_bytes.decode('utf-8').strip()
        
        if not line:  # Empty line signifies end of an event
            if event_field_value == "close" and data_lines:
                return {"type": "close_event", "data": "\n".join(data_lines)}
            
            if data_lines:
                full_data = "\n".join(data_lines)
                try:
                    parsed_json = json.loads(full_data)
                    # If 'event:' field was present and JSON doesn't have 'type', add it.
                    # However, this adapter always includes 'type' in JSON.
                    if event_field_value and isinstance(parsed_json, dict) and 'type' not in parsed_json:
                        # This path is unlikely to be hit with the current adapter
                        parsed_json['type_from_event_field'] = event_field_value
                    return parsed_json
                except json.JSONDecodeError:
                    return {"error": "JSONDecodeError", "raw_data": full_data}
            
            # Reset for next event, in case of keep-alive pings (just newlines)
            event_field_value = None
            data_lines = []
            # If it was just an empty line (e.g. comment processed and ignored), continue reading
            # This handles SSE comments (lines starting with ':') followed by an empty line.
            continue

        if line.startswith("event:"):
            event_field_value = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].strip())
        elif line.startswith(":"):  # SSE Comment
            pass  # Ignore comments
        # Silently ignore other fields like 'id:' or 'retry:' for this helper


class TestSSETransportAdapter(AioHTTPTestCase):

    async def get_application(self):
        # Create an app instance for the test case
        self.app_instance = web.Application()
        # Initialize adapter with the app_instance, so _app_owned is False
        self.adapter = SSETransportAdapter(
            transport_id="test_sse_adapter",
            app=self.app_instance,
            host="127.0.0.1", # Will be overridden by test server
            port=0 # Will be overridden by test server
        )
        # The adapter adds its routes in __init__
        return self.app_instance

    async def test_connection_established_and_headers(self):
        client_id_val = "client123"
        async with self.client.get(f"/events?client_id={client_id_val}") as resp:
            self.assertEqual(resp.status, 200)
            self.assertEqual(resp.headers["Content-Type"], "text/event-stream")
            self.assertEqual(resp.headers["Cache-Control"], "no-cache")
            self.assertEqual(resp.headers["Connection"], "keep-alive")

            # Check for "connection_established" event
            event = await read_sse_event_from_stream(resp.content)
            self.assertIsNotNone(event)
            self.assertEqual(event.get("type"), "connection_established")
            self.assertEqual(event.get("client_id"), client_id_val)
        
        # Check client is tracked
        self.assertIn(client_id_val, self.adapter._clients)
        # Clean up (client disconnects automatically when 'async with' block exits)
        # Wait a moment for the server to process the disconnect
        await asyncio.sleep(0.01)
        self.assertNotIn(client_id_val, self.adapter._clients)


    async def test_connection_default_client_id(self):
        async with self.client.get("/events") as resp:
            self.assertEqual(resp.status, 200)
            event = await read_sse_event_from_stream(resp.content)
            self.assertIsNotNone(event)
            self.assertEqual(event.get("type"), "connection_established")
            # Default client_id is str(id(request)), check it's present
            self.assertIn("client_id", event) 
            client_id_val = event["client_id"] # Get the generated client_id
        
        await asyncio.sleep(0.01) # Allow disconnect to process
        self.assertNotIn(client_id_val, self.adapter._clients)


    async def test_send_event_broadcasts_to_all_clients(self):
        clients = []
        client_ids = ["c1", "c2"]
        try:
            for cid in client_ids:
                resp = await self.client.get(f"/events?client_id={cid}")
                self.assertEqual(resp.status, 200)
                # Read "connection_established"
                await read_sse_event_from_stream(resp.content)
                clients.append(resp)
            
            self.assertEqual(len(self.adapter._clients), 2)

            event_payload = {"message": "hello world"}
            await self.adapter.send_event(EventTypes.STATUS, event_payload)

            for i, resp in enumerate(clients):
                event = await read_sse_event_from_stream(resp.content)
                self.assertIsNotNone(event)
                self.assertEqual(event.get("type"), EventTypes.STATUS.value)
                self.assertEqual(event.get("message"), event_payload["message"])
        finally:
            for resp in clients:
                resp.close()
        
        await asyncio.sleep(0.01) # Allow disconnects
        self.assertEqual(len(self.adapter._clients), 0)

    async def test_event_formatting_and_type_override(self):
        async with self.client.get("/events?client_id=cformat") as resp:
            self.assertEqual(resp.status, 200)
            await read_sse_event_from_stream(resp.content) # connection_established

            # Test data["type"] overrides event_type.value
            custom_type_payload = {"type": "custom_status", "detail": "override"}
            await self.adapter.send_event(EventTypes.STATUS, custom_type_payload)
            event1 = await read_sse_event_from_stream(resp.content)
            self.assertEqual(event1.get("type"), "custom_status")
            self.assertEqual(event1.get("detail"), "override")

            # Test event_type.value is used if data["type"] is not present
            no_type_payload = {"detail": "no_override"}
            await self.adapter.send_event(EventTypes.PROGRESS, no_type_payload)
            event2 = await read_sse_event_from_stream(resp.content)
            self.assertEqual(event2.get("type"), EventTypes.PROGRESS.value)
            self.assertEqual(event2.get("detail"), "no_override")
        
        await asyncio.sleep(0.01)
        self.assertNotIn("cformat", self.adapter._clients)

    async def test_heartbeat_event_sent_periodically(self):
        # This test verifies the connection stays open and heartbeats are configured.
        # Precise timing of the 30s heartbeat is hard to test reliably in unit tests.
        # We will check that a heartbeat event is received if we wait.
        # For faster testing, we mock asyncio.sleep.
        
        mock_sleep = mock.AsyncMock() # Mock asyncio.sleep

        with mock.patch("asyncio.sleep", mock_sleep):
            async with self.client.get("/events?client_id=hb_client") as resp:
                self.assertEqual(resp.status, 200)
                await read_sse_event_from_stream(resp.content) # connection_established

                # Simulate time passing for one heartbeat cycle
                # The loop is: sleep(30) -> send_heartbeat
                # We need sleep to return once to trigger the send.
                
                # Allow the _handle_sse_connection loop to run once
                # by making the first call to sleep return immediately.
                # Subsequent calls can raise an exception to break the loop for test cleanup.
                call_count = 0
                async def side_effect(*args, **kwargs):
                    nonlocal call_count
                    call_count += 1
                    if call_count == 1: # First sleep in the loop
                        return
                    raise asyncio.CancelledError # Stop the loop after first heartbeat

                mock_sleep.side_effect = side_effect
                
                heartbeat_event = await read_sse_event_from_stream(resp.content)
                self.assertIsNotNone(heartbeat_event, "Heartbeat event not received")
                self.assertEqual(heartbeat_event.get("type"), EventTypes.HEARTBEAT.value)
                
                # Ensure sleep was called (at least once for the heartbeat)
                mock_sleep.assert_called()

        await asyncio.sleep(0.01)
        self.assertNotIn("hb_client", self.adapter._clients)


    async def test_client_disconnect_removes_client(self):
        resp = await self.client.get("/events?client_id=disconnect_test")
        self.assertEqual(resp.status, 200)
        await read_sse_event_from_stream(resp.content) # connection_established
        self.assertIn("disconnect_test", self.adapter._clients)
        
        resp.close() # Simulate client disconnect
        await asyncio.sleep(0.01) # Allow server to process disconnect
        
        self.assertNotIn("disconnect_test", self.adapter._clients)

    async def test_server_shutdown_sends_close_event_and_clears_clients(self):
        clients_to_close = []
        client_ids_for_shutdown = ["shutdown_c1", "shutdown_c2"]
        try:
            for cid in client_ids_for_shutdown:
                resp = await self.client.get(f"/events?client_id={cid}")
                self.assertEqual(resp.status, 200)
                await read_sse_event_from_stream(resp.content) # connection_established
                clients_to_close.append(resp)
            
            self.assertEqual(len(self.adapter._clients), 2)

            # Call adapter's shutdown (which is part of ITransportAdapter)
            await self.adapter.shutdown()

            for resp in clients_to_close:
                event = await read_sse_event_from_stream(resp.content)
                self.assertIsNotNone(event)
                self.assertEqual(event.get("type"), "close_event") # Custom type from helper
                self.assertEqual(event.get("data"), "Server shutting down")
        finally:
            for resp in clients_to_close:
                resp.close() # Ensure connections are closed test-side

        self.assertEqual(len(self.adapter._clients), 0, "Clients not cleared after shutdown")

    # --- ITransportAdapter Interface Compliance Tests ---
    def test_get_transport_id(self):
        self.assertEqual(self.adapter.get_transport_id(), "test_sse_adapter")

    def test_get_transport_type(self):
        self.assertEqual(self.adapter.get_transport_type(), "sse")

    async def test_initialize_method(self):
        # initialize is a no-op as per spec, just ensure it runs without error
        await self.adapter.initialize() 
        # No specific state change to assert for this adapter's initialize

    async def test_start_listening_app_not_owned(self):
        # Adapter was initialized with self.app_instance, so _app_owned is False
        self.assertFalse(self.adapter._app_owned)
        initial_runner = self.adapter._runner
        await self.adapter.start_listening()
        # Should not start a new runner or change existing one if not owned
        self.assertIs(self.adapter._runner, initial_runner) 

    def test_get_capabilities(self):
        expected_capabilities = {
            EventTypes.STATUS,
            EventTypes.PROGRESS,
            EventTypes.TOOL_RESULT,
            EventTypes.HEARTBEAT,
        }
        self.assertEqual(self.adapter.get_capabilities(), expected_capabilities)

    def test_should_receive_event(self):
        # Default behavior is True
        self.assertTrue(self.adapter.should_receive_event(EventTypes.STATUS, {}))

    async def test_handle_message_request_raises_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            await self.adapter.handle_message_request({})

    def test_validate_request_security(self):
        context = self.adapter.validate_request_security({})
        self.assertIsInstance(context, SecurityContext)
        self.assertTrue(context.is_anonymous)
        self.assertIsNone(context.user_id)
        self.assertEqual(context.transport_id, "test_sse_adapter")

    async def test_send_event_to_disconnected_client_gracefully_handled(self):
        client_id_dis = "client_to_disconnect_early"
        resp = await self.client.get(f"/events?client_id={client_id_dis}")
        self.assertEqual(resp.status, 200)
        await read_sse_event_from_stream(resp.content) # connection_established
        self.assertIn(client_id_dis, self.adapter._clients)

        # Mock the StreamResponse's write method for this specific client to simulate error
        
        # Close the client connection from the client side
        resp.close()
        # Wait for server to potentially process this, though it might not be immediate
        await asyncio.sleep(0.01) 
        # At this point, the client might be removed or might still be there if send_event is fast

        # Now, attempt to send an event. The adapter should handle ConnectionResetError.
        # The send_event itself iterates and tries to send. If a client is gone,
        # _send_sse_event will raise ConnectionResetError, which is caught in send_event's loop.
        # The _handle_sse_connection's finally block is what removes the client.
        
        # To ensure the client is removed by the time send_event is called,
        # or to test the error handling within send_event if write fails:
        # We can patch the specific client's response object's write method.
        
        # If client is still in adapter._clients, patch its response object
        # This is a bit intrusive but tests the error handling path.
        # A less intrusive way is to rely on resp.close() and timing.
        
        # Let's rely on resp.close() and the adapter's natural error handling.
        # The client should be removed by _handle_sse_connection's finally block.
        # If not removed yet, _send_sse_event will fail, and it should be handled.
        
        try:
            await self.adapter.send_event(EventTypes.STATUS, {"data": "test_after_disconnect"})
        except Exception as e:
            self.fail(f"adapter.send_event raised an unexpected exception: {e}")

        # The client should definitely be removed now if it wasn't already
        self.assertNotIn(client_id_dis, self.adapter._clients)


class TestSSETransportAdapterAppOwned(AioHTTPTestCase):
    # Separate test case for when the adapter owns the app, as AioHTTPTestCase
    # normally runs its own app. We can't use AioHTTPTestCase's client for this directly
    # if the adapter starts its own server on a different port.
    # So, these tests will be more unit-style for start_listening/shutdown.

    async def get_application(self):
        # Provide a dummy app for AioHTTPTestCase.
        # Tests in this class focus on the adapter's self-managed app and server.
        app = web.Application()
        return app

    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.test_host = "127.0.0.1"
        # For app-owned tests, we create adapter without an app
        self.owned_adapter = SSETransportAdapter(
            transport_id="owned_adapter",
            app=None, # App is owned by adapter
            host=self.test_host,
            port=0 # OS will pick a free port
        )
        # The adapter's app is self.owned_adapter._app
        # Routes are added to self.owned_adapter._app in its __init__

    async def asyncTearDown(self):
        if self.owned_adapter and self.owned_adapter._runner: # Ensure cleanup if adapter started its server
            await self.owned_adapter.shutdown()
        await super().asyncTearDown() # Important for AioHTTPTestCase cleanup

    async def test_start_listening_app_owned(self): # This test doesn't use self.client from AioHTTPTestCase
        self.assertTrue(self.owned_adapter._app_owned)
        self.assertIsNone(self.owned_adapter._runner)
        
        await self.owned_adapter.start_listening()
        
        self.assertIsNotNone(self.owned_adapter._runner, "Runner should be created")
        self.assertIsNotNone(self.owned_adapter._site, "Site should be created")
        self.assertIsNotNone(self.owned_adapter._site._server, "Site server should be created and started")
        self.assertTrue(self.owned_adapter._site._server.sockets, "Site server should have active sockets")
        
        # Check if the server is actually listening (optional, basic check)
        # This requires knowing the actual port. self.owned_adapter._port might still be 0.
        # The actual port is in self.owned_adapter._site._server.sockets[0].getsockname()[1]
        actual_port = self.owned_adapter._site._server.sockets[0].getsockname()[1]
        self.assertGreater(actual_port, 0)

        # Try a quick connection to see if it's up (very basic)
        try:
            reader, writer = await asyncio.open_connection(self.test_host, actual_port)
            writer.close()
            await writer.wait_closed()
        except ConnectionRefusedError:
            self.fail("Server did not seem to start correctly (connection refused)")

        # Calling again should be a no-op if already running
        runner_before = self.owned_adapter._runner
        site_before = self.owned_adapter._site
        await self.owned_adapter.start_listening()
        self.assertIs(self.owned_adapter._runner, runner_before)
        self.assertIs(self.owned_adapter._site, site_before)

        await self.owned_adapter.shutdown() # Clean up
        self.assertIsNone(self.owned_adapter._runner)
        self.assertIsNone(self.owned_adapter._site)


    async def test_shutdown_app_owned(self):
        # Start it first
        await self.owned_adapter.start_listening()
        self.assertTrue(self.owned_adapter._app_owned)
        self.assertIsNotNone(self.owned_adapter._runner)
        
        # Add a mock client to test cleanup (not a real connection)
        mock_stream_response = mock.MagicMock(spec=web.StreamResponse)
        mock_stream_response.write = mock.AsyncMock()
        mock_stream_response.drain = mock.AsyncMock()
        self.owned_adapter._clients["mock_client_for_shutdown"] = mock_stream_response

        await self.owned_adapter.shutdown()

        self.assertIsNone(self.owned_adapter._runner, "Runner should be cleaned up")
        self.assertIsNone(self.owned_adapter._site, "Site should be cleaned up")
        self.assertEqual(len(self.owned_adapter._clients), 0, "Clients should be cleared")
        
        # Check if the "close" event was attempted on the mock client
        mock_stream_response.write.assert_any_call(b"event: close\ndata: Server shutting down\n\n")

    def test_app_property_returns_app(self): # Synchronous test
        # Case 1: App provided externally
        external_app = web.Application()
        adapter_with_external_app = SSETransportAdapter("id1", app=external_app)
        self.assertIs(adapter_with_external_app.app, external_app)

        # Case 2: App owned by adapter (uses self.owned_adapter from asyncSetUp)
        self.assertTrue(self.owned_adapter._app_owned) # Sanity check
        self.assertIsInstance(self.owned_adapter.app, web.Application)
        # Ensure it's the one created internally by the adapter
        self.assertIs(self.owned_adapter.app, self.owned_adapter._app)

# To run these tests with pytest, it will discover AioHTTPTestCase.
# If not using pytest, you might need a main block:
# if __name__ == '__main__':
#     import unittest
#     unittest.main()
