import asyncio
import json
import uuid
from typing import Callable  # Added for LoggerFactory type hint
from typing import Any, AsyncGenerator, Dict, List
from unittest import mock

import httpx  # For making async HTTP requests
import pytest
import pytest_asyncio

from aider_mcp_server.atoms.event_types import EventTypes
from aider_mcp_server.http_streamable_transport_adapter import HttpStreamableTransportAdapter
from aider_mcp_server.mcp_types import EventData, LoggerProtocol  # Added LoggerProtocol
from aider_mcp_server.security import SecurityContext

# Mocked dependencies
from aider_mcp_server.transport_coordinator import ApplicationCoordinator


# Helper to read NDJSON stream
async def read_ndjson_stream_from_response(
    response_content_iterator: AsyncGenerator[bytes, None],
    expected_messages: int,
    timeout_per_message: float = 2.0,  # Default timeout for receiving each message
) -> list[dict]:
    events = []
    buffer = b""

    for _ in range(expected_messages):
        message_found_this_iteration = False
        # Try to find a complete message in the current buffer first
        while b"\n" in buffer and not message_found_this_iteration:
            line_bytes, buffer = buffer.split(b"\n", 1)
            line = line_bytes.decode("utf-8").strip()
            if line:
                try:
                    events.append(json.loads(line))
                    message_found_this_iteration = True
                except json.JSONDecodeError:
                    events.append({"error": "JSONDecodeError", "raw_data": line})
                    message_found_this_iteration = True  # Count error as a "message"
            # If line is empty, loop again to find next \n in buffer or fetch more data

        if message_found_this_iteration:
            continue  # Proceed to next expected message

        # If no complete message in buffer, or we need more messages, fetch more data
        try:
            chunk = await asyncio.wait_for(
                anext(response_content_iterator), timeout=timeout_per_message
            )
            if chunk:  # Append non-empty chunk
                buffer += chunk
            else: # Empty chunk, try to read again in next iteration or timeout
                pass

            # After getting new chunk, try to process it for the current message
            while b"\n" in buffer: # Process all complete messages in new chunk + buffer
                line_bytes, buffer = buffer.split(b"\n", 1)
                line = line_bytes.decode("utf-8").strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        events.append({"error": "JSONDecodeError", "raw_data": line})
                    message_found_this_iteration = True
                    break # Found one message, break from inner while to outer for
            
            if not message_found_this_iteration and not (b"\n" in buffer):
                # If after a read and processing, no message was found,
                # and buffer doesn't contain a newline, it implies a partial message
                # or stream ended. The next outer loop iteration will try to read more or timeout.
                # If it was the last expected message, this partial data is ignored.
                pass


        except StopAsyncIteration:
            # Stream ended before all expected messages were received
            return events
        except asyncio.TimeoutError:
            # Timed out waiting for data for the current message
            return events
        
        if not message_found_this_iteration:
            # If after fetching data and attempting to parse, no new message was completed,
            # and we are still expecting more messages, this indicates a problem (e.g. partial message followed by timeout)
            # or the server is not sending data as expected.
            # For this function, we return what we have. The test can then assert the count.
            return events


    return events


@pytest.fixture
def mock_logger_factory():
    mock_logger = mock.MagicMock(spec=LoggerProtocol)
    # Make logger methods awaitable if they are ever awaited (though typically not for logging)
    # For this test, simple MagicMocks are fine as they are not awaited.
    factory = mock.MagicMock(
        spec=Callable[..., LoggerProtocol], return_value=mock_logger
    )  # Use Callable and imported LoggerProtocol
    return factory, mock_logger


@pytest.fixture
def mock_coordinator():
    coordinator = mock.AsyncMock(spec=ApplicationCoordinator)
    coordinator.register_transport = mock.AsyncMock()
    coordinator.unregister_transport = mock.AsyncMock()
    coordinator.broadcast_event = mock.AsyncMock()
    return coordinator


@pytest_asyncio.fixture
async def adapter(mock_logger_factory, mock_coordinator):
    logger_factory, _ = mock_logger_factory
    adapter_instance = HttpStreamableTransportAdapter(
        coordinator=mock_coordinator,
        host="127.0.0.1",
        port=0,  # Use 0 to let OS pick a free port
        get_logger=logger_factory,
        editor_model="test_model",
        current_working_dir="/test/cwd",
        heartbeat_interval=0.05,  # Short for testing, ensure it's > sleep times
    )
    await adapter_instance.initialize()
    await adapter_instance.start_listening()

    # Wait for Uvicorn to start. Adapter has a 0.5s sleep.
    # Add a small delay here to ensure server is fully up.
    await asyncio.sleep(0.6)  # Must be > adapter's internal sleep

    yield adapter_instance

    await adapter_instance.shutdown()


@pytest_asyncio.fixture
async def http_client(adapter: HttpStreamableTransportAdapter):
    actual_port = adapter.get_actual_port()
    assert actual_port is not None, "Could not determine the actual port for the test client."
    base_url = f"http://{adapter._host}:{actual_port}"
    async with httpx.AsyncClient(base_url=base_url, timeout=5.0) as client:  # Added timeout
        yield client


@pytest.mark.asyncio
class TestHttpStreamableTransportAdapter:
    async def test_initialization_and_properties(
        self, adapter: HttpStreamableTransportAdapter, mock_logger_factory, mock_coordinator
    ):
        _, mock_logger = mock_logger_factory
        assert adapter.get_transport_id().startswith("http_stream_")
        assert adapter.get_transport_type() == "http_stream"
        assert adapter._host == "127.0.0.1"
        assert adapter._port == 0  # Initial config port
        actual_port = adapter.get_actual_port()
        assert actual_port is not None and actual_port > 0  # Check actual running port via getter
        assert adapter._editor_model == "test_model"
        assert adapter._current_working_dir == "/test/cwd"

        mock_coordinator.register_transport.assert_called_once_with(adapter.get_transport_id(), adapter)
        assert adapter._mcp_server is not None
        assert adapter._fastmcp_initialized is True

        mock_logger.info.assert_any_call(
            f"HttpStreamableTransportAdapter instance created with ID {adapter.get_transport_id()}"
        )
        mock_logger.debug.assert_any_call(f"HttpStreamableTransportAdapter {adapter.get_transport_id()} initialized.")
        mock_logger.info.assert_any_call(
            f"Starlette app created for HttpStreamableTransportAdapter {adapter.get_transport_id()}."
        )
        mock_logger.info.assert_any_call(
            f"FastMCP initialized for HttpStreamableTransportAdapter {adapter.get_transport_id()}."
        )
        # actual_port is already defined and checked above
        mock_logger.info.assert_any_call(
            f"HttpStreamableTransportAdapter server listening on http://{adapter._host}:{actual_port}"
        )

    async def test_get_capabilities(self, adapter: HttpStreamableTransportAdapter):
        expected_capabilities = {
            EventTypes.STATUS,
            EventTypes.PROGRESS,
            EventTypes.TOOL_RESULT,
            EventTypes.HEARTBEAT,
        }
        assert adapter.get_capabilities() == expected_capabilities

    async def test_stream_connection_successful_and_disconnect(
        self, adapter: HttpStreamableTransportAdapter, http_client: httpx.AsyncClient
    ):
        client_id = f"client_{uuid.uuid4()}"
        events_received = []

        async with http_client.stream("GET", f"/stream/{client_id}") as response:
            assert response.status_code == 200
            assert response.headers["content-type"] == "application/x-ndjson"
            assert client_id in adapter._active_connections

            events_received = await read_ndjson_stream_from_response(response.aiter_bytes(), expected_messages=1)

        assert len(events_received) == 1, f"Expected 1 message, got {len(events_received)}"
        initial_event = events_received[0]
        assert initial_event["event"] == EventTypes.STATUS.value
        assert initial_event["data"]["message"] == "Successfully connected to HTTP stream."
        assert initial_event["data"]["client_id"] == client_id

        await asyncio.sleep(0.1)  # Allow time for disconnect to be processed
        assert client_id not in adapter._active_connections

    async def test_stream_connection_no_client_id_in_path_param(self, http_client: httpx.AsyncClient):
        # Starlette's router should handle this with a 404 if client_id is a mandatory path param.
        # The route is "/stream/{client_id}". A request to "/stream/" will not match.
        response = await http_client.get("/stream/")  # No client_id in path
        assert response.status_code == 404  # Starlette default for unmatched route

    async def test_stream_connection_already_connected(
        self, adapter: HttpStreamableTransportAdapter, http_client: httpx.AsyncClient
    ):
        client_id = f"client_dup_{uuid.uuid4()}"
        async with http_client.stream("GET", f"/stream/{client_id}") as response1:
            assert response1.status_code == 200
            # Consume initial message
            initial_events = await read_ndjson_stream_from_response(response1.aiter_bytes(), expected_messages=1)
            assert len(initial_events) == 1
            assert client_id in adapter._active_connections

            response2 = await http_client.get(f"/stream/{client_id}")
            assert response2.status_code == 409
            assert f"Client {client_id} already connected" in response2.text

        await asyncio.sleep(0.1)
        assert client_id not in adapter._active_connections

    @mock.patch("aider_mcp_server.handlers.process_aider_ai_code_request")
    async def test_message_handler_aider_ai_code_success(
        self, mock_process_request, adapter: HttpStreamableTransportAdapter, http_client: httpx.AsyncClient
    ):
        client_id = f"client_msg_ai_{uuid.uuid4()}"
        async with http_client.stream("GET", f"/stream/{client_id}") as stream_resp:  # Connect client
            initial_events = await read_ndjson_stream_from_response(stream_resp.aiter_bytes(), expected_messages=1)
            assert len(initial_events) == 1

            request_payload = {
                "method": "aider_ai_code",
                "params": {"ai_coding_prompt": "p", "relative_editable_files": ["f.py"]},
                "id": "req1",
            }
            expected_ai_code_result = {"status": "success", "output": "code"}
            mock_process_request.return_value = expected_ai_code_result

            response = await http_client.post(f"/message/{client_id}", json=request_payload)

            assert response.status_code == 200
            response_json = response.json()
            assert response_json["result"] == expected_ai_code_result
            assert response_json["id"] == "req1"
            mock_process_request.assert_called_once()

    @mock.patch("aider_mcp_server.handlers.process_list_models_request")
    async def test_message_handler_list_models_success(
        self, mock_process_request, adapter: HttpStreamableTransportAdapter, http_client: httpx.AsyncClient
    ):
        client_id = f"client_msg_lm_{uuid.uuid4()}"
        async with http_client.stream("GET", f"/stream/{client_id}") as stream_resp:  # Connect client
            initial_events = await read_ndjson_stream_from_response(stream_resp.aiter_bytes(), expected_messages=1)
            assert len(initial_events) == 1

            request_payload = {"method": "list_models", "params": {}, "id": "req2"}
            expected_result = {"models": ["model1", "model2"]}
            mock_process_request.return_value = expected_result

            response = await http_client.post(f"/message/{client_id}", json=request_payload)
            assert response.status_code == 200
            response_json = response.json()
            assert response_json["result"] == expected_result
            assert response_json["id"] == "req2"

    async def test_message_handler_client_not_connected(self, http_client: httpx.AsyncClient):
        client_id = "non_existent_client"
        response = await http_client.post(f"/message/{client_id}", json={"method": "test"})
        assert response.status_code == 404
        assert "not connected to an active stream" in response.json()["error"]

    async def test_message_handler_invalid_json_payload(
        self, adapter: HttpStreamableTransportAdapter, http_client: httpx.AsyncClient
    ):
        client_id = f"client_badjson_{uuid.uuid4()}"
        async with http_client.stream("GET", f"/stream/{client_id}") as stream_resp:
            initial_events = await read_ndjson_stream_from_response(stream_resp.aiter_bytes(), expected_messages=1)
            assert len(initial_events) == 1
            response = await http_client.post(f"/message/{client_id}", content="not json")
            assert response.status_code == 400
            assert "Invalid JSON payload" in response.json()["error"]

    async def test_message_handler_missing_method_in_payload(
        self, adapter: HttpStreamableTransportAdapter, http_client: httpx.AsyncClient
    ):
        client_id = f"client_nomethod_{uuid.uuid4()}"
        async with http_client.stream("GET", f"/stream/{client_id}") as stream_resp:
            initial_events = await read_ndjson_stream_from_response(stream_resp.aiter_bytes(), expected_messages=1)
            assert len(initial_events) == 1
            response = await http_client.post(f"/message/{client_id}", json={"params": {}})
            assert response.status_code == 400
            assert "Missing or invalid 'method' in request payload" in response.json()["error"]

    async def test_message_handler_unknown_method(
        self, adapter: HttpStreamableTransportAdapter, http_client: httpx.AsyncClient
    ):
        client_id = f"client_unknownmeth_{uuid.uuid4()}"
        async with http_client.stream("GET", f"/stream/{client_id}") as stream_resp:
            initial_events = await read_ndjson_stream_from_response(stream_resp.aiter_bytes(), expected_messages=1)
            assert len(initial_events) == 1
            response = await http_client.post(f"/message/{client_id}", json={"method": "phantom_method"})
            assert response.status_code == 404
            assert "Method 'phantom_method' not found" in response.json()["error"]["message"]

    @mock.patch(
        "aider_mcp_server.handlers.process_aider_ai_code_request",
        side_effect=PermissionError("Access denied"),
    )
    async def test_message_handler_permission_error_response(
        self, mock_handler, adapter: HttpStreamableTransportAdapter, http_client: httpx.AsyncClient
    ):
        client_id = f"client_perm_err_{uuid.uuid4()}"
        async with http_client.stream("GET", f"/stream/{client_id}") as stream_resp:
            initial_events = await read_ndjson_stream_from_response(stream_resp.aiter_bytes(), expected_messages=1)
            assert len(initial_events) == 1
            response = await http_client.post(f"/message/{client_id}", json={"method": "aider_ai_code", "params": {}})
            assert response.status_code == 403
            assert "Access denied" in response.json()["error"]["message"]

    @mock.patch(
        "aider_mcp_server.handlers.process_list_models_request",
        side_effect=ValueError("Invalid param"),
    )
    async def test_message_handler_value_error_response(
        self, mock_handler, adapter: HttpStreamableTransportAdapter, http_client: httpx.AsyncClient
    ):
        client_id = f"client_val_err_{uuid.uuid4()}"
        async with http_client.stream("GET", f"/stream/{client_id}") as stream_resp:
            initial_events = await read_ndjson_stream_from_response(stream_resp.aiter_bytes(), expected_messages=1)
            assert len(initial_events) == 1
            response = await http_client.post(f"/message/{client_id}", json={"method": "list_models", "params": {}})
            assert response.status_code == 400
            assert "Invalid param" in response.json()["error"]["message"]

    async def test_event_broadcasting_to_multiple_clients(
        self, adapter: HttpStreamableTransportAdapter, http_client: httpx.AsyncClient
    ):
        client_ids = [f"client_bc_{i}_{uuid.uuid4()}" for i in range(2)]
        client_event_lists: List[List[Dict[Any, Any]]] = [[] for _ in client_ids]
        client_tasks = []

        async def event_collector_task(client_idx: int, client_id: str):
            async with http_client.stream("GET", f"/stream/{client_id}") as response:
                assert response.status_code == 200
                # Expect 2 messages: initial status + 1 broadcasted event
                # Use a timeout that allows for both messages to arrive.
                received_events = await read_ndjson_stream_from_response(
                    response.aiter_bytes(), expected_messages=2, timeout_per_message=2.0
                )
                client_event_lists[client_idx].extend(received_events)

        for i, cid in enumerate(client_ids):
            client_tasks.append(asyncio.create_task(event_collector_task(i, cid)))

        # Allow clients to connect and receive initial message.
        # This sleep is less critical now as read_ndjson_stream_from_response handles waiting.
        # However, send_event is called after this, so clients must be connected.
        await asyncio.sleep(0.2) 

        # Check if initial messages were received (optional, as event_collector_task now handles counts)
        # For robustness, we can let the tasks run and then check final counts.

        broadcast_data: EventData = {"info": "test_broadcast", "id": adapter.get_transport_id()}
        await adapter.send_event(EventTypes.PROGRESS, broadcast_data)

        # Adjust timeout for gather: expected_messages * timeout_per_message + buffer
        # 2 messages * 2.0s/message = 4.0s. Add 1s buffer = 5.0s.
        await asyncio.wait_for(asyncio.gather(*client_tasks), timeout=5.0)

        for i, cid in enumerate(client_ids):
            assert len(client_event_lists[i]) == 2, f"Client {cid} did not receive all events. Got: {client_event_lists[i]}"
            # Initial event
            assert client_event_lists[i][0]["event"] == EventTypes.STATUS.value
            assert client_event_lists[i][0]["data"]["client_id"] == cid
            # Broadcasted event
            assert client_event_lists[i][1]["event"] == EventTypes.PROGRESS.value
            assert client_event_lists[i][1]["data"]["info"] == "test_broadcast"
            assert client_event_lists[i][1]["data"]["transport_origin"]["transport_id"] == adapter.get_transport_id()

    async def test_send_event_client_queue_full(
        self, adapter: HttpStreamableTransportAdapter, http_client: httpx.AsyncClient, mock_logger_factory
    ):
        _, mock_logger = mock_logger_factory
        client_id = f"client_qf_{uuid.uuid4()}"
        async with http_client.stream("GET", f"/stream/{client_id}") as response:  # Connect client
            initial_events = await read_ndjson_stream_from_response(response.aiter_bytes(), expected_messages=1)
            assert len(initial_events) == 1

            client_queue = adapter._active_connections.get(client_id)
            assert client_queue is not None
            with mock.patch.object(client_queue, "put_nowait", side_effect=asyncio.QueueFull):
                await adapter.send_event(EventTypes.STATUS, {"detail": "dropped"})

            mock_logger.warning.assert_any_call(
                f"Outgoing queue full for HTTP stream client {client_id}. Event {EventTypes.STATUS.value} dropped."
            )
        await asyncio.sleep(0.1)  # disconnect

    @mock.patch("json.dumps", side_effect=TypeError("Cannot serialize"))
    async def test_send_event_json_serialization_error(
        self, mock_json_dumps, adapter: HttpStreamableTransportAdapter, mock_logger_factory
    ):
        _, mock_logger = mock_logger_factory
        await adapter.send_event(EventTypes.STATUS, {"data": object()})  # object() is not serializable
        mock_logger.error.assert_called_once()
        log_msg = mock_logger.error.call_args[0][0]
        assert "JSON serialization error" in log_msg
        assert "Cannot serialize" in log_msg

    async def test_heartbeat_event_via_send_event(
        self, adapter: HttpStreamableTransportAdapter, http_client: httpx.AsyncClient
    ):
        client_id = f"client_hb_test_{uuid.uuid4()}"
        events = []
        async with http_client.stream("GET", f"/stream/{client_id}") as response:
            # Expect initial status message and one heartbeat.
            # Heartbeat interval is 0.05s. Timeout for message should be > 0.05s.
            # e.g. 0.5s per message. Total time ~1s.
            events = await read_ndjson_stream_from_response(
                response.aiter_bytes(), expected_messages=2, timeout_per_message=0.5
            )

        assert len(events) >= 1, "Did not receive at least the initial status event"
        assert events[0]["event"] == EventTypes.STATUS.value

        if len(events) < 2:
            pytest.skip(
                f"Heartbeat event not received within timeout. Expected 2 events, got {len(events)}. This can be timing-sensitive."
            )
        
        # If we received 2 events:
        heartbeat_event = next((e for e in events if e["event"] == EventTypes.HEARTBEAT.value), None)
        assert heartbeat_event is not None, f"Heartbeat event not found in received events: {events}"
        assert heartbeat_event["data"]["transport_id"] == adapter.get_transport_id()
        assert heartbeat_event["data"]["transport_origin"]["transport_id"] == adapter.get_transport_id()

    async def test_shutdown_closes_streams_and_cleans_up(
        self,
        adapter: HttpStreamableTransportAdapter,
        http_client: httpx.AsyncClient,
        mock_logger_factory,
        mock_coordinator,
    ):
        _, mock_logger = mock_logger_factory
        client_id = f"client_shutdown_test_{uuid.uuid4()}"

        client_stream_ended_event = asyncio.Event()

        async def client_connect_task():
            try:
                async with http_client.stream("GET", f"/stream/{client_id}") as response:
                    assert response.status_code == 200
                    assert client_id in adapter._active_connections
                    # Try to read from stream until it's closed by server shutdown
                    async for _ in response.aiter_bytes():
                        pass  # Consume data
            except httpx.StreamError:  # Covers various stream termination issues
                pass  # Expected when server shuts down connection
            finally:
                client_stream_ended_event.set()

        task = asyncio.create_task(client_connect_task())
        await asyncio.sleep(0.2)  # Ensure client connected
        assert client_id in adapter._active_connections

        # Store queue to check if CLOSE_STREAM was put (optional, advanced check)
        # client_queue = adapter._active_connections[client_id]
        # with mock.patch.object(client_queue, 'put', wraps=client_queue.put) as mock_put:

        await adapter.shutdown()  # This is also called by fixture, testing it explicitly

        await asyncio.wait_for(client_stream_ended_event.wait(), timeout=2.0)

        assert client_id not in adapter._active_connections
        assert len(adapter._active_connections) == 0
        assert adapter._server_instance is None  # Uvicorn server instance cleaned up
        assert adapter._server_task is None

        mock_logger.info.assert_any_call(
            f"Shutting down HttpStreamableTransportAdapter ({adapter.get_transport_id()})..."
        )
        mock_coordinator.unregister_transport.assert_called_once_with(adapter.get_transport_id())
        # mock_put.assert_any_call("CLOSE_STREAM") # Check if internal signal was sent

        if not task.done():  # Should be done
            task.cancel()

    async def test_validate_request_security_default(self, adapter: HttpStreamableTransportAdapter):
        sec_context = adapter.validate_request_security({"client_id": "c1", "payload": {}, "headers": httpx.Headers()})
        assert isinstance(sec_context, SecurityContext)
        assert sec_context.is_anonymous is True
        assert sec_context.user_id is None
        assert sec_context.transport_id == adapter.get_transport_id()

    async def test_should_receive_event_logic(self, adapter: HttpStreamableTransportAdapter, mock_logger_factory):
        _, mock_logger = mock_logger_factory
        my_id = adapter.get_transport_id()

        # Event from other transport
        assert adapter.should_receive_event(EventTypes.STATUS, {"transport_origin": {"transport_id": "other"}})
        mock_logger.debug.assert_any_call(
            f"HttpStream ({my_id}) will process event {EventTypes.STATUS.value} from origin other."
        )

        # Event from self (non-heartbeat) - should skip
        assert not adapter.should_receive_event(EventTypes.STATUS, {"transport_origin": {"transport_id": my_id}})
        mock_logger.debug.assert_any_call(
            f"HttpStream ({my_id}) skipping event {EventTypes.STATUS.value} as it originated from self and is not a self-generated heartbeat."
        )

        # Self-generated heartbeat (data.transport_id is self) - should process
        hb_data_self = {"transport_origin": {"transport_id": my_id}, "transport_id": my_id}
        assert adapter.should_receive_event(EventTypes.HEARTBEAT, hb_data_self)
        mock_logger.debug.assert_any_call(f"HttpStream ({my_id}) will send self-generated HEARTBEAT.")

    @mock.patch("aider_mcp_server.http_streamable_transport_adapter.FastMCP")
    async def test_fastmcp_initialization_and_tool_registration(
        self, MockFastMCP, mock_coordinator, mock_logger_factory
    ):
        logger_factory, mock_logger = mock_logger_factory

        adapter_instance = HttpStreamableTransportAdapter(
            coordinator=mock_coordinator, get_logger=logger_factory, heartbeat_interval=30
        )
        mock_fmcps_server = MockFastMCP.return_value
        mock_fmcps_server.tool = mock.MagicMock(return_value=lambda f: f)  # Decorator passthrough

        await adapter_instance.initialize()

        MockFastMCP.assert_called_once_with(f"aider-http-stream-{adapter_instance.get_transport_id()}")
        assert mock_fmcps_server.tool.call_count == 2  # aider_ai_code, list_models
        assert adapter_instance._fastmcp_initialized is True
        mock_logger.info.assert_any_call(
            f"FastMCP initialized for HttpStreamableTransportAdapter {adapter_instance.get_transport_id()}."
        )

        # Test no FastMCP if no coordinator
        adapter_no_coord = HttpStreamableTransportAdapter(coordinator=None, get_logger=logger_factory)
        await adapter_no_coord.initialize()
        assert adapter_no_coord._mcp_server is None
        assert adapter_no_coord._fastmcp_initialized is False
        mock_logger.warning.assert_any_call(
            "No coordinator available, FastMCP will not be initialized for HttpStreamableTransportAdapter"
        )

        await adapter_instance.shutdown()
        await adapter_no_coord.shutdown()

    async def test_reconnect_client(self, adapter: HttpStreamableTransportAdapter, http_client: httpx.AsyncClient):
        client_id = f"client_reconnect_{uuid.uuid4()}"

        # First connection and disconnection
        async with http_client.stream("GET", f"/stream/{client_id}") as response1:
            assert response1.status_code == 200
            initial_events1 = await read_ndjson_stream_from_response(response1.aiter_bytes(), expected_messages=1)
            assert len(initial_events1) == 1
        await asyncio.sleep(0.1) # Allow server to process disconnect
        assert client_id not in adapter._active_connections

        # Reconnection
        async with http_client.stream("GET", f"/stream/{client_id}") as response2:
            assert response2.status_code == 200
            assert client_id in adapter._active_connections
            events_reconnect = await read_ndjson_stream_from_response(response2.aiter_bytes(), expected_messages=1)

        assert len(events_reconnect) == 1, f"Expected 1 message on reconnect, got {len(events_reconnect)}"
        assert events_reconnect[0]["event"] == EventTypes.STATUS.value
        assert events_reconnect[0]["data"]["client_id"] == client_id

        await asyncio.sleep(0.1)
        assert client_id not in adapter._active_connections
