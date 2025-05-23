import asyncio
import json
import sys
from unittest import mock

import pytest
import pytest_asyncio

from aider_mcp_server.atoms.event_types import EventTypes
from aider_mcp_server.interfaces.transport_adapter import ITransportAdapter
from aider_mcp_server.security import SecurityContext
from aider_mcp_server.stdio_transport_adapter_task7 import StdioTransportAdapter


@pytest.fixture
def mock_request_handler():
    return mock.AsyncMock()


class TestStdioTransportAdapter:
    TRANSPORT_ID = "test_stdio_adapter"

    @pytest_asyncio.fixture(autouse=True)
    async def setup_mocks_and_adapter(self, mock_request_handler, monkeypatch):
        self.mock_stdout = mock.MagicMock(spec=sys.stdout)
        self.mock_stderr = mock.MagicMock(spec=sys.stderr)
        self.mock_stdin_reader = mock.create_autospec(asyncio.StreamReader)

        monkeypatch.setattr(sys, "stdout", self.mock_stdout)
        monkeypatch.setattr(sys, "stderr", self.mock_stderr)

        # Mock connect_read_pipe to do nothing successfully by default
        self.mock_connect_read_pipe = mock.AsyncMock()
        
        # Mock asyncio.StreamReader to return our mock_stdin_reader
        # This is a bit tricky as StreamReader is usually instantiated.
        # We'll mock the loop's connect_read_pipe to set up the reader.
        # The adapter does:
        # reader = asyncio.StreamReader()
        # protocol = asyncio.StreamReaderProtocol(reader)
        # await loop.connect_read_pipe(lambda: protocol, sys.stdin)
        # So, we need to ensure 'reader' becomes self.mock_stdin_reader by patching
        # asyncio.StreamReader() to return self.mock_stdin_reader.

        self.mock_loop = mock.MagicMock(spec=asyncio.AbstractEventLoop)
        self.mock_loop.connect_read_pipe = self.mock_connect_read_pipe

        monkeypatch.setattr(asyncio, "get_running_loop", lambda: self.mock_loop)
        
        # Patch StreamReader constructor
        def new_stream_reader_init(*args, **kwargs):
            # Return our pre-configured mock reader instance
            return self.mock_stdin_reader
        monkeypatch.setattr(asyncio, "StreamReader", new_stream_reader_init)


        self.mock_request_handler = mock_request_handler
        self.adapter = StdioTransportAdapter(
            transport_id=self.TRANSPORT_ID, request_handler=mock_request_handler
        )

        yield

        if self.adapter._running:
            await self.adapter.shutdown()
        # monkeypatch automatically undoes its changes, no need to manually restore asyncio.StreamReader


    def _get_written_json_output(self, mock_stdio_stream):
        """Helper to get all JSON objects written to a mocked stdio stream (stdout/stderr)."""
        outputs = []
        for call_args in mock_stdio_stream.write.call_args_list:
            written_text = call_args[0][0]
            # Assuming each write call is a single JSON line + newline
            if written_text.strip():
                try:
                    outputs.append(json.loads(written_text.strip()))
                except json.JSONDecodeError:
                    outputs.append({"raw_text": written_text.strip()}) # For non-JSON error messages
        return outputs

    async def _feed_stdin_and_wait(self, lines_bytes: list[bytes]):
        """Feeds lines to stdin mock and allows the reader task to process."""
        # Make a copy to avoid modifying the original list if it's reused
        input_queue = list(lines_bytes)

        async def mock_readline_impl():
            if input_queue:
                return input_queue.pop(0)
            return b''  # EOF

        self.mock_stdin_reader.readline.side_effect = mock_readline_impl
        
        # If the adapter's stdin reading task is active, wait for it to complete.
        # It should complete after processing all lines and the EOF.
        if self.adapter._stdin_task and not self.adapter._stdin_task.done():
            try:
                # The task should naturally finish once EOF (b'') is read from mock_readline_impl
                # and _running is set to False by the adapter's _read_stdin loop.
                await asyncio.wait_for(self.adapter._stdin_task, timeout=2.0) # Timeout for safety
            except asyncio.TimeoutError:
                # This is a failure condition, means the _read_stdin loop didn't terminate as expected.
                if not self.adapter._stdin_task.done(): # Check again, could have finished during timeout exception handling
                    self.adapter._stdin_task.cancel() # Attempt to clean up the task
                raise TimeoutError(
                    f"_feed_stdin_and_wait: Timeout waiting for _stdin_task to complete. "
                    f"Adapter running: {self.adapter._running}. "
                    f"Input queue still has {len(input_queue)} items. "
                    f"Stdin task cancelled: {self.adapter._stdin_task.cancelled() if self.adapter._stdin_task else 'N/A'}."
                )
        else:
            # If there's no active task (e.g., start_listening failed, was not called, or task already done),
            # yield control a few times. This might be for tests not expecting a running stdin loop
            # but still using this helper.
            num_yields = len(lines_bytes) + 2 # Yield for each line, EOF, and a bit more margin
            for _ in range(num_yields):
                await asyncio.sleep(0)

    # ITransportAdapter Interface Tests
    def test_itransportadapter_implemented(self):
        assert isinstance(self.adapter, ITransportAdapter)

    def test_get_transport_id(self):
        assert self.adapter.get_transport_id() == self.TRANSPORT_ID

    def test_get_transport_type(self):
        assert self.adapter.get_transport_type() == "stdio"

    def test_get_capabilities(self):
        expected_capabilities = {
            EventTypes.STATUS,
            EventTypes.PROGRESS,
            EventTypes.TOOL_RESULT,
        }
        assert self.adapter.get_capabilities() == expected_capabilities

    def test_should_receive_event(self):
        assert self.adapter.should_receive_event(EventTypes.STATUS, {}) is True

    @pytest.mark.asyncio
    async def test_handle_sse_request_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            await self.adapter.handle_sse_request({})

    @pytest.mark.asyncio
    async def test_handle_message_request_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            await self.adapter.handle_message_request({})

    def test_validate_request_security(self):
        context = self.adapter.validate_request_security({})
        assert isinstance(context, SecurityContext)
        assert context.user_id == "stdio_user"
        assert not context.is_anonymous
        assert context.transport_id == self.TRANSPORT_ID

    # Lifecycle Tests
    @pytest.mark.asyncio
    async def test_initialize(self):
        await self.adapter.initialize()  # Should run without error
        assert not self.adapter._running # Initialize doesn't start listening

    @pytest.mark.asyncio
    async def test_start_listening_creates_task_and_sets_running(self):
        assert not self.adapter._running
        assert self.adapter._stdin_task is None

        await self.adapter.start_listening()

        assert self.adapter._running
        assert self.adapter._stdin_task is not None
        assert not self.adapter._stdin_task.done()
        self.mock_connect_read_pipe.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_listening_idempotent(self):
        await self.adapter.start_listening()
        first_task = self.adapter._stdin_task
        self.mock_connect_read_pipe.assert_called_once()

        await self.adapter.start_listening()  # Call again
        assert self.adapter._stdin_task is first_task  # Task should be the same
        self.mock_connect_read_pipe.assert_called_once() # Should not be called again

    @pytest.mark.asyncio
    async def test_shutdown_cancels_task_and_resets_state(self):
        await self.adapter.start_listening()
        stdin_task = self.adapter._stdin_task
        assert stdin_task is not None and not stdin_task.done()

        await self.adapter.shutdown()

        assert not self.adapter._running
        assert stdin_task.cancelled() or stdin_task.done() # Task should be cancelled or completed
        assert self.adapter._stdin_task is None # Reset after awaiting

    @pytest.mark.asyncio
    async def test_shutdown_idempotent(self):
        await self.adapter.start_listening()
        await self.adapter.shutdown()
        # Call again
        await self.adapter.shutdown()
        assert not self.adapter._running
        assert self.adapter._stdin_task is None

    @pytest.mark.asyncio
    async def test_shutdown_before_start(self):
        await self.adapter.shutdown() # Should not raise error
        assert not self.adapter._running
        assert self.adapter._stdin_task is None

    # Event Sending Tests
    @pytest.mark.asyncio
    async def test_send_event_writes_json_to_stdout(self):
        event_data = {"key": "value", "num": 123}
        await self.adapter.send_event(EventTypes.PROGRESS, event_data)

        expected_json_str = json.dumps({"type": EventTypes.PROGRESS.value, **event_data})
        self.mock_stdout.write.assert_called_once_with(expected_json_str + "\n")
        self.mock_stdout.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_event_merges_data_correctly(self):
        # If data contains 'type', it should be part of the data payload,
        # the main 'type' is from event_type.value
        event_data = {"type": "custom_type_in_data", "message": "hello"}
        await self.adapter.send_event(EventTypes.STATUS, event_data)
        
        # The adapter's send_event structure is:
        # event_payload = {"type": event_type.value}
        # event_payload.update(data) -> this means data["type"] overwrites event_type.value
        # This is consistent with Task 7's StdioTransportAdapter.send_event
        expected_payload = {"type": "custom_type_in_data", "message": "hello"}
        
        expected_json_str = json.dumps(expected_payload)
        self.mock_stdout.write.assert_called_once_with(expected_json_str + "\n")

    @pytest.mark.asyncio
    async def test_send_event_error_during_json_dumps(self, monkeypatch):
        # This is hard to trigger with dicts, but simulate non-serializable data
        event_data = {"key": object()} # object() is not JSON serializable
        
        # Mock json.dumps to raise TypeError
        mock_json_dumps = mock.MagicMock(side_effect=TypeError("Test TypeError"))
        monkeypatch.setattr(json, "dumps", mock_json_dumps)

        await self.adapter.send_event(EventTypes.STATUS, event_data)

        self.mock_stdout.write.assert_not_called() # Stdout write should not happen
        self.mock_stderr.write.assert_called_once()
        # Check that the error message contains the exception string
        error_call_arg = self.mock_stderr.write.call_args[0][0]
        assert "Error sending event to stdout: Test TypeError" in error_call_arg
        self.mock_stderr.flush.assert_called_once()

    # Stdin Reading and Processing Tests
    @pytest.mark.asyncio
    async def test_read_stdin_valid_json_no_handler(self):
        self.adapter._request_handler = None # Ensure no handler
        await self.adapter.start_listening()

        input_json_str = '{"request": "test"}'
        await self._feed_stdin_and_wait([input_json_str.encode('utf-8') + b'\n', b'']) # EOF

        self.mock_request_handler.assert_not_called()
        # No output expected if no handler and no errors
        self.mock_stdout.write.assert_not_called()

    @pytest.mark.asyncio
    async def test_read_stdin_valid_json_with_handler_responds(self):
        await self.adapter.start_listening()
        
        input_message = {"request": "echo", "data": "hello"}
        handler_response = {"type": EventTypes.STATUS.value, "response_data": "world"}
        self.mock_request_handler.return_value = handler_response.copy() # Return a copy

        await self._feed_stdin_and_wait([json.dumps(input_message).encode('utf-8') + b'\n', b''])

        self.mock_request_handler.assert_called_once_with(input_message)
        
        # Check that send_event was called with handler's response (minus 'type')
        # The adapter's _read_stdin extracts 'type' and passes the rest as data
        expected_sent_data = {"response_data": "world"}
        expected_json_output = json.dumps({"type": EventTypes.STATUS.value, **expected_sent_data})
        
        self.mock_stdout.write.assert_called_once_with(expected_json_output + "\n")

    @pytest.mark.asyncio
    async def test_read_stdin_handler_returns_none(self):
        await self.adapter.start_listening()
        self.mock_request_handler.return_value = None
        input_message = {"request": "no_reply"}

        await self._feed_stdin_and_wait([json.dumps(input_message).encode('utf-8') + b'\n', b''])

        self.mock_request_handler.assert_called_once_with(input_message)
        self.mock_stdout.write.assert_not_called() # No response sent

    @pytest.mark.asyncio
    async def test_read_stdin_handler_returns_invalid_event_type(self):
        await self.adapter.start_listening()
        input_message = {"request": "bad_type_response"}
        # Handler returns a dict with a 'type' that is not a valid EventTypes member
        handler_response_payload = {"some_data": "payload"}
        self.mock_request_handler.return_value = {"type": "INVALID_EVENT_TYPE", **handler_response_payload}

        await self._feed_stdin_and_wait([json.dumps(input_message).encode('utf-8') + b'\n', b''])

        self.mock_request_handler.assert_called_once_with(input_message)
        
        # Expect an error event to be sent via stdout
        written_jsons = self._get_written_json_output(self.mock_stdout)
        assert len(written_jsons) == 1
        error_event = written_jsons[0]
        assert error_event["type"] == EventTypes.STATUS.value
        assert error_event["level"] == "error"
        assert "Handler returned unknown event type: INVALID_EVENT_TYPE" in error_event["message"]
        assert error_event["original_response"] == handler_response_payload

    @pytest.mark.asyncio
    async def test_read_stdin_handler_returns_non_dict_response(self):
        await self.adapter.start_listening()
        input_message = {"request": "non_dict_response"}
        self.mock_request_handler.return_value = "This is not a dict" # Invalid response

        await self._feed_stdin_and_wait([json.dumps(input_message).encode('utf-8') + b'\n', b''])

        self.mock_request_handler.assert_called_once_with(input_message)
        
        written_jsons = self._get_written_json_output(self.mock_stdout)
        assert len(written_jsons) == 1
        error_event = written_jsons[0]
        assert error_event["type"] == EventTypes.STATUS.value
        assert error_event["level"] == "error"
        assert "Handler returned invalid or non-dictionary response format" in error_event["message"]
        assert error_event["original_response"] == "This is not a dict"

    @pytest.mark.asyncio
    async def test_read_stdin_handler_raises_exception(self):
        await self.adapter.start_listening()
        self.mock_request_handler.side_effect = Exception("Handler error!")
        input_message = {"request": "trigger_error"}

        await self._feed_stdin_and_wait([json.dumps(input_message).encode('utf-8') + b'\n', b''])

        self.mock_request_handler.assert_called_once_with(input_message)
        
        # Expect an error event to be sent via stdout
        # This is because the outer exception handler in _read_stdin catches it.
        written_jsons = self._get_written_json_output(self.mock_stdout)
        assert len(written_jsons) == 1
        error_event = written_jsons[0]
        assert error_event["type"] == EventTypes.STATUS.value
        assert error_event["level"] == "error"
        assert "Error processing stdin: Handler error!" in error_event["message"]

    @pytest.mark.asyncio
    async def test_read_stdin_invalid_json_input(self):
        await self.adapter.start_listening()
        invalid_json_str = '{"request": "test",,}' # Invalid JSON

        await self._feed_stdin_and_wait([invalid_json_str.encode('utf-8') + b'\n', b''])

        self.mock_request_handler.assert_not_called()
        
        written_jsons = self._get_written_json_output(self.mock_stdout)
        assert len(written_jsons) == 1
        error_event = written_jsons[0]
        assert error_event["type"] == EventTypes.STATUS.value
        assert error_event["level"] == "error"
        assert error_event["message"] == "Invalid JSON message received on stdin"

    @pytest.mark.asyncio
    async def test_read_stdin_empty_line_is_skipped(self):
        await self.adapter.start_listening()
        input_message = {"request": "data"}
        
        lines = [
            b'\n', # Empty line
            json.dumps(input_message).encode('utf-8') + b'\n',
            b'\n', # Another empty line
            b'' # EOF
        ]
        self.mock_request_handler.return_value = None # Keep it simple
        await self._feed_stdin_and_wait(lines)

        self.mock_request_handler.assert_called_once_with(input_message)
        self.mock_stdout.write.assert_not_called() # No output if handler returns None

    @pytest.mark.asyncio
    async def test_read_stdin_eof_stops_loop(self):
        await self.adapter.start_listening()
        stdin_task = self.adapter._stdin_task
        assert stdin_task is not None, "Stdin task should exist after start_listening"
        assert not stdin_task.done(), "Stdin task should be running initially"

        await self._feed_stdin_and_wait([b'']) # EOF immediately

        # _feed_stdin_and_wait now waits for the task to complete or times out.
        # So, no additional sleep should be needed here.
        assert stdin_task.done(), "_stdin_task should be done after processing EOF"
        assert not self.adapter._running, "_running should be false after loop exits due to EOF"

    @pytest.mark.asyncio
    async def test_connect_read_pipe_error_stops_adapter(self):
        # Configure connect_read_pipe mock to raise an error
        self.mock_connect_read_pipe.side_effect = OSError("Failed to connect pipe")
        
        await self.adapter.start_listening() # This will call _read_stdin which calls connect_read_pipe

        assert not self.adapter._running # Should be set to False due to error
        assert self.adapter._stdin_task is not None # Task is created
        
        # Wait for the task to complete (it should exit quickly due to the error)
        await asyncio.wait_for(self.adapter._stdin_task, timeout=1.0)
        assert self.adapter._stdin_task.done()

        # Check stderr for the fatal error message
        self.mock_stderr.write.assert_any_call(
            "Fatal: Error connecting stdin reader: Failed to connect pipe. Stdio adapter will not run.\n"
        )
        self.mock_stderr.flush.assert_called()
        self.mock_request_handler.assert_not_called()
        self.mock_stdout.write.assert_not_called()

    @pytest.mark.asyncio
    async def test_multiple_lines_processed_sequentially(self):
        await self.adapter.start_listening()

        msg1 = {"id": 1, "data": "first"}
        msg2 = {"id": 2, "data": "second"}
        
        handler_resp1 = {"type": EventTypes.STATUS.value, "ack": 1}
        handler_resp2 = {"type": EventTypes.PROGRESS.value, "ack": 2}

        # Configure handler to respond differently based on input
        async def side_effect_handler(message):
            if message["id"] == 1:
                return handler_resp1.copy()
            if message["id"] == 2:
                return handler_resp2.copy()
            return None
        self.mock_request_handler.side_effect = side_effect_handler

        lines = [
            json.dumps(msg1).encode('utf-8') + b'\n',
            json.dumps(msg2).encode('utf-8') + b'\n',
            b'' # EOF
        ]
        await self._feed_stdin_and_wait(lines)

        assert self.mock_request_handler.call_count == 2
        self.mock_request_handler.assert_any_call(msg1)
        self.mock_request_handler.assert_any_call(msg2)

        written_jsons = self._get_written_json_output(self.mock_stdout)
        assert len(written_jsons) == 2
        
        # Check responses (order might matter if stdout is strictly ordered)
        # The adapter processes and sends immediately.
        expected_output1_data = {"ack": 1}
        expected_output1 = {"type": EventTypes.STATUS.value, **expected_output1_data}
        
        expected_output2_data = {"ack": 2}
        expected_output2 = {"type": EventTypes.PROGRESS.value, **expected_output2_data}

        assert written_jsons[0] == expected_output1
        assert written_jsons[1] == expected_output2
