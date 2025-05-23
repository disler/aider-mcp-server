import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aider_mcp_server.request_processor import RequestProcessor


@pytest.fixture
def mock_logger():
    """Fixture for a mock logger."""
    logger = MagicMock()
    logger.debug = MagicMock()
    logger.info = MagicMock()
    logger.warning = MagicMock()
    logger.error = MagicMock()
    return logger


@pytest.fixture
@patch("aider_mcp_server.request_processor.get_logger")
def request_processor(mock_get_logger, mock_logger):
    """Fixture for a RequestProcessor instance with a mocked logger."""
    mock_get_logger.return_value = mock_logger
    processor = RequestProcessor()
    return processor


class TestRequestProcessorBasics:
    @pytest.mark.asyncio
    async def test_initialization(self, request_processor: RequestProcessor, mock_logger: MagicMock):
        assert isinstance(request_processor._handlers, dict)
        assert len(request_processor._handlers) == 0
        assert isinstance(request_processor._active_requests, dict)
        assert len(request_processor._active_requests) == 0
        assert isinstance(request_processor._lock, asyncio.Lock)
        assert request_processor._logger is mock_logger

    @pytest.mark.asyncio
    async def test_register_handler(self, request_processor: RequestProcessor, mock_logger: MagicMock):
        handler_mock = AsyncMock()
        request_type = "test_request"
        request_processor.register_handler(request_type, handler_mock)
        assert request_type in request_processor._handlers
        assert request_processor._handlers[request_type] is handler_mock
        mock_logger.debug.assert_called_with(f"Registered handler for request type: {request_type}")

    @pytest.mark.asyncio
    async def test_register_duplicate_handler_overwrites(
        self, request_processor: RequestProcessor, mock_logger: MagicMock
    ):
        request_type = "test_request"
        handler_mock1 = AsyncMock(name="handler1")
        handler_mock2 = AsyncMock(name="handler2")

        request_processor.register_handler(request_type, handler_mock1)
        assert request_processor._handlers[request_type] is handler_mock1
        mock_logger.debug.assert_called_with(f"Registered handler for request type: {request_type}")

        request_processor.register_handler(request_type, handler_mock2)
        assert request_processor._handlers[request_type] is handler_mock2  # Should be overwritten
        mock_logger.debug.assert_called_with(f"Registered handler for request type: {request_type}")  # Called again

    @pytest.mark.asyncio
    async def test_unregister_handler(self, request_processor: RequestProcessor, mock_logger: MagicMock):
        handler_mock = AsyncMock()
        request_type = "test_request_unreg"
        request_processor.register_handler(request_type, handler_mock)
        assert request_type in request_processor._handlers

        request_processor.unregister_handler(request_type)
        assert request_type not in request_processor._handlers
        mock_logger.debug.assert_called_with(f"Unregistered handler for request type: {request_type}")

    @pytest.mark.asyncio
    async def test_unregister_non_existent_handler(self, request_processor: RequestProcessor, mock_logger: MagicMock):
        request_type = "non_existent_request"
        request_processor.unregister_handler(request_type)
        # Should not raise an error, but log a warning
        mock_logger.warning.assert_called_with(f"Attempted to unregister non-existent handler for type: {request_type}")


class TestRequestProcessorValidation:
    @pytest.mark.asyncio
    async def test_process_request_missing_type(self, request_processor: RequestProcessor, mock_logger: MagicMock):
        request_data = {"id": "123"}
        response = await request_processor.process_request(request_data)

        assert not response["success"]
        assert response["error"] == "Missing request type"
        mock_logger.error.assert_called_with("Request processing failed: Missing 'type' field.")

    @pytest.mark.asyncio
    async def test_process_request_unknown_type(self, request_processor: RequestProcessor, mock_logger: MagicMock):
        request_data = {"id": "123", "type": "unknown_type"}
        response = await request_processor.process_request(request_data)

        assert not response["success"]
        assert response["error"] == "Unknown request type: unknown_type"
        mock_logger.error.assert_called_with("Request processing failed: Unknown request type 'unknown_type'.")


class TestRequestProcessorRoutingAndExecution:
    @pytest.mark.asyncio
    async def test_process_request_successful(self, request_processor: RequestProcessor, mock_logger: MagicMock):
        request_type = "echo_request"

        async def mock_handler(req_data):
            return {"success": True, "data": req_data.get("data", {})}

        handler_mock = AsyncMock(side_effect=mock_handler)
        request_processor.register_handler(request_type, handler_mock)

        request_data = {"id": "req1", "type": request_type, "data": {"input": "hello"}}
        response = await request_processor.process_request(request_data)

        handler_mock.assert_awaited_once_with(request_data)  # request_data now includes 'id'
        assert response["success"]
        assert response["id"] == "req1"
        assert response["data"] == {"input": "hello"}
        mock_logger.info.assert_any_call(f"Started processing request_id: req1, type: {request_type}")
        mock_logger.info.assert_any_call(f"Finished processing request_id: req1, type: {request_type}. Success: True")

    @pytest.mark.asyncio
    @patch("aider_mcp_server.request_processor.uuid.uuid4")
    async def test_process_request_generates_id_if_missing(
        self, mock_uuid4: MagicMock, request_processor: RequestProcessor, mock_logger: MagicMock
    ):
        mock_uuid4.return_value = uuid.UUID("12345678-1234-5678-1234-567812345678")
        generated_id = "12345678-1234-5678-1234-567812345678"
        request_type = "test_gen_id"

        async def mock_handler(req_data):
            return {"success": True, "message": "handled"}

        handler_mock = AsyncMock(side_effect=mock_handler)
        request_processor.register_handler(request_type, handler_mock)

        request_data = {"type": request_type, "data": "some_data"}
        response = await request_processor.process_request(request_data)

        expected_request_data_with_id = {"type": request_type, "data": "some_data", "id": generated_id}
        handler_mock.assert_awaited_once_with(expected_request_data_with_id)
        assert response["id"] == generated_id
        assert response["success"]
        # Use assert_any_call as other debug logs might exist
        mock_logger.debug.assert_any_call(f"Generated new request_id: {generated_id} for request type {request_type}")


class TestRequestProcessorLifecycle:
    @pytest.mark.asyncio
    async def test_process_request_tracks_and_cleans_active_request(self, request_processor: RequestProcessor):
        request_type = "lifecycle_test"
        request_id = "lc1"

        # Handler that allows us to inspect state mid-execution
        handler_execution_started = asyncio.Event()
        handler_can_finish = asyncio.Event()

        async def slow_handler(req_data):
            handler_execution_started.set()
            await handler_can_finish.wait()
            return {"success": True, "message": "done"}

        handler_mock = AsyncMock(side_effect=slow_handler)
        request_processor.register_handler(request_type, handler_mock)

        request_data = {"id": request_id, "type": request_type}

        # Start processing in background
        process_task = asyncio.create_task(request_processor.process_request(request_data))

        await handler_execution_started.wait()
        assert request_id in request_processor._active_requests
        assert isinstance(request_processor._active_requests[request_id], asyncio.Task)

        handler_can_finish.set()
        response = await process_task  # Wait for completion

        assert response["success"]
        assert request_id not in request_processor._active_requests  # Should be cleaned up

    @pytest.mark.asyncio
    async def test_cleanup_after_handler_failure(self, request_processor: RequestProcessor):
        request_type = "fail_lifecycle"
        request_id = "flc1"

        async def failing_handler(req_data):
            raise ValueError("Handler error")

        handler_mock = AsyncMock(side_effect=failing_handler)
        request_processor.register_handler(request_type, handler_mock)
        request_data = {"id": request_id, "type": request_type}

        response = await request_processor.process_request(request_data)

        assert not response["success"]
        assert "Error processing request: Handler error" in response["error"]
        assert request_id not in request_processor._active_requests

    @pytest.mark.asyncio
    async def test_cleanup_after_invalid_handler_response(self, request_processor: RequestProcessor):
        request_type = "invalid_resp_lifecycle"
        request_id = "irlc1"

        async def invalid_response_handler(req_data):
            return "not a dict"  # Invalid response

        handler_mock = AsyncMock(side_effect=invalid_response_handler)
        request_processor.register_handler(request_type, handler_mock)
        request_data = {"id": request_id, "type": request_type}

        response = await request_processor.process_request(request_data)

        assert not response["success"]
        assert response["error"] == "Handler returned invalid response"
        assert request_id not in request_processor._active_requests


class TestRequestProcessorErrorHandling:
    @pytest.mark.asyncio
    async def test_process_request_handler_exception(self, request_processor: RequestProcessor, mock_logger: MagicMock):
        request_type = "error_handler"
        request_id = "err1"
        error_message = "Simulated handler error"

        async def failing_handler(req_data):
            raise RuntimeError(error_message)

        handler_mock = AsyncMock(side_effect=failing_handler)
        request_processor.register_handler(request_type, handler_mock)

        request_data = {"id": request_id, "type": request_type}
        response = await request_processor.process_request(request_data)

        assert not response["success"]
        # The _error_response used in this path does not include 'id'
        assert f"Error processing request: {error_message}" in response["error"]
        mock_logger.error.assert_called_with(
            f"Error processing request_id {request_id} (type: {request_type}): {error_message}", exc_info=True
        )

    @pytest.mark.asyncio
    async def test_process_request_handler_returns_invalid_response(
        self, request_processor: RequestProcessor, mock_logger: MagicMock
    ):
        request_type = "invalid_response_handler"
        request_id = "irh1"

        async def bad_response_handler(req_data):
            return "This is not a dictionary"

        handler_mock = AsyncMock(side_effect=bad_response_handler)
        request_processor.register_handler(request_type, handler_mock)

        request_data = {"id": request_id, "type": request_type}
        response = await request_processor.process_request(request_data)

        assert not response["success"]
        # The _error_response used in this path does not include 'id'
        assert response["error"] == "Handler returned invalid response"
        mock_logger.error.assert_called_with(
            f"Handler for request_id {request_id} (type: {request_type}) "
            f"returned invalid response type: <class 'str'>. Expected dict."
        )

    def test_error_response_format(self, request_processor: RequestProcessor):
        error_message = "A test error occurred."
        response = request_processor._error_response(error_message)
        assert response == {"success": False, "error": error_message}


class TestRequestProcessorCancellation:
    @pytest.mark.asyncio
    async def test_cancel_request_successful(self, request_processor: RequestProcessor, mock_logger: MagicMock):
        request_type = "cancellable_request"
        request_id = "cancel1"

        handler_started = asyncio.Event()
        task_cancelled_event = asyncio.Event()

        async def long_running_handler(req_data):
            handler_started.set()
            try:
                await asyncio.sleep(5)  # Simulate long work
                return {"success": True, "message": "completed normally"}
            except asyncio.CancelledError:
                task_cancelled_event.set()
                raise  # Important to re-raise

        handler_mock = AsyncMock(side_effect=long_running_handler)
        request_processor.register_handler(request_type, handler_mock)
        request_data = {"id": request_id, "type": request_type}

        # Start the request processing in the background
        process_task = asyncio.create_task(request_processor.process_request(request_data))

        await handler_started.wait()  # Ensure handler has started and task is in _active_requests
        assert request_id in request_processor._active_requests

        cancel_result = await request_processor.cancel_request(request_id)
        assert cancel_result is True

        # Wait for the process_request task to finish and get its response
        response = await process_task

        assert await asyncio.wait_for(task_cancelled_event.wait(), timeout=1)  # Handler saw cancellation
        assert request_id not in request_processor._active_requests

        assert not response["success"]
        assert f"Request {request_id} was cancelled" in response["error"]

        mock_logger.info.assert_any_call(f"Attempting to cancel request_id: {request_id}")
        mock_logger.info.assert_any_call(f"Cancelled and removed request_id: {request_id} from active_requests.")
        mock_logger.warning.assert_any_call(f"Request_id: {request_id}, type: {request_type} was cancelled.")

    @pytest.mark.asyncio
    async def test_cancel_request_non_existent(self, request_processor: RequestProcessor, mock_logger: MagicMock):
        request_id = "non_existent_cancel"
        cancel_result = await request_processor.cancel_request(request_id)
        assert cancel_result is False
        mock_logger.warning.assert_called_with(
            f"Attempted to cancel non-existent or already completed request_id: {request_id}"
        )

    @pytest.mark.asyncio
    async def test_cancel_request_already_completed(self, request_processor: RequestProcessor, mock_logger: MagicMock):
        request_type = "completed_request_cancel"
        request_id = "completed1"

        async def simple_handler(req_data):
            return {"success": True, "message": "done"}

        handler_mock = AsyncMock(side_effect=simple_handler)
        request_processor.register_handler(request_type, handler_mock)
        request_data = {"id": request_id, "type": request_type}

        await request_processor.process_request(request_data)  # Process and complete
        assert request_id not in request_processor._active_requests

        cancel_result = await request_processor.cancel_request(request_id)
        assert cancel_result is False
        mock_logger.warning.assert_called_with(
            f"Attempted to cancel non-existent or already completed request_id: {request_id}"
        )


class TestRequestProcessorConcurrency:
    @pytest.mark.asyncio
    async def test_concurrent_process_requests(self, request_processor: RequestProcessor):
        request_type = "concurrent_req"
        num_requests = 10

        async def concurrent_handler(req_data):
            await asyncio.sleep(0.01)  # Simulate some async work
            return {"success": True, "data": req_data["data"]}

        handler_mock = AsyncMock(side_effect=concurrent_handler)
        request_processor.register_handler(request_type, handler_mock)

        tasks = []
        for i in range(num_requests):
            req_data = {"type": request_type, "data": {"index": i}, "id": f"conc_req_{i}"}
            tasks.append(request_processor.process_request(req_data))

        responses = await asyncio.gather(*tasks)

        assert len(responses) == num_requests
        assert handler_mock.call_count == num_requests
        for i, response in enumerate(responses):
            assert response["success"]
            assert response["data"]["index"] == i  # Assuming order is preserved by gather or checking IDs
            assert response["id"] == f"conc_req_{i}"

        assert len(request_processor._active_requests) == 0  # All should be cleaned up

    @pytest.mark.asyncio
    async def test_concurrent_process_and_cancel(self, request_processor: RequestProcessor):
        """
        Tests concurrent request processing, some of which are cancelled,
        using events for deterministic control.
        """
        NUM_REQUESTS = 5
        REQUEST_IDS_TO_CANCEL = {f"req_{i}" for i in range(0, NUM_REQUESTS, 2)}  # e.g., req_0, req_2, req_4
        REQUEST_TYPE = "concurrent_controlled_task"

        handler_controls = {}  # req_id -> (started_event, finish_event)

        async def controllable_side_effect(req_data):
            req_id = req_data["id"]
            if req_id not in handler_controls:
                # Should not happen in a correctly set up test
                return {"success": False, "error": f"Test setup error: {req_id} not in handler_controls", "id": req_id}

            started_event, finish_event = handler_controls[req_id]
            started_event.set()
            try:
                # Wait for the finish_event, with a timeout to prevent test hangs
                await asyncio.wait_for(finish_event.wait(), timeout=2.0)
                return {"success": True, "message": "controlled_finish", "id": req_id}
            except asyncio.CancelledError:
                # This block is reached if the task wrapping this handler is cancelled
                request_processor._logger.debug(f"Handler for {req_id} was internally cancelled.")
                raise  # Re-raise for RequestProcessor to catch and format its standard cancellation response
            except asyncio.TimeoutError:
                # This block means finish_event was not set for a non-cancelled task
                request_processor._logger.error(f"Handler for {req_id} timed out waiting for finish_event.")
                return {"success": False, "error": f"Handler for {req_id} timed out", "id": req_id}

        request_processor.register_handler(REQUEST_TYPE, AsyncMock(side_effect=controllable_side_effect))

        process_tasks = []
        request_details = []

        for i in range(NUM_REQUESTS):
            req_id = f"req_{i}"
            started_event = asyncio.Event()
            finish_event = asyncio.Event()
            handler_controls[req_id] = (started_event, finish_event)

            request_data = {"id": req_id, "type": REQUEST_TYPE, "payload": f"data_{i}"}
            request_details.append(request_data)
            process_tasks.append(asyncio.create_task(request_processor.process_request(request_data)))

        # Wait for all handlers to have started execution
        await asyncio.gather(*(handler_controls[f"req_{i}"][0].wait() for i in range(NUM_REQUESTS)))
        request_processor._logger.debug("All handlers started.")

        # Attempt to cancel specified requests
        cancel_call_results = {}
        for req_id in REQUEST_IDS_TO_CANCEL:
            request_processor._logger.debug(f"Test attempting to cancel {req_id}.")
            cancel_call_results[req_id] = await request_processor.cancel_request(req_id)

        # Allow non-cancelled requests to complete
        for i in range(NUM_REQUESTS):
            req_id = f"req_{i}"
            if req_id not in REQUEST_IDS_TO_CANCEL:
                request_processor._logger.debug(f"Test allowing {req_id} to finish.")
                _, finish_event = handler_controls[req_id]
                finish_event.set()
            else:
                # For cancelled requests, ensure their finish_event is NOT set by the test logic itself,
                # so the handler remains in `await finish_event.wait()` when cancellation hits.
                pass

        # Gather all responses
        # process_request catches CancelledError and returns a dict, so no exceptions here from gather
        responses = await asyncio.gather(*process_tasks)

        # Assertions
        assert len(responses) == NUM_REQUESTS

        num_actually_cancelled = 0
        num_successfully_completed = 0

        for i, response in enumerate(responses):
            req_id = request_details[i]["id"]

            if req_id in REQUEST_IDS_TO_CANCEL:
                assert cancel_call_results[req_id] is True, f"cancel_request for {req_id} should have returned True"
                assert not response.get("success"), f"Response for cancelled {req_id} should indicate failure"
                assert f"Request {req_id} was cancelled" in response.get("error", ""), (
                    f"Error message for {req_id} incorrect"
                )
                # As per current RequestProcessor, ID is not in error response from _error_response
                assert "id" not in response or response.get("id") is None, (
                    f"ID should not be in cancellation error response for {req_id}"
                )
                num_actually_cancelled += 1
            else:
                assert response.get("success"), (
                    f"Response for non-cancelled {req_id} should indicate success. Got: {response}"
                )
                assert response.get("message") == "controlled_finish", f"Message for {req_id} incorrect"
                assert response.get("id") == req_id, f"ID mismatch for successful {req_id}"
                num_successfully_completed += 1

        assert num_actually_cancelled == len(REQUEST_IDS_TO_CANCEL)
        assert num_successfully_completed == NUM_REQUESTS - len(REQUEST_IDS_TO_CANCEL)
        assert len(request_processor._active_requests) == 0, "All requests should be cleaned up from _active_requests"
