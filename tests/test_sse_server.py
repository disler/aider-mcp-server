import asyncio
import json
import signal
import uuid
from pathlib import Path # Import Path
from unittest.mock import AsyncMock, MagicMock, patch, call, ANY

import pytest
import pytest_asyncio # Import pytest_asyncio
import uvicorn # Import uvicorn
# Use absolute imports from the package root
from starlette.datastructures import Headers
from starlette.requests import Request
from starlette.responses import JSONResponse, Response # Import Response
from starlette.routing import Route # Import Route for assertion checks
from sse_starlette.sse import EventSourceResponse

from aider_mcp_server.atoms.event_types import EventTypes
from aider_mcp_server.security import SecurityContext, Permissions, create_context_from_credentials
from aider_mcp_server.sse_transport_adapter import SSETransportAdapter
from aider_mcp_server.transport_coordinator import ApplicationCoordinator
# Import the specific functions/classes being tested or mocked
# Import the module itself to allow patching its global variable
import aider_mcp_server.sse_server as sse_server_module
from aider_mcp_server.sse_server import serve_sse, handle_shutdown_signal, _create_shutdown_task_wrapper, is_git_repository # Import is_git_repository
# Import Logger for spec - use LoggerProtocol as defined elsewhere if Logger is concrete
# Assuming LoggerProtocol is the intended type hint standard
from aider_mcp_server.transport_adapter import LoggerProtocol


# --- Fixtures ---

@pytest.fixture
def mock_coordinator():
    """Fixture for a mocked ApplicationCoordinator instance returned by getInstance."""
    # Create an instance mock *with* spec to ensure methods exist
    coordinator_instance = MagicMock(spec=ApplicationCoordinator)

    # Add sync methods needed
    coordinator_instance.register_transport = MagicMock()
    coordinator_instance.unregister_transport = MagicMock()
    # coordinator_instance.register_handler = MagicMock() # Incorrect: register_handler is async
    coordinator_instance.subscribe_to_event_type = MagicMock()
    coordinator_instance.is_shutting_down = MagicMock(return_value=False) # Needed by endpoints

    # Add async methods needed (use AsyncMock for awaitables)
    coordinator_instance.start_request = AsyncMock()
    coordinator_instance.fail_request = AsyncMock()
    coordinator_instance.shutdown = AsyncMock()
    coordinator_instance.wait_for_initialization = AsyncMock() # Needed by start_request etc.
    coordinator_instance.register_handler = AsyncMock() # Correct: register_handler is async

    # Add async context manager methods explicitly
    coordinator_instance.__aenter__ = AsyncMock(return_value=coordinator_instance)
    coordinator_instance.__aexit__ = AsyncMock()

    # Patch the class's getInstance method where it's used (sse_server)
    # Use new_callable=AsyncMock because getInstance is async
    # Patch in 'sse_server' as that's where serve_sse calls it
    patcher_sse_server = patch(
        'aider_mcp_server.sse_server.ApplicationCoordinator.getInstance',
        new_callable=AsyncMock,
        return_value=coordinator_instance
    )
    # REMOVED: Patch in 'sse_transport_adapter' as the adapter does NOT call getInstance
    # patcher_adapter = patch(
    #     'aider_mcp_server.sse_transport_adapter.ApplicationCoordinator.getInstance',
    #     new_callable=AsyncMock,
    #     return_value=coordinator_instance
    # )

    # Start the patcher(s)
    mock_get_instance_sse = patcher_sse_server.start()
    # mock_get_instance_adapter = patcher_adapter.start() # Removed

    yield coordinator_instance # Yield the configured instance mock

    # Stop the patcher(s)
    patcher_sse_server.stop()
    # patcher_adapter.stop() # Removed


@pytest.fixture
def mock_security_context():
    """Fixture for a basic SecurityContext."""
    return SecurityContext(user_id="test_user", permissions={Permissions.EXECUTE_AIDER})

@pytest_asyncio.fixture # Use pytest_asyncio.fixture for async fixtures
async def sse_adapter(mock_coordinator):
    """
    Fixture for an SSETransportAdapter instance with a mocked coordinator.
    This fixture now initializes the adapter asynchronously.
    """
    adapter = None
    # Patch the logger within the adapter's scope
    with patch('aider_mcp_server.transport_adapter.get_logger_func') as mock_get_logger:
        # Use LoggerProtocol for spec matching the type hints
        mock_logger = MagicMock(spec=LoggerProtocol)
        # Ensure log methods are MagicMock, not AsyncMock, as they are sync
        mock_logger.info = MagicMock()
        mock_logger.debug = MagicMock()
        mock_logger.warning = MagicMock()
        mock_logger.error = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Create adapter instance - pass the *mocked* coordinator instance
        adapter = SSETransportAdapter(coordinator=mock_coordinator, heartbeat_interval=0.1)
        adapter.logger = mock_logger # Ensure mock logger is attached
        # Assign coordinator explicitly for tests that might check adapter._coordinator
        adapter._coordinator = mock_coordinator

        # Initialize the adapter (this calls coordinator.register_transport)
        await adapter.initialize()

        yield adapter # Provide the initialized adapter to the test

        # Cleanup: Shutdown the adapter after the test runs
        if adapter:
            # Ensure shutdown is awaited
            await adapter.shutdown()


@pytest.fixture
def mock_request():
    """Fixture for a mocked Starlette Request."""
    request = MagicMock(spec=Request)
    request.client = MagicMock()
    request.client.host = "127.0.0.1"
    request.client.port = 5000
    request.headers = Headers({})
    # Mock json() as an async method
    request.json = AsyncMock(return_value={}) # Default mock
    return request

@pytest.fixture
def mock_uvicorn_server():
    """Fixture for a mocked Uvicorn Server instance."""
    # This fixture provides a basic mock, but test_serve_sse_startup_and_run
    # now creates its own instance mock via the patched class.
    server = MagicMock()
    server.serve = AsyncMock()
    server.shutdown = AsyncMock()
    server.started = True
    return server

@pytest.fixture
def mock_uvicorn_config():
    """Fixture for a mocked Uvicorn Config."""
    return MagicMock(spec=uvicorn.Config)

# --- Test SSETransportAdapter ---
# These tests focus on the adapter logic itself, using the mocked coordinator

@pytest.mark.asyncio
async def test_sse_adapter_init_and_initialize(sse_adapter, mock_coordinator):
    """Test SSETransportAdapter initialization and async initialize via fixture."""
    # The sse_adapter fixture now handles initialization
    assert sse_adapter._coordinator == mock_coordinator
    assert sse_adapter.transport_type == "sse"
    assert isinstance(sse_adapter.transport_id, str) and sse_adapter.transport_id.startswith("sse_")
    assert sse_adapter.logger is not None # Logger is set by fixture

    # Check that initialize was called by the fixture (mock_coordinator interactions)
    # register_transport is sync
    mock_coordinator.register_transport.assert_called_once_with(sse_adapter.transport_id, sse_adapter)

    # Check logging during init and initialize (logger methods are sync)
    sse_adapter.logger.info.assert_any_call(f"SSETransportAdapter created with ID: {sse_adapter.transport_id}. Max queue size: 100") # Default size
    sse_adapter.logger.info.assert_any_call(f"Initializing SSE transport adapter {sse_adapter.transport_id}...")
    sse_adapter.logger.info.assert_any_call(f"SSE transport adapter {sse_adapter.transport_id} initialized.")


@pytest.mark.asyncio
async def test_sse_adapter_shutdown_real(mock_coordinator):
    """Test the real shutdown method of SSETransportAdapter (calls coordinator)."""
    # Create adapter manually for this test to control init/shutdown explicitly
    adapter = None
    with patch('aider_mcp_server.transport_adapter.get_logger_func') as mock_get_logger:
        mock_logger = MagicMock(spec=LoggerProtocol)
        mock_logger.info = MagicMock()
        mock_logger.debug = MagicMock()
        mock_logger.warning = MagicMock()
        mock_logger.error = MagicMock()
        mock_get_logger.return_value = mock_logger
        adapter = SSETransportAdapter(coordinator=mock_coordinator, heartbeat_interval=None)
        adapter.logger = mock_logger
        adapter._coordinator = mock_coordinator

    # Manually initialize (await async method)
    await adapter.initialize()
    # register_transport is sync
    mock_coordinator.register_transport.assert_called_once_with(adapter.transport_id, adapter)
    # logger.info is sync
    adapter.logger.info.assert_any_call(f"Initializing SSE transport adapter {adapter.transport_id}...")

    # Simulate an active connection
    conn_id = "test_conn_1"
    real_queue = asyncio.Queue()
    adapter._active_connections[conn_id] = real_queue

    # Mock _signal_queue_close to check its call without queue internals
    # Assume _signal_queue_close is async based on usage in shutdown
    with patch.object(adapter, '_signal_queue_close', new_callable=AsyncMock, wraps=adapter._signal_queue_close) as mock_signal_close:
        await adapter.shutdown() # Call real shutdown (await async method)
        # Assert awaitable mock was awaited
        mock_signal_close.assert_awaited_once_with(real_queue, conn_id)

    # Check queue received signal
    assert await real_queue.get() == "CLOSE_CONNECTION"
    assert conn_id not in adapter._active_connections # Check connection removed

    # Check unregistration call on the coordinator mock (unregister_transport is sync)
    mock_coordinator.unregister_transport.assert_called_once_with(adapter.transport_id)
    # logger.info is sync
    adapter.logger.info.assert_any_call(f"Shutting down SSE transport adapter {adapter.transport_id}...")
    adapter.logger.info.assert_any_call(f"SSE transport adapter {adapter.transport_id} shut down.")


@pytest.mark.asyncio
async def test_sse_adapter_send_event(sse_adapter):
    """Test sending an event to active connections (using initialized adapter from fixture)."""
    # Fixture provides initialized adapter
    sse_adapter._active_connections.clear() # Ensure clean state for this test
    conn_id_1 = "conn1"
    conn_id_2 = "conn2"
    # Use real queues to test put_nowait behavior correctly
    queue_1 = asyncio.Queue()
    queue_2 = asyncio.Queue()
    sse_adapter._active_connections = {conn_id_1: queue_1, conn_id_2: queue_2}

    event_type = EventTypes.PROGRESS
    event_data = {"request_id": "req123", "status": "running", "message": "Working..."}
    expected_json = json.dumps(event_data)
    expected_sse_message = f"event: {event_type.value}\ndata: {expected_json}\n\n"

    # send_event is async
    await sse_adapter.send_event(event_type, event_data)

    # Check items were put in the queues
    assert await queue_1.get() == expected_sse_message
    assert await queue_2.get() == expected_sse_message
    assert queue_1.empty()
    assert queue_2.empty()


@pytest.mark.asyncio
async def test_sse_adapter_send_event_no_connections(sse_adapter):
    # Fixture provides initialized adapter
    sse_adapter._active_connections = {} # Clear connections for this test
    event_type = EventTypes.STATUS
    event_data = {"request_id": "req456", "status": "starting"}
    # send_event is async
    await sse_adapter.send_event(event_type, event_data)
    # logger.debug is sync
    sse_adapter.logger.debug.assert_any_call(f"No active SSE connections to send event {event_type.value}")


@pytest.mark.asyncio
async def test_sse_adapter_send_event_queue_full(sse_adapter):
    # Fixture provides initialized adapter
    sse_adapter._active_connections.clear() # Ensure clean state
    conn_id = "conn_full"
    # Use a real queue and fill it
    queue = asyncio.Queue(maxsize=1)
    await queue.put("dummy") # Fill the queue
    sse_adapter._active_connections = {conn_id: queue}

    event_type = EventTypes.PROGRESS
    event_data = {"request_id": "req789", "status": "running"}

    # send_event is async
    await sse_adapter.send_event(event_type, event_data)

    # Check logger warning (sync)
    sse_adapter.logger.warning.assert_called_once_with(f"Queue full for connection {conn_id}. Event {event_type.value} dropped.")
    # Ensure the queue still only contains the original item
    assert not queue.empty()
    assert await queue.get() == "dummy"
    assert queue.empty()


@pytest.mark.asyncio
async def test_sse_adapter_validate_request_security_success(sse_adapter, mock_security_context):
    """Test successful security validation (using initialized adapter)."""
    # Fixture provides initialized adapter
    request_data = {"request_id": "req_sec_ok", "auth_token": "VALID_TEST_TOKEN", "parameters": {}}
    # Assume create_context_from_credentials is sync for this test setup
    with patch('aider_mcp_server.sse_transport_adapter.create_context_from_credentials', return_value=mock_security_context) as mock_create:
        # validate_request_security is sync
        context = sse_adapter.validate_request_security(request_data)
        # Check sync mock call
        mock_create.assert_called_once_with({"auth_token": "VALID_TEST_TOKEN"})
        assert context == mock_security_context
        # logger.debug is sync
        sse_adapter.logger.debug.assert_any_call(f"Validating security for request req_sec_ok with keys: ['request_id', 'auth_token', 'parameters']")
        sse_adapter.logger.debug.assert_any_call(f"Security context created for request req_sec_ok: {mock_security_context}")

@pytest.mark.asyncio
async def test_sse_adapter_validate_request_security_failure(sse_adapter):
    """Test failed security validation (using initialized adapter)."""
    # Fixture provides initialized adapter
    request_data = {"request_id": "req_sec_fail", "auth_token": "invalid_token"}
    validation_error = ValueError("Invalid token format")
    # Assume create_context_from_credentials is sync for this test setup
    with patch('aider_mcp_server.sse_transport_adapter.create_context_from_credentials', side_effect=validation_error) as mock_create:
        with pytest.raises(ValueError, match="Security validation failed: Invalid token format"):
            # validate_request_security is sync
            sse_adapter.validate_request_security(request_data)
        # Check sync mock call
        mock_create.assert_called_once_with({"auth_token": "invalid_token"})
        # logger.error is sync
        sse_adapter.logger.error.assert_called_once_with(f"Security validation failed for request req_sec_fail: {validation_error}")

@pytest.mark.asyncio
async def test_sse_adapter_handle_sse_request(sse_adapter, mock_request):
    """Test handling a new SSE connection request (using initialized adapter)."""
    # Fixture provides initialized adapter
    sse_adapter._active_connections.clear() # Ensure clean state
    # handle_sse_request is async
    response = await sse_adapter.handle_sse_request(mock_request)

    assert isinstance(response, EventSourceResponse)
    assert len(sse_adapter._active_connections) == 1
    conn_id = list(sse_adapter._active_connections.keys())[0]
    assert conn_id.startswith("sse-conn-")
    assert isinstance(sse_adapter._active_connections[conn_id], asyncio.Queue)
    # logger.info is sync
    sse_adapter.logger.info.assert_any_call(f"New SSE client connection request received by transport {sse_adapter.transport_id}. Assigning ID: {conn_id} to client {mock_request.client.host}")

    # Basic generator test
    queue = sse_adapter._active_connections[conn_id]
    test_message = "event: test\ndata: {}\n\n"
    gen = response.body_iterator
    try:
        # __anext__ is async
        initial_event = await gen.__anext__()
        assert isinstance(initial_event, dict) and initial_event['event'] == 'connected'
        await queue.put(test_message)
        # __anext__ is async
        next_event = await gen.__anext__()
        assert next_event == test_message
        await queue.put("CLOSE_CONNECTION")
        with pytest.raises(StopAsyncIteration):
            # __anext__ is async
            await gen.__anext__()
    finally:
        await asyncio.sleep(0.01) # Allow cleanup task in generator to run
        assert conn_id not in sse_adapter._active_connections
        # logger.info is sync
        sse_adapter.logger.info.assert_any_call(f"Cleaning up resources for SSE connection {conn_id}")


@pytest.mark.asyncio
async def test_sse_adapter_handle_message_request_success(sse_adapter, mock_request, mock_coordinator):
    """Test handling a valid message request (using initialized adapter)."""
    # Fixture provides initialized adapter
    request_id = "req_msg_ok"
    operation_name = "test_op"
    params = {"param1": "value1"}
    request_data = {"request_id": request_id, "name": operation_name, "parameters": params, "auth_token": "valid_token"}
    # Configure the async mock method
    mock_request.json = AsyncMock(return_value=request_data)

    # handle_message_request is async
    response = await sse_adapter.handle_message_request(mock_request)

    # Check coordinator.start_request was awaited (it's async)
    mock_coordinator.start_request.assert_awaited_once_with(
        request_id=request_id,
        transport_id=sse_adapter.transport_id,
        operation_name=operation_name,
        request_data=request_data,
    )
    assert isinstance(response, JSONResponse)
    assert response.status_code == 202
    response_body = json.loads(response.body)
    assert response_body == {"success": True, "status": "accepted", "request_id": request_id}
    # logger.info is sync
    sse_adapter.logger.info.assert_any_call(f"Request {request_id} accepted for processing.")

@pytest.mark.asyncio
async def test_sse_adapter_handle_message_request_invalid_json(sse_adapter, mock_request):
    # Fixture provides initialized adapter
    # Configure the async mock method to raise error
    mock_request.json = AsyncMock(side_effect=json.JSONDecodeError("Expecting value", "doc", 0))
    # handle_message_request is async
    response = await sse_adapter.handle_message_request(mock_request)
    assert isinstance(response, JSONResponse)
    assert response.status_code == 400
    assert json.loads(response.body) == {"success": False, "error": "Invalid JSON payload"}
    # logger.error is sync
    sse_adapter.logger.error.assert_called_once_with(f"Invalid JSON received in message request body: {json.JSONDecodeError('Expecting value', 'doc', 0)}")

@pytest.mark.asyncio
async def test_sse_adapter_handle_message_request_missing_name(sse_adapter, mock_request, mock_coordinator):
    # Fixture provides initialized adapter
    request_data = {"parameters": {}} # Missing 'name' and 'request_id'
    # Configure the async mock method
    mock_request.json = AsyncMock(return_value=request_data)
    # handle_message_request is async
    response = await sse_adapter.handle_message_request(mock_request)
    assert isinstance(response, JSONResponse)
    assert response.status_code == 400

    # Check logger.error was called (sync) - less specific assertion
    sse_adapter.logger.error.assert_called()
    # Check the structure of the first call's arguments if needed, e.g., check the message starts correctly
    first_error_call_args = sse_adapter.logger.error.call_args[0]
    assert isinstance(first_error_call_args[0], str)
    assert first_error_call_args[0].startswith("Missing or invalid 'name' field (operation name) in request")

    response_body = json.loads(response.body)
    assert response_body["success"] is False
    assert "error" in response_body
    assert response_body["error"].startswith("Missing or invalid 'name' field")
    # Assert that request_id is present in the response body
    assert "request_id" in response_body
    assert isinstance(response_body["request_id"], str) # Check it's a string (UUID format)

    # Check fail_request was awaited (it's async)
    mock_coordinator.fail_request.assert_awaited_once_with(
        request_id=response_body["request_id"], # Use the ID returned in the response
        operation_name="unknown", # Use placeholder as name was missing
        error="Invalid request", # Error category used by adapter
        error_details=ANY, # Detailed message can vary slightly with generated ID
        originating_transport_id=sse_adapter.transport_id,
        request_details={}, # Parameters from request
    )


@pytest.mark.asyncio
async def test_sse_adapter_handle_message_request_security_fail(sse_adapter, mock_request, mock_coordinator):
    """Test handling message where security validation (simulated via start_request failure) fails."""
    # Fixture provides initialized adapter
    request_id = "req_sec_fail_handler"
    operation_name = "op_sec_fail"
    request_data = {"request_id": request_id, "name": operation_name, "auth_token": "bad", "parameters": {}} # Added parameters
    # Configure the async mock method
    mock_request.json = AsyncMock(return_value=request_data)
    security_error = ValueError("Token expired") # Simulate security failure

    # Mock the coordinator's start_request (async) to raise the security error
    mock_coordinator.start_request.side_effect = security_error

    # handle_message_request is async
    response = await sse_adapter.handle_message_request(mock_request)

    # Check start_request was awaited (async)
    mock_coordinator.start_request.assert_awaited_once_with(
        request_id=request_id,
        transport_id=sse_adapter.transport_id,
        operation_name=operation_name,
        request_data=request_data,
    )
    assert isinstance(response, JSONResponse)
    # The adapter catches ValueError from start_request and returns 401
    assert response.status_code == 401
    response_body = json.loads(response.body)
    assert response_body == {"success": False, "error": "Security validation failed", "details": str(security_error)}

    # Check fail_request *was* awaited (it's async) - Adapter calls fail_request in this case
    mock_coordinator.fail_request.assert_awaited_once_with(
        request_id=request_id,
        operation_name=operation_name,
        error="Security validation failed",
        error_details=str(security_error),
        originating_transport_id=sse_adapter.transport_id,
        request_details={}, # Parameters from request (empty in this case)
    )


@pytest.mark.asyncio
async def test_sse_adapter_handle_message_request_coordinator_unavailable(sse_adapter, mock_request):
    """Test handling message when coordinator is None (using initialized adapter)."""
    # Fixture provides initialized adapter
    sse_adapter._coordinator = None # Simulate coordinator not being set *after* init for this test
    request_data = {"request_id": "req_no_coord", "name": "op"}
    # Configure the async mock method
    mock_request.json = AsyncMock(return_value=request_data)

    # handle_message_request is async
    response = await sse_adapter.handle_message_request(mock_request)

    assert isinstance(response, JSONResponse)
    assert response.status_code == 503
    response_body = json.loads(response.body)
    expected_error = f"Application coordinator not available for transport {sse_adapter.transport_id}."
    assert response_body == {"success": False, "error": expected_error}
    # logger.error is sync
    sse_adapter.logger.error.assert_called_once_with(expected_error)


# --- Test process_aider_ai_code_request ---

@pytest.mark.asyncio
async def test_process_aider_ai_code_request():
    """Test that process_aider_ai_code_request properly awaits the async code_with_aider function."""
    from aider_mcp_server.handlers import process_aider_ai_code_request
    from aider_mcp_server.security import SecurityContext, Permissions
    
    # Create a mock for the async code_with_aider function
    with patch('aider_mcp_server.handlers.code_with_aider', new_callable=AsyncMock) as mock_code_with_aider:
        # Configure the mock to return a valid JSON string when awaited
        mock_code_with_aider.return_value = json.dumps({
            "success": True,
            "diff": "Sample diff content"
        })
        
        # Create test parameters
        request_id = "test_req_123"
        transport_id = "test_transport"
        params = {
            "ai_coding_prompt": "Test prompt",
            "relative_editable_files": ["test.py"],
            "relative_readonly_files": []
        }
        security_context = SecurityContext(user_id="test_user", permissions={Permissions.EXECUTE_AIDER})
        editor_model = "test-model"
        current_working_dir = "/test/dir"
        
        # Call the function under test
        result = await process_aider_ai_code_request(
            request_id=request_id,
            transport_id=transport_id,
            params=params,
            security_context=security_context,
            editor_model=editor_model,
            current_working_dir=current_working_dir
        )
        
        # Verify code_with_aider was awaited (AsyncMock tracks this automatically)
        mock_code_with_aider.assert_awaited_once_with(
            ai_coding_prompt="Test prompt",
            relative_editable_files=["test.py"],
            relative_readonly_files=[],
            model=editor_model,
            working_dir=current_working_dir,
        )
        
        # Verify result was properly processed
        assert result["success"] is True
        assert result["diff"] == "Sample diff content"

# --- Test serve_sse ---

@pytest.mark.asyncio
@patch('aider_mcp_server.sse_server.uvicorn.Server') # Patches the class
@patch('aider_mcp_server.sse_server.uvicorn.Config') # Patches the class
# Patch Starlette where it's used in sse_server
@patch('aider_mcp_server.sse_server.Starlette')
# Patch ApplicationCoordinator.getInstance where it's used in sse_server - USE ASYNCMOCK
@patch('aider_mcp_server.sse_server.ApplicationCoordinator.getInstance', new_callable=AsyncMock)
# Patch logger used by sse_server module
@patch('aider_mcp_server.sse_server.get_logger_func')
# Patch logger used by the adapter module (as serve_sse creates a real adapter)
@patch('aider_mcp_server.transport_adapter.get_logger_func')
@patch('aider_mcp_server.sse_server.asyncio.get_running_loop')
@patch('aider_mcp_server.sse_server.signal.signal')
@patch('aider_mcp_server.sse_server.signal.getsignal')
@patch('aider_mcp_server.sse_server._create_shutdown_task_wrapper')
@patch('aider_mcp_server.sse_server.is_git_repository')
# REMOVED: Patch for handle_shutdown_signal - we will mock via global variable now
# @patch('aider_mcp_server.sse_server.handle_shutdown_signal', new_callable=AsyncMock)
@patch('aider_mcp_server.sse_server.asyncio.Event') # Patches the Event class
async def test_serve_sse_startup_and_run(
    mock_asyncio_event_cls, # Mock for asyncio.Event CLASS
    # REMOVED: mock_handle_shutdown_signal_func parameter
    mock_is_git_repo,
    mock_create_shutdown_wrapper,
    mock_getsignal,
    mock_signal_signal,
    mock_get_loop,
    mock_adapter_get_logger, # Mock for adapter's logger
    mock_server_get_logger, # Mock for sse_server's logger
    mock_get_coordinator_instance, # Mock for ApplicationCoordinator.getInstance (now AsyncMock)
    mock_starlette_cls, # Renamed from mock_fastapi_cls
    mock_uvicorn_config_cls, # Mock for Config class
    mock_uvicorn_server_cls, # Mock for Server class
    # mock_uvicorn_server fixture is removed as it's not directly used here
    mock_uvicorn_config): # Fixture for Uvicorn Config instance
    """Test the serve_sse function startup sequence and running the server."""

    # --- Mock Setup ---
    # Mock loggers (ensure methods are sync MagicMocks)
    mock_server_logger = MagicMock(spec=LoggerProtocol)
    mock_server_logger.info = MagicMock()
    mock_server_logger.debug = MagicMock()
    mock_server_logger.warning = MagicMock()
    mock_server_logger.error = MagicMock()
    mock_server_get_logger.return_value = mock_server_logger
    mock_adapter_logger = MagicMock(spec=LoggerProtocol)
    mock_adapter_logger.info = MagicMock()
    mock_adapter_logger.debug = MagicMock()
    mock_adapter_logger.warning = MagicMock()
    mock_adapter_logger.error = MagicMock()
    mock_adapter_get_logger.return_value = mock_adapter_logger

    # Mock Coordinator Instance (returned by the patched getInstance AsyncMock)
    mock_coordinator_instance = MagicMock(spec=ApplicationCoordinator)
    # Sync methods
    mock_coordinator_instance.register_transport = MagicMock()
    mock_coordinator_instance.unregister_transport = MagicMock()
    # mock_coordinator_instance.register_handler = MagicMock() # Incorrect: register_handler is async
    # Async methods
    mock_coordinator_instance.shutdown = AsyncMock()
    mock_coordinator_instance.wait_for_initialization = AsyncMock()
    mock_coordinator_instance.register_handler = AsyncMock() # Correct: register_handler is async
    mock_coordinator_instance.__aenter__ = AsyncMock(return_value=mock_coordinator_instance)
    mock_coordinator_instance.__aexit__ = AsyncMock()
    # Configure the AsyncMock patch to return our instance mock when awaited
    mock_get_coordinator_instance.return_value = mock_coordinator_instance

    # Mock Starlette app instance
    mock_starlette_app = MagicMock()
    mock_starlette_cls.return_value = mock_starlette_app
    # No need to mock decorators like .get or .post for Starlette route list initialization

    # Mock Uvicorn Config class to return the fixture instance
    mock_uvicorn_config_cls.return_value = mock_uvicorn_config

    # Create the Uvicorn Server *instance* mock manually (without spec)
    mock_uvicorn_server_instance = MagicMock()
    mock_uvicorn_server_instance.serve = AsyncMock()
    mock_uvicorn_server_instance.shutdown = AsyncMock() # Use AsyncMock for shutdown
    mock_uvicorn_server_instance.started = True # Simulate server starts
    # Configure the patched Server *class* to return our instance mock
    mock_uvicorn_server_cls.return_value = mock_uvicorn_server_instance

    # Mock loop and signal handling setup
    mock_loop = MagicMock()
    mock_loop.add_signal_handler = MagicMock()
    mock_loop.remove_signal_handler = MagicMock() # Add mock for remove_signal_handler
    mock_get_loop.return_value = mock_loop
    mock_getsignal.return_value = None
    # Create distinct mocks for the sync wrappers returned by _create_shutdown_task_wrapper
    mock_sync_sigint_handler = MagicMock(name="sync_sigint_handler")
    mock_sync_sigterm_handler = MagicMock(name="sync_sigterm_handler")
    # The wrapper needs to accept the signal, the *async* handler function it wraps, and the event
    def wrapper_side_effect(sig, async_handler_func, event):
        # Return the appropriate sync handler mock based on the signal
        if sig == signal.SIGINT:
            # Store the async handler it's supposed to call for later verification
            mock_sync_sigint_handler._calls_async = async_handler_func
            mock_sync_sigint_handler._event = event
            return mock_sync_sigint_handler
        elif sig == signal.SIGTERM:
            mock_sync_sigterm_handler._calls_async = async_handler_func
            mock_sync_sigterm_handler._event = event
            return mock_sync_sigterm_handler
        return MagicMock() # Default fallback
    mock_create_shutdown_wrapper.side_effect = wrapper_side_effect

    # --- Configure the mock Event returned by the patched class ---
    # Get the mock instance that will be returned when asyncio.Event() is called
    mock_event_instance_returned = mock_asyncio_event_cls.return_value
    mock_event_instance_returned.is_set = MagicMock(return_value=False) # Make is_set a mock
    # Make wait() async
    async def wait_side_effect():
        # Simulate running: check is_set until it becomes true
        while not mock_event_instance_returned.is_set(): # Call the mock is_set
            await asyncio.sleep(0.001)
    mock_event_instance_returned.wait = AsyncMock(side_effect=wait_side_effect)
    # Make set() sync and define its side effect using a helper function
    def set_side_effect():
        # This function will be called when mock_event_instance_returned.set() is called
        mock_event_instance_returned.is_set.return_value = True # Update the return value of the is_set mock
    mock_event_instance_returned.set = MagicMock(side_effect=set_side_effect) # Use the helper function

    # Simulate server.serve() running until event.wait() finishes
    # Also mock the server task for cancellation checks
    mock_server_task = asyncio.create_task(asyncio.sleep(0.01)) # Dummy task initially
    async def serve_side_effect():
        nonlocal mock_server_task
        # Start a task that will eventually set the event, simulating signal reception
        async def trigger_shutdown():
            await asyncio.sleep(0.01) # Give time for wait() to start
            # Simulate the signal handler being called, which sets the event
            mock_event_instance_returned.set() # Call the mocked set()
        trigger_task = asyncio.create_task(trigger_shutdown())
        # Wait for the event to be set
        await mock_event_instance_returned.wait() # Await the mocked wait()
        # Update the state of the *instance* mock
        mock_uvicorn_server_instance.started = False # Mark server as stopped
        await trigger_task # Ensure trigger task completes
        return None
    # Assign the side effect to the serve method of the *instance* mock
    mock_uvicorn_server_instance.serve.side_effect = serve_side_effect

    # Mock create_task to capture the server task
    original_create_task = asyncio.create_task
    def create_task_wrapper(coro, *, name=None):
        nonlocal mock_server_task
        task = original_create_task(coro, name=name)
        if name == "uvicorn-server":
            mock_server_task = task # Capture the actual server task
        return task
    mock_loop.create_task.side_effect = create_task_wrapper

    mock_is_git_repo.return_value = (True, None)

    # --- Create mock for handle_shutdown_signal and patch global ---
    mock_handle_shutdown_signal_func = AsyncMock(name="mock_handle_shutdown_signal")

    # --- Call the function ---
    host, port, editor_model, cwd, heartbeat = "127.0.0.1", 8888, "test_model", "/test/repo", 20.0
    # Patch the global variable in the sse_server module for the duration of the call
    with patch.object(sse_server_module, '_test_handle_shutdown_signal', mock_handle_shutdown_signal_func):
        # serve_sse is async
        await serve_sse(host, port, editor_model, cwd, heartbeat_interval=heartbeat)

    # --- Assertions ---
    mock_is_git_repo.assert_called_once_with(Path(cwd))
    # Check singleton access (awaiting the AsyncMock patch)
    mock_get_coordinator_instance.assert_awaited_once()

    # Check coordinator context manager usage (async)
    mock_coordinator_instance.__aenter__.assert_awaited_once()

    # Check handler registration *inside* the coordinator context (async)
    # Use assert_has_calls to check both expected registrations occurred (awaiting AsyncMock)
    expected_handler_calls = [
        call("aider_ai_code", ANY, required_permission=Permissions.EXECUTE_AIDER),
        call("list_models", ANY) # Assuming no permission needed for list_models
    ]
    # Assert that the async mock was awaited for each call
    mock_coordinator_instance.register_handler.assert_has_awaits(expected_handler_calls, any_order=True)

    # Check adapter registration (called by coordinator inside serve_sse) (sync)
    # register_transport is sync in the coordinator mock setup.
    mock_coordinator_instance.register_transport.assert_called_once() # Check it was called
    adapter_instance_registered = mock_coordinator_instance.register_transport.call_args[0][1]
    assert isinstance(adapter_instance_registered, SSETransportAdapter)
    assert adapter_instance_registered.transport_id.startswith("sse_")

    # Check Starlette app creation with routes
    mock_starlette_cls.assert_called_once()
    call_args, call_kwargs = mock_starlette_cls.call_args
    # Check the 'routes' keyword argument passed to Starlette constructor
    assert 'routes' in call_kwargs
    routes_arg = call_kwargs['routes']
    assert isinstance(routes_arg, list)
    assert len(routes_arg) == 2 # Expecting /sse and /message routes
    # Check route details
    assert isinstance(routes_arg[0], Route)
    assert routes_arg[0].path == "/sse"
    assert isinstance(routes_arg[1], Route)
    assert routes_arg[1].path == "/message"
    assert routes_arg[1].methods == {"POST"}

    # Check Uvicorn config and server creation
    mock_uvicorn_config_cls.assert_called_once_with(
        app=mock_starlette_app, # Check the mocked Starlette app instance is passed
        host=host,
        port=port,
        log_config=None, # Check log_config is None
        handle_signals=False # Check signals are handled manually
    )
    # Check the Server class was called with the config instance
    mock_uvicorn_server_cls.assert_called_once_with(mock_uvicorn_config)

    # Signal handling setup
    mock_get_loop.assert_called_once()
    assert mock_create_shutdown_wrapper.call_count == 2
    # Check it was called with the signal, the *mock* async handler function, and the event instance
    mock_create_shutdown_wrapper.assert_any_call(signal.SIGINT, mock_handle_shutdown_signal_func, mock_event_instance_returned)
    mock_create_shutdown_wrapper.assert_any_call(signal.SIGTERM, mock_handle_shutdown_signal_func, mock_event_instance_returned)
    # Check signal handlers were added using the *results* of the wrapper (the sync mocks)
    if hasattr(mock_loop, 'add_signal_handler') and mock_loop.add_signal_handler.called:
        mock_loop.add_signal_handler.assert_any_call(signal.SIGINT, mock_sync_sigint_handler)
        mock_loop.add_signal_handler.assert_any_call(signal.SIGTERM, mock_sync_sigterm_handler)
    else: # Fallback check for systems without add_signal_handler
        mock_signal_signal.assert_any_call(signal.SIGINT, mock_sync_sigint_handler)
        mock_signal_signal.assert_any_call(signal.SIGTERM, mock_sync_sigterm_handler)

    # Server run and wait (async)
    # Assert serve was awaited on the *instance* mock
    mock_uvicorn_server_instance.serve.assert_awaited_once()
    # Check we waited for shutdown on the event instance returned by the patch
    mock_event_instance_returned.wait.assert_awaited_once()

    # Shutdown sequence
    # Check server shutdown logic (should_exit and wait_for)
    # Assert on the *instance* mock
    assert mock_uvicorn_server_instance.should_exit is True
    # Check the server task was awaited (via wait_for)
    # This requires mocking asyncio.wait_for if we want to check it directly,
    # or ensuring the mock_server_task completes. Our side effect handles completion.
    assert mock_server_task.done() # Check the captured server task finished

    # Check coordinator context exit (async)
    mock_coordinator_instance.__aexit__.assert_awaited_once()

    # Check signal handler removal
    if hasattr(mock_loop, 'remove_signal_handler') and mock_loop.remove_signal_handler.called:
        mock_loop.remove_signal_handler.assert_any_call(signal.SIGINT)
        mock_loop.remove_signal_handler.assert_any_call(signal.SIGTERM)
    else: # Fallback check
        # Check signal.signal was called again to restore handlers (likely SIG_DFL)
        mock_signal_signal.assert_any_call(signal.SIGINT, signal.SIG_DFL)
        mock_signal_signal.assert_any_call(signal.SIGTERM, signal.SIG_DFL)


    # Check that the mock handle_shutdown_signal function was NOT called directly
    # because the signal mechanism calls the *wrapper* created by _create_shutdown_task_wrapper
    mock_handle_shutdown_signal_func.assert_not_awaited()


@pytest.mark.asyncio
async def test_handle_shutdown_signal():
    """Test the handle_shutdown_signal function correctly sets the event."""
    # Use MagicMock for event as set() is sync.
    mock_event = MagicMock(spec=asyncio.Event)
    mock_event.is_set.return_value = False # Ensure event starts as not set
    mock_event.set = MagicMock() # set() is sync
    mock_logger = MagicMock(spec=LoggerProtocol)
    mock_logger.warning = MagicMock()
    mock_logger.info = MagicMock()

    # Patch logger within the function's scope
    with patch('aider_mcp_server.sse_server.logger', mock_logger):
        # Call the actual async function, passing signal, event, and optionally signum/frame
        # handle_shutdown_signal is async
        await handle_shutdown_signal(signal.SIGINT, mock_event, signum=signal.SIGINT, frame=None)

    # --- Assertions ---
    # Check ONLY the event was set (sync method)
    mock_event.set.assert_called_once()

    # Check logging (sync methods)
    mock_logger.warning.assert_any_call("Received signal SIGINT. Initiating graceful shutdown...")
    mock_logger.info.assert_any_call("Signaling main loop to shut down via event...")


def test_create_shutdown_task_wrapper_with_event():
    """Test the _create_shutdown_task_wrapper helper when an event is provided."""
    # The handler being wrapped is async
    mock_async_handler = AsyncMock()
    mock_event = MagicMock(spec=asyncio.Event)
    sig = signal.SIGINT
    signum_received = signal.SIGINT # Signal number passed by signal mechanism
    frame_received = None # Frame passed by signal mechanism

    # Create the sync wrapper by calling the function under test, providing the event
    sync_wrapper = _create_shutdown_task_wrapper(sig, mock_async_handler, mock_event)

    # Mock the running loop and create_task
    mock_loop = MagicMock(spec=asyncio.AbstractEventLoop)
    mock_loop.is_running.return_value = True
    mock_loop.is_closed.return_value = False
    mock_loop.create_task = MagicMock()

    # Patch get_running_loop to return our mock loop
    with patch('aider_mcp_server.sse_server.asyncio.get_running_loop', return_value=mock_loop):
        # Call the generated sync wrapper, simulating a signal event
        sync_wrapper(signum_received, frame_received)

        # Assert get_running_loop was called
        # asyncio.get_running_loop.assert_called_once() # This mock is tricky with context manager

        # Assert loop.create_task was called once
        mock_loop.create_task.assert_called_once()
        call_args, call_kwargs = mock_loop.create_task.call_args
        # Get the coroutine object passed to create_task
        coro = call_args[0]
        assert asyncio.iscoroutine(coro)

        # --- Verify the coroutine calls the original async handler ---
        # We need to run the coroutine to check if it awaits the mock_async_handler
        async def run_coro():
            await coro

        # Run the captured coroutine
        asyncio.run(run_coro())

        # Assert the original async handler was awaited with the correct arguments
        # When event is provided, it expects (sig, event, signum, frame)
        mock_async_handler.assert_awaited_once_with(sig, mock_event, signum_received, frame_received)


def test_create_shutdown_task_wrapper_no_event_test_mode():
    """Test the _create_shutdown_task_wrapper helper when no event is provided (test mode)."""
    # The handler being wrapped is async (use a different mock for clarity)
    mock_async_handler_test = AsyncMock()
    sig = signal.SIGTERM
    signum_received = signal.SIGTERM
    frame_received = None

    # Create the sync wrapper *without* providing the event
    sync_wrapper = _create_shutdown_task_wrapper(sig, mock_async_handler_test, event=None)

    # Patch asyncio.create_task directly (as used in the 'else' block)
    with patch('aider_mcp_server.sse_server.asyncio.create_task') as mock_create_task_direct:
        # Call the generated sync wrapper
        sync_wrapper(signum_received, frame_received)

        # Assert asyncio.create_task was called directly
        mock_create_task_direct.assert_called_once()
        call_args, call_kwargs = mock_create_task_direct.call_args
        coro = call_args[0]
        assert asyncio.iscoroutine(coro)

        # --- Verify the coroutine calls the original async handler ---
        async def run_coro():
            await coro
        asyncio.run(run_coro())

        # Assert the original async handler was awaited with the arguments expected in test mode
        # When event is None, it expects (sig, signum, frame)
        mock_async_handler_test.assert_awaited_once_with(sig, signum_received, frame_received)

