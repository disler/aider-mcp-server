import asyncio
import json
import signal
import uuid
from pathlib import Path # Import Path
from unittest.mock import AsyncMock, MagicMock, patch, call, ANY

import pytest
import uvicorn # Import uvicorn
# Use absolute imports from the package root
from starlette.datastructures import Headers
from starlette.requests import Request
from starlette.responses import JSONResponse, Response # Import Response
from sse_starlette.sse import EventSourceResponse

from aider_mcp_server.atoms.event_types import EventTypes
from aider_mcp_server.security import SecurityContext, Permissions, create_context_from_credentials
from aider_mcp_server.sse_transport_adapter import SSETransportAdapter
from aider_mcp_server.transport_coordinator import ApplicationCoordinator
# Import the specific functions/classes being tested or mocked
from aider_mcp_server.sse_server import serve_sse, handle_shutdown_signal, _create_shutdown_task_wrapper
# Import Logger for spec - use LoggerProtocol as defined elsewhere if Logger is concrete
# Assuming LoggerProtocol is the intended type hint standard
from aider_mcp_server.transport_adapter import LoggerProtocol


# --- Fixtures ---

@pytest.fixture
def mock_coordinator():
    """Fixture for a mocked ApplicationCoordinator instance returned by getInstance."""
    # Create an instance mock *with* spec to ensure methods exist
    # This instance will be returned by the patched getInstance
    coordinator_instance = MagicMock(spec=ApplicationCoordinator)

    # Add async methods needed (use AsyncMock for awaitables)
    coordinator_instance.register_transport = MagicMock()
    coordinator_instance.unregister_transport = MagicMock()
    coordinator_instance.register_handler = MagicMock()
    coordinator_instance.subscribe_to_event_type = MagicMock()
    coordinator_instance.start_request = AsyncMock()
    coordinator_instance.fail_request = AsyncMock()
    coordinator_instance.shutdown = AsyncMock()
    coordinator_instance.wait_for_initialization = AsyncMock() # Needed by start_request etc.
    coordinator_instance.is_shutting_down = MagicMock(return_value=False) # Needed by endpoints

    # Add async context manager methods explicitly
    coordinator_instance.__aenter__ = AsyncMock(return_value=coordinator_instance)
    # __aexit__ should call shutdown in real impl, ensure mock does too if needed, or just mock __aexit__
    coordinator_instance.__aexit__ = AsyncMock()

    # Patch the class's getInstance method in the module where it's used (sse_server)
    with patch('aider_mcp_server.sse_server.ApplicationCoordinator.getInstance', return_value=coordinator_instance) as mock_get_instance:
        # Also patch the class itself for spec checks if needed elsewhere, though getInstance is primary
        with patch('aider_mcp_server.sse_server.ApplicationCoordinator', spec=ApplicationCoordinator) as mock_coord_cls:
             # Configure the patched class to return the instance mock via getInstance
             mock_coord_cls.getInstance.return_value = coordinator_instance
             yield coordinator_instance # Yield the configured instance mock

@pytest.fixture
def mock_security_context():
    """Fixture for a basic SecurityContext."""
    return SecurityContext(user_id="test_user", permissions={Permissions.EXECUTE_AIDER})

@pytest.fixture
def sse_adapter(mock_coordinator):
    """Fixture for an SSETransportAdapter instance with a mocked coordinator."""
    # Patch the logger within the adapter's scope
    with patch('aider_mcp_server.transport_adapter.get_logger_func') as mock_get_logger:
        # Use LoggerProtocol for spec matching the type hints
        mock_logger = MagicMock(spec=LoggerProtocol)
        mock_get_logger.return_value = mock_logger
        # Create adapter instance - pass the *mocked* coordinator instance
        adapter = SSETransportAdapter(coordinator=mock_coordinator, heartbeat_interval=0.1)
        # Mock async methods/tasks if needed for specific tests focusing *only* on adapter
        # For integration tests like serve_sse, we might want the real methods.
        # adapter._heartbeat_task = AsyncMock() # Base class handles this
        adapter.initialize = AsyncMock() # Mock initialize to prevent real registration in simple tests
        adapter.shutdown = AsyncMock() # Mock shutdown in simple tests
        adapter.logger = mock_logger # Ensure mock logger is attached
        # Assign coordinator explicitly for tests that might check adapter._coordinator
        adapter._coordinator = mock_coordinator
        yield adapter

@pytest.fixture
def mock_request():
    """Fixture for a mocked Starlette Request."""
    request = MagicMock(spec=Request)
    request.client = MagicMock()
    request.client.host = "127.0.0.1"
    request.client.port = 5000
    request.headers = Headers({})
    request.json = AsyncMock(return_value={}) # Default mock
    return request

@pytest.fixture
def mock_uvicorn_server():
    """Fixture for a mocked Uvicorn Server instance."""
    server = MagicMock(spec=uvicorn.Server) # Use spec for better mocking
    server.serve = AsyncMock()
    server.shutdown = AsyncMock()
    # Simulate properties used in serve_sse shutdown logic
    server.started = True # Assume server starts for shutdown tests
    return server

@pytest.fixture
def mock_uvicorn_config():
    """Fixture for a mocked Uvicorn Config."""
    return MagicMock(spec=uvicorn.Config)

# --- Test SSETransportAdapter ---
# These tests focus on the adapter logic itself, using the mocked coordinator

@pytest.mark.asyncio
async def test_sse_adapter_init(mock_coordinator):
    """Test SSETransportAdapter initialization."""
    with patch('aider_mcp_server.transport_adapter.get_logger_func') as mock_get_logger:
        mock_logger = MagicMock(spec=LoggerProtocol)
        mock_get_logger.return_value = mock_logger
        adapter = SSETransportAdapter(coordinator=mock_coordinator, sse_queue_size=50)

    assert adapter._coordinator == mock_coordinator
    assert adapter.transport_type == "sse"
    assert adapter._sse_queue_size == 50
    assert isinstance(adapter.transport_id, str) and adapter.transport_id.startswith("sse_")
    assert adapter.logger is mock_logger
    assert adapter._active_connections == {}
    # Check logger calls if needed, e.g.,
    mock_logger.info.assert_called_with(f"SSETransportAdapter created with ID: {adapter.transport_id}. Max queue size: 50")

@pytest.mark.asyncio
async def test_sse_adapter_initialize_real(mock_coordinator):
    """Test the real initialize method of SSETransportAdapter (calls coordinator)."""
    # Use the real adapter, don't mock initialize/shutdown
    with patch('aider_mcp_server.transport_adapter.get_logger_func') as mock_get_logger:
        mock_logger = MagicMock(spec=LoggerProtocol)
        mock_get_logger.return_value = mock_logger
        # Create real adapter with the mocked coordinator instance
        adapter = SSETransportAdapter(coordinator=mock_coordinator, heartbeat_interval=None) # Disable heartbeat

    await adapter.initialize() # Call the real initialize

    # Assert registration calls on the coordinator mock
    # Note: register_transport now also updates subscriptions by default
    mock_coordinator.register_transport.assert_called_once_with(adapter.transport_id, adapter)
    # Check that update_transport_subscriptions was called (implicitly by register_transport)
    # This requires the mock_coordinator fixture to have this method mocked if we want to assert it.
    # Let's add it to the fixture for completeness, although register_transport is the main check.
    # mock_coordinator.update_transport_subscriptions.assert_called_once_with(adapter.transport_id, adapter.get_capabilities())

    # Check logging
    mock_logger.info.assert_any_call(f"Initializing SSE transport adapter {adapter.transport_id}...")
    mock_logger.info.assert_any_call(f"SSE transport adapter {adapter.transport_id} initialized.")


@pytest.mark.asyncio
async def test_sse_adapter_shutdown_real(mock_coordinator):
    """Test the real shutdown method of SSETransportAdapter (calls coordinator)."""
    with patch('aider_mcp_server.transport_adapter.get_logger_func') as mock_get_logger:
        mock_logger = MagicMock(spec=LoggerProtocol)
        mock_get_logger.return_value = mock_logger
        adapter = SSETransportAdapter(coordinator=mock_coordinator, heartbeat_interval=None)

    # Simulate an active connection
    conn_id = "test_conn_1"
    real_queue = asyncio.Queue()
    adapter._active_connections[conn_id] = real_queue

    # Mock _signal_queue_close to check its call without queue internals
    with patch.object(adapter, '_signal_queue_close', wraps=adapter._signal_queue_close) as mock_signal_close:
        await adapter.shutdown() # Call real shutdown
        mock_signal_close.assert_awaited_once_with(real_queue, conn_id)

    # Check queue received signal
    assert await real_queue.get() == "CLOSE_CONNECTION"
    assert conn_id not in adapter._active_connections # Check connection removed

    # Check unregistration call on the coordinator mock
    mock_coordinator.unregister_transport.assert_called_once_with(adapter.transport_id)
    mock_logger.info.assert_any_call(f"Shutting down SSE transport adapter {adapter.transport_id}...")
    mock_logger.info.assert_any_call(f"SSE transport adapter {adapter.transport_id} shut down.")


@pytest.mark.asyncio
async def test_sse_adapter_send_event(sse_adapter):
    """Test sending an event to active connections (using mocked adapter)."""
    sse_adapter._active_connections.clear() # Ensure clean state
    conn_id_1 = "conn1"
    conn_id_2 = "conn2"
    queue_1 = MagicMock(spec=asyncio.Queue)
    queue_2 = MagicMock(spec=asyncio.Queue)
    sse_adapter._active_connections = {conn_id_1: queue_1, conn_id_2: queue_2}

    event_type = EventTypes.PROGRESS
    event_data = {"request_id": "req123", "status": "running", "message": "Working..."}
    expected_json = json.dumps(event_data)
    expected_sse_message = f"event: {event_type.value}\ndata: {expected_json}\n\n"

    await sse_adapter.send_event(event_type, event_data)

    queue_1.put_nowait.assert_called_once_with(expected_sse_message)
    queue_2.put_nowait.assert_called_once_with(expected_sse_message)

@pytest.mark.asyncio
async def test_sse_adapter_send_event_no_connections(sse_adapter):
    sse_adapter._active_connections = {}
    event_type = EventTypes.STATUS
    event_data = {"request_id": "req456", "status": "starting"}
    await sse_adapter.send_event(event_type, event_data)
    sse_adapter.logger.debug.assert_any_call(f"No active SSE connections to send event {event_type.value}")


@pytest.mark.asyncio
async def test_sse_adapter_send_event_queue_full(sse_adapter):
    sse_adapter._active_connections.clear()
    conn_id = "conn_full"
    queue = MagicMock(spec=asyncio.Queue)
    queue.put_nowait.side_effect = asyncio.QueueFull
    sse_adapter._active_connections = {conn_id: queue}
    event_type = EventTypes.PROGRESS
    event_data = {"request_id": "req789", "status": "running"}
    expected_json = json.dumps(event_data)
    expected_sse_message = f"event: {event_type.value}\ndata: {expected_json}\n\n"
    await sse_adapter.send_event(event_type, event_data)
    queue.put_nowait.assert_called_once_with(expected_sse_message)
    sse_adapter.logger.warning.assert_called_once_with(f"Queue full for connection {conn_id}. Event {event_type.value} dropped.")

@pytest.mark.asyncio
async def test_sse_adapter_validate_request_security_success(sse_adapter, mock_security_context):
    """Test successful security validation (using mocked adapter)."""
    request_data = {"request_id": "req_sec_ok", "auth_token": "VALID_TEST_TOKEN", "parameters": {}}
    with patch('aider_mcp_server.sse_transport_adapter.create_context_from_credentials', return_value=mock_security_context) as mock_create:
        context = sse_adapter.validate_request_security(request_data)
        mock_create.assert_called_once_with({"auth_token": "VALID_TEST_TOKEN"})
        assert context == mock_security_context
        sse_adapter.logger.debug.assert_any_call(f"Validating security for request req_sec_ok with keys: ['request_id', 'auth_token', 'parameters']")
        sse_adapter.logger.debug.assert_any_call(f"Security context created for request req_sec_ok: {mock_security_context}")

@pytest.mark.asyncio
async def test_sse_adapter_validate_request_security_failure(sse_adapter):
    """Test failed security validation (using mocked adapter)."""
    request_data = {"request_id": "req_sec_fail", "auth_token": "invalid_token"}
    validation_error = ValueError("Invalid token format")
    with patch('aider_mcp_server.sse_transport_adapter.create_context_from_credentials', side_effect=validation_error) as mock_create:
        with pytest.raises(ValueError, match="Security validation failed: Invalid token format"):
            sse_adapter.validate_request_security(request_data)
        mock_create.assert_called_once_with({"auth_token": "invalid_token"})
        sse_adapter.logger.error.assert_called_once_with(f"Security validation failed for request req_sec_fail: {validation_error}")

@pytest.mark.asyncio
async def test_sse_adapter_handle_sse_request(sse_adapter, mock_request):
    """Test handling a new SSE connection request (using mocked adapter)."""
    sse_adapter._active_connections.clear()
    response = await sse_adapter.handle_sse_request(mock_request)

    assert isinstance(response, EventSourceResponse)
    assert len(sse_adapter._active_connections) == 1
    conn_id = list(sse_adapter._active_connections.keys())[0]
    assert conn_id.startswith("sse-conn-")
    assert isinstance(sse_adapter._active_connections[conn_id], asyncio.Queue)
    sse_adapter.logger.info.assert_any_call(f"New SSE client connection request received by transport {sse_adapter.transport_id}. Assigning ID: {conn_id} to client {mock_request.client.host}")

    # Basic generator test
    queue = sse_adapter._active_connections[conn_id]
    test_message = "event: test\ndata: {}\n\n"
    gen = response.body_iterator
    try:
        initial_event = await gen.__anext__()
        assert isinstance(initial_event, dict) and initial_event['event'] == 'connected'
        await queue.put(test_message)
        next_event = await gen.__anext__()
        assert next_event == test_message
        await queue.put("CLOSE_CONNECTION")
        with pytest.raises(StopAsyncIteration):
            await gen.__anext__()
    finally:
        await asyncio.sleep(0.01) # Allow cleanup
        assert conn_id not in sse_adapter._active_connections
        sse_adapter.logger.info.assert_any_call(f"Cleaning up resources for SSE connection {conn_id}")


@pytest.mark.asyncio
async def test_sse_adapter_handle_message_request_success(sse_adapter, mock_request, mock_coordinator):
    """Test handling a valid message request (using mocked adapter)."""
    request_id = "req_msg_ok"
    operation_name = "test_op"
    params = {"param1": "value1"}
    request_data = {"request_id": request_id, "name": operation_name, "parameters": params, "auth_token": "valid_token"}
    mock_request.json = AsyncMock(return_value=request_data)

    response = await sse_adapter.handle_message_request(mock_request)

    # Check coordinator.start_request was called via the mock_coordinator fixture
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
    sse_adapter.logger.info.assert_any_call(f"Request {request_id} accepted for processing.")

@pytest.mark.asyncio
async def test_sse_adapter_handle_message_request_invalid_json(sse_adapter, mock_request):
    mock_request.json = AsyncMock(side_effect=json.JSONDecodeError("Expecting value", "doc", 0))
    response = await sse_adapter.handle_message_request(mock_request)
    assert isinstance(response, JSONResponse)
    assert response.status_code == 400
    assert json.loads(response.body) == {"success": False, "error": "Invalid JSON payload"}
    sse_adapter.logger.error.assert_called_once_with(f"Invalid JSON received in message request body: {json.JSONDecodeError('Expecting value', 'doc', 0)}")

@pytest.mark.asyncio
async def test_sse_adapter_handle_message_request_missing_name(sse_adapter, mock_request, mock_coordinator):
    request_data = {"parameters": {}} # Missing 'name' and 'request_id'
    mock_request.json = AsyncMock(return_value=request_data)
    response = await sse_adapter.handle_message_request(mock_request)
    assert isinstance(response, JSONResponse)
    assert response.status_code == 400

    # Check logger call (using ANY for generated ID)
    sse_adapter.logger.error.assert_called_once_with(
        f"Missing or invalid 'name' field (operation name) in request {ANY}."
    )
    response_body = json.loads(response.body)
    assert response_body["success"] is False
    assert response_body["error"].startswith("Missing or invalid 'name' field")

    # Check fail_request was called via the mock_coordinator fixture
    # The adapter now calls fail_request with specific arguments
    mock_coordinator.fail_request.assert_awaited_once_with(
        request_id=ANY,
        operation_name=None, # Name was missing
        error="Invalid request", # Error category used by adapter
        error_details=ANY, # Detailed message
        originating_transport_id=sse_adapter.transport_id,
        request_details={}, # Parameters from request
    )


@pytest.mark.asyncio
async def test_sse_adapter_handle_message_request_security_fail(sse_adapter, mock_request, mock_coordinator):
    """Test handling message where security validation (simulated via start_request failure) fails."""
    request_id = "req_sec_fail_handler"
    operation_name = "op_sec_fail"
    request_data = {"request_id": request_id, "name": operation_name, "auth_token": "bad"}
    mock_request.json = AsyncMock(return_value=request_data)
    security_error = ValueError("Token expired") # Simulate security failure

    # Mock the coordinator's start_request (via fixture) to raise the security error
    mock_coordinator.start_request.side_effect = security_error

    response = await sse_adapter.handle_message_request(mock_request)

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

    # Check fail_request was NOT called by the adapter in this case
    # The adapter returns the error response directly when start_request fails this way.
    # The coordinator's start_request itself *would* call fail_request internally if the
    # validate_request_security call failed there, but here we mock start_request itself failing.
    mock_coordinator.fail_request.assert_not_called()


@pytest.mark.asyncio
async def test_sse_adapter_handle_message_request_coordinator_unavailable(sse_adapter, mock_request):
    """Test handling message when coordinator is None (using mocked adapter)."""
    sse_adapter._coordinator = None # Simulate coordinator not being set
    request_data = {"request_id": "req_no_coord", "name": "op"}
    mock_request.json = AsyncMock(return_value=request_data)

    response = await sse_adapter.handle_message_request(mock_request)

    assert isinstance(response, JSONResponse)
    assert response.status_code == 503
    response_body = json.loads(response.body)
    expected_error = f"Application coordinator not available for transport {sse_adapter.transport_id}."
    assert response_body == {"success": False, "error": expected_error}
    sse_adapter.logger.error.assert_called_once_with(expected_error)


# --- Test serve_sse ---

@pytest.mark.asyncio
@patch('aider_mcp_server.sse_server.uvicorn.Server')
@patch('aider_mcp_server.sse_server.uvicorn.Config')
@patch('aider_mcp_server.sse_server.FastAPI')
# Patch ApplicationCoordinator.getInstance where it's used in sse_server
@patch('aider_mcp_server.sse_server.ApplicationCoordinator.getInstance')
# Patch logger used by sse_server module
@patch('aider_mcp_server.sse_server.get_logger_func')
# Patch logger used by the adapter module (as serve_sse creates a real adapter)
@patch('aider_mcp_server.transport_adapter.get_logger_func')
@patch('aider_mcp_server.sse_server.asyncio.get_running_loop')
@patch('aider_mcp_server.sse_server.signal.signal')
@patch('aider_mcp_server.sse_server.signal.getsignal')
@patch('aider_mcp_server.sse_server._create_shutdown_task_wrapper')
@patch('aider_mcp_server.sse_server.is_git_repository')
@patch('aider_mcp_server.sse_server.handle_shutdown_signal', new_callable=AsyncMock)
@patch('aider_mcp_server.sse_server.asyncio.Event')
async def test_serve_sse_startup_and_run(
    mock_asyncio_event, # Mock for asyncio.Event
    mock_handle_shutdown_signal, # Mock for the async handler func
    mock_is_git_repo,
    mock_create_shutdown_wrapper,
    mock_getsignal,
    mock_signal_signal,
    mock_get_loop,
    mock_adapter_get_logger, # Mock for adapter's logger
    mock_server_get_logger, # Mock for sse_server's logger
    mock_get_coordinator_instance, # Mock for ApplicationCoordinator.getInstance
    mock_fastapi_cls, mock_uvicorn_config_cls, mock_uvicorn_server_cls,
    mock_uvicorn_server, mock_uvicorn_config): # Fixtures for instances
    """Test the serve_sse function startup sequence and running the server."""

    # --- Mock Setup ---
    # Mock loggers
    mock_server_logger = MagicMock(spec=LoggerProtocol)
    mock_server_get_logger.return_value = mock_server_logger
    mock_adapter_logger = MagicMock(spec=LoggerProtocol)
    mock_adapter_get_logger.return_value = mock_adapter_logger

    # Mock Coordinator Instance (returned by the patched getInstance)
    mock_coordinator_instance = MagicMock(spec=ApplicationCoordinator)
    mock_coordinator_instance.register_handler = MagicMock()
    mock_coordinator_instance.shutdown = AsyncMock()
    mock_coordinator_instance.wait_for_initialization = AsyncMock()
    mock_coordinator_instance.__aenter__ = AsyncMock(return_value=mock_coordinator_instance)
    mock_coordinator_instance.__aexit__ = AsyncMock()
    mock_get_coordinator_instance.return_value = mock_coordinator_instance

    # Mock FastAPI app
    mock_fastapi_app = MagicMock()
    mock_fastapi_cls.return_value = mock_fastapi_app
    # Mock endpoint registration decorators
    mock_fastapi_app.get.return_value = lambda func: func # Return the function itself
    mock_fastapi_app.post.return_value = lambda func: func

    # Mock Uvicorn classes to return the fixture instances
    mock_uvicorn_config_cls.return_value = mock_uvicorn_config
    mock_uvicorn_server_cls.return_value = mock_uvicorn_server

    # Mock loop and signal handling setup
    mock_loop = MagicMock()
    mock_loop.add_signal_handler = MagicMock()
    mock_get_loop.return_value = mock_loop
    mock_getsignal.return_value = None
    mock_sync_sigint_handler = MagicMock(name="sync_sigint_handler")
    mock_sync_sigterm_handler = MagicMock(name="sync_sigterm_handler")
    def wrapper_side_effect(sig, handler):
        if sig == signal.SIGINT: return mock_sync_sigint_handler
        if sig == signal.SIGTERM: return mock_sync_sigterm_handler
        return MagicMock()
    mock_create_shutdown_wrapper.side_effect = wrapper_side_effect

    # Mock asyncio.Event behavior for shutdown control
    mock_event_instance = MagicMock(spec=asyncio.Event)
    mock_event_instance.is_set.return_value = False
    async def wait_side_effect():
        await asyncio.sleep(0.01) # Simulate running
        mock_event_instance.is_set.return_value = True # Simulate signal received
    mock_event_instance.wait = AsyncMock(side_effect=wait_side_effect)
    mock_asyncio_event.return_value = mock_event_instance

    # Simulate server.serve() running until event.wait() finishes
    async def serve_side_effect():
        await mock_event_instance.wait()
        mock_uvicorn_server.started = False # Mark server as stopped
        return None
    mock_uvicorn_server.serve.side_effect = serve_side_effect

    mock_is_git_repo.return_value = (True, None)

    # --- Call the function ---
    host, port, editor_model, cwd, heartbeat = "127.0.0.1", 8888, "test_model", "/test/repo", 20.0
    await serve_sse(host, port, editor_model, cwd, heartbeat_interval=heartbeat)

    # --- Assertions ---
    mock_is_git_repo.assert_called_once_with(Path(cwd))
    mock_get_coordinator_instance.assert_called_once() # Check singleton access

    # Check coordinator context manager usage
    mock_coordinator_instance.__aenter__.assert_awaited_once()

    # Check handler registration *inside* the coordinator context
    mock_coordinator_instance.register_handler.assert_any_call("aider_ai_code", ANY, required_permission=Permissions.EXECUTE_AIDER)
    mock_coordinator_instance.register_handler.assert_any_call("list_models", ANY)

    # Check FastAPI app creation and endpoint registration
    mock_fastapi_cls.assert_called_once()
    # Check decorators were called (implies endpoints were defined)
    mock_fastapi_app.post.assert_any_call("/message", status_code=202)
    mock_fastapi_app.get.assert_any_call("/sse")

    # Check Uvicorn config and server creation
    mock_uvicorn_config_cls.assert_called_once_with(
        app=mock_fastapi_app,
        host=host,
        port=port,
        log_config=None, # Check log_config is None
        handle_signals=False # Check signals are handled manually
    )
    mock_uvicorn_server_cls.assert_called_once_with(mock_uvicorn_config)

    # Signal handling setup
    mock_get_loop.assert_called_once()
    assert mock_create_shutdown_wrapper.call_count == 2
    mock_create_shutdown_wrapper.assert_any_call(signal.SIGINT, ANY)
    mock_create_shutdown_wrapper.assert_any_call(signal.SIGTERM, ANY)
    # Check signal handlers were added using the *results* of the wrapper
    if mock_loop.add_signal_handler.called:
        mock_loop.add_signal_handler.assert_any_call(signal.SIGINT, mock_sync_sigint_handler)
        mock_loop.add_signal_handler.assert_any_call(signal.SIGTERM, mock_sync_sigterm_handler)
    else: # Fallback check
        mock_signal_signal.assert_any_call(signal.SIGINT, mock_sync_sigint_handler)
        mock_signal_signal.assert_any_call(signal.SIGTERM, mock_sync_sigterm_handler)

    # Server run and wait
    mock_uvicorn_server.serve.assert_awaited_once()
    mock_event_instance.wait.assert_awaited_once() # Check we waited for shutdown

    # Shutdown sequence
    mock_uvicorn_server.shutdown.assert_awaited_once() # Check server shutdown called
    mock_coordinator_instance.__aexit__.assert_awaited_once() # Check coordinator context exit

    # Check that the actual handle_shutdown_signal function (mocked) was NOT called directly
    # because the signal mechanism calls the *wrapper* created by _create_shutdown_task_wrapper
    mock_handle_shutdown_signal.assert_not_called()


@pytest.mark.asyncio
async def test_handle_shutdown_signal():
    """Test the handle_shutdown_signal function correctly sets the event."""
    mock_event = AsyncMock(spec=asyncio.Event)
    mock_event.is_set.return_value = False # Ensure event starts as not set
    mock_logger = MagicMock(spec=LoggerProtocol)

    # Patch logger within the function's scope
    with patch('aider_mcp_server.sse_server.logger', mock_logger):
        # Call the actual async function, passing only signal and event
        await handle_shutdown_signal(signal.SIGINT, mock_event) # Removed coordinator arg

    # --- Assertions ---
    # Check ONLY the event was set
    mock_event.set.assert_called_once()

    # Check logging
    mock_logger.warning.assert_any_call("Received signal SIGINT. Initiating graceful shutdown...")
    mock_logger.info.assert_any_call("Signaling main loop to shut down via event...")


def test_create_shutdown_task_wrapper():
    """Test the _create_shutdown_task_wrapper helper."""
    mock_async_handler = AsyncMock()
    sig = signal.SIGINT
    arg1 = signal.SIGINT
    arg2 = None # Frame

    sync_wrapper = _create_shutdown_task_wrapper(sig, mock_async_handler)

    with patch('aider_mcp_server.sse_server.asyncio.create_task') as mock_create_task:
        sync_wrapper(arg1, arg2) # Call sync wrapper

        # Assert create_task was called with a coroutine
        mock_create_task.assert_called_once()
        call_args, call_kwargs = mock_create_task.call_args
        coro = call_args[0]
        assert asyncio.iscoroutine(coro)

        # Run the coroutine to check args passed to the original async handler
        async def run_coro(): await coro
        asyncio.run(run_coro())

        # Assert the original async handler was awaited with the signal enum/object
        # and the extra args passed to the sync wrapper.
        mock_async_handler.assert_awaited_once_with(sig, arg1, arg2)

