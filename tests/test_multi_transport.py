import asyncio
import logging
import uuid
from typing import Any, Dict, List, Set, Tuple
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

# Use absolute imports from the package root
from aider_mcp_server.atoms.event_types import EventTypes
# ProgressReporter might be used internally by handlers, ensure import works
from aider_mcp_server.progress_reporter import ProgressReporter
from aider_mcp_server.security import SecurityContext, Permissions # Import Permissions
from aider_mcp_server.transport_adapter import AbstractTransportAdapter, LoggerProtocol # Import LoggerProtocol
from aider_mcp_server.transport_coordinator import ApplicationCoordinator

# Configure logging for tests if needed
# logger = logging.getLogger(__name__)

# --- Mock Transport Implementation ---

class MockTransportAdapter(AbstractTransportAdapter):
    """A mock transport adapter for testing."""
    def __init__(
        self,
        transport_id: str,
        transport_type: str,
        coordinator: ApplicationCoordinator,
        capabilities: Set[EventTypes],
    ):
        # Call super().__init__ correctly
        super().__init__(
            transport_id=transport_id,
            transport_type=transport_type,
            coordinator=coordinator,
            heartbeat_interval=None, # Disable heartbeat for mocks
        )
        self._sent_events: List[Tuple[EventTypes, Dict[str, Any]]] = []
        self._capabilities = capabilities
        # Ensure logger is initialized by super() before use
        self.logger.debug(f"MockTransportAdapter {self.transport_id} created.")

    async def send_event(self, event: EventTypes, data: Dict[str, Any]) -> None:
        """Captures calls to send_event."""
        self._sent_events.append((event, data))
        # self.logger.debug(f"MockTransport {self.transport_id} captured event: {event.value}, Data: {data}")

    def get_capabilities(self) -> Set[EventTypes]:
        """Returns the predefined capabilities for this mock transport."""
        return self._capabilities

    def validate_request_security(
        self, request_data: Dict[str, Any]
    ) -> SecurityContext:
        """Returns a default permissive SecurityContext for testing."""
        self.logger.debug(f"MockTransport {self.transport_id} validating security (returning default permissive context).")
        # Return a context with some basic permissions for testing handlers
        return SecurityContext(user_id=f"mock_user_{self.transport_id}", permissions={Permissions.EXECUTE_AIDER, Permissions.LIST_MODELS}, transport_id=self.transport_id)

    # REMOVED initialize override - let the real initialize run for registration/subscription
    # async def initialize(self):
    #     # Don't call super().initialize()
    #     self.logger.debug(f"MockTransport {self.transport_id} initialized (mock override).")

    # REMOVED shutdown override - let the real shutdown run for unregistration
    # async def shutdown(self):
    #     # Don't call super().shutdown()
    #     self.logger.debug(f"MockTransport {self.transport_id} shut down (mock override).")

    def get_sent_events(self) -> List[Tuple[EventTypes, Dict[str, Any]]]:
        return self._sent_events

    def clear_sent_events(self):
        self._sent_events = []


# --- Mock Handlers ---
# (Keep mock handlers as they are, ensuring they accept SecurityContext)
async def mock_simple_handler(request_id: str, transport_id: str, parameters: Dict[str, Any], security_context: SecurityContext) -> Dict[str, Any]:
    await asyncio.sleep(0.01)
    return {"success": True, "message": "Operation completed", "params_received": parameters, "request_id": request_id, "context_user": security_context.user_id}

async def mock_progress_handler(request_id: str, transport_id: str, parameters: Dict[str, Any], security_context: SecurityContext) -> Dict[str, Any]:
    coordinator = ApplicationCoordinator.getInstance()
    # Use get_progress_reporter for context management
    async with coordinator.get_progress_reporter(request_id, "progress_op") as reporter:
        await reporter.update("Step 1/3: Processing...")
        await asyncio.sleep(0.01)
        await reporter.update("Step 2/3: Working...", details={"progress": 0.5})
        await asyncio.sleep(0.01)
        await reporter.update("Step 3/3: Finalizing...", details={"progress": 1.0})
        await asyncio.sleep(0.01)
    return {"success": True, "message": "Progress operation completed", "request_id": request_id}

async def mock_error_handler(request_id: str, transport_id: str, parameters: Dict[str, Any], security_context: SecurityContext) -> Dict[str, Any]:
    await asyncio.sleep(0.01)
    raise ValueError("Something went wrong in the handler")

async def mock_long_running_handler(request_id: str, transport_id: str, parameters: Dict[str, Any], security_context: SecurityContext) -> Dict[str, Any]:
    duration = parameters.get("duration", 0.1)
    await asyncio.sleep(duration)
    return {"success": True, "message": f"Slept for {duration} seconds", "request_id": request_id}

async def mock_progress_reporter_handler(request_id: str, transport_id: str, parameters: Dict[str, Any], security_context: SecurityContext) -> Dict[str, Any]:
    coordinator = ApplicationCoordinator.getInstance()
    # Use get_progress_reporter for context management
    async with coordinator.get_progress_reporter(request_id, "mock_progress_reporter_handler") as reporter:
        await reporter.update("Using reporter: Step 1...")
        await asyncio.sleep(0.01)
        await reporter.update("Using reporter: Step 2...", details={"stage": 2})
        await asyncio.sleep(0.01)
    return {"success": True, "message": "ProgressReporter operation completed", "request_id": request_id}


# --- Fixtures ---

@pytest_asyncio.fixture
async def coordinator():
    """Provides a fresh ApplicationCoordinator instance for each test."""
    # Reset singleton before getting instance
    ApplicationCoordinator._instance = None
    ApplicationCoordinator._initialized = False # Reset initialization flag too
    coord = ApplicationCoordinator.getInstance()
    # Mock the logger used *within* the coordinator instance
    coord_logger_mock = MagicMock(spec=LoggerProtocol) # Use LoggerProtocol spec
    with patch('aider_mcp_server.transport_coordinator.logger', coord_logger_mock):
        # Ensure coordinator is initialized (simulates startup)
        await coord.wait_for_initialization()
        yield coord
        # Clean up by calling the coordinator's shutdown method
        # This is implicitly called by __aexit__ if using async with,
        # but calling explicitly ensures cleanup even if test fails before __aexit__.
        if not coord._shutdown_event.is_set():
            await coord.shutdown()
        # Reset singleton state after test
        ApplicationCoordinator._instance = None
        ApplicationCoordinator._initialized = False


@pytest_asyncio.fixture
async def mock_sse_transport(coordinator):
    """Provides a mock SSE transport adapter."""
    capabilities = {EventTypes.STATUS, EventTypes.PROGRESS, EventTypes.TOOL_RESULT, EventTypes.HEARTBEAT}
    # Patch the logger used by the *adapter* instance
    adapter_logger_mock = MagicMock(spec=LoggerProtocol)
    with patch('aider_mcp_server.transport_adapter.get_logger_func', return_value=adapter_logger_mock):
        transport = MockTransportAdapter("sse-1", "sse", coordinator, capabilities)
        # Attach the mock logger to the instance for potential assertions if needed
        transport.logger = adapter_logger_mock
        yield transport


@pytest_asyncio.fixture
async def mock_stdio_transport(coordinator):
    """Provides a mock StdIO transport adapter."""
    capabilities = {EventTypes.STATUS, EventTypes.PROGRESS, EventTypes.TOOL_RESULT} # No heartbeat typically
    # Patch the logger used by the *adapter* instance
    adapter_logger_mock = MagicMock(spec=LoggerProtocol)
    with patch('aider_mcp_server.transport_adapter.get_logger_func', return_value=adapter_logger_mock):
        transport = MockTransportAdapter("stdio-1", "stdio", coordinator, capabilities)
        # Attach the mock logger to the instance
        transport.logger = adapter_logger_mock
        yield transport


@pytest_asyncio.fixture
async def registered_transports(coordinator, mock_sse_transport, mock_stdio_transport):
    """Registers mock transports with the coordinator and initializes them."""
    # Use the real register_transport method
    coordinator.register_transport(mock_sse_transport.transport_id, mock_sse_transport)
    coordinator.register_transport(mock_stdio_transport.transport_id, mock_stdio_transport)

    # Use the real initialize method (which subscribes based on capabilities)
    # This is now handled automatically by register_transport updating subscriptions
    # await mock_sse_transport.initialize() # No longer needed if register handles it
    # await mock_stdio_transport.initialize() # No longer needed if register handles it

    yield {"sse": mock_sse_transport, "stdio": mock_stdio_transport}
    # Unregistration is handled by coordinator shutdown


@pytest_asyncio.fixture
async def registered_handlers(coordinator):
    """Registers mock operation handlers with the coordinator."""
    coordinator.register_handler("simple_op", mock_simple_handler)
    coordinator.register_handler("progress_op", mock_progress_handler)
    coordinator.register_handler("error_op", mock_error_handler)
    coordinator.register_handler("long_op", mock_long_running_handler)
    coordinator.register_handler("reporter_op", mock_progress_reporter_handler)
    yield


@pytest.fixture
def mock_uuid4():
    """Fixture to mock uuid.uuid4() and return predictable UUIDs."""
    test_uuid_counter = 0
    def mock_func():
        nonlocal test_uuid_counter
        test_uuid_counter += 1
        base_uuid = uuid.UUID("12345678-1234-5678-1234-000000000000")
        return uuid.UUID(int=base_uuid.int + test_uuid_counter)

    # Patch uuid4 where it's used (transport_coordinator and potentially elsewhere)
    # Patching it globally might be simpler if multiple modules use it.
    with patch("uuid.uuid4", side_effect=mock_func) as mock_uuid:
        yield mock_uuid


# --- Helper for asserting events ---
# (Keep compare_dicts_with_any and assert_events_sent as they are)
def compare_dicts_with_any(sent_data: Dict[str, Any], expected_data: Dict[str, Any]) -> bool:
    if expected_data is ANY: return True
    if not isinstance(sent_data, dict) or not isinstance(expected_data, dict): return sent_data == expected_data
    # Check if all expected keys are present and match
    for key, expected_value in expected_data.items():
        if key not in sent_data:
            print(f"Missing key: {key} in sent data {sent_data}")
            return False
        sent_value = sent_data[key]
        if expected_value is ANY: continue
        elif isinstance(expected_value, dict) and isinstance(sent_value, dict):
            if not compare_dicts_with_any(sent_value, expected_value): return False
        elif isinstance(expected_value, list) and isinstance(sent_value, list):
             # Basic list comparison (order matters)
             if len(expected_value) != len(sent_value):
                  print(f"List length mismatch for key '{key}': Sent={len(sent_value)}, Expected={len(expected_value)}")
                  return False
             for i, item in enumerate(expected_value):
                  if isinstance(item, dict) and isinstance(sent_value[i], dict):
                       if not compare_dicts_with_any(sent_value[i], item): return False
                  elif item != sent_value[i]:
                       print(f"List item mismatch for key '{key}' at index {i}: Sent='{sent_value[i]}', Expected='{item}'")
                       return False
        elif sent_value != expected_value:
            print(f"Value mismatch for key '{key}': Sent='{sent_value}' ({type(sent_value)}), Expected='{expected_value}' ({type(expected_value)})")
            return False
    # Optionally, check if sent_data has extra keys (strict comparison)
    # if len(sent_data) != len(expected_data):
    #     print(f"Sent data has extra keys: {sent_data.keys() - expected_data.keys()}")
    #     return False
    return True

def assert_events_sent(mock_transport: MockTransportAdapter, expected_events: List[Tuple[EventTypes, Dict[str, Any]]]):
    sent_events = mock_transport.get_sent_events()
    assert len(sent_events) == len(expected_events), (
        f"Expected {len(expected_events)} events, but got {len(sent_events)} for {mock_transport.transport_id}\n"
        f"Sent: {sent_events}\nExpected: {expected_events}"
    )
    for i, (expected_event_type, expected_data) in enumerate(expected_events):
        assert i < len(sent_events), f"Missing event at index {i}"
        sent_event_type, sent_data = sent_events[i]
        assert sent_event_type == expected_event_type, (
            f"Event type enum mismatch at index {i}. Expected {expected_event_type.value}, got {sent_event_type.value}"
        )
        assert compare_dicts_with_any(sent_data, expected_data), (
            f"Event data mismatch at index {i}.\nSent: {sent_data}\nExpected: {expected_data}"
        )


# --- Test Cases ---

@pytest.mark.asyncio
async def test_transports_registered(coordinator, registered_transports):
    """Test that transports are correctly registered and subscribed."""
    sse_transport = registered_transports["sse"]
    stdio_transport = registered_transports["stdio"]

    # Check registry
    with coordinator._transports_lock:
        assert sse_transport.transport_id in coordinator._transports
        assert stdio_transport.transport_id in coordinator._transports
        assert coordinator._transports[sse_transport.transport_id] is sse_transport
        assert coordinator._transports[stdio_transport.transport_id] is stdio_transport

    # Check capabilities stored
    with coordinator._transport_capabilities_lock:
        assert coordinator._transport_capabilities.get(sse_transport.transport_id) == sse_transport.get_capabilities()
        assert coordinator._transport_capabilities.get(stdio_transport.transport_id) == stdio_transport.get_capabilities()

    # Check subscriptions match capabilities after registration (default behavior)
    with coordinator._transport_subscriptions_lock:
        assert coordinator._transport_subscriptions.get(sse_transport.transport_id) == sse_transport.get_capabilities()
        assert coordinator._transport_subscriptions.get(stdio_transport.transport_id) == stdio_transport.get_capabilities()


@pytest.mark.asyncio
async def test_simple_request_routing(coordinator, registered_transports, registered_handlers, mock_uuid4):
    """Test routing a simple request to the correct handler and getting a result."""
    sse_transport = registered_transports["sse"]
    stdio_transport = registered_transports["stdio"]
    # Use the mocked UUID generator's first predictable UUID
    request_id = str(uuid.UUID("12345678-1234-5678-1234-000000000001"))
    operation_name = "simple_op"
    params = {"data": 123}
    # Simulate request data structure (might include auth, etc., handled by validate_request_security mock)
    request_data = {"request_id": request_id, "name": operation_name, "parameters": params, "auth_token": "mock_token"}

    # Start request via SSE transport
    await coordinator.start_request(
        request_id=request_id,
        transport_id=sse_transport.transport_id,
        operation_name=operation_name,
        request_data=request_data, # Pass full data
    )

    # Wait for handler task to complete (needs slight delay)
    await asyncio.sleep(0.05)

    # Assert events sent to SSE transport
    assert_events_sent(sse_transport, [
        (EventTypes.STATUS, {"type": "status", "request_id": request_id, "operation": operation_name, "status": "starting", "message": ANY, "details": {"parameters": params}}),
        (EventTypes.TOOL_RESULT, {"type": "tool_result", "request_id": request_id, "tool_name": operation_name, "result": {"success": True, "message": "Operation completed", "params_received": params, "request_id": request_id, "context_user": f"mock_user_{sse_transport.transport_id}"}}),
    ])

    # Assert no events sent to Stdio transport
    assert_events_sent(stdio_transport, [])

    # Assert request state is cleaned up
    with coordinator._active_requests_lock:
        assert request_id not in coordinator._active_requests


@pytest.mark.asyncio
async def test_progress_request_routing(coordinator, registered_transports, registered_handlers, mock_uuid4):
    """Test routing a request that reports progress using ProgressReporter."""
    stdio_transport = registered_transports["stdio"]
    sse_transport = registered_transports["sse"]
    request_id = str(uuid.UUID("12345678-1234-5678-1234-000000000001"))
    operation_name = "progress_op"
    params = {"input": "abc"}
    request_data = {"request_id": request_id, "name": operation_name, "parameters": params, "auth_token": "mock_token"}

    # Start request via Stdio transport
    await coordinator.start_request(
        request_id=request_id,
        transport_id=stdio_transport.transport_id,
        operation_name=operation_name,
        request_data=request_data,
    )

    # Wait for handler task to complete
    await asyncio.sleep(0.1) # Allow time for progress updates

    # Assert events sent to Stdio transport
    assert_events_sent(stdio_transport, [
        # 1. Initial status from start_request
        (EventTypes.STATUS, {"type": "status", "request_id": request_id, "operation": operation_name, "status": "starting", "message": ANY, "details": {"parameters": params}}),
        # 2. ProgressReporter sends 'starting' on __aenter__
        (EventTypes.PROGRESS, {"type": "progress", "request_id": request_id, "operation": operation_name, "status": "starting", "message": f"Operation '{operation_name}' started.", "details": {"parameters": params}}),
        # 3. First update from handler via reporter
        (EventTypes.PROGRESS, {"type": "progress", "request_id": request_id, "operation": operation_name, "status": "in_progress", "message": "Step 1/3: Processing...", "details": {"parameters": params}}),
        # 4. Second update from handler via reporter
        (EventTypes.PROGRESS, {"type": "progress", "request_id": request_id, "operation": operation_name, "status": "in_progress", "message": "Step 2/3: Working...", "details": {"parameters": params, "progress": 0.5}}),
        # 5. Third update from handler via reporter
        (EventTypes.PROGRESS, {"type": "progress", "request_id": request_id, "operation": operation_name, "status": "in_progress", "message": "Step 3/3: Finalizing...", "details": {"parameters": params, "progress": 1.0}}),
        # 6. ProgressReporter sends 'completed' on __aexit__
        (EventTypes.PROGRESS, {"type": "progress", "request_id": request_id, "operation": operation_name, "status": "completed", "message": "Operation completed successfully.", "details": {"parameters": params}}),
        # 7. Final result from the handler itself
        (EventTypes.TOOL_RESULT, {"type": "tool_result", "request_id": request_id, "tool_name": operation_name, "result": {"success": True, "message": "Progress operation completed", "request_id": request_id}}),
    ])

    # Assert events also sent to SSE transport (since it's subscribed to PROGRESS and TOOL_RESULT)
    # Note: SSE transport did not originate the request, so it gets the same progress/result events.
    assert_events_sent(sse_transport, [
        # SSE is subscribed, so it should receive these too
        (EventTypes.PROGRESS, {"type": "progress", "request_id": request_id, "operation": operation_name, "status": "starting", "message": f"Operation '{operation_name}' started.", "details": {"parameters": params}}),
        (EventTypes.PROGRESS, {"type": "progress", "request_id": request_id, "operation": operation_name, "status": "in_progress", "message": "Step 1/3: Processing...", "details": {"parameters": params}}),
        (EventTypes.PROGRESS, {"type": "progress", "request_id": request_id, "operation": operation_name, "status": "in_progress", "message": "Step 2/3: Working...", "details": {"parameters": params, "progress": 0.5}}),
        (EventTypes.PROGRESS, {"type": "progress", "request_id": request_id, "operation": operation_name, "status": "in_progress", "message": "Step 3/3: Finalizing...", "details": {"parameters": params, "progress": 1.0}}),
        (EventTypes.PROGRESS, {"type": "progress", "request_id": request_id, "operation": operation_name, "status": "completed", "message": "Operation completed successfully.", "details": {"parameters": params}}),
        (EventTypes.TOOL_RESULT, {"type": "tool_result", "request_id": request_id, "tool_name": operation_name, "result": {"success": True, "message": "Progress operation completed", "request_id": request_id}}),
    ])


    # Assert request state is cleaned up
    with coordinator._active_requests_lock:
        assert request_id not in coordinator._active_requests


@pytest.mark.asyncio
async def test_error_request_routing(coordinator, registered_transports, registered_handlers, mock_uuid4):
    """Test routing a request that results in an error."""
    sse_transport = registered_transports["sse"]
    stdio_transport = registered_transports["stdio"] # Also subscribed to TOOL_RESULT
    request_id = str(uuid.UUID("12345678-1234-5678-1234-000000000001"))
    operation_name = "error_op"
    params = {}
    request_data = {"request_id": request_id, "name": operation_name, "parameters": params, "auth_token": "mock_token"}

    await coordinator.start_request(
        request_id=request_id,
        transport_id=sse_transport.transport_id,
        operation_name=operation_name,
        request_data=request_data,
    )

    await asyncio.sleep(0.05) # Allow handler to run and fail

    expected_error_result = (EventTypes.TOOL_RESULT, {"type": "tool_result", "request_id": request_id, "tool_name": operation_name, "result": {"success": False, "error": "Operation failed: ValueError", "details": "Something went wrong in the handler"}})

    # Assert events sent to originating transport (SSE)
    assert_events_sent(sse_transport, [
        (EventTypes.STATUS, {"type": "status", "request_id": request_id, "operation": operation_name, "status": "starting", "message": ANY, "details": {"parameters": params}}),
        expected_error_result,
    ])

    # Assert error result also sent to other subscribed transport (Stdio)
    assert_events_sent(stdio_transport, [
        expected_error_result,
    ])


    with coordinator._active_requests_lock:
        assert request_id not in coordinator._active_requests


@pytest.mark.asyncio
async def test_unknown_operation(coordinator, registered_transports, mock_uuid4):
    """Test requesting an operation without a registered handler."""
    sse_transport = registered_transports["sse"]
    stdio_transport = registered_transports["stdio"] # Also subscribed to TOOL_RESULT
    request_id = str(uuid.UUID("12345678-1234-5678-1234-000000000001"))
    operation_name = "non_existent_op"
    params = {}
    request_data = {"request_id": request_id, "name": operation_name, "parameters": params, "auth_token": "mock_token"}

    await coordinator.start_request(
        request_id=request_id,
        transport_id=sse_transport.transport_id,
        operation_name=operation_name,
        request_data=request_data,
    )

    await asyncio.sleep(0.01) # Allow fail_request to process within start_request

    expected_error_result = (EventTypes.TOOL_RESULT, {"type": "tool_result", "request_id": request_id, "tool_name": operation_name, "result": {"success": False, "error": "Operation not supported", "details": f"No handler registered for operation '{operation_name}'."}})

    # Assert events sent ONLY to originating transport (SSE) because start_request fails early
    # The "starting" status is sent *before* the handler check now.
    assert_events_sent(sse_transport, [
         (EventTypes.STATUS, {"type": "status", "request_id": request_id, "operation": operation_name, "status": "starting", "message": ANY, "details": {"parameters": params}}),
         expected_error_result,
    ])

    # Assert NO events sent to other transport (Stdio)
    assert_events_sent(stdio_transport, [])

    with coordinator._active_requests_lock:
        assert request_id not in coordinator._active_requests


@pytest.mark.asyncio
async def test_permission_denied(coordinator, registered_transports, registered_handlers, mock_uuid4):
    """Test requesting an operation without required permissions."""
    sse_transport = registered_transports["sse"]
    stdio_transport = registered_transports["stdio"] # Also subscribed to TOOL_RESULT

    # Mock the security validation for this specific transport to return a context *without* execute_aider
    # Ensure the mock returns a SecurityContext object
    limited_context = SecurityContext(user_id="limited_user", permissions={Permissions.LIST_MODELS}, transport_id=sse_transport.transport_id)
    sse_transport.validate_request_security = MagicMock(return_value=limited_context)

    # Register a handler that requires execute_aider permission
    async def protected_op_handler(request_id: str, transport_id: str, parameters: Dict[str, Any], security_context: SecurityContext) -> Dict[str, Any]:
        # This should not be called
        pytest.fail("Protected handler should not be executed")
        return {"success": True, "message": "Protected operation executed"} # pragma: no cover
    coordinator.register_handler("protected_op", protected_op_handler, required_permission=Permissions.EXECUTE_AIDER)

    request_id = str(uuid.UUID("12345678-1234-5678-1234-000000000001"))
    operation_name = "protected_op"
    params = {}
    request_data = {"request_id": request_id, "name": operation_name, "parameters": params, "auth_token": "mock_token_limited"} # Token content doesn't matter due to mock

    await coordinator.start_request(
        request_id=request_id,
        transport_id=sse_transport.transport_id,
        operation_name=operation_name,
        request_data=request_data,
    )

    await asyncio.sleep(0.01) # Allow fail_request to process within start_request

    expected_error_result = (EventTypes.TOOL_RESULT, {"type": "tool_result", "request_id": request_id, "tool_name": operation_name, "result": {"success": False, "error": "Permission denied", "details": f"User does not have the required permission '{Permissions.EXECUTE_AIDER.name}' for operation '{operation_name}'."}})

    # Assert error event sent ONLY to originating transport (SSE)
    # start_request now detects permission failure *after* security validation but *before* running the handler or sending 'starting' status.
    # It sends the TOOL_RESULT directly back.
    assert_events_sent(sse_transport, [
        expected_error_result,
    ])

    # Assert NO events sent to other transport (Stdio)
    assert_events_sent(stdio_transport, [])

    with coordinator._active_requests_lock:
        assert request_id not in coordinator._active_requests


@pytest.mark.asyncio
async def test_reporter_handler_routing(coordinator, registered_transports, registered_handlers, mock_uuid4):
    """Test routing a request that uses ProgressReporter."""
    sse_transport = registered_transports["sse"]
    stdio_transport = registered_transports["stdio"] # Also subscribed
    request_id = str(uuid.UUID("12345678-1234-5678-1234-000000000001"))
    operation_name = "reporter_op"
    params = {}
    request_data = {"request_id": request_id, "name": operation_name, "parameters": params, "auth_token": "mock_token"}

    await coordinator.start_request(
        request_id=request_id,
        transport_id=sse_transport.transport_id,
        operation_name=operation_name,
        request_data=request_data,
    )

    await asyncio.sleep(0.1) # Allow time for progress updates

    expected_events = [
        # 1. Initial status from start_request (sent to originating transport only initially, but broadcast later?)
        #    Correction: STATUS is not broadcast by default, only PROGRESS/TOOL_RESULT are typically broadcast.
        #    Let's assume STATUS is only sent to origin for now.
        # (EventTypes.STATUS, {"type": "status", "request_id": request_id, "operation": operation_name, "status": "starting", "message": ANY, "details": {"parameters": params}}), # Sent only to SSE

        # 2. ProgressReporter sends 'starting' via update on __aenter__ (broadcast)
        (EventTypes.PROGRESS, {"type": "progress", "request_id": request_id, "operation": operation_name, "status": "starting", "message": f"Operation '{operation_name}' started.", "details": {"parameters": params}}),
        # 3. First update from handler via reporter (broadcast)
        (EventTypes.PROGRESS, {"type": "progress", "request_id": request_id, "operation": operation_name, "status": "in_progress", "message": "Using reporter: Step 1...", "details": {"parameters": params}}),
        # 4. Second update from handler via reporter (broadcast)
        (EventTypes.PROGRESS, {"type": "progress", "request_id": request_id, "operation": operation_name, "status": "in_progress", "message": "Using reporter: Step 2...", "details": {"parameters": params, "stage": 2}}),
        # 5. ProgressReporter sends 'completed' via update on __aexit__ (broadcast)
        (EventTypes.PROGRESS, {"type": "progress", "request_id": request_id, "operation": operation_name, "status": "completed", "message": "Operation completed successfully.", "details": {"parameters": params}}),
        # 6. Final result from the handler itself (broadcast)
        (EventTypes.TOOL_RESULT, {"type": "tool_result", "request_id": request_id, "tool_name": operation_name, "result": {"success": True, "message": "ProgressReporter operation completed", "request_id": request_id}}),
    ]

    # Assert events sent to originating transport (SSE)
    # It should receive the initial STATUS plus all broadcast events
    assert_events_sent(sse_transport, [
         (EventTypes.STATUS, {"type": "status", "request_id": request_id, "operation": operation_name, "status": "starting", "message": ANY, "details": {"parameters": params}}),
         *expected_events # Add the broadcast events
    ])

    # Assert events sent to other subscribed transport (Stdio)
    # It should receive only the broadcast events (PROGRESS, TOOL_RESULT)
    assert_events_sent(stdio_transport, expected_events)


    with coordinator._active_requests_lock:
        assert request_id not in coordinator._active_requests
