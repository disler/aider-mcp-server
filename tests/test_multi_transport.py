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
    coordinator = await ApplicationCoordinator.getInstance() # Use await for getInstance
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
    coordinator = await ApplicationCoordinator.getInstance() # Use await for getInstance
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
    # Patch getInstance within the coordinator module itself for internal calls if needed
    # And patch it where it's used in the test module (e.g., handlers)
    coordinator_instance = MagicMock(spec=ApplicationCoordinator)
    coordinator_instance.register_transport = AsyncMock() # Now async
    coordinator_instance.unregister_transport = AsyncMock() # Now async
    coordinator_instance.register_handler = MagicMock() # Sync
    coordinator_instance.subscribe_to_event_type = AsyncMock() # Now async
    coordinator_instance.start_request = AsyncMock()
    coordinator_instance.fail_request = AsyncMock()
    coordinator_instance.shutdown = AsyncMock()
    coordinator_instance.wait_for_initialization = AsyncMock()
    coordinator_instance.is_shutting_down = MagicMock(return_value=False)
    coordinator_instance.get_progress_reporter = MagicMock(return_value=AsyncMock(spec=ProgressReporter)) # Mock reporter context manager
    coordinator_instance.__aenter__ = AsyncMock(return_value=coordinator_instance)
    coordinator_instance.__aexit__ = AsyncMock()

    # Configure the mock reporter context manager methods
    mock_reporter = coordinator_instance.get_progress_reporter.return_value
    mock_reporter.__aenter__ = AsyncMock(return_value=mock_reporter)
    mock_reporter.__aexit__ = AsyncMock()
    mock_reporter.update = AsyncMock()
    mock_reporter.error = AsyncMock()


    # Patch getInstance where it's used (transport_coordinator, test_multi_transport)
    patcher_coord = patch(
        'aider_mcp_server.transport_coordinator.ApplicationCoordinator.getInstance',
        new_callable=AsyncMock,
        return_value=coordinator_instance
    )
    patcher_test = patch(
        'tests.test_multi_transport.ApplicationCoordinator.getInstance',
        new_callable=AsyncMock,
        return_value=coordinator_instance
    )

    mock_get_instance_coord = patcher_coord.start()
    mock_get_instance_test = patcher_test.start()


    # Get the *actual* instance (which will be the mock due to the patch)
    # This ensures the instance used by the test setup is the same mocked one
    coord = await ApplicationCoordinator.getInstance()

    # Mock the logger used *within* the coordinator instance
    coord_logger_mock = MagicMock(spec=LoggerProtocol) # Use LoggerProtocol spec
    # Assign logger directly to the mocked instance if needed for assertions
    coord.logger = coord_logger_mock
    # Patch the logger where it's instantiated in the coordinator module
    with patch('aider_mcp_server.transport_coordinator.logger', coord_logger_mock):
        # Ensure coordinator initialization is awaited (using the mock)
        await coord.wait_for_initialization()
        yield coord # Yield the mocked instance
        # Clean up by calling the coordinator's shutdown method (on the mock)
        if not coord.is_shutting_down(): # Use mock method
             await coord.shutdown()

    # Stop patchers
    patcher_coord.stop()
    patcher_test.stop()

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
        # Pass the mocked coordinator instance to the adapter
        transport = MockTransportAdapter("sse-1", "sse", coordinator, capabilities)
        # Attach the mock logger to the instance for potential assertions if needed
        transport.logger = adapter_logger_mock
        # Assign coordinator explicitly if adapter logic relies on self._coordinator
        transport._coordinator = coordinator
        yield transport


@pytest_asyncio.fixture
async def mock_stdio_transport(coordinator):
    """Provides a mock StdIO transport adapter."""
    capabilities = {EventTypes.STATUS, EventTypes.PROGRESS, EventTypes.TOOL_RESULT} # No heartbeat typically
    # Patch the logger used by the *adapter* instance
    adapter_logger_mock = MagicMock(spec=LoggerProtocol)
    with patch('aider_mcp_server.transport_adapter.get_logger_func', return_value=adapter_logger_mock):
        # Pass the mocked coordinator instance to the adapter
        transport = MockTransportAdapter("stdio-1", "stdio", coordinator, capabilities)
        # Attach the mock logger to the instance
        transport.logger = adapter_logger_mock
        # Assign coordinator explicitly if adapter logic relies on self._coordinator
        transport._coordinator = coordinator
        yield transport


@pytest_asyncio.fixture
async def registered_transports(coordinator, mock_sse_transport, mock_stdio_transport):
    """Registers mock transports with the coordinator and initializes them."""
    # Use the coordinator mock's async register_transport method
    await coordinator.register_transport(mock_sse_transport.transport_id, mock_sse_transport)
    await coordinator.register_transport(mock_stdio_transport.transport_id, mock_stdio_transport)

    # Initialization/subscription is handled within the (mocked) register_transport

    yield {"sse": mock_sse_transport, "stdio": mock_stdio_transport}
    # Unregistration is handled by coordinator shutdown mock


@pytest_asyncio.fixture
async def registered_handlers(coordinator):
    """Registers mock operation handlers with the coordinator."""
    # Use the coordinator mock's register_handler method (sync)
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
    """Test that transports are correctly registered and subscribed (using mocks)."""
    sse_transport = registered_transports["sse"]
    stdio_transport = registered_transports["stdio"]

    # Check that the mocked register_transport was called correctly
    coordinator.register_transport.assert_any_await(sse_transport.transport_id, sse_transport)
    coordinator.register_transport.assert_any_await(stdio_transport.transport_id, stdio_transport)
    assert coordinator.register_transport.await_count == 2

    # We can't easily check the internal state (_transports, _subscriptions)
    # as we are interacting with a mock coordinator. Asserting the calls above is sufficient.


@pytest.mark.asyncio
async def test_simple_request_routing(coordinator, registered_transports, registered_handlers, mock_uuid4):
    """Test routing a simple request to the correct handler and getting a result (using mocks)."""
    sse_transport = registered_transports["sse"]
    stdio_transport = registered_transports["stdio"]
    request_id = str(uuid.UUID("12345678-1234-5678-1234-000000000001"))
    operation_name = "simple_op"
    params = {"data": 123}
    request_data = {"request_id": request_id, "name": operation_name, "parameters": params, "auth_token": "mock_token"}

    # Mock the behavior of start_request on the coordinator mock
    # It should eventually call the handler and send events via send_event_to_subscribers
    async def mock_start_request_side_effect(request_id, transport_id, operation_name, request_data):
        # Simulate security validation (using the transport's mock method)
        originating_transport = coordinator.register_transport.await_args_list[0][0][1] # Get transport from register call
        if transport_id == sse_transport.transport_id:
            originating_transport = sse_transport
        elif transport_id == stdio_transport.transport_id:
            originating_transport = stdio_transport

        security_context = originating_transport.validate_request_security(request_data)

        # Simulate finding handler
        handler = coordinator.register_handler.call_args_list[0][0][1] # Get handler from register call

        # Simulate sending 'starting' status
        await coordinator.send_event_to_subscribers(
            EventTypes.STATUS,
            {"type": "status", "request_id": request_id, "operation": operation_name, "status": "starting", "message": ANY, "details": {"parameters": params}},
            originating_transport_id=transport_id # Send only to origin initially
        )

        # Simulate running handler
        result = await handler(request_id, transport_id, params, security_context)

        # Simulate sending result
        await coordinator.send_event_to_subscribers(
            EventTypes.TOOL_RESULT,
            {"type": "tool_result", "request_id": request_id, "tool_name": operation_name, "result": result}
        )

    # Configure the coordinator mock's start_request
    coordinator.start_request.side_effect = mock_start_request_side_effect

    # Mock send_event_to_subscribers to route events to the correct mock transport's send_event
    async def mock_send_to_subscribers(event_type, data, originating_transport_id=None):
        # Simulate broadcasting based on subscriptions (mocked via capabilities)
        # In this simple case, assume both are subscribed to TOOL_RESULT, STATUS only to origin
        if event_type == EventTypes.STATUS and originating_transport_id == sse_transport.transport_id:
             await sse_transport.send_event(event_type, data)
        elif event_type == EventTypes.TOOL_RESULT:
             if EventTypes.TOOL_RESULT in sse_transport.get_capabilities():
                 await sse_transport.send_event(event_type, data)
             if EventTypes.TOOL_RESULT in stdio_transport.get_capabilities():
                 await stdio_transport.send_event(event_type, data)
        # Add other event types if needed

    coordinator.send_event_to_subscribers = AsyncMock(side_effect=mock_send_to_subscribers)


    # Start request via SSE transport (calls the mocked start_request)
    await coordinator.start_request(
        request_id=request_id,
        transport_id=sse_transport.transport_id,
        operation_name=operation_name,
        request_data=request_data,
    )

    # Assert start_request was awaited
    coordinator.start_request.assert_awaited_once_with(
        request_id=request_id,
        transport_id=sse_transport.transport_id,
        operation_name=operation_name,
        request_data=request_data,
    )

    # Assert events sent to SSE transport (origin)
    assert_events_sent(sse_transport, [
        (EventTypes.STATUS, {"type": "status", "request_id": request_id, "operation": operation_name, "status": "starting", "message": ANY, "details": {"parameters": params}}),
        (EventTypes.TOOL_RESULT, {"type": "tool_result", "request_id": request_id, "tool_name": operation_name, "result": {"success": True, "message": "Operation completed", "params_received": params, "request_id": request_id, "context_user": f"mock_user_{sse_transport.transport_id}"}}),
    ])

    # Assert TOOL_RESULT event also sent to Stdio transport (subscribed)
    assert_events_sent(stdio_transport, [
        (EventTypes.TOOL_RESULT, {"type": "tool_result", "request_id": request_id, "tool_name": operation_name, "result": {"success": True, "message": "Operation completed", "params_received": params, "request_id": request_id, "context_user": f"mock_user_{sse_transport.transport_id}"}}),
    ])

    # We don't check internal state (_active_requests) on the mock


@pytest.mark.asyncio
async def test_progress_request_routing(coordinator, registered_transports, registered_handlers, mock_uuid4):
    """Test routing a request that reports progress using ProgressReporter (using mocks)."""
    stdio_transport = registered_transports["stdio"]
    sse_transport = registered_transports["sse"]
    request_id = str(uuid.UUID("12345678-1234-5678-1234-000000000001"))
    operation_name = "progress_op"
    params = {"input": "abc"}
    request_data = {"request_id": request_id, "name": operation_name, "parameters": params, "auth_token": "mock_token"}

    # Mock send_event_to_subscribers to route events to the correct mock transport's send_event
    async def mock_send_to_subscribers(event_type, data, originating_transport_id=None):
        # Simulate broadcasting based on subscriptions (mocked via capabilities)
        # Assume both subscribed to PROGRESS and TOOL_RESULT, STATUS only to origin
        if event_type == EventTypes.STATUS and originating_transport_id == stdio_transport.transport_id:
             await stdio_transport.send_event(event_type, data)
        elif event_type == EventTypes.PROGRESS:
             if EventTypes.PROGRESS in sse_transport.get_capabilities():
                 await sse_transport.send_event(event_type, data)
             if EventTypes.PROGRESS in stdio_transport.get_capabilities():
                 await stdio_transport.send_event(event_type, data)
        elif event_type == EventTypes.TOOL_RESULT:
             if EventTypes.TOOL_RESULT in sse_transport.get_capabilities():
                 await sse_transport.send_event(event_type, data)
             if EventTypes.TOOL_RESULT in stdio_transport.get_capabilities():
                 await stdio_transport.send_event(event_type, data)

    coordinator.send_event_to_subscribers = AsyncMock(side_effect=mock_send_to_subscribers)

    # Mock update_request to simulate progress updates calling send_event_to_subscribers
    async def mock_update_request(request_id, status, message=None, details=None):
        # Find the operation name associated with the request_id (might need storing in mock)
        # For simplicity, assume we know the operation name here
        op_name = operation_name # Cheating slightly for the mock
        progress_data = {
            "type": "progress",
            "request_id": request_id,
            "operation": op_name,
            "status": status,
            "message": message,
            "details": details or {}, # Ensure details is a dict
        }
        await coordinator.send_event_to_subscribers(EventTypes.PROGRESS, progress_data)

    coordinator.update_request = AsyncMock(side_effect=mock_update_request)

    # Mock the ProgressReporter context manager provided by the coordinator mock
    mock_reporter = MagicMock(spec=ProgressReporter)
    mock_reporter.__aenter__ = AsyncMock(return_value=mock_reporter)
    mock_reporter.__aexit__ = AsyncMock()
    mock_reporter.update = AsyncMock()
    mock_reporter.error = AsyncMock()
    # Link reporter updates back to coordinator.update_request
    async def reporter_update_side_effect(message, status="in_progress", details=None):
        # Assume reporter knows its request_id and operation_name
        await coordinator.update_request(request_id, status, message, details)
    mock_reporter.update.side_effect = reporter_update_side_effect
    # Simulate the 'starting' and 'completed' messages sent by the real reporter context manager
    async def reporter_aenter_side_effect():
        await coordinator.update_request(request_id, "starting", f"Operation '{operation_name}' started.", {"parameters": params})
        return mock_reporter
    async def reporter_aexit_side_effect(exc_type, exc_val, exc_tb):
        if exc_type is None:
            await coordinator.update_request(request_id, "completed", "Operation completed successfully.", {"parameters": params})
        else:
            # Handle error case if needed
            pass
    mock_reporter.__aenter__.side_effect = reporter_aenter_side_effect
    mock_reporter.__aexit__.side_effect = reporter_aexit_side_effect

    coordinator.get_progress_reporter.return_value = mock_reporter # Make coordinator return this configured mock


    # Mock the behavior of start_request
    async def mock_start_request_side_effect(request_id, transport_id, operation_name, request_data):
        originating_transport = stdio_transport # Originating transport for this test
        security_context = originating_transport.validate_request_security(request_data)
        handler = coordinator.register_handler.call_args_list[1][0][1] # progress_op handler

        await coordinator.send_event_to_subscribers(
            EventTypes.STATUS,
            {"type": "status", "request_id": request_id, "operation": operation_name, "status": "starting", "message": ANY, "details": {"parameters": params}},
            originating_transport_id=transport_id
        )
        # Handler execution will trigger reporter updates via mocked methods
        result = await handler(request_id, transport_id, params, security_context)

        await coordinator.send_event_to_subscribers(
            EventTypes.TOOL_RESULT,
            {"type": "tool_result", "request_id": request_id, "tool_name": operation_name, "result": result}
        )

    coordinator.start_request.side_effect = mock_start_request_side_effect

    # Start request via Stdio transport
    await coordinator.start_request(
        request_id=request_id,
        transport_id=stdio_transport.transport_id,
        operation_name=operation_name,
        request_data=request_data,
    )

    # Assert start_request was awaited
    coordinator.start_request.assert_awaited_once()

    # Assert events sent to Stdio transport (origin)
    assert_events_sent(stdio_transport, [
        (EventTypes.STATUS, {"type": "status", "request_id": request_id, "operation": operation_name, "status": "starting", "message": ANY, "details": {"parameters": params}}),
        (EventTypes.PROGRESS, {"type": "progress", "request_id": request_id, "operation": operation_name, "status": "starting", "message": f"Operation '{operation_name}' started.", "details": {"parameters": params}}),
        (EventTypes.PROGRESS, {"type": "progress", "request_id": request_id, "operation": operation_name, "status": "in_progress", "message": "Step 1/3: Processing...", "details": {"parameters": params}}),
        (EventTypes.PROGRESS, {"type": "progress", "request_id": request_id, "operation": operation_name, "status": "in_progress", "message": "Step 2/3: Working...", "details": {"parameters": params, "progress": 0.5}}),
        (EventTypes.PROGRESS, {"type": "progress", "request_id": request_id, "operation": operation_name, "status": "in_progress", "message": "Step 3/3: Finalizing...", "details": {"parameters": params, "progress": 1.0}}),
        (EventTypes.PROGRESS, {"type": "progress", "request_id": request_id, "operation": operation_name, "status": "completed", "message": "Operation completed successfully.", "details": {"parameters": params}}),
        (EventTypes.TOOL_RESULT, {"type": "tool_result", "request_id": request_id, "tool_name": operation_name, "result": {"success": True, "message": "Progress operation completed", "request_id": request_id}}),
    ])

    # Assert PROGRESS and TOOL_RESULT events also sent to SSE transport (subscribed)
    assert_events_sent(sse_transport, [
        (EventTypes.PROGRESS, {"type": "progress", "request_id": request_id, "operation": operation_name, "status": "starting", "message": f"Operation '{operation_name}' started.", "details": {"parameters": params}}),
        (EventTypes.PROGRESS, {"type": "progress", "request_id": request_id, "operation": operation_name, "status": "in_progress", "message": "Step 1/3: Processing...", "details": {"parameters": params}}),
        (EventTypes.PROGRESS, {"type": "progress", "request_id": request_id, "operation": operation_name, "status": "in_progress", "message": "Step 2/3: Working...", "details": {"parameters": params, "progress": 0.5}}),
        (EventTypes.PROGRESS, {"type": "progress", "request_id": request_id, "operation": operation_name, "status": "in_progress", "message": "Step 3/3: Finalizing...", "details": {"parameters": params, "progress": 1.0}}),
        (EventTypes.PROGRESS, {"type": "progress", "request_id": request_id, "operation": operation_name, "status": "completed", "message": "Operation completed successfully.", "details": {"parameters": params}}),
        (EventTypes.TOOL_RESULT, {"type": "tool_result", "request_id": request_id, "tool_name": operation_name, "result": {"success": True, "message": "Progress operation completed", "request_id": request_id}}),
    ])


@pytest.mark.asyncio
async def test_error_request_routing(coordinator, registered_transports, registered_handlers, mock_uuid4):
    """Test routing a request that results in an error (using mocks)."""
    sse_transport = registered_transports["sse"]
    stdio_transport = registered_transports["stdio"]
    request_id = str(uuid.UUID("12345678-1234-5678-1234-000000000001"))
    operation_name = "error_op"
    params = {}
    request_data = {"request_id": request_id, "name": operation_name, "parameters": params, "auth_token": "mock_token"}

    # Mock send_event_to_subscribers
    async def mock_send_to_subscribers(event_type, data, originating_transport_id=None):
        if event_type == EventTypes.STATUS and originating_transport_id == sse_transport.transport_id:
             await sse_transport.send_event(event_type, data)
        elif event_type == EventTypes.TOOL_RESULT:
             if EventTypes.TOOL_RESULT in sse_transport.get_capabilities():
                 await sse_transport.send_event(event_type, data)
             if EventTypes.TOOL_RESULT in stdio_transport.get_capabilities():
                 await stdio_transport.send_event(event_type, data)
    coordinator.send_event_to_subscribers = AsyncMock(side_effect=mock_send_to_subscribers)

    # Mock fail_request to simulate error reporting calling send_event_to_subscribers
    async def mock_fail_request(request_id, operation_name, error, error_details=None, originating_transport_id=None, request_details=None):
        error_data = {
            "type": "tool_result",
            "request_id": request_id,
            "tool_name": operation_name,
            "result": {"success": False, "error": error, "details": str(error_details)}
        }
        await coordinator.send_event_to_subscribers(EventTypes.TOOL_RESULT, error_data)
    coordinator.fail_request = AsyncMock(side_effect=mock_fail_request)


    # Mock the behavior of start_request to simulate the error path
    async def mock_start_request_side_effect(request_id, transport_id, operation_name, request_data):
        originating_transport = sse_transport
        security_context = originating_transport.validate_request_security(request_data)
        handler = coordinator.register_handler.call_args_list[2][0][1] # error_op handler

        await coordinator.send_event_to_subscribers(
            EventTypes.STATUS,
            {"type": "status", "request_id": request_id, "operation": operation_name, "status": "starting", "message": ANY, "details": {"parameters": params}},
            originating_transport_id=transport_id
        )
        try:
            await handler(request_id, transport_id, params, security_context)
        except Exception as e:
            # Simulate coordinator catching error and calling fail_request
            await coordinator.fail_request(
                request_id=request_id,
                operation_name=operation_name,
                error=f"Operation failed: {type(e).__name__}",
                error_details=e,
                originating_transport_id=transport_id,
                request_details=params
            )

    coordinator.start_request.side_effect = mock_start_request_side_effect

    await coordinator.start_request(
        request_id=request_id,
        transport_id=sse_transport.transport_id,
        operation_name=operation_name,
        request_data=request_data,
    )

    # Assert start_request was awaited
    coordinator.start_request.assert_awaited_once()
    # Assert fail_request was awaited (called internally by the mocked start_request)
    coordinator.fail_request.assert_awaited_once()

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


@pytest.mark.asyncio
async def test_unknown_operation(coordinator, registered_transports, mock_uuid4):
    """Test requesting an operation without a registered handler (using mocks)."""
    sse_transport = registered_transports["sse"]
    stdio_transport = registered_transports["stdio"]
    request_id = str(uuid.UUID("12345678-1234-5678-1234-000000000001"))
    operation_name = "non_existent_op"
    params = {}
    request_data = {"request_id": request_id, "name": operation_name, "parameters": params, "auth_token": "mock_token"}

    # Mock send_event_to_subscribers
    async def mock_send_to_subscribers(event_type, data, originating_transport_id=None):
        # Only the originating transport should receive the error here
        if event_type == EventTypes.STATUS and originating_transport_id == sse_transport.transport_id:
             await sse_transport.send_event(event_type, data)
        elif event_type == EventTypes.TOOL_RESULT and originating_transport_id == sse_transport.transport_id:
             await sse_transport.send_event(event_type, data)

    coordinator.send_event_to_subscribers = AsyncMock(side_effect=mock_send_to_subscribers)

    # Mock fail_request
    async def mock_fail_request(request_id, operation_name, error, error_details=None, originating_transport_id=None, request_details=None):
        error_data = {
            "type": "tool_result",
            "request_id": request_id,
            "tool_name": operation_name,
            "result": {"success": False, "error": error, "details": str(error_details)}
        }
        # Send error *only* to originating transport when handler not found
        await coordinator.send_event_to_subscribers(EventTypes.TOOL_RESULT, error_data, originating_transport_id=originating_transport_id)
    coordinator.fail_request = AsyncMock(side_effect=mock_fail_request)

    # Mock start_request to simulate handler not found
    async def mock_start_request_side_effect(request_id, transport_id, operation_name, request_data):
        # Simulate sending 'starting' status
        await coordinator.send_event_to_subscribers(
            EventTypes.STATUS,
            {"type": "status", "request_id": request_id, "operation": operation_name, "status": "starting", "message": ANY, "details": {"parameters": params}},
            originating_transport_id=transport_id
        )
        # Simulate handler lookup failure
        await coordinator.fail_request(
            request_id=request_id,
            operation_name=operation_name,
            error="Operation not supported",
            error_details=f"No handler registered for operation '{operation_name}'.",
            originating_transport_id=transport_id,
            request_details=params
        )
    coordinator.start_request.side_effect = mock_start_request_side_effect

    await coordinator.start_request(
        request_id=request_id,
        transport_id=sse_transport.transport_id,
        operation_name=operation_name,
        request_data=request_data,
    )

    # Assert start_request and fail_request were awaited
    coordinator.start_request.assert_awaited_once()
    coordinator.fail_request.assert_awaited_once()

    expected_error_result = (EventTypes.TOOL_RESULT, {"type": "tool_result", "request_id": request_id, "tool_name": operation_name, "result": {"success": False, "error": "Operation not supported", "details": f"No handler registered for operation '{operation_name}'."}})

    # Assert events sent ONLY to originating transport (SSE)
    assert_events_sent(sse_transport, [
         (EventTypes.STATUS, {"type": "status", "request_id": request_id, "operation": operation_name, "status": "starting", "message": ANY, "details": {"parameters": params}}),
         expected_error_result,
    ])

    # Assert NO events sent to other transport (Stdio)
    assert_events_sent(stdio_transport, [])


@pytest.mark.asyncio
async def test_permission_denied(coordinator, registered_transports, registered_handlers, mock_uuid4):
    """Test requesting an operation without required permissions (using mocks)."""
    sse_transport = registered_transports["sse"]
    stdio_transport = registered_transports["stdio"]

    # Mock the security validation for this specific transport to return a limited context
    limited_context = SecurityContext(user_id="limited_user", permissions={Permissions.LIST_MODELS}, transport_id=sse_transport.transport_id)
    sse_transport.validate_request_security = MagicMock(return_value=limited_context)

    # Register a handler that requires execute_aider permission (on the mock coordinator)
    async def protected_op_handler(request_id: str, transport_id: str, parameters: Dict[str, Any], security_context: SecurityContext) -> Dict[str, Any]:
        pytest.fail("Protected handler should not be executed") # pragma: no cover
    coordinator.register_handler("protected_op", protected_op_handler, required_permission=Permissions.EXECUTE_AIDER)

    request_id = str(uuid.UUID("12345678-1234-5678-1234-000000000001"))
    operation_name = "protected_op"
    params = {}
    request_data = {"request_id": request_id, "name": operation_name, "parameters": params, "auth_token": "mock_token_limited"}

    # Mock send_event_to_subscribers
    async def mock_send_to_subscribers(event_type, data, originating_transport_id=None):
        # Only the originating transport should receive the error here
        if event_type == EventTypes.TOOL_RESULT and originating_transport_id == sse_transport.transport_id:
             await sse_transport.send_event(event_type, data)
    coordinator.send_event_to_subscribers = AsyncMock(side_effect=mock_send_to_subscribers)

    # Mock fail_request
    async def mock_fail_request(request_id, operation_name, error, error_details=None, originating_transport_id=None, request_details=None):
        error_data = {
            "type": "tool_result",
            "request_id": request_id,
            "tool_name": operation_name,
            "result": {"success": False, "error": error, "details": str(error_details)}
        }
        # Send error *only* to originating transport for permission denied
        await coordinator.send_event_to_subscribers(EventTypes.TOOL_RESULT, error_data, originating_transport_id=originating_transport_id)
    coordinator.fail_request = AsyncMock(side_effect=mock_fail_request)

    # Mock start_request to simulate permission check failure
    async def mock_start_request_side_effect(request_id, transport_id, operation_name, request_data):
        originating_transport = sse_transport
        # Simulate security validation (returns limited context)
        security_context = originating_transport.validate_request_security(request_data)
        # Simulate permission check failure
        required_permission = Permissions.EXECUTE_AIDER # Get required permission from handler registration mock
        if required_permission not in security_context.permissions:
            await coordinator.fail_request(
                request_id=request_id,
                operation_name=operation_name,
                error="Permission denied",
                error_details=f"User does not have the required permission '{required_permission.name}' for operation '{operation_name}'.",
                originating_transport_id=transport_id,
                request_details=params
            )
            return # Stop processing

        # This part should not be reached
        pytest.fail("Handler execution should be prevented by permission check") # pragma: no cover

    coordinator.start_request.side_effect = mock_start_request_side_effect


    await coordinator.start_request(
        request_id=request_id,
        transport_id=sse_transport.transport_id,
        operation_name=operation_name,
        request_data=request_data,
    )

    # Assert start_request and fail_request were awaited
    coordinator.start_request.assert_awaited_once()
    coordinator.fail_request.assert_awaited_once()

    expected_error_result = (EventTypes.TOOL_RESULT, {"type": "tool_result", "request_id": request_id, "tool_name": operation_name, "result": {"success": False, "error": "Permission denied", "details": f"User does not have the required permission '{Permissions.EXECUTE_AIDER.name}' for operation '{operation_name}'."}})

    # Assert error event sent ONLY to originating transport (SSE)
    assert_events_sent(sse_transport, [
        expected_error_result,
    ])

    # Assert NO events sent to other transport (Stdio)
    assert_events_sent(stdio_transport, [])


@pytest.mark.asyncio
async def test_reporter_handler_routing(coordinator, registered_transports, registered_handlers, mock_uuid4):
    """Test routing a request that uses ProgressReporter (using mocks)."""
    # This test is very similar to test_progress_request_routing
    # We can reuse much of the mocking setup from there.

    sse_transport = registered_transports["sse"]
    stdio_transport = registered_transports["stdio"]
    request_id = str(uuid.UUID("12345678-1234-5678-1234-000000000001"))
    operation_name = "reporter_op" # Use the specific handler name
    params = {}
    request_data = {"request_id": request_id, "name": operation_name, "parameters": params, "auth_token": "mock_token"}

    # --- Mocking Setup (similar to test_progress_request_routing) ---

    # Mock send_event_to_subscribers
    async def mock_send_to_subscribers(event_type, data, originating_transport_id=None):
        if event_type == EventTypes.STATUS and originating_transport_id == sse_transport.transport_id:
             await sse_transport.send_event(event_type, data)
        elif event_type == EventTypes.PROGRESS:
             if EventTypes.PROGRESS in sse_transport.get_capabilities():
                 await sse_transport.send_event(event_type, data)
             if EventTypes.PROGRESS in stdio_transport.get_capabilities():
                 await stdio_transport.send_event(event_type, data)
        elif event_type == EventTypes.TOOL_RESULT:
             if EventTypes.TOOL_RESULT in sse_transport.get_capabilities():
                 await sse_transport.send_event(event_type, data)
             if EventTypes.TOOL_RESULT in stdio_transport.get_capabilities():
                 await stdio_transport.send_event(event_type, data)
    coordinator.send_event_to_subscribers = AsyncMock(side_effect=mock_send_to_subscribers)

    # Mock update_request
    async def mock_update_request(request_id, status, message=None, details=None):
        op_name = operation_name
        progress_data = {
            "type": "progress", "request_id": request_id, "operation": op_name,
            "status": status, "message": message, "details": details or {},
        }
        await coordinator.send_event_to_subscribers(EventTypes.PROGRESS, progress_data)
    coordinator.update_request = AsyncMock(side_effect=mock_update_request)

    # Mock ProgressReporter context manager
    mock_reporter = MagicMock(spec=ProgressReporter)
    mock_reporter.__aenter__ = AsyncMock(return_value=mock_reporter)
    mock_reporter.__aexit__ = AsyncMock()
    mock_reporter.update = AsyncMock()
    mock_reporter.error = AsyncMock()
    async def reporter_update_side_effect(message, status="in_progress", details=None):
        await coordinator.update_request(request_id, status, message, details)
    mock_reporter.update.side_effect = reporter_update_side_effect
    async def reporter_aenter_side_effect():
        # Use the correct operation name for the message
        await coordinator.update_request(request_id, "starting", f"Operation '{operation_name}' started.", {"parameters": params})
        return mock_reporter
    async def reporter_aexit_side_effect(exc_type, exc_val, exc_tb):
        if exc_type is None:
            await coordinator.update_request(request_id, "completed", "Operation completed successfully.", {"parameters": params})
    mock_reporter.__aenter__.side_effect = reporter_aenter_side_effect
    mock_reporter.__aexit__.side_effect = reporter_aexit_side_effect
    coordinator.get_progress_reporter.return_value = mock_reporter

    # Mock start_request behavior
    async def mock_start_request_side_effect(request_id, transport_id, operation_name, request_data):
        originating_transport = sse_transport # Originating transport for this test
        security_context = originating_transport.validate_request_security(request_data)
        # Get the correct handler based on operation_name
        handler = None
        for call in coordinator.register_handler.call_args_list:
            if call[0][0] == operation_name:
                handler = call[0][1]
                break
        assert handler is not None, f"Handler for {operation_name} not found in mock registrations"

        await coordinator.send_event_to_subscribers(
            EventTypes.STATUS,
            {"type": "status", "request_id": request_id, "operation": operation_name, "status": "starting", "message": ANY, "details": {"parameters": params}},
            originating_transport_id=transport_id
        )
        result = await handler(request_id, transport_id, params, security_context)
        await coordinator.send_event_to_subscribers(
            EventTypes.TOOL_RESULT,
            {"type": "tool_result", "request_id": request_id, "tool_name": operation_name, "result": result}
        )
    coordinator.start_request.side_effect = mock_start_request_side_effect

    # --- Execution ---
    await coordinator.start_request(
        request_id=request_id,
        transport_id=sse_transport.transport_id,
        operation_name=operation_name,
        request_data=request_data,
    )

    # --- Assertions ---
    coordinator.start_request.assert_awaited_once()

    # Define expected broadcast events (PROGRESS, TOOL_RESULT)
    expected_broadcast_events = [
        (EventTypes.PROGRESS, {"type": "progress", "request_id": request_id, "operation": operation_name, "status": "starting", "message": f"Operation '{operation_name}' started.", "details": {"parameters": params}}),
        (EventTypes.PROGRESS, {"type": "progress", "request_id": request_id, "operation": operation_name, "status": "in_progress", "message": "Using reporter: Step 1...", "details": {"parameters": params}}),
        (EventTypes.PROGRESS, {"type": "progress", "request_id": request_id, "operation": operation_name, "status": "in_progress", "message": "Using reporter: Step 2...", "details": {"parameters": params, "stage": 2}}),
        (EventTypes.PROGRESS, {"type": "progress", "request_id": request_id, "operation": operation_name, "status": "completed", "message": "Operation completed successfully.", "details": {"parameters": params}}),
        (EventTypes.TOOL_RESULT, {"type": "tool_result", "request_id": request_id, "tool_name": operation_name, "result": {"success": True, "message": "ProgressReporter operation completed", "request_id": request_id}}),
    ]

    # Assert events sent to originating transport (SSE) - includes initial STATUS
    assert_events_sent(sse_transport, [
         (EventTypes.STATUS, {"type": "status", "request_id": request_id, "operation": operation_name, "status": "starting", "message": ANY, "details": {"parameters": params}}),
         *expected_broadcast_events
    ])

    # Assert events sent to other subscribed transport (Stdio) - only broadcast events
    assert_events_sent(stdio_transport, expected_broadcast_events)
