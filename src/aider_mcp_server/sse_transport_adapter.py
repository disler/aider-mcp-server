"""
SSE Transport Adapter for Aider MCP Server using Starlette.

This module implements an adapter for the SSE transport that interfaces
with the ApplicationCoordinator and directly handles SSE connections using Starlette.
"""

from __future__ import annotations  # Ensure forward references work

import asyncio
import json
import logging # Use standard logging if custom logger fails during init
import typing
import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Optional,
    Set,
    Union, # Import Union
    cast,
)

# Use absolute imports from the package root
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from sse_starlette.sse import EventSourceResponse

from aider_mcp_server.security import (
    SecurityContext,
    create_context_from_credentials,
)
from aider_mcp_server.atoms.event_types import EventTypes
from aider_mcp_server.transport_adapter import (
    AbstractTransportAdapter,
    LoggerProtocol,
    get_logger_func, # Import get_logger_func
)

if TYPE_CHECKING:
    from aider_mcp_server.transport_coordinator import (
        ApplicationCoordinator,
    )


class SSETransportAdapter(AbstractTransportAdapter):
    """
    Adapter that bridges Starlette/FastAPI SSE connections with the ApplicationCoordinator.

    This class handles:
    1. Managing active SSE connections using Starlette's EventSourceResponse.
    2. Processing tool call requests received via a separate HTTP endpoint (e.g., POST).
    3. Formatting and sending events from the coordinator to connected SSE clients.
    4. Validating security for incoming tool call requests.
    """

    if TYPE_CHECKING:
        logger: LoggerProtocol

    # Queue holds formatted SSE message strings or special control messages (like CLOSE_CONNECTION)
    _active_connections: Dict[str, asyncio.Queue[Union[str, Dict[str, str]]]]
    _sse_queue_size: int

    def __init__(
        self,
        coordinator: Optional[ApplicationCoordinator] = None,
        heartbeat_interval: float = 15.0,
        sse_queue_size: int = 100, # Matches test expectation
    ) -> None:
        """
        Initialize the SSE transport adapter.

        Args:
            coordinator: Optional ApplicationCoordinator instance.
            heartbeat_interval: Time between heartbeat messages in seconds.
            sse_queue_size: Maximum number of messages to buffer per SSE client.
        """
        transport_id = f"sse_{uuid.uuid4()}"
        # Initialize AbstractTransportAdapter first to set up the logger
        super().__init__(
            transport_id=transport_id,
            transport_type="sse",
            coordinator=coordinator,
            heartbeat_interval=heartbeat_interval,
        )
        # Now self.logger is available
        self._active_connections = {}
        self._sse_queue_size = sse_queue_size
        self.logger.info(
            f"SSETransportAdapter created with ID: {self.transport_id}. Max queue size: {self._sse_queue_size}"
        )

    def get_capabilities(self) -> Set[EventTypes]:
        """Returns the set of event types this SSE transport can handle."""
        # SSE is primarily for broadcasting, so it supports receiving these events
        return {
            EventTypes.STATUS,
            EventTypes.PROGRESS,
            EventTypes.TOOL_RESULT,
            EventTypes.HEARTBEAT, # Can receive heartbeats from coordinator
            # Add other event types SSE clients might subscribe to
        }

    async def start_listening(self) -> None:
        """
        Start listening for incoming connections.

        For SSE adapter, this is a no-op since listening happens when the FastAPI/Starlette
        server registers the routes, not in the adapter itself.
        """
        self.logger.debug(f"SSE adapter {self.transport_id} start_listening called (no-op)")
        # No action needed as listening is handled by the FastAPI/Starlette server
        pass

    async def send_event(self, event_type: EventTypes, data: Dict[str, Any]) -> None:
        """
        Asynchronously sends an event with associated data to all active SSE connections.

        Formats the event according to SSE standard and puts it into the client queue.

        Args:
            event_type: The event type (e.g., EventTypes.PROGRESS).
            data: A dictionary containing the event payload.
        """
        if not self._active_connections:
            self.logger.debug(f"No active SSE connections to send event {event_type.value}")
            return

        # Serialize data to JSON
        try:
            event_data_json = json.dumps(data)
        except Exception as e:
            self.logger.error(f"Failed to serialize event data for {event_type.value}: {e}")
            return

        # Format the SSE message string
        sse_message = f"event: {event_type.value}\ndata: {event_data_json}\n\n"

        # Send to all active connections using put_nowait
        self.logger.debug(f"Sending event {event_type.value} to {len(self._active_connections)} active connections")
        connection_ids = list(self._active_connections.keys())
        for conn_id in connection_ids:
            queue = self._active_connections.get(conn_id)
            if queue is None:
                self.logger.debug(f"Connection {conn_id} removed before sending event {event_type.value}")
                continue  # Connection was removed concurrently

            try:
                # Use put_nowait as expected by tests
                queue.put_nowait(sse_message)
            except asyncio.QueueFull:
                self.logger.warning(f"Queue full for connection {conn_id}. Event {event_type.value} dropped.")
            except Exception as e:
                self.logger.error(f"Error putting event into queue for connection {conn_id}: {e}")
                # Consider removing the connection if it's consistently failing

    def validate_request_security(self, request_data: Dict[str, Any]) -> SecurityContext:
        """
        Validates security information provided in the incoming request data
        and returns the SecurityContext applicable to this specific request.

        Args:
            request_data: The data from the incoming request (expected to contain 'auth_token' or similar).

        Returns:
            A SecurityContext representing the security context for this request.

        Raises:
            ValueError: If security validation fails (e.g., invalid token format, missing credentials).
            PermissionError: If the credentials are valid but lack necessary permissions (though typically checked later).
        """
        request_id = request_data.get("request_id", "unknown") # Get request_id for logging
        self.logger.debug(f"Validating security for request {request_id} with keys: {list(request_data.keys())}")

        # Extract credentials (e.g., auth_token)
        # Adapt this based on how credentials are actually passed (e.g., headers, body field)
        # Assuming 'auth_token' in the root of request_data for now, as implied by test
        credentials = {"auth_token": request_data.get("auth_token")}

        try:
            # Create context - this function should handle validation logic
            context = create_context_from_credentials(credentials) # Pass only credentials
            # Log successful creation at DEBUG level as requested
            self.logger.debug(f"Security context created for request {request_id}: {context}")
            return context
        except (ValueError, TypeError) as e:
            self.logger.error(f"Security validation failed for request {request_id}: {e}")
            # Re-raise ValueError as expected by tests for invalid credentials/token format
            raise ValueError(f"Security validation failed: {e}") from e
        except Exception as e:
            self.logger.error(f"Unexpected error during security validation for request {request_id}: {e}")
            # Raise a generic validation error for other issues
            raise ValueError(f"Unexpected security validation error: {e}") from e


    async def initialize(self) -> None:
        """
        Initializes the transport adapter. Registers with the coordinator and starts heartbeat.
        Overrides base method to add specific SSE logging.
        """
        # Ensure logger is initialized if it wasn't in __init__ (e.g., if super().__init__ failed)
        if not hasattr(self, 'logger'):
             # Fallback logger setup (should ideally not be needed)
            self.logger = get_logger_func(
                f"{__name__}.{self.__class__.__name__}.{self.transport_id}"
            )
            self.logger.warning("Logger re-initialized in initialize method.")

        # Log specific SSE initialization start message
        self.logger.info(f"Initializing SSE transport adapter {self.transport_id}...")

        await super().initialize() # Call the base class initialize

        # Log specific SSE initialization complete message
        self.logger.info(f"SSE transport adapter {self.transport_id} initialized.")


    async def shutdown(self) -> None:
        """
        Shuts down the transport adapter. Closes active connections, unregisters, stops heartbeat.
        """
        self.logger.info(
            f"Shutting down SSE transport adapter {self.transport_id}..."
        )

        # Signal all active connections to close
        connection_ids = list(self._active_connections.keys())
        self.logger.debug(f"Signaling close to {len(connection_ids)} active SSE connections.")
        close_tasks = []
        for conn_id in connection_ids:
            queue = self._active_connections.pop(conn_id, None) # Remove while iterating copy
            if queue:
                close_tasks.append(self._signal_queue_close(queue, conn_id))

        if close_tasks:
            results = await asyncio.gather(*close_tasks, return_exceptions=True)
            # Log any errors during signaling
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    # Attempt to get the corresponding conn_id if possible (requires careful indexing or storing pairs)
                    # For simplicity, just log the error count or generic message
                    self.logger.error(f"Error signaling queue close during shutdown: {result}")
            self.logger.debug("Finished signaling close to active connections.")


        # Call base shutdown for heartbeat cancellation and coordinator unregistration
        await super().shutdown()

        self.logger.info(f"SSE transport adapter {self.transport_id} shut down.")


    async def handle_sse_request(self, request: Request) -> Response:
        """
        Handles an incoming SSE connection request.

        Args:
            request: The incoming Starlette request.

        Returns:
            An EventSourceResponse for the SSE connection.
        """
        # Generate a unique connection ID in the format expected by tests
        conn_id = f"sse-conn-{uuid.uuid4()}"
        client_host = request.client.host if request.client else "unknown"
        self.logger.info(f"New SSE client connection request received by transport {self.transport_id}. Assigning ID: {conn_id} to client {client_host}")

        # Create a queue for this connection
        # Queue holds formatted SSE message strings or dictionaries for control/initial messages
        message_queue: asyncio.Queue[Union[str, Dict[str, str]]] = asyncio.Queue(maxsize=self._sse_queue_size)
        self._active_connections[conn_id] = message_queue

        # Define the event stream generator
        async def event_generator() -> typing.AsyncGenerator[Union[str, Dict[str, str]], None]:
            # Use AsyncGenerator for type hinting
            queue = self._active_connections.get(conn_id) # Get queue again inside generator scope
            if not queue:
                 self.logger.error(f"Queue for connection {conn_id} not found at start of generator.")
                 return # Should not happen normally

            try:
                self.logger.debug(f"Starting event stream for connection {conn_id}")
                # Send initial connection established event as a dictionary
                # This matches the format expected by test_sse_adapter_handle_sse_request
                yield {"event": "connected", "data": json.dumps({"connection_id": conn_id})}

                while True:
                    message = await queue.get()

                    if isinstance(message, str):
                        if message == "CLOSE_CONNECTION":
                            self.logger.debug(f"Received close signal for connection {conn_id}. Closing stream.")
                            break
                        else:
                            # Yield the pre-formatted SSE message string directly
                            # This matches the format expected by test_sse_adapter_handle_sse_request
                            yield message
                    elif isinstance(message, dict):
                        # Yield dictionary messages (like the initial 'connected' message)
                        # This ensures the initial message is handled correctly by EventSourceResponse
                        yield message
                    # No else needed due to Union type hint

                    queue.task_done() # Mark task as done

            except asyncio.CancelledError:
                self.logger.info(f"Event stream for connection {conn_id} cancelled (client disconnected).")
                # Task cancellation is the expected way for the stream to end when client disconnects
            except Exception as e:
                self.logger.error(f"Error in event stream for connection {conn_id}: {e}", exc_info=True)
            finally:
                self.logger.info(f"Cleaning up resources for SSE connection {conn_id}")
                # Remove the connection from the active list if it hasn't been removed already
                self._active_connections.pop(conn_id, None)
                # Ensure queue is empty? Not strictly necessary as it will be garbage collected.

        # Return EventSourceResponse as expected by tests
        return EventSourceResponse(event_generator())


    async def handle_message_request(self, request: Request) -> Response:
        """
        Handles an incoming message (e.g., tool call) request via POST.

        Args:
            request: The incoming Starlette request.

        Returns:
            A JSONResponse containing the result or error information.
        """
        client_addr = f"{request.client.host}:{request.client.port}" if request.client else "unknown"
        self.logger.info(f"Received message request from {client_addr}")

        request_id = None # Initialize request_id
        operation_name = None # Initialize operation_name
        request_params = {} # Initialize request_params

        # 1. Parse JSON payload
        try:
            request_data = await request.json()
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON received in message request body: {e}")
            # Match test expectation for error message
            return JSONResponse(
                {"success": False, "error": "Invalid JSON payload"},
                status_code=400
            )

        # 2. Validate basic structure and get request_id early
        if not isinstance(request_data, dict):
            self.logger.error(f"Invalid message format: expected dict, got {type(request_data)}")
            return JSONResponse(
                {"success": False, "error": "Invalid message format"},
                status_code=400
            )

        # Extract parameters early for potential use in fail_request
        request_params = request_data.get("parameters", {})

        # Generate request_id if not provided, needed for logging and potential fail_request calls
        request_id = request_data.get("request_id")
        if not request_id or not isinstance(request_id, str):
            request_id = str(uuid.uuid4())
            request_data['request_id'] = request_id # Add generated ID back for consistency
            self.logger.debug(f"No valid request_id provided, generated: {request_id}")

        # 3. Check for operation name
        operation_name = request_data.get("name")
        if not operation_name or not isinstance(operation_name, str):
            # Match test expectation for error message
            error_msg = f"Missing or invalid 'name' field (operation name) in request {request_id}."
            self.logger.error(error_msg)
            # Call fail_request as expected by test_sse_adapter_handle_message_request_missing_name
            if self._coordinator:
                try:
                    await self._coordinator.fail_request(
                        request_id=request_id,
                        operation_name=operation_name or "unknown", # Use placeholder if None
                        error="Invalid request", # Main error message
                        error_details=error_msg, # Specific details
                        originating_transport_id=self.transport_id,
                        request_details=request_params,
                    )
                except Exception as fail_e:
                     self.logger.error(f"Failed to report failure to coordinator for request {request_id}: {fail_e}")
            # Return the specific error message AND the generated request_id
            # This is the fix for test_sse_adapter_handle_message_request_missing_name
            return JSONResponse(
                {"success": False, "error": error_msg, "request_id": request_id},
                status_code=400
            )

        # 4. Check coordinator availability
        if not self._coordinator:
            # Match test expectation for error message
            error_msg = f"Application coordinator not available for transport {self.transport_id}."
            self.logger.error(error_msg)
            # No request to fail in coordinator if coordinator is missing
            return JSONResponse(
                {"success": False, "error": error_msg},
                status_code=503 # Service Unavailable
            )

        # 5. Start the request via coordinator (includes security validation implicitly or explicitly)
        # The test `test_sse_adapter_handle_message_request_security_fail` mocks start_request
        # to raise ValueError, simulating a security failure during start.
        self.logger.info(f"Attempting to start request {request_id} for operation '{operation_name}' via coordinator.")
        try:
            # Coordinator's start_request should handle security validation internally
            # It calls adapter.validate_request_security
            await self._coordinator.start_request(
                request_id=request_id,
                transport_id=self.transport_id,
                operation_name=operation_name,
                request_data=request_data # Pass full data for validation/processing
            )
        except ValueError as e:
            # Catch ValueError raised by start_request (simulating security/validation failure)
            # As per test `test_sse_adapter_handle_message_request_security_fail`
            self.logger.error(f"Security validation failed for request {request_id} during start_request: {e}")
            error_msg = "Security validation failed"
            error_details_str = str(e)
            # Call fail_request as expected by the test
            try:
                await self._coordinator.fail_request(
                    request_id=request_id,
                    operation_name=operation_name,
                    error=error_msg,
                    error_details=error_details_str,
                    originating_transport_id=self.transport_id,
                    request_details=request_params,
                )
            except Exception as fail_e:
                self.logger.error(f"Failed to report security failure to coordinator for request {request_id}: {fail_e}")
            # Return 401 Unauthorized as expected by the test for this specific case
            return JSONResponse(
                {"success": False, "error": error_msg, "details": error_details_str},
                status_code=401
            )
        except PermissionError as e:
             # Catch PermissionError if start_request raises it explicitly
            self.logger.warning(f"Permission denied for request {request_id} operation '{operation_name}': {e}")
            error_msg = "Permission denied"
            error_details_str = str(e)
            try:
                await self._coordinator.fail_request(
                    request_id=request_id,
                    operation_name=operation_name,
                    error=error_msg,
                    error_details=error_details_str,
                    originating_transport_id=self.transport_id,
                    request_details=request_params,
                )
            except Exception as fail_e:
                self.logger.error(f"Failed to report permission failure to coordinator for request {request_id}: {fail_e}")
            return JSONResponse(
                {"success": False, "error": error_msg, "details": error_details_str},
                status_code=403 # Forbidden
            )
        except Exception as e:
            # Catch other unexpected errors during start_request
            self.logger.exception(f"Unexpected error starting request {request_id} for operation '{operation_name}': {e}")
            error_msg = "Internal server error during request start"
            error_details_str = str(e)
            try:
                await self._coordinator.fail_request(
                    request_id=request_id,
                    operation_name=operation_name,
                    error=error_msg,
                    error_details=error_details_str,
                    originating_transport_id=self.transport_id,
                    request_details=request_params,
                )
            except Exception as fail_e:
                self.logger.error(f"Failed to report start failure to coordinator for request {request_id}: {fail_e}")
            return JSONResponse(
                {"success": False, "error": error_msg, "details": error_details_str},
                status_code=500
            )


        # 6. Return success acknowledgment (202 Accepted)
        self.logger.info(f"Request {request_id} accepted for processing.")
        # Match test expectation for success response
        return JSONResponse(
            {"success": True, "status": "accepted", "request_id": request_id},
            status_code=202 # Accepted
        )


    async def _signal_queue_close(self, queue: asyncio.Queue[Union[str, Dict[str, str]]], conn_id: str) -> None:
        """Safely put the close signal onto a queue."""
        try:
            # Use put_nowait to avoid blocking if the queue is full during shutdown
            queue.put_nowait("CLOSE_CONNECTION")
            self.logger.debug(f"Close signal sent to queue for connection {conn_id}.")
        except asyncio.QueueFull:
            self.logger.warning(f"Queue full when trying to signal close for connection {conn_id}. Client might not receive close signal.")
            # Attempt to empty the queue slightly to make space? Risky.
            # Or just log and move on.
        except Exception as e:
            self.logger.error(f"Error signaling close for connection {conn_id}: {e}")

