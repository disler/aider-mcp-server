"""
SSE Transport Adapter for Aider MCP Server using Starlette.

This module implements an adapter for the SSE transport that interfaces
with the ApplicationCoordinator and directly handles SSE connections using Starlette.
"""

from __future__ import annotations  # Ensure forward references work

import asyncio
import json
import time
import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Optional,
    Set,
    Union,
)

# Use absolute imports from the package root
from aider_mcp_server.atoms.event_types import EventTypes
from aider_mcp_server.mcp_types import (
    EventData,
    LoggerProtocol,
)
from aider_mcp_server.transport_adapter import (
    AbstractTransportAdapter,
)

if TYPE_CHECKING:
    from aider_mcp_server.transport_coordinator import ApplicationCoordinator


class SSETransportAdapter(AbstractTransportAdapter):
    """
    Adapter that bridges Starlette/FastAPI SSE connections with the ApplicationCoordinator.

    This class handles:
    1. Managing active SSE connections using Starlette's EventSourceResponse.
    2. Processing tool call requests received via a separate HTTP endpoint (e.g., POST).
    3. Formatting and sending events from the coordinator to connected SSE clients.
    4. Validating security for incoming tool call requests.
    """

    @property
    def transport_id(self) -> str:
        """
        Property for accessing the transport ID.
        Used for compatibility with code that expects a transport_id attribute.
        """
        return self._transport_id

    @classmethod
    def get_default_capabilities(cls) -> Set[EventTypes]:
        """
        Get the default capabilities for this transport adapter class without instantiation.

        This allows the TransportAdapterRegistry to determine capabilities
        without instantiating the adapter.

        Returns:
            A set of event types that this adapter supports by default.
        """
        # SSE is primarily for broadcasting, so it supports receiving these events
        return {
            EventTypes.STATUS,
            EventTypes.PROGRESS,
            EventTypes.TOOL_RESULT,
            EventTypes.HEARTBEAT,  # Can receive heartbeats from coordinator
            # Add other event types SSE clients might subscribe to
        }

    if TYPE_CHECKING:
        logger: LoggerProtocol

    # Queue holds formatted SSE message strings or special control messages (like CLOSE_CONNECTION)
    _active_connections: Dict[str, asyncio.Queue[Union[str, Dict[str, str]]]]
    _sse_queue_size: int

    def __init__(
        self,
        coordinator: Optional[ApplicationCoordinator] = None,
        heartbeat_interval: float = 15.0,
        sse_queue_size: int = 100,  # Matches test expectation
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

        # Get event loop for task creation
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self.logger.warning("No running event loop found. Creating a new one.")
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

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
            EventTypes.HEARTBEAT,  # Can receive heartbeats from coordinator
            # Add other event types SSE clients might subscribe to
        }

    def validate_request_security(self, request_data: Dict[str, Any]) -> Any:
        """
        Validates security information provided in the incoming request data.

        For SSE connections, we use a simple anonymous security context.

        Args:
            request_data: The data from the incoming request.

        Returns:
            A SecurityContext representing the security validation result.
        """
        from aider_mcp_server.security import ANONYMOUS_SECURITY_CONTEXT

        return ANONYMOUS_SECURITY_CONTEXT

    async def start_listening(self) -> None:
        """
        Start listening for incoming connections.

        For SSE adapter, this is a no-op since listening happens when the FastAPI/Starlette
        server registers the routes, not in the adapter itself.
        """
        self.logger.debug(
            f"SSE adapter {self.transport_id} start_listening called (no-op)"
        )
        # No action needed as listening is handled by the FastAPI/Starlette server
        pass

    async def send_event(self, event_type: EventTypes, data: EventData) -> None:
        """
        Asynchronously sends an event with associated data to all active SSE connections.

        Formats the event according to SSE standard and puts it into the client queue.

        Args:
            event_type: The event type (e.g., EventTypes.PROGRESS).
            data: A dictionary containing the event payload.
        """
        if not self._active_connections:
            self.logger.debug(
                f"No active SSE connections to send event {event_type.value}"
            )
            return

        # Serialize data to JSON
        try:
            event_data_json = json.dumps(data)
        except Exception as e:
            self.logger.error(
                f"Failed to serialize event data for {event_type.value}: {e}"
            )
            return

        # Format the SSE message string
        sse_message = f"event: {event_type.value}\ndata: {event_data_json}\n\n"

        # Send to all active connections using put_nowait
        self.logger.debug(
            f"Sending event {event_type.value} to {len(self._active_connections)} active connections"
        )
        connection_ids = list(self._active_connections.keys())
        for conn_id in connection_ids:
            queue = self._active_connections.get(conn_id)
            if queue is None:
                self.logger.debug(
                    f"Connection {conn_id} removed before sending event {event_type.value}"
                )
                continue  # Connection was removed concurrently

            try:
                # Use put_nowait as expected by tests
                queue.put_nowait(sse_message)
            except asyncio.QueueFull:
                self.logger.warning(
                    f"Queue full for connection {conn_id}. Event {event_type.value} dropped."
                )
            except Exception as e:
                self.logger.error(
                    f"Error putting event into queue for connection {conn_id}: {e}"
                )
                # Consider removing the connection if it's consistently failing

    async def handle_sse_request(self, request: Any) -> Any:
        """
        Handle a new SSE connection request from a client.

        Creates a new connection ID, sets up an event queue, and returns an
        EventSourceResponse that will send events to the client.

        Args:
            request: The Starlette/FastAPI request object

        Returns:
            An EventSourceResponse object that will stream events to the client
        """
        # Import the EventSourceResponse here to avoid potential circular imports
        from sse_starlette.sse import EventSourceResponse

        # Create a connection ID for this client
        client_id = f"client_{uuid.uuid4()}"
        queue: asyncio.Queue[Union[str, Dict[str, str]]] = asyncio.Queue(
            maxsize=self._sse_queue_size
        )

        # Register this connection
        self._active_connections[client_id] = queue
        self.logger.info(f"SSE connection established: {client_id}")

        # Define the event generator
        async def event_generator() -> (
            Any
        ):  # Using Any for the async generator return type
            try:
                # Send initial connection event
                yield {
                    "event": "connected",
                    "data": json.dumps({"client_id": client_id}),
                }

                # Loop to pull messages from the queue
                while True:
                    try:
                        message = await asyncio.wait_for(queue.get(), timeout=30.0)
                        yield message
                        queue.task_done()
                    except asyncio.TimeoutError:
                        # Heartbeat to keep connection alive
                        yield {
                            "event": "heartbeat",
                            "data": json.dumps({"timestamp": time.time()}),
                        }
                    except Exception as e:
                        self.logger.error(
                            f"Error in SSE event stream for {client_id}: {e}"
                        )
                        break
            except asyncio.CancelledError:
                # Client disconnected
                self.logger.info(f"SSE connection closed: {client_id}")
            finally:
                # Clean up the connection
                self._active_connections.pop(client_id, None)
                self.logger.info(f"SSE connection cleaned up: {client_id}")

        # Return the EventSourceResponse with our generator
        return EventSourceResponse(event_generator())

    async def handle_message_request(self, request: Any) -> Any:
        """
        Handle a message request from a client, such as a tool call.

        This method will process the request, validate it, and forward it to
        the coordinator for execution. The response will be sent back via SSE.

        Args:
            request: The Starlette/FastAPI request object containing the message

        Returns:
            A JSONResponse with a confirmation of the message processing
        """
        from fastapi.responses import JSONResponse

        try:
            # Parse the request body
            request_body = await request.json()

            # Validate the request structure
            if not isinstance(request_body, dict):
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "error": "Request body must be a JSON object",
                    },
                )

            # Extract request fields
            request_id = request_body.get("request_id", str(uuid.uuid4()))
            operation = request_body.get("operation")
            params = request_body.get("parameters", {})

            if not operation:
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "error": "Missing required field 'operation'",
                    },
                )

            # Log the request
            self.logger.info(
                f"Received message request {request_id} for operation '{operation}'"
            )

            # Make sure we have a coordinator
            if not self._coordinator:
                self.logger.error("No coordinator available to process message request")
                return JSONResponse(
                    status_code=503,
                    content={
                        "success": False,
                        "error": "Server not ready to process requests",
                    },
                )

            # Start the request processing in the coordinator (non-blocking)
            # This will handle the request and send results via SSE
            self._loop.create_task(
                self._coordinator.start_request(
                    request_id=request_id,
                    transport_id=self.get_transport_id(),
                    operation_name=operation,
                    request_data={"parameters": params},
                )
            )

            # Return a 202 Accepted response immediately
            return JSONResponse(
                status_code=202,  # Accepted
                content={
                    "success": True,
                    "message": f"Request {request_id} accepted for processing",
                    "request_id": request_id,
                },
            )

        except json.JSONDecodeError:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Invalid JSON in request body"},
            )
        except Exception as e:
            self.logger.exception(f"Error processing message request: {e}")
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": "Internal server error processing request",
                },
            )


...
