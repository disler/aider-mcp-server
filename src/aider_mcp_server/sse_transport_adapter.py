"""
SSE Transport Adapter for MCP Server.

This module provides Server-Sent Events (SSE) transport capabilities for the MCP server,
allowing web clients to connect and receive real-time events through an HTTP connection.
"""

from __future__ import annotations  # Ensure forward references work

import asyncio
import json
import time
import typing
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
)
from aider_mcp_server.security import SecurityContext
from aider_mcp_server.transport_adapter import (
    AbstractTransportAdapter,
)

if TYPE_CHECKING:
    from aider_mcp_server.transport_coordinator import ApplicationCoordinator


class SSETransportAdapter(AbstractTransportAdapter):
    """SSE transport adapter for MCP server."""

    def __init__(
        self,
        coordinator: Optional["ApplicationCoordinator"] = None,
        host: str = "127.0.0.1",  # noqa: S104
        port: int = 8808,
        sse_queue_size: int = 10,
        get_logger: Optional[Any] = None,
        **kwargs: Any,  # Accept and ignore additional keyword arguments
    ):
        """
        Initialize the SSE transport adapter.

        Args:
            coordinator: The coordinator to use for transport operations
            host: The hostname/IP to bind the SSE server to (default: "0.0.0.0")
            port: The port to bind the SSE server to (default: 8808)
            sse_queue_size: Maximum size of the SSE event queue (default: 10)
            get_logger: Function to create logger instance
            **kwargs: Additional keyword arguments (ignored)
        """
        if get_logger is None:
            from aider_mcp_server.atoms.logging import get_logger

        super().__init__(
            transport_id="sse",
            transport_type="sse",
            coordinator=coordinator,
        )
        if callable(get_logger):
            self.logger = get_logger(__name__)
        else:
            import logging

            self.logger = logging.getLogger(__name__)  # type: ignore[assignment]

        self._host = host
        self._port = port
        self._sse_queue_size = sse_queue_size
        self._active_connections: Dict[str, asyncio.Queue[Union[str, Dict[str, str]]]] = {}
        self._server: Optional[Any] = None  # Starlette/Uvicorn server
        self._monitor_connections: Set[str] = set()
        self.monitor_stdio_transport_id: Optional[str] = None
        self._app: Optional[Any] = None  # Starlette app instance
        self._server_instance: Optional[Any] = None  # Uvicorn server instance

    async def initialize(self) -> None:
        """
        Initialize the SSE transport adapter.

        Sets up the Starlette app with SSE endpoints and prepares it for serving.
        """
        self.logger.info(f"Initializing SSE transport adapter on {self._host}:{self._port}")
        # Call parent initialization
        await super().initialize()

        # Create the Starlette app with SSE endpoints
        await self._create_app()
        self.logger.info("SSE transport adapter initialized")

    async def _create_app(self) -> None:
        """Create the Starlette application with SSE endpoints."""
        from starlette.applications import Starlette
        from starlette.routing import Route

        async def handle_sse(request: Any) -> Any:
            """Handle SSE connection requests."""
            return await self.handle_sse_request(request)

        self._app = Starlette(
            routes=[
                Route("/sse", handle_sse),
            ],
            debug=True,  # Enable debug mode for better error messages
        )

    async def start_listening(self) -> None:
        """Start listening for SSE connections."""
        self.logger.info(f"Starting SSE transport on {self._host}:{self._port}")

        # Import here to avoid circular dependency issues
        import uvicorn

        # Create server configuration
        config = uvicorn.Config(
            app=self._app,
            host=self._host,
            port=self._port,
            log_level="warning",  # Reduce Uvicorn's verbosity
            access_log=False,  # Disable access logs
        )

        # Create and start server
        self._server_instance = uvicorn.Server(config)

        # Run server in background task
        asyncio.create_task(self._server_instance.serve())
        self.logger.info(f"SSE transport started on {self._host}:{self._port}")

    async def shutdown(self) -> None:
        """Shutdown the SSE transport adapter."""
        self.logger.info("Shutting down SSE transport adapter")

        # Close all active connections by sending a close signal
        for client_id, queue in list(self._active_connections.items()):
            try:
                # Put a special message to signal connection close
                await queue.put("CLOSE_CONNECTION")
            except Exception as e:
                self.logger.debug(f"Error sending close signal to {client_id}: {e}")

        # Give some time for connections to close gracefully
        await asyncio.sleep(0.1)

        # Now force-clear all connections
        self._active_connections.clear()

        # Shutdown the server if it's running
        if self._server_instance:
            self.logger.debug("Shutting down Uvicorn server")
            await self._server_instance.shutdown()
            self._server_instance = None

        # Call parent shutdown
        await super().shutdown()
        self.logger.info("SSE transport adapter shut down")

    def get_capabilities(self) -> Set[EventTypes]:
        """
        Return the event types supported by this transport adapter.

        Returns:
            A set of EventTypes that this transport supports.
        """
        return {
            EventTypes.STATUS,
            EventTypes.PROGRESS,
            EventTypes.TOOL_RESULT,
            EventTypes.HEARTBEAT,
        }

    async def send_event(self, event_type: EventTypes, data: EventData) -> None:
        """Send an event to all connected SSE clients."""
        # Format as SSE message
        sse_message = f"event: {event_type.value}\ndata: {json.dumps(data)}\n\n"

        if event_type == EventTypes.PROGRESS:
            self.logger.debug(f"Broadcasting progress event to SSE clients: {data}")

        # Use list() to create a copy to avoid modification during iteration
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
        queue: asyncio.Queue[Union[str, Dict[str, str]]] = asyncio.Queue(maxsize=self._sse_queue_size)

        # Register this connection
        self._active_connections[client_id] = queue
        self.logger.info(f"SSE connection established: {client_id}")

        # Define the event stream generator
        async def event_generator() -> typing.AsyncGenerator[Union[str, Dict[str, str]], None]:
            """
            Generates SSE events for clients.

            Waits for messages from queue with a 30-second timeout.
            If no messages are received within the timeout, sends a heartbeat event
            to keep the connection alive and prevent client timeouts.
            """
            try:
                # Send initial connection event
                yield {
                    "event": "connected",
                    "data": json.dumps({"client_id": client_id}),
                }

                # Loop to pull messages from the queue
                while True:
                    try:
                        # Wait for a message with a 30-second timeout
                        message = await asyncio.wait_for(queue.get(), timeout=30.0)

                        if isinstance(message, str):
                            if message == "CLOSE_CONNECTION":
                                self.logger.debug(f"Received close signal for connection {client_id}. Closing stream.")
                                break
                            else:
                                # Yield the pre-formatted SSE message string directly
                                # This matches the format expected by test_sse_adapter_handle_sse_request
                                yield message
                        elif isinstance(message, dict):
                            # If it's a dict, yield it directly (for initial connection events)
                            yield message
                        queue.task_done()
                    except asyncio.TimeoutError:
                        # Heartbeat to keep connection alive
                        yield {
                            "event": "heartbeat",
                            "data": json.dumps({"timestamp": time.time()}),
                        }
                    except Exception as e:
                        self.logger.error(f"Error in SSE event stream for {client_id}: {e}")
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
        """Handle incoming message requests."""
        self.logger.debug("Handling message request (not implemented for SSE)")
        # SSE is unidirectional - messages go from server to client only
        # Return an error or appropriate response
        return {"error": "SSE transport does not support incoming messages"}

    def register_monitor_connection(self, connection_id: str) -> None:
        """
        Register a connection ID as a monitor that should receive events from stdio.

        Args:
            connection_id: The connection ID to register as a monitor
        """
        self._monitor_connections.add(connection_id)
        self.logger.info(f"Registered monitor connection: {connection_id}")

    def should_receive_event(
        self,
        event_type: EventTypes,
        data: EventData,
        request_details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Determine if this transport adapter should receive and handle a given event.

        For SSE transport, we want to receive events that:
        1. Should be forwarded to SSE clients (like stdio events when monitoring)
        2. Are not originated from SSE itself (to prevent loops)

        Args:
            event_type: The type of event being checked
            data: The event data
            request_details: Optional request details (not used for SSE)

        Returns:
            True if the event should be received, False otherwise
        """
        # If we're monitoring stdio transport, receive events from it
        if self.monitor_stdio_transport_id and data.get("transport_origin"):
            origin = data["transport_origin"]
            if origin.get("transport_id") == self.monitor_stdio_transport_id:
                self.logger.debug(f"SSE accepting event from monitored stdio transport: {event_type.value}")
                return True

        # Skip events that originated from us to prevent loops
        if data.get("transport_origin", {}).get("transport_id") == self.get_transport_id():
            return False

        # All other events should be received
        return True

    def validate_request_security(self, request_details: Dict[str, Any]) -> SecurityContext:
        """
        Validate the security of an incoming request.

        Args:
            request_details: Details about the incoming request.

        Returns:
            SecurityContext containing security validation information.
        """
        # For SSE, we typically don't have authentication
        # But we can validate origin headers and other security measures
        return SecurityContext(
            user_id=None,
            permissions=set(),
            is_anonymous=True,  # SSE connections are typically anonymous
            transport_id=self.get_transport_id(),
        )
