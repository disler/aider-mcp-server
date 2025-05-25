from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Optional, Set

from aiohttp import web

from aider_mcp_server.atoms.security.context import SecurityContext
from aider_mcp_server.atoms.types.event_types import EventTypes

logger = logging.getLogger(__name__)


class SSETransportAdapter:  # Implements ITransportAdapter protocol
    """
    SSE Transport Adapter based on Task 6 specification.
    Uses aiohttp.web for SSE communication.
    """

    _app: web.Application
    _clients: Dict[str, web.StreamResponse]
    _client_lock: asyncio.Lock
    _transport_id: str
    _app_owned: bool
    _host: str
    _port: int
    _runner: Optional[web.AppRunner]
    _site: Optional[web.BaseSite]

    def __init__(
        self,
        transport_id: str,
        app: Optional[web.Application] = None,
        host: str = "127.0.0.1",
        port: int = 8000,
    ):
        self._transport_id = transport_id
        self._app_owned = app is None
        self._app = app or web.Application()
        self._host = host
        self._port = port
        self._clients = {}
        self._client_lock = asyncio.Lock()
        self._runner = None
        self._site = None
        self._setup_routes()

    def _setup_routes(self) -> None:
        """Set up the routes for the SSE endpoint."""
        self._app.router.add_get("/events", self._handle_sse_connection)

    # --- ITransportAdapter Methods ---

    def get_transport_id(self) -> str:
        """Get the unique identifier for this transport adapter."""
        return self._transport_id

    def get_transport_type(self) -> str:
        """Get the type of this transport adapter (e.g., 'sse', 'stdio')."""
        return "sse"

    async def initialize(self) -> None:
        """Initialize the SSE transport adapter."""
        # As per Task 6 spec: "Nothing to do here as routes are set up in __init__"
        pass

    async def start_listening(self) -> None:
        """Start listening for incoming connections if the app is owned by this adapter."""
        if self._app_owned and not self._runner:
            self._runner = web.AppRunner(self._app)
            await self._runner.setup()
            self._site = web.TCPSite(self._runner, self._host, self._port)
            await self._site.start()
            # print(f"SSE server running on http://{self._host}:{self._port}/events") # Optional: for debugging

    async def shutdown(self) -> None:
        """Clean up resources and shut down the adapter."""
        async with self._client_lock:
            for _client_id, response in list(self._clients.items()):  # Iterate over a copy
                try:
                    # Task 6 Spec: specific shutdown message format
                    await response.write(b"event: close\ndata: Server shutting down\n\n")
                except Exception:
                    # Ignore errors during shutdown message sending
                    logger.warning(
                        "Failed to send shutdown message to a client during overall shutdown.", exc_info=True
                    )
                    pass
            self._clients.clear()

        if self._runner:  # If we started the app runner
            await self._runner.cleanup()
            self._runner = None
            self._site = None

    def get_capabilities(self) -> Set[EventTypes]:
        """Return the event types supported by this transport adapter."""
        return {
            EventTypes.STATUS,
            EventTypes.PROGRESS,
            EventTypes.TOOL_RESULT,
            EventTypes.HEARTBEAT,
        }

    async def send_event(self, event_type: EventTypes, data: Dict[str, Any]) -> None:
        """
        Send an event to all connected SSE clients.
        This implements the ITransportAdapter.send_event method.
        """
        event_to_send = {"type": event_type.value}
        event_to_send.update(data)  # Merge data into the event payload

        async with self._client_lock:
            # Broadcast to all clients
            # Iterate over a copy of values in case of modification during iteration
            for response in list(self._clients.values()):
                try:
                    await self._send_sse_event(response, event_to_send)
                except Exception:
                    # Errors (like ConnectionResetError) will be handled by _handle_sse_connection's finally block
                    # which removes the client.
                    logger.warning("Failed to send SSE event to a client.", exc_info=True)
                    pass

    def should_receive_event(
        self,
        event_type: EventTypes,
        data: Dict[str, Any],
        request_details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Determine if this transport adapter should receive and handle a given event."""
        # Default: SSE transport typically broadcasts events it's given.
        # Specific filtering logic can be added here if needed.
        return True

    async def _handle_sse_connection(self, request: web.Request) -> web.StreamResponse:
        """Handle a new SSE connection from a client."""
        # Task 6 Spec: client_id generation
        client_id = request.query.get("client_id", str(id(request)))

        response = web.StreamResponse()
        response.headers["Content-Type"] = "text/event-stream"
        response.headers["Cache-Control"] = "no-cache"
        response.headers["Connection"] = "keep-alive"
        await response.prepare(request)

        async with self._client_lock:
            self._clients[client_id] = response

        # Send initial connection established event (custom type for this adapter)
        await self._send_sse_event(response, {"type": "connection_established", "client_id": client_id})

        try:
            # Keep connection alive until client disconnects
            while True:
                await asyncio.sleep(30)  # Heartbeat every 30 seconds
                # Use EventTypes for heartbeat
                await self._send_sse_event(response, {"type": EventTypes.HEARTBEAT.value})
        except ConnectionResetError:
            # Client disconnected
            pass
        except asyncio.CancelledError:
            # Task cancelled, e.g. server shutting down
            pass
        finally:
            # Unregister client on disconnect
            async with self._client_lock:
                if client_id in self._clients:
                    del self._clients[client_id]
        return response

    async def _send_sse_event(self, response: web.StreamResponse, event: Dict[str, Any]) -> None:
        """Send an SSE event to a client."""
        # Task 6 Spec: JSON data format for SSE
        event_data_json = json.dumps(event)
        try:
            await response.write(f"data: {event_data_json}\n\n".encode("utf-8"))
        except ConnectionResetError:
            # Connection might have been closed between check and write
            # This will be handled by the calling context (_handle_sse_connection or send_event)
            raise

    async def handle_message_request(self, request_details: Dict[str, Any]) -> Any:
        """Handle incoming message requests (not typical for SSE)."""
        # SSE is primarily server-to-client.
        # This method is part of ITransportAdapter, so it needs to be implemented.
        raise NotImplementedError("SSETransportAdapter does not support incoming messages via handle_message_request.")

    def validate_request_security(self, request_details: Dict[str, Any]) -> SecurityContext:
        """Validate the security of an incoming request."""
        # For SSE, the initial HTTP connection might be authenticated.
        # The event stream itself usually doesn't carry per-message auth.
        # This provides a default anonymous context.
        return SecurityContext(
            user_id=None,  # Or extract from request_details if available/applicable
            permissions=set(),
            is_anonymous=True,
            transport_id=self._transport_id,
        )

    # --- Methods from ITransportAdapter not explicitly in Task 6 spec, but needed for protocol ---
    # handle_sse_request is effectively _handle_sse_connection, which is wired to the router.
    # The ITransportAdapter's handle_sse_request signature is (self, request_details: Dict[str, Any]).
    # Our _handle_sse_connection takes (self, request: web.Request).
    # For aiohttp, the router directly calls _handle_sse_connection with web.Request.
    # If direct invocation via ITransportAdapter.handle_sse_request was needed, a wrapper would be required.
    # For now, assuming router-based invocation is primary for this method.

    # Expose the app object if it was created internally, so it can be run by an external manager if desired.
    @property
    def app(self) -> web.Application:
        return self._app
