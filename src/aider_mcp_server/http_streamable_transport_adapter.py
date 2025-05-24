"""
HTTP Streamable Transport Adapter for MCP Server.

This module provides an HTTP-based streaming transport for the MCP server,
allowing clients to connect and receive real-time events over a persistent HTTP connection
and send messages via HTTP POST.
"""

from __future__ import annotations  # Ensure forward references work

import asyncio
import json
import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Dict,
    Optional,
    Set,
)

import uvicorn

# For MCP SDK integration
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from aider_mcp_server.atoms.types.event_types import EventTypes
from aider_mcp_server.atoms.types.mcp_types import (
    EventData,
    LoggerFactory,
    LoggerProtocol,
    RequestParameters,
)
from aider_mcp_server.atoms.security.context import SecurityContext
from aider_mcp_server.transport_adapter import (
    AbstractTransportAdapter,
)

if TYPE_CHECKING:
    from aider_mcp_server.transport_coordinator import ApplicationCoordinator

    # LoggerFactory and LoggerProtocol are now imported directly from mcp_types


# Custom exception for method dispatch
class MethodNotFoundError(Exception):
    """Raised when a requested MCP method is not found."""

    pass


# Pydantic model for parsed message payload
class ParsedMessagePayload(BaseModel):
    """Structure for a successfully parsed client message."""

    method_name: str
    params: Dict[str, Any]
    request_id_from_client: Optional[str]


class HttpStreamableTransportAdapter(AbstractTransportAdapter):
    """HTTP Streamable transport adapter for MCP server."""

    logger: LoggerProtocol

    def __init__(
        self,
        coordinator: Optional[ApplicationCoordinator] = None,
        host: str = "127.0.0.1",  # noqa: S104
        port: int = 8766,  # Default port, different from SSE
        stream_queue_size: int = 100,
        get_logger: Optional[LoggerFactory] = None,
        editor_model: str = "",
        current_working_dir: str = "",
        heartbeat_interval: float = 30.0,  # Default heartbeat interval in seconds
        **kwargs: Any,  # Accept and ignore additional keyword arguments
    ):
        """
        Initialize the HTTP Streamable transport adapter.

        Args:
            coordinator: The coordinator to use for transport operations.
            host: The hostname/IP to bind the server to.
            port: The port to bind the server to.
            stream_queue_size: Maximum size of the outgoing event queue for each client.
            get_logger: Function to create logger instance.
            editor_model: Configuration for the editor model.
            current_working_dir: Current working directory for operations.
            heartbeat_interval: Interval for sending heartbeats.
            **kwargs: Additional keyword arguments (ignored).
        """
        effective_transport_id = kwargs.pop("transport_id", f"http_stream_{uuid.uuid4()}")

        super().__init__(
            transport_id=effective_transport_id,
            transport_type="http_stream",
            coordinator=coordinator,
            heartbeat_interval=heartbeat_interval,
        )

        # Logger setup
        from aider_mcp_server.atoms.logging.logger import get_logger as default_get_logger

        # Determine the logger function to use (either the passed one or the default)
        effective_get_logger = default_get_logger if get_logger is None else get_logger
        self.logger = effective_get_logger(f"{__name__}.{self.__class__.__name__}.{self.get_transport_id()}")

        self._host = host
        self._port = port
        self._stream_queue_size = stream_queue_size
        self._editor_model = editor_model
        self._current_working_dir = current_working_dir

        self._active_connections: Dict[str, asyncio.Queue[str]] = {}  # client_id -> queue for outgoing messages
        self._app: Optional[Starlette] = None
        self._server_instance: Optional[uvicorn.Server] = None
        self._server_task: Optional[asyncio.Task[None]] = None

        self._mcp_server: Optional[FastMCP] = None
        self._fastmcp_initialized = False
        self.logger.info(f"HttpStreamableTransportAdapter instance created with ID {self.get_transport_id()}")

    async def initialize(self) -> None:
        """Initialize the adapter: parent init, FastMCP, Starlette app."""
        await super().initialize()  # Registers with coordinator, starts heartbeat
        if self._coordinator:
            self._initialize_fastmcp()
        else:
            self.logger.warning(
                "No coordinator available, FastMCP will not be initialized for HttpStreamableTransportAdapter"
            )
        await self._create_app()
        self.logger.debug(f"HttpStreamableTransportAdapter {self.get_transport_id()} initialized.")

    def _initialize_fastmcp(self) -> None:
        """Initialize FastMCP and register tools."""
        if self._fastmcp_initialized or not self._coordinator:
            self.logger.debug("FastMCP already initialized or no coordinator, skipping.")
            return

        self._mcp_server = FastMCP(f"aider-http-stream-{self.get_transport_id()}")

        # Import handlers for registration
        from aider_mcp_server.handlers import (
            process_aider_ai_code_request,
            process_list_models_request,
        )

        @self._mcp_server.tool()
        async def aider_ai_code(
            ai_coding_prompt: str,
            relative_editable_files: list[str],
            relative_readonly_files: Optional[list[str]] = None,
            model: Optional[str] = None,
        ) -> dict[str, Any]:
            """MCP Tool: Run Aider to perform AI coding tasks."""
            self.logger.debug(f"HTTP Stream (MCP Tool): aider_ai_code called with prompt: {ai_coding_prompt[:50]}...")
            params = {
                "ai_coding_prompt": ai_coding_prompt,
                "relative_editable_files": relative_editable_files,
                "relative_readonly_files": relative_readonly_files or [],
                "model": model,
            }
            request_id = f"mcp_http_stream_{uuid.uuid4()}"
            # This security context is for the tool execution if called via FastMCP directly.
            # Requests via /message endpoint will have their own context.
            security_context = SecurityContext(
                user_id=None, permissions=set(), is_anonymous=True, transport_id=self.get_transport_id()
            )
            try:
                result = await process_aider_ai_code_request(
                    request_id=request_id,
                    transport_id=self.get_transport_id(),  # Or a generic MCP transport ID
                    params=params,
                    security_context=security_context,
                    editor_model=self._editor_model,
                    current_working_dir=self._current_working_dir,
                )
                return result.result if hasattr(result, "result") else result
            except Exception as e:
                self.logger.error(f"Error in HTTP Stream (MCP Tool) aider_ai_code: {e}", exc_info=True)
                return {"error": str(e)}

        @self._mcp_server.tool()
        async def list_models(substring: Optional[str] = None) -> list[str]:
            """MCP Tool: List available models."""
            self.logger.debug(f"HTTP Stream (MCP Tool): list_models called with substring: {substring}")
            params = {"substring": substring} if substring else {}
            request_id = f"mcp_http_stream_{uuid.uuid4()}"
            security_context = SecurityContext(
                user_id=None, permissions=set(), is_anonymous=True, transport_id=self.get_transport_id()
            )
            try:
                result = await process_list_models_request(
                    request_id=request_id,
                    transport_id=self.get_transport_id(),  # Or a generic MCP transport ID
                    params=params,
                    security_context=security_context,
                )
                if isinstance(result, dict) and "models" in result:
                    models_list = result.get("models", [])
                    if isinstance(models_list, list):
                        return models_list
                return []
            except Exception as e:
                self.logger.error(f"Error in HTTP Stream (MCP Tool) list_models: {e}", exc_info=True)
                return []

        self._fastmcp_initialized = True
        self.logger.info(f"FastMCP initialized for HttpStreamableTransportAdapter {self.get_transport_id()}.")

    async def _create_app(self) -> None:
        """Create the Starlette application with streaming and message endpoints."""
        self._app = Starlette(
            routes=[
                Route("/stream/{client_id}", self.handle_stream_request, methods=["GET"]),
                Route("/message/{client_id}", self.handle_message_request, methods=["POST"]),
            ],
            debug=True,  # Enable debug for development; consider disabling for production
        )
        self.logger.info(f"Starlette app created for HttpStreamableTransportAdapter {self.get_transport_id()}.")

    async def start_listening(self) -> None:
        """Start the Uvicorn server to listen for HTTP connections."""
        if self._app is None:
            self.logger.error("App not initialized. Call _create_app() before start_listening().")
            raise RuntimeError("App not initialized. Call _create_app() first.")

        config = uvicorn.Config(
            app=self._app,
            host=self._host,
            port=self._port,
            log_level="warning",  # Uvicorn's own log level
            access_log=False,  # Disable Uvicorn's access logs if our logging is sufficient
        )
        self._server_instance = uvicorn.Server(config)
        self._server_task = asyncio.create_task(self._server_instance.serve())

        # Wait for the server to start and bind to a port
        # Uvicorn's 'started' flag is a reliable way to know it's ready
        while not self._server_instance.started:
            await asyncio.sleep(0.01)

        actual_port = self.get_actual_port()
        if actual_port is None:
            self.logger.error(
                f"HttpStreamableTransportAdapter server started but could not determine actual listening port. Configured port was {self._port}."
            )
            # Potentially raise an error or handle this case more robustly
        else:
            self.logger.info(f"HttpStreamableTransportAdapter server listening on http://{self._host}:{actual_port}")

    def get_actual_port(self) -> Optional[int]:
        """Return the actual port the Uvicorn server is listening on."""
        if self._server_instance and self._server_instance.started and self._server_instance.servers:
            # Uvicorn server can have multiple sockets (e.g., for HTTP and HTTPS)
            # We assume the first one is what we're interested in for this basic HTTP setup.
            # The socket object might vary, so we check for 'getsockname'
            for server_socket_group in self._server_instance.servers:
                if hasattr(server_socket_group, "sockets") and server_socket_group.sockets:
                    socket_info = server_socket_group.sockets[0].getsockname()
                    # Ensure socket_info is a tuple with at least host and port
                    if isinstance(socket_info, tuple) and len(socket_info) >= 2:
                        port = socket_info[1]
                        if isinstance(port, int):
                            return port
                        else:
                            self.logger.warning(f"Unexpected port type: {type(port)}")
                            return None
                    else:
                        self.logger.warning(f"Unexpected socket_info format: {socket_info}")
                        return None
        self.logger.debug("Server instance not available or not started, or sockets not found, cannot get actual port.")
        return None

    async def shutdown(self) -> None:
        """Shutdown the adapter: close connections, stop server, unregister."""
        self.logger.info(f"Shutting down HttpStreamableTransportAdapter ({self.get_transport_id()})...")

        # Signal all active stream generators to close and clear connections
        client_ids = list(self._active_connections.keys())
        for client_id in client_ids:
            queue = self._active_connections.get(client_id)
            if queue:
                try:
                    # Special message to signal the stream generator to terminate
                    await queue.put("CLOSE_STREAM")
                except Exception as e:
                    self.logger.debug(f"Error sending CLOSE_STREAM to client {client_id} queue: {e}")

        # Allow a moment for streams to process the close signal
        if client_ids:
            await asyncio.sleep(0.2)
        self._active_connections.clear()
        self.logger.debug(f"Cleared all active connections for {self.get_transport_id()}.")

        # Shutdown Uvicorn server
        if self._server_instance and self._server_instance.started:
            self.logger.debug(f"Shutting down Uvicorn server for {self.get_transport_id()}.")
            self._server_instance.should_exit = True
            if self._server_task and not self._server_task.done():
                try:
                    await asyncio.wait_for(self._server_task, timeout=5.0)
                except asyncio.TimeoutError:
                    self.logger.warning(
                        f"HttpStreamableTransportAdapter Uvicorn server shutdown timed out for {self.get_transport_id()}."
                    )
                except asyncio.CancelledError:
                    self.logger.debug(f"Uvicorn server task was cancelled for {self.get_transport_id()}.")
            self._server_instance = None
            self._server_task = None

        # Call parent shutdown (handles coordinator unregistration and heartbeat cancellation)
        await super().shutdown()
        self.logger.info(f"HttpStreamableTransportAdapter ({self.get_transport_id()}) shut down successfully.")

    async def _stream_generator(self, client_id: str, queue: asyncio.Queue[str]) -> AsyncGenerator[str, None]:
        """Async generator for sending messages to a single client stream."""
        self.logger.debug(f"Stream generator started for client {client_id}.")
        try:
            while True:
                message = await queue.get()
                if message == "CLOSE_STREAM":
                    self.logger.info(f"Received CLOSE_STREAM for client {client_id}, closing stream.")
                    break
                yield f"{message}\n"  # Newline delimited JSON objects
                queue.task_done()
        except asyncio.CancelledError:
            self.logger.info(f"Stream generator for client {client_id} was cancelled (e.g., client disconnected).")
        except Exception as e:
            self.logger.error(f"Error in stream generator for client {client_id}: {e}", exc_info=True)
        finally:
            self.logger.debug(f"Stream generator for client {client_id} finished.")
            # Ensure connection is removed if it still exists (e.g., if loop broke due to error)
            if client_id in self._active_connections:
                del self._active_connections[client_id]
                self.logger.info(f"Removed active connection for client {client_id} after stream ended/error.")

    async def handle_stream_request(self, request: Request) -> Response:
        """Handle new client connection requests for the HTTP stream."""
        client_id = request.path_params.get("client_id")
        if not client_id:
            self.logger.warning("Stream request received without client_id in path.")
            return Response("client_id path parameter is required.", status_code=400, media_type="text/plain")

        if client_id in self._active_connections:
            self.logger.warning(f"Client {client_id} attempted to connect but already has an active stream.")
            return Response(
                f"Client {client_id} already connected. Reconnection not yet supported this way.",
                status_code=409,
                media_type="text/plain",
            )

        queue: asyncio.Queue[str] = asyncio.Queue(self._stream_queue_size)
        self._active_connections[client_id] = queue
        self.logger.info(
            f"Client {client_id} connected to stream. Total active connections: {len(self._active_connections)}"
        )

        # Send an initial status message
        try:
            initial_event_data = {
                "message": "Successfully connected to HTTP stream.",
                "client_id": client_id,
                "transport_id": self.get_transport_id(),
            }
            initial_message_payload = {"event": EventTypes.STATUS.value, "data": initial_event_data}
            await queue.put(json.dumps(initial_message_payload))
        except Exception as e:
            self.logger.error(f"Failed to send initial connection message to {client_id}: {e}", exc_info=True)
            # Proceed with stream anyway, or terminate? For now, proceed.

        return StreamingResponse(
            self._stream_generator(client_id, queue),
            media_type="application/x-ndjson",  # Newline Delimited JSON
        )

    async def _parse_message_payload(
        self, request: Request, client_id: str
    ) -> tuple[Optional[ParsedMessagePayload], Optional[JSONResponse]]:
        """Parses and validates the JSON payload from an HTTP request."""
        try:
            payload_dict = await request.json()
            self.logger.debug(f"Received message from client {client_id}: {str(payload_dict)[:200]}...")
        except json.JSONDecodeError:
            self.logger.error(f"Invalid JSON payload received from client {client_id}.")
            return None, JSONResponse({"error": "Invalid JSON payload.", "id": None}, status_code=400)
        except Exception as e:
            self.logger.error(f"Error reading message request body from client {client_id}: {e}", exc_info=True)
            return None, JSONResponse({"error": "Failed to process request body.", "id": None}, status_code=500)

        method_name = payload_dict.get("method")
        params = payload_dict.get("params", {})
        request_id_from_client = payload_dict.get("id")

        if not method_name or not isinstance(method_name, str):
            return None, JSONResponse(
                {"error": "Missing or invalid 'method' in request payload.", "id": request_id_from_client},
                status_code=400,
            )
        if not isinstance(params, dict):
            return None, JSONResponse(
                {"error": "Invalid 'params' in request payload, must be an object.", "id": request_id_from_client},
                status_code=400,
            )

        return (
            ParsedMessagePayload(method_name=method_name, params=params, request_id_from_client=request_id_from_client),
            None,  # No error response means success
        )

    async def _dispatch_request_to_handler(
        self,
        method_name: str,
        params: RequestParameters,
        internal_request_id: str,
        security_context: SecurityContext,
    ) -> Any:
        """Dispatches the request to the appropriate handler based on method_name."""
        if method_name == "aider_ai_code":
            from aider_mcp_server.handlers import process_aider_ai_code_request

            result_obj = await process_aider_ai_code_request(
                request_id=internal_request_id,
                transport_id=self.get_transport_id(),
                params=params,
                security_context=security_context,
                editor_model=self._editor_model,
                current_working_dir=self._current_working_dir,
            )
            return result_obj.result if hasattr(result_obj, "result") else result_obj
        elif method_name == "list_models":
            from aider_mcp_server.handlers import process_list_models_request

            return await process_list_models_request(
                request_id=internal_request_id,
                transport_id=self.get_transport_id(),
                params=params,
                security_context=security_context,
            )
        else:
            raise MethodNotFoundError(f"Method '{method_name}' not found.")

    async def handle_message_request(self, request: Request) -> JSONResponse:
        """Handle incoming messages from clients via HTTP POST."""
        client_id = request.path_params.get("client_id")
        if not client_id:
            self.logger.warning("Message POST request received without client_id in path.")
            return JSONResponse({"error": "client_id path parameter is required.", "id": None}, status_code=400)

        if client_id not in self._active_connections:
            self.logger.warning(f"Message received from client {client_id} which has no active stream.")
            return JSONResponse(
                {"error": f"Client {client_id} not connected to an active stream.", "id": None}, status_code=404
            )

        parsed_payload, error_response = await self._parse_message_payload(request, client_id)
        if error_response:
            return error_response

        # At this point, parsed_payload should be valid since error_response is None
        if parsed_payload is None:
            # This should not happen based on the logic in _parse_message_payload
            return JSONResponse({"error": "Internal error: payload parsing failed", "id": None}, status_code=500)

        method_name = parsed_payload.method_name
        params = parsed_payload.params
        request_id_from_client = parsed_payload.request_id_from_client

        internal_request_id = f"http_msg_{self.get_transport_id()}_{uuid.uuid4()}"
        self.logger.info(
            f"Processing method '{method_name}' for client {client_id}, req_id_client: {request_id_from_client}, req_id_internal: {internal_request_id}"
        )

        try:
            # Reconstruct a payload dict for security validation if it expects the original structure
            # This might need adjustment based on what validate_request_security actually needs
            validation_payload_dict = {
                "method": method_name,
                "params": params,
                "id": request_id_from_client,
            }
            security_context = self.validate_request_security(
                {"client_id": client_id, "payload": validation_payload_dict, "headers": request.headers}
            )

            response_data = await self._dispatch_request_to_handler(
                method_name, params, internal_request_id, security_context
            )
            return JSONResponse({"result": response_data, "id": request_id_from_client})

        except MethodNotFoundError as e:
            self.logger.warning(f"Unknown method '{method_name}' requested by client {client_id}: {e}")
            return JSONResponse(
                {"error": {"message": str(e), "code": -32601}, "id": request_id_from_client},
                status_code=404,
            )
        except PermissionError as e:
            self.logger.warning(
                f"Permission denied for method '{method_name}' from client {client_id}: {e}"
            )  # exc_info=True removed for brevity on permission errors
            return JSONResponse(
                {"error": {"message": str(e), "code": -32001}, "id": request_id_from_client}, status_code=403
            )
        except ValueError as e:  # Parameter validation errors from handlers or dispatch
            self.logger.warning(
                f"Invalid parameters for method '{method_name}' from client {client_id}: {e}", exc_info=True
            )
            return JSONResponse(
                {"error": {"message": str(e), "code": -32602}, "id": request_id_from_client}, status_code=400
            )
        except Exception as e:
            self.logger.error(
                f"Unexpected error processing method '{method_name}' for client {client_id}: {e}", exc_info=True
            )
            return JSONResponse(
                {
                    "error": {"message": f"Internal server error: {str(e)}", "code": -32000},
                    "id": request_id_from_client,
                },
                status_code=500,
            )

    async def send_event(self, event_type: EventTypes, data: EventData) -> None:
        """Send an event to all connected HTTP stream clients."""
        # Ensure transport_origin is set, preferably by the coordinator or source.
        # If not, this adapter can add its own origin info.
        final_data = data.copy()  # Avoid modifying the input dict if it's reused
        if "transport_origin" not in final_data:
            final_data["transport_origin"] = {
                "transport_id": self.get_transport_id(),
                "transport_type": self.get_transport_type(),
            }

        message_payload = {"event": event_type.value, "data": final_data}

        try:
            json_message_str = json.dumps(message_payload)
        except TypeError as e:
            self.logger.error(
                f"JSON serialization error for event type {event_type.value}: {e}. Data: {str(final_data)[:200]}...",
                exc_info=True,
            )
            return

        if event_type == EventTypes.PROGRESS:
            # Avoid overly verbose logging for progress, log snippet or less frequently
            progress_message = final_data.get(
                "message", str(final_data.get("completed_steps", "")) + "/" + str(final_data.get("total_steps", ""))
            )
            self.logger.debug(f"Broadcasting PROGRESS to HTTP stream clients: {progress_message}")
        elif event_type == EventTypes.HEARTBEAT:
            self.logger.debug(f"Broadcasting HEARTBEAT to {len(self._active_connections)} HTTP stream clients.")
        else:
            self.logger.debug(
                f"Broadcasting {event_type.value} to {len(self._active_connections)} HTTP stream clients."
            )

        for client_id in list(self._active_connections.keys()):  # Iterate on copy for safe modification
            queue = self._active_connections.get(client_id)
            if queue:
                try:
                    queue.put_nowait(json_message_str)
                except asyncio.QueueFull:
                    self.logger.warning(
                        f"Outgoing queue full for HTTP stream client {client_id}. Event {event_type.value} dropped."
                    )
                    # Future: consider strategies for persistent queue full (e.g., disconnect client)
                except Exception as e:  # Should be rare for put_nowait
                    self.logger.error(
                        f"Error putting event into queue for HTTP stream client {client_id}: {e}", exc_info=True
                    )
            else:
                # This case should ideally not happen if cleanup is correct
                self.logger.debug(f"Client {client_id} connection queue not found while sending {event_type.value}.")

    def get_capabilities(self) -> Set[EventTypes]:
        """Return the set of event types supported by this transport."""
        return {
            EventTypes.STATUS,
            EventTypes.PROGRESS,
            EventTypes.TOOL_RESULT,
            EventTypes.HEARTBEAT,
        }

    def validate_request_security(self, request_details: RequestParameters) -> SecurityContext:
        """
        Validate security of an incoming request. Placeholder implementation.

        Args:
            request_details: Dict containing details like 'client_id', 'payload', 'headers'.

        Returns:
            SecurityContext object.

        Raises:
            PermissionError: If authentication or authorization fails.
        """
        # client_id = request_details.get("client_id", "unknown_client")
        # headers = request_details.get("headers") # Starlette Headers object
        # Example: token = headers.get("Authorization", "").replace("Bearer ", "")
        # user_id, permissions = self._auth_service.validate_token(token) ...

        self.logger.debug(
            f"Performing placeholder security validation for request from client {request_details.get('client_id')}."
        )
        # For now, all requests are treated as anonymous with no specific permissions.
        return SecurityContext(
            user_id=None,
            permissions=set(),
            is_anonymous=True,
            transport_id=self.get_transport_id(),
            # additional_context = {"client_ip": request.client.host} # If request object is passed
        )

    def should_receive_event(
        self,
        event_type: EventTypes,
        data: EventData,
        request_details: Optional[Dict[str, Any]] = None,  # Not typically used by this method
    ) -> bool:
        """
        Determine if this transport adapter should receive and handle a given event.
        Prevents event loops by not re-processing events originating from itself,
        unless it's a self-generated heartbeat intended for its clients.
        """
        origin_transport_info = data.get("transport_origin", {})
        origin_transport_id = origin_transport_info.get("transport_id")

        if origin_transport_id == self.get_transport_id():
            # If the event originated from this transport instance:
            # Allow heartbeats that this instance generated for its own clients.
            # These heartbeats have data["transport_id"] == self.get_transport_id().
            if event_type == EventTypes.HEARTBEAT and data.get("transport_id") == self.get_transport_id():
                self.logger.debug(f"HttpStream ({self.get_transport_id()}) will send self-generated HEARTBEAT.")
                return True

            # Otherwise, skip the event to prevent loops.
            self.logger.debug(
                f"HttpStream ({self.get_transport_id()}) skipping event {event_type.value} as it originated from self and is not a self-generated heartbeat."
            )
            return False

        # For events from other transports, accept them.
        # Future: Add logic for monitoring specific other transports if needed.
        self.logger.debug(
            f"HttpStream ({self.get_transport_id()}) will process event {event_type.value} from origin {origin_transport_id}."
        )
        return True
