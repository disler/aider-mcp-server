import asyncio
import json
import time
import uuid
from typing import Any, Dict, Optional, cast

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from aider_mcp_server.atoms.logging.logger import get_logger
from aider_mcp_server.interfaces.transport_registry import TransportAdapterRegistry
from aider_mcp_server.molecules.transport.discovery import CoordinatorDiscovery  # Import CoordinatorDiscovery
from ...pages.application.coordinator import ApplicationCoordinator
from aider_mcp_server.organisms.transports.sse.sse_transport_adapter import SSETransportAdapter

# Define global adapter variable for use in route handlers
_adapter: Optional[SSETransportAdapter] = None

# Active SSE connections for event streaming
_event_connections: Dict[str, Dict[str, asyncio.Queue[Any]]] = {
    "aider": {},  # General AIDER events
    "errors": {},  # Error-specific events
    "progress": {},  # Progress update events
}

# Get logger using the pattern established in other modules
logger = get_logger("app")

# Global event listener for coordinator broadcasts
_coordinator_event_listener: Optional[asyncio.Task[None]] = None


def _check_adapter_availability() -> Optional[Response]:
    """Check if the adapter is available and not shutting down."""
    if not _adapter:
        logger.error("Request received but SSE Transport Adapter is not initialized.")
        return Response(content="Server not ready", status_code=503, media_type="text/plain")

    if _adapter._coordinator and _adapter._coordinator.is_shutting_down():
        logger.warning("Request rejected: Server is shutting down.")
        return Response(content="Server is shutting down", status_code=503, media_type="text/plain")

    return None


async def _handle_sse_request(request: Request) -> Response:
    """Handle SSE connection requests with error handling."""
    if error_response := _check_adapter_availability():
        return error_response

    # Ensure adapter is available before calling its methods
    if _adapter is None:
        logger.error("Adapter unexpectedly None despite availability check")
        return Response(content="Server not ready", status_code=503, media_type="text/plain")

    try:
        # Delegate the request handling to the adapter
        return cast(Response, await _adapter.handle_sse_request(request))
    except HTTPException as http_exc:
        # Re-raise FastAPI/Starlette HTTP exceptions directly
        logger.warning(f"HTTPException during SSE request: {http_exc.status_code} - {http_exc.detail}")
        raise http_exc
    except Exception as e:
        # Catch any other unexpected errors during adapter processing
        logger.exception(f"Unexpected error handling SSE connection request: {e}")
        # Return a generic 500 Internal Server Error
        return Response(
            content="Internal server error handling SSE request",
            status_code=500,
            media_type="text/plain",
        )


async def _extract_request_id(request: Request) -> str:
    """Extract request_id from request body for better error logging."""
    request_id = "unknown"  # Default request_id for logging if parsing fails
    try:
        # Attempt to parse the request body to get request_id for better logging context
        body = await request.json()
        if isinstance(body, dict):
            request_id = body.get("request_id", "unknown")
    except Exception:
        # Ignore errors during body parsing for logging purposes
        logger.warning("Failed to parse request body to extract request_id for logging during error handling.")
    return request_id


async def _handle_message_request(request: Request) -> Response:
    """Handle message requests with error handling."""
    # Check adapter availability
    if not _adapter:
        logger.error("Message request received but SSE Transport Adapter is not initialized.")
        # Raise 503 Service Unavailable if adapter isn't ready
        raise HTTPException(status_code=503, detail="Server not ready")

    # Ensure adapter is not None before accessing its attributes
    adapter = _adapter  # Local variable to satisfy type checker
    if adapter is None:
        # This should never happen due to the check above, but keeps mypy happy
        raise HTTPException(status_code=503, detail="Server not ready")

    if adapter._coordinator and adapter._coordinator.is_shutting_down():
        logger.warning("Message request rejected: Server is shutting down.")
        raise HTTPException(status_code=503, detail="Server is shutting down")

    try:
        # Delegate the request handling to the adapter
        return cast(Response, await adapter.handle_message_request(request))
    except HTTPException as http_exc:
        # Re-raise FastAPI/Starlette HTTP exceptions directly
        logger.warning(f"HTTPException during message request: {http_exc.status_code} - {http_exc.detail}")
        raise http_exc
    except Exception as e:
        # Catch any other unexpected errors during adapter processing
        request_id = await _extract_request_id(request)
        logger.exception(f"Unexpected error handling message request {request_id}: {e}")
        # Return a generic 500 Internal Server Error as a JSON response
        return JSONResponse(
            content={
                "success": False,
                "error": "Internal server error processing message request",
            },
            status_code=500,
        )


async def _broadcast_to_sse_clients(event_type_str: str, event_data: Dict[str, Any]) -> None:
    """Broadcast coordinator events to relevant SSE clients."""
    try:
        # Determine which SSE endpoint types should receive this event
        event_routing = {
            "aider.rate_limit_detected": ["aider", "errors"],
            "aider.session_started": ["aider", "progress"],
            "aider.session_progress": ["aider", "progress"],
            "aider.session_completed": ["aider", "progress"],
            "aider.throttling_detected": ["aider", "errors"],
            "aider.error_occurred": ["aider", "errors"],
        }

        target_endpoints = event_routing.get(event_type_str, [])
        if not target_endpoints:
            logger.debug(f"No SSE endpoints configured for event type: {event_type_str}")
            return

        # Create SSE event payload
        sse_event = {
            "type": event_type_str,
            "data": {**event_data, "sse_timestamp": time.time(), "correlation_id": str(uuid.uuid4())},
            "id": str(uuid.uuid4()),
        }

        # Send to all relevant client queues
        delivered_count = 0
        for endpoint_type in target_endpoints:
            if endpoint_type in _event_connections:
                for client_id, client_queue in _event_connections[endpoint_type].items():
                    try:
                        # Non-blocking put with immediate discard if queue is full
                        client_queue.put_nowait(sse_event)
                        delivered_count += 1
                    except asyncio.QueueFull:
                        logger.warning(
                            f"Queue full for client {client_id} on {endpoint_type}, dropping event {event_type_str}"
                        )
                    except Exception as e:
                        logger.error(f"Error delivering event to client {client_id}: {e}")

        if delivered_count > 0:
            logger.debug(f"Delivered event {event_type_str} to {delivered_count} SSE clients")

    except Exception as e:
        logger.error(f"Error broadcasting event {event_type_str} to SSE clients: {e}")


async def _start_coordinator_event_listener() -> None:
    """Start listening for coordinator events and relay them to SSE clients."""
    global _coordinator_event_listener

    if _coordinator_event_listener is not None:
        logger.warning("Coordinator event listener already running")
        return

    if not _adapter or not _adapter._coordinator:
        logger.error("Cannot start event listener: no coordinator available")
        return

    async def event_listener_task() -> None:
        """Background task to listen for coordinator events."""
        logger.info("Starting coordinator event listener for SSE broadcasting")

        # This is a simplified polling approach for demonstration
        # In a production system, this should use proper event subscription
        # when coordinator.subscribe_to_event() is available

        try:
            while True:
                await asyncio.sleep(0.1)  # Prevent busy waiting

                # For now, this is a placeholder that demonstrates the integration point
                # Real implementation will connect to coordinator's event bus
                # when Phase 1 coordinator.broadcast_event() is extended with subscription support

        except asyncio.CancelledError:
            logger.info("Coordinator event listener cancelled")
        except Exception as e:
            logger.error(f"Error in coordinator event listener: {e}")

    _coordinator_event_listener = asyncio.create_task(event_listener_task())
    logger.info("Coordinator event listener started")


async def _stop_coordinator_event_listener() -> None:
    """Stop the coordinator event listener."""
    global _coordinator_event_listener

    if _coordinator_event_listener is not None:
        _coordinator_event_listener.cancel()
        try:
            await _coordinator_event_listener
        except asyncio.CancelledError:
            pass
        _coordinator_event_listener = None
        logger.info("Coordinator event listener stopped")


async def _create_event_stream(event_type: str, client_id: str) -> None:
    """Create event stream generator for SSE monitoring endpoints."""
    if _check_adapter_availability():
        return

    if not _adapter or not _adapter._coordinator:
        logger.error(f"No coordinator available for {event_type} event stream")
        return

    # Create client queue for events
    client_queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=100)
    _event_connections[event_type][client_id] = client_queue

    # Define event types to subscribe to based on endpoint
    event_filters = {
        "aider": [
            "aider.rate_limit_detected",
            "aider.session_started",
            "aider.session_progress",
            "aider.session_completed",
            "aider.throttling_detected",
            "aider.error_occurred",
        ],
        "errors": ["aider.rate_limit_detected", "aider.error_occurred", "aider.throttling_detected"],
        "progress": ["aider.session_started", "aider.session_progress", "aider.session_completed"],
    }

    target_events = event_filters.get(event_type, [])
    logger.info(f"Client {client_id} subscribing to {event_type} events: {target_events}")

    # Store subscription info for the client
    try:
        client_queue._subscribed_events = target_events  # type: ignore[attr-defined]
        client_queue._event_type = event_type  # type: ignore[attr-defined]
        logger.debug(f"Event stream setup complete for client {client_id} on {event_type}")

        # Ensure coordinator event listener is running
        await _start_coordinator_event_listener()

    except Exception as e:
        logger.error(f"Error setting up event stream for {event_type}: {e}")
        if client_id in _event_connections[event_type]:
            del _event_connections[event_type][client_id]


async def _generate_sse_events(event_type: str, client_id: str) -> Any:
    """Generate SSE events for monitoring endpoints."""
    try:
        # Get client queue
        if client_id not in _event_connections[event_type]:
            logger.warning(f"Client {client_id} not found in {event_type} connections")
            return

        client_queue = _event_connections[event_type][client_id]

        # Send initial connection event
        connection_event = {
            "type": f"{event_type}.connection_established",
            "data": {
                "client_id": client_id,
                "timestamp": time.time(),
                "event_types": getattr(client_queue, "_subscribed_events", []),
                "status": "connected",
            },
            "id": str(uuid.uuid4()),
        }

        yield f"event: {connection_event['type']}\n"
        yield f"data: {json.dumps(connection_event['data'])}\n"
        yield f"id: {connection_event['id']}\n\n"

        # Keep connection alive and send heartbeat events
        heartbeat_interval = 30.0  # seconds
        last_heartbeat = time.time()

        while True:
            try:
                # Check for new events with timeout for heartbeat
                current_time = time.time()
                timeout = max(0.1, heartbeat_interval - (current_time - last_heartbeat))

                try:
                    event = await asyncio.wait_for(client_queue.get(), timeout=timeout)

                    # Send the actual event
                    yield f"event: {event['type']}\n"
                    yield f"data: {json.dumps(event['data'])}\n"
                    yield f"id: {event['id']}\n\n"

                except asyncio.TimeoutError:
                    # Send heartbeat
                    if time.time() - last_heartbeat >= heartbeat_interval:
                        heartbeat_event = {
                            "type": f"{event_type}.heartbeat",
                            "data": {"timestamp": time.time(), "client_id": client_id, "status": "alive"},
                            "id": str(uuid.uuid4()),
                        }

                        yield f"event: {heartbeat_event['type']}\n"
                        yield f"data: {json.dumps(heartbeat_event['data'])}\n"
                        yield f"id: {heartbeat_event['id']}\n\n"

                        last_heartbeat = time.time()

            except asyncio.CancelledError:
                logger.info(f"SSE connection cancelled for client {client_id} on {event_type}")
                break
            except Exception as e:
                logger.error(f"Error in SSE event stream for client {client_id}: {e}")
                error_event = {
                    "type": f"{event_type}.error",
                    "data": {"error": str(e), "timestamp": time.time(), "client_id": client_id},
                    "id": str(uuid.uuid4()),
                }

                yield f"event: {error_event['type']}\n"
                yield f"data: {json.dumps(error_event['data'])}\n"
                yield f"id: {error_event['id']}\n\n"
                break

    except Exception as e:
        logger.error(f"Fatal error in SSE event generator for {event_type}: {e}")
    finally:
        # Cleanup connection
        if client_id in _event_connections[event_type]:
            del _event_connections[event_type][client_id]
            logger.info(f"Cleaned up SSE connection for client {client_id} on {event_type}")


async def _create_sse_monitoring_endpoint(event_type: str) -> EventSourceResponse:
    """Helper to create SSE monitoring endpoints."""
    if _check_adapter_availability():
        raise HTTPException(status_code=503, detail="Server not ready")

    client_id = str(uuid.uuid4())
    logger.info(f"New client {client_id} connecting to /events/{event_type}")

    # Set up event stream
    await _create_event_stream(event_type, client_id)

    return EventSourceResponse(
        _generate_sse_events(event_type, client_id),
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Client-ID": client_id},
    )


def _setup_core_routes(app: FastAPI) -> None:
    """Set up core SSE and message endpoints."""

    @app.get("/sse")
    async def sse_endpoint(request: Request) -> Response:
        """Endpoint for clients to establish an SSE connection and receive events."""
        return await _handle_sse_request(request)

    @app.post("/message", status_code=202)
    async def message_endpoint(request: Request) -> Response:
        """Endpoint to handle incoming message requests (like tool calls)."""
        return await _handle_message_request(request)


def _setup_monitoring_routes(app: FastAPI) -> None:
    """Set up SSE monitoring endpoints for Phase 2.1."""

    @app.get("/events/aider")
    async def aider_events_stream(request: Request) -> EventSourceResponse:
        """Stream general AIDER events (rate limits, progress, errors) to clients."""
        return await _create_sse_monitoring_endpoint("aider")

    @app.get("/events/errors")
    async def error_events_stream(request: Request) -> EventSourceResponse:
        """Stream error-specific AIDER events to clients."""
        return await _create_sse_monitoring_endpoint("errors")

    @app.get("/events/progress")
    async def progress_events_stream(request: Request) -> EventSourceResponse:
        """Stream progress update AIDER events to clients."""
        return await _create_sse_monitoring_endpoint("progress")

    @app.get("/health")
    async def health_check() -> JSONResponse:
        """Health check endpoint with streaming status information."""
        if _check_adapter_availability():
            return JSONResponse(content={"status": "unhealthy", "reason": "Server not ready"}, status_code=503)

        # Count active connections
        total_connections = sum(len(conns) for conns in _event_connections.values())

        return JSONResponse(
            content={
                "status": "healthy",
                "timestamp": time.time(),
                "active_sse_connections": {
                    "total": total_connections,
                    "aider_events": len(_event_connections["aider"]),
                    "error_events": len(_event_connections["errors"]),
                    "progress_events": len(_event_connections["progress"]),
                },
                "coordinator_status": "active" if _adapter and _adapter._coordinator else "unavailable",
            }
        )


def _setup_routes(app: FastAPI) -> None:
    """Set up the FastAPI routes for SSE and message endpoints."""
    _setup_core_routes(app)
    _setup_monitoring_routes(app)


async def create_app(
    coordinator: ApplicationCoordinator,
    editor_model: str,  # Parameter kept for signature consistency, may be used elsewhere
    current_working_dir: str,  # Parameter kept for signature consistency
    heartbeat_interval: float = 15.0,
    host: str = "127.0.0.1",  # Add host parameter for discovery registration
    port: int = 8000,  # Add port parameter for discovery registration
) -> FastAPI:
    """Create and configure the FastAPI application with SSE routes."""
    global _adapter  # Indicate modification of the global variable

    # Create FastAPI app
    app = FastAPI()

    # Get the transport adapter registry
    registry = await TransportAdapterRegistry.get_instance()

    # Create SSETransportAdapter instance via registry
    adapter = await registry.create_adapter(
        adapter_type="sse",
        coordinator=coordinator,
        heartbeat_interval=heartbeat_interval,
    )

    if not adapter:
        logger.error("Failed to create SSE transport adapter via registry")
        raise RuntimeError("Failed to create SSE transport adapter")

    _adapter = cast(SSETransportAdapter, adapter)
    logger.info(f"Created FastAPI app with SSE adapter {adapter.get_transport_id()} (heartbeat: {heartbeat_interval}s)")

    # Define streaming capabilities metadata
    streaming_caps = {
        "sse_endpoints": {
            "aider_events": "/events/aider",
            "error_events": "/events/errors",
            "progress_events": "/events/progress",
            "health_check": "/health",
        },
        # Add other streaming-related metadata here if needed
        "supported_event_types": list(_event_connections.keys()),  # Example: list endpoint types
    }

    # Register coordinator in discovery with streaming metadata
    try:
        # Instantiate CoordinatorDiscovery directly
        discovery_instance = CoordinatorDiscovery()
        coordinator_id = await discovery_instance.register_coordinator(
            host=host,
            port=port,
            transport_type="sse",
            metadata={"version": "1.0.0"},  # Include existing metadata
            streaming_capabilities=streaming_caps,  # Add streaming capabilities
        )
        logger.info(f"Registered coordinator {coordinator_id} in discovery with streaming capabilities.")
        logger.info("Phase 2.2 Transport Discovery Integration complete (Streaming metadata added)")
    except Exception as e:
        logger.error(f"Failed to register coordinator in discovery: {e}")
        # Continue without discovery registration if it fails

    # Initialize event broadcasting integration
    # Ensure adapter and its coordinator are available before setting up broadcasting
    if _adapter and _adapter._coordinator:
        # Start coordinator event listener for SSE broadcasting
        await _start_coordinator_event_listener()

        # Register SSE broadcast function with coordinator for Phase 1 integration
        # This is a temporary integration point until the coordinator has a proper subscribe method
        if hasattr(_adapter._coordinator, "_sse_event_broadcaster"):
            # Check if the attribute exists before assigning
            _adapter._coordinator._sse_event_broadcaster = broadcast_event_to_sse_clients
            logger.debug("SSE event broadcaster registered with coordinator")
        else:
            logger.warning(
                "Coordinator instance does not have _sse_event_broadcaster attribute. SSE broadcasting from coordinator to app will not work."
            )

        logger.info("SSE monitoring endpoints ready - Phase 2.1 Event Broadcasting Integration complete")
    else:
        logger.warning("Adapter or Coordinator not available, skipping event broadcasting setup.")

    # Set up the routes
    _setup_routes(app)

    logger.info("FastAPI app with SSE monitoring endpoints created successfully")
    return app


# Public API for coordinator integration
async def broadcast_event_to_sse_clients(event_type: str, event_data: Dict[str, Any]) -> None:
    """Public function for coordinator to broadcast events to SSE clients."""
    await _broadcast_to_sse_clients(event_type, event_data)


def get_sse_connection_stats() -> Dict[str, int]:
    """Get current SSE connection statistics."""
    return {
        "total": sum(len(conns) for conns in _event_connections.values()),
        "aider_events": len(_event_connections["aider"]),
        "error_events": len(_event_connections["errors"]),
        "progress_events": len(_event_connections["progress"]),
    }
