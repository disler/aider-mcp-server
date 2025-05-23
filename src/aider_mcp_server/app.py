from typing import Optional, cast

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from .interfaces.transport_registry import TransportAdapterRegistry
from .sse_transport_adapter import SSETransportAdapter
from .transport_coordinator import ApplicationCoordinator, get_logger_func

# Define global adapter variable for use in route handlers
_adapter: Optional[SSETransportAdapter] = None

# Get logger using the pattern established in other modules
logger = get_logger_func("app")


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


def _setup_routes(app: FastAPI) -> None:
    """Set up the FastAPI routes for SSE and message endpoints."""

    # FastAPI typing without type: ignore
    @app.get("/sse")
    async def sse_endpoint(request: Request) -> Response:
        """Endpoint for clients to establish an SSE connection and receive events."""
        return await _handle_sse_request(request)

    # FastAPI typing without type: ignore
    @app.post("/message", status_code=202)  # Use 202 Accepted status code
    async def message_endpoint(request: Request) -> Response:
        """Endpoint to handle incoming message requests (like tool calls)."""
        return await _handle_message_request(request)


async def create_app(
    coordinator: ApplicationCoordinator,
    editor_model: str,  # Parameter kept for signature consistency, may be used elsewhere
    current_working_dir: str,  # Parameter kept for signature consistency
    heartbeat_interval: float = 15.0,
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

    _adapter = adapter  # type: ignore  # We know this is an SSETransportAdapter
    logger.info(f"Created FastAPI app with SSE adapter {adapter.get_transport_id()} (heartbeat: {heartbeat_interval}s)")

    # Set up the routes
    _setup_routes(app)

    return app
