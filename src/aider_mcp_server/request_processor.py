"""
RequestProcessor component for handling request processing functionality.
"""

import asyncio
from typing import Any, Dict, Optional, Union

from aider_mcp_server.atoms.event_types import EventTypes
from aider_mcp_server.event_coordinator import EventCoordinator
from aider_mcp_server.handler_registry import HandlerRegistry
from aider_mcp_server.mcp_types import LoggerFactory
from aider_mcp_server.response_formatter import ResponseFormatter
from aider_mcp_server.security import SecurityContext
from aider_mcp_server.session_manager import SessionManager


class RequestProcessor:
    """
    Handles the request lifecycle, validation, and execution.

    Responsibilities:
    - Processes incoming operation requests
    - Manages request handlers
    - Validates permissions
    - Tracks request state
    - Reports request status updates
    """

    def __init__(
        self,
        event_coordinator: EventCoordinator,
        session_manager: SessionManager,
        logger_factory: LoggerFactory,
        handler_registry: HandlerRegistry,
        response_formatter: ResponseFormatter,
    ) -> None:
        """
        Initialize the RequestProcessor.

        Args:
            event_coordinator: Coordinator for sending events
            session_manager: Manager for session information
            logger_factory: Factory function to create loggers
            handler_registry: Registry for managing operation handlers
            response_formatter: Formatter for standardizing response formatting
        """
        self._event_coordinator = event_coordinator
        self._session_manager = session_manager
        self._handler_registry = handler_registry
        self._response_formatter = response_formatter
        self._active_requests: Dict[str, Dict[str, Any]] = {}
        self._active_requests_lock = asyncio.Lock()
        self._logger = logger_factory(__name__)

    async def process_request(
        self,
        request_id: str,
        transport_id: str,
        operation_name: str,
        request_data: Dict[str, Any],
    ) -> None:
        """
        Process an incoming request.

        Args:
            request_id: Unique identifier for the request
            transport_id: Identifier for the transport that made the request
            operation_name: Name of the operation to execute
            request_data: Request parameters and data
        """
        handler = await self._handler_registry.get_handler(operation_name)
        if not handler:
            self._logger.error(f"No handler found for operation '{operation_name}'")
            await self.fail_request(
                request_id,
                operation_name,
                "OperationNotFound",
                f"No handler registered for operation '{operation_name}'",
                transport_id,
                request_data,
            )
            return

        required_permission = await self._handler_registry.get_required_permission(
            operation_name
        )

        if required_permission:
            # Check permissions
            has_permission = await self._session_manager.check_permission(
                transport_id, required_permission
            )
            if not has_permission:
                self._logger.error(
                    f"Permission denied for operation '{operation_name}'"
                )
                await self.fail_request(
                    request_id,
                    operation_name,
                    "PermissionDenied",
                    f"Permission '{required_permission}' required for operation '{operation_name}'",
                    transport_id,
                    request_data,
                )
                return

        # Create a task to run the handler
        async def run_handler() -> None:
            try:
                # Send status update that processing has started
                await self._event_coordinator.send_event_to_transport(
                    transport_id,
                    EventTypes.STATUS,
                    {
                        "request_id": request_id,
                        "status": "processing",
                        "message": f"Processing {operation_name} request",
                    },
                )

                # Extract parameters for the handler
                # Get security context - Use a properly typed security context
                from aider_mcp_server.security import ANONYMOUS_SECURITY_CONTEXT

                security_context: SecurityContext = ANONYMOUS_SECURITY_CONTEXT

                # Extract parameters from request_data
                params = request_data.get("parameters", {})

                # Call the handler with all required parameters plus defaults
                # Handler expects 6 parameters:
                # request_id, transport_id, params, security_context, use_diff_cache, clear_cached_for_unchanged
                result = await handler(
                    request_id,
                    transport_id,
                    params,
                    security_context,
                    True,  # use_diff_cache
                    True,  # clear_cached_for_unchanged
                )

                # Format success response
                success_response = self._response_formatter.format_success_response(
                    request_id, transport_id, result
                )

                # Send the result back to the client
                await self._event_coordinator.send_event_to_transport(
                    transport_id,
                    EventTypes.TOOL_RESULT,
                    {
                        "request_id": request_id,
                        "tool_name": operation_name,
                        "result": success_response,
                    },
                )
            except Exception as e:
                self._logger.error(f"Error processing request: {e}")
                await self.fail_request(
                    request_id,
                    operation_name,
                    "ProcessingError",
                    str(e),
                    transport_id,
                    request_data,
                )
            finally:
                await self._cleanup_request(request_id)

        task = asyncio.create_task(run_handler())
        # Store the request state
        async with self._active_requests_lock:
            self._active_requests[request_id] = {
                "operation_name": operation_name,
                "transport_id": transport_id,
                "task": task,
                "status": "processing",
                "details": request_data,
            }

    async def fail_request(
        self,
        request_id: str,
        operation_name: str,
        error: str,
        error_details: Union[str, Dict[str, Any]],
        originating_transport_id: Optional[str] = None,
        request_details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Mark a request as failed.

        Args:
            request_id: Unique identifier for the request
            operation_name: Name of the operation
            error: Error code or short description
            error_details: Detailed error information
            originating_transport_id: Identifier for the transport that made the request
            request_details: Original request parameters and data
        """
        if originating_transport_id is None:
            self._logger.warning(
                f"Cannot send failure notification for request {request_id}: no transport ID provided"
            )
            return

        # Format error response - convert error_details to appropriate format
        error_message = error
        error_code = None

        # If error_details is a dict, use it for details parameter
        # If it's a string, use it as the error message instead
        details = None
        if isinstance(error_details, dict):
            details = error_details
        else:
            error_message = error_details

        error_response = self._response_formatter.format_error_response(
            request_id, originating_transport_id, error_message, error_code, details
        )

        # Send failure message as a TOOL_RESULT event
        await self._event_coordinator.send_event_to_transport(
            originating_transport_id,
            EventTypes.TOOL_RESULT,
            {
                "request_id": request_id,
                "tool_name": operation_name,
                "result": error_response,
            },
        )

        # Also send a STATUS event to indicate failure
        await self._event_coordinator.send_event_to_transport(
            originating_transport_id,
            EventTypes.STATUS,
            {
                "request_id": request_id,
                "status": "failed",
                "message": f"Operation {operation_name} failed: {error}",
            },
        )

        await self._cleanup_request(request_id)

    async def _cleanup_request(self, request_id: str) -> None:
        """
        Clean up the request state.
        """
        async with self._active_requests_lock:
            if request_id in self._active_requests:
                del self._active_requests[request_id]
