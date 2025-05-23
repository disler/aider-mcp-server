"""
RequestProcessor component for handling request processing functionality.
Based on Task 5 specification from reference/tasks/tasks.json.
"""

import asyncio
import uuid
from typing import Any, Awaitable, Callable, Dict

from aider_mcp_server.atoms.logging import get_logger

# Type alias for request handlers, as specified in Task 5
RequestHandler = Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]


class RequestProcessor:
    """
    Implements the RequestProcessor as specified in Task 5.
    1. Validates incoming requests
    2. Routes requests to appropriate handlers
    3. Manages request processing lifecycle
    4. Handles response formatting
    """

    def __init__(self) -> None:
        """
        Initialize the RequestProcessor.
        """
        self._handlers: Dict[str, RequestHandler] = {}
        self._active_requests: Dict[str, asyncio.Task[Dict[str, Any]]] = {}
        self._lock = asyncio.Lock()
        self._logger = get_logger(__name__) # Adhering to project logging patterns

    def register_handler(self, request_type: str, handler: RequestHandler) -> None:
        """
        Register a handler for a specific request type.

        Args:
            request_type: The type of request (e.g., "echo", "listFiles").
            handler: The asynchronous function to handle this request type.
        """
        self._handlers[request_type] = handler
        self._logger.debug(f"Registered handler for request type: {request_type}")

    def unregister_handler(self, request_type: str) -> None:
        """
        Unregister a handler for a specific request type.

        Args:
            request_type: The type of request to unregister.
        """
        if request_type in self._handlers:
            del self._handlers[request_type]
            self._logger.debug(f"Unregistered handler for request type: {request_type}")
        else:
            self._logger.warning(f"Attempted to unregister non-existent handler for type: {request_type}")

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an incoming request and return a response.

        Validates the request, routes it to the appropriate handler,
        manages its lifecycle, and formats the response.

        Args:
            request: The request dictionary. Expected to have a 'type' key.
                     May have an 'id' key for request tracking.

        Returns:
            A dictionary representing the response.
        """
        self._logger.debug(f"Received request: {request}")

        if 'type' not in request:
            self._logger.error("Request processing failed: Missing 'type' field.")
            return self._error_response("Missing request type")

        request_type = request['type']
        if request_type not in self._handlers:
            self._logger.error(f"Request processing failed: Unknown request type '{request_type}'.")
            return self._error_response(f"Unknown request type: {request_type}")

        request_id = request.get('id', str(uuid.uuid4()))
        if 'id' not in request:
            self._logger.debug(f"Generated new request_id: {request_id} for request type {request_type}")
        
        request['id'] = request_id # Ensure request object passed to handler has the ID

        handler = self._handlers[request_type]
        task: asyncio.Task[Dict[str, Any]] = asyncio.create_task(
            self._handle_request(handler, request, request_id)
        )

        async with self._lock:
            self._active_requests[request_id] = task
        self._logger.info(f"Started processing request_id: {request_id}, type: {request_type}")

        try:
            response = await task
            self._logger.info(f"Finished processing request_id: {request_id}, type: {request_type}. Success: {response.get('success', 'N/A')}")
            return response
        except asyncio.CancelledError:
            self._logger.warning(f"Request_id: {request_id}, type: {request_type} was cancelled.")
            # Ensure it's removed if cancellation happened before finally block in process_request
            # cancel_request should have already removed it if called.
            async with self._lock:
                if request_id in self._active_requests:
                    del self._active_requests[request_id]
            return self._error_response(f"Request {request_id} was cancelled")
        finally:
            # This finally block ensures removal if the task completes normally or with an unhandled exception
            # not caught by _handle_request.
            async with self._lock:
                if request_id in self._active_requests:
                    self._logger.debug(f"Cleaning up request_id: {request_id} from active_requests in process_request.finally.")
                    del self._active_requests[request_id]

    async def _handle_request(
        self, handler: RequestHandler, request: Dict[str, Any], request_id: str
    ) -> Dict[str, Any]:
        """
        Internal method to execute the handler and manage its outcome.

        Args:
            handler: The handler function to execute.
            request: The request dictionary (guaranteed to have 'id' and 'type').
            request_id: The unique ID for this request.

        Returns:
            A dictionary representing the response from the handler or an error response.
        """
        try:
            self._logger.debug(f"Executing handler for request_id: {request_id}, type: {request['type']}")
            response = await handler(request)

            if not isinstance(response, dict):
                self._logger.error(
                    f"Handler for request_id {request_id} (type: {request['type']}) "
                    f"returned invalid response type: {type(response)}. Expected dict."
                )
                return self._error_response("Handler returned invalid response")

            response['id'] = request_id # Ensure response has the request ID, as per Task 5
            self._logger.debug(f"Handler for request_id: {request_id} completed successfully.")
            return response
        except asyncio.CancelledError:
            self._logger.warning(f"Handler for request_id: {request_id} (type: {request['type']}) was cancelled during execution.")
            raise # Re-raise to be caught by process_request or propagate
        except Exception as e:
            self._logger.error(
                f"Error processing request_id {request_id} (type: {request['type']}): {str(e)}",
                exc_info=True # Include traceback for server logs
            )
            return self._error_response(f"Error processing request: {str(e)}")

    def _error_response(self, message: str) -> Dict[str, Any]:
        """
        Create a standardized error response as per Task 5.

        Args:
            message: The error message.

        Returns:
            A dictionary representing the error response.
        """
        return {
            "success": False,
            "error": message
        }

    async def cancel_request(self, request_id: str) -> bool:
        """
        Cancel an active request by ID, as per Task 5.

        Args:
            request_id: The ID of the request to cancel.

        Returns:
            True if the request was found and cancellation was attempted, False otherwise.
        """
        async with self._lock:
            if request_id in self._active_requests:
                task_to_cancel = self._active_requests[request_id]
                self._logger.info(f"Attempting to cancel request_id: {request_id}")
                task_to_cancel.cancel()
                # Task 5 specifies deleting the entry from _active_requests here.
                del self._active_requests[request_id]
                self._logger.info(f"Cancelled and removed request_id: {request_id} from active_requests.")
                return True
            else:
                self._logger.warning(f"Attempted to cancel non-existent or already completed request_id: {request_id}")
                return False
