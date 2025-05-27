"""
Handler Registry for the Aider MCP Server.

This module implements the HandlerRegistry as specified in Task 8.
It manages registration, lifecycle, and access to request handlers.
"""

import inspect
from typing import Any, Awaitable, Callable, Dict, List, Optional, Type

from aider_mcp_server.atoms.logging.logger import get_logger

# Type alias for handler functions, as specified in Task 8
RequestHandler = Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]


class HandlerRegistry:
    """
    Manages request handler registration and dispatch.

    This class is responsible for:
    1. Registering request handlers for different request types.
    2. Managing handler lifecycle (registration, unregistration).
    3. Providing access to registered handlers.
    4. Supporting registration of individual handlers and methods from handler classes.
    5. Handling incoming requests by dispatching them to the appropriate handler.
    """

    def __init__(self) -> None:
        """Initialize the HandlerRegistry."""
        self._handlers: Dict[str, RequestHandler] = {}
        self._logger = get_logger(__name__)
        self._logger.info("HandlerRegistry initialized")

    def register_handler(self, request_type: str, handler: RequestHandler) -> None:
        """
        Register a handler for a specific request type.

        Args:
            request_type: The type of request (e.g., "echo", "listFiles").
            handler: The asynchronous function to handle this request type.
        """
        if request_type in self._handlers:
            self._logger.warning(f"Handler for request type '{request_type}' already registered. Overwriting.")
        self._handlers[request_type] = handler
        self._logger.info(f"Handler registered for request type: '{request_type}'")

    def register_handler_class(self, handler_class: Type[Any]) -> None:
        """
        Register all handler methods from a class.
        Handler methods are identified by names starting with 'handle_'.
        The request type is derived by removing the 'handle_' prefix.

        Args:
            handler_class: The class containing handler methods.
        """
        instance = handler_class()
        registered_count = 0
        for name, method in inspect.getmembers(instance, inspect.ismethod):
            if name.startswith("handle_"):
                request_type = name[7:]  # Remove 'handle_' prefix
                if request_type:  # Ensure the request_type is not empty
                    self.register_handler(request_type, method)
                    registered_count += 1
                else:
                    self._logger.warning(
                        f"Method '{name}' in class '{handler_class.__name__}' "
                        "has 'handle_' prefix but no subsequent request type. Skipping."
                    )
        if registered_count > 0:
            self._logger.info(f"Registered {registered_count} handlers from class '{handler_class.__name__}'.")
        else:
            self._logger.info(
                f"No handler methods (starting with 'handle_') found in class '{handler_class.__name__}'."
            )

    def unregister_handler(self, request_type: str) -> None:
        """
        Unregister a handler for a specific request type.

        Args:
            request_type: The type of request to unregister.
        """
        if request_type in self._handlers:
            del self._handlers[request_type]
            self._logger.info(f"Handler unregistered for request type: '{request_type}'")
        else:
            self._logger.warning(f"Attempted to unregister non-existent handler for request type: '{request_type}'")

    def get_handler(self, request_type: str) -> Optional[RequestHandler]:
        """
        Get a handler for a specific request type.

        Args:
            request_type: The type of request.

        Returns:
            The registered handler function, or None if not found.
        """
        return self._handlers.get(request_type)

    def get_supported_request_types(self) -> List[str]:
        """
        Get a list of all supported (registered) request types.

        Returns:
            A list of strings, where each string is a registered request type.
        """
        return list(self._handlers.keys())

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a request using the appropriate registered handler.

        Args:
            request: The request dictionary, expected to have a 'type' field.

        Returns:
            A dictionary representing the response from the handler or an error response.
        """
        self._logger.debug(f"Handling request: {request}")

        if "type" not in request:
            self._logger.error("Request handling failed: Missing 'type' field.")
            return {"success": False, "error": "Missing request type"}

        request_type = request["type"]
        handler = self.get_handler(request_type)

        if not handler:
            self._logger.error(f"Request handling failed: Unknown request type '{request_type}'.")
            return {"success": False, "error": f"Unknown request type: {request_type}"}

        try:
            self._logger.debug(f"Dispatching request type '{request_type}' to handler.")
            response = await handler(request)
            # Task 8 spec doesn't require modifying handler response (e.g. adding ID)
            # It's assumed the handler itself formats the response correctly.
            return response
        except Exception as e:
            self._logger.error(
                f"Error handling request type '{request_type}': {str(e)}",
                exc_info=self._logger.is_verbose(),  # Log traceback if verbose
            )
            return {"success": False, "error": f"Error handling request: {str(e)}"}
