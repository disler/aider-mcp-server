"""
Handler Registry for the Aider MCP Server.

This module handles registration and management of operation handlers.
Extracted from ApplicationCoordinator to improve modularity and maintainability.
"""

import inspect
import logging
import typing
from typing import Any, Awaitable, Callable, Dict, List, Optional, Type

from aider_mcp_server.mcp_types import (
    LoggerFactory,
    LoggerProtocol,
)

# from aider_mcp_server.security import Permissions, SecurityContext # Removed

# Initialize the logger factory
get_logger_func: LoggerFactory

try:
    from aider_mcp_server.atoms.logging import get_logger as custom_get_logger

    get_logger_func = typing.cast(LoggerFactory, custom_get_logger)
except ImportError:

    def fallback_get_logger(name: str, *args: typing.Any, **kwargs: typing.Any) -> LoggerProtocol:
        logger = logging.getLogger(name)
        if not logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            if logger.level == logging.NOTSET:
                logger.setLevel(logging.INFO)

        class CustomLogger(LoggerProtocol):
            def debug(self, message: str, **kwargs: typing.Any) -> None:
                logger.debug(message, **kwargs)

            def info(self, message: str, **kwargs: typing.Any) -> None:
                logger.info(message, **kwargs)

            def warning(self, message: str, **kwargs: typing.Any) -> None:
                logger.warning(message, **kwargs)

            def error(self, message: str, **kwargs: typing.Any) -> None:
                logger.error(message, **kwargs)

            def critical(self, message: str, **kwargs: typing.Any) -> None:
                logger.critical(message, **kwargs)

            def exception(self, message: str, **kwargs: typing.Any) -> None:
                logger.exception(message, **kwargs)

            def verbose(self, message: str, **kwargs: typing.Any) -> None:
                logger.debug(message, **kwargs)

        return CustomLogger()

    get_logger_func = fallback_get_logger

logger = get_logger_func(__name__)

# Type alias for request handlers, consistent with Task 5 (RequestProcessor) and Task 8 spec
RequestHandler = Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]


class HandlerRegistry:
    """
    Manages operation handler registration and retrieval.

    This class is responsible for:
    1. Registering and unregistering operation handlers
    2. Tracking required permissions for operations
    3. Retrieving handlers and their required permissions
    """

    def __init__(self) -> None:
        """Initialize the HandlerRegistry."""
        self._handlers: Dict[str, RequestHandler] = {}
        logger.info("HandlerRegistry initialized")

    def register_handler(self, request_type: str, handler: RequestHandler) -> None:
        """
        Register a handler for a specific request type.

        Args:
            request_type: The type of request (e.g., "echo", "listFiles").
            handler: The asynchronous function to handle this request type.
        """
        # Overwrites existing handler for the same request_type without warning, as per Task 8 spec.
        self._handlers[request_type] = handler
        logger.info(f"Handler registered for request type: '{request_type}'")

    def register_handler_class(self, handler_class: Type[Any]) -> None:
        """
        Register all handler methods from a class.
        Handler methods are expected to be named 'handle_<request_type>'.

        Args:
            handler_class: The class containing handler methods.
        """
        instance = handler_class()
        for name, method in inspect.getmembers(instance, inspect.ismethod):
            if name.startswith("handle_"):
                request_type = name[7:]  # Remove 'handle_' prefix
                self.register_handler(request_type, method)
                logger.debug(
                    f"Registered method {name} from class {handler_class.__name__} for request type '{request_type}'"
                )

    def unregister_handler(self, request_type: str) -> None:
        """
        Unregister a handler for a specific request type.

        Args:
            request_type: The type of request to unregister.
        """
        if request_type in self._handlers:
            del self._handlers[request_type]
            logger.info(f"Handler unregistered for request type: '{request_type}'")
        else:
            logger.warning(f"Attempted to unregister non-existent handler for request type: '{request_type}'")

    def get_handler(self, request_type: str) -> Optional[RequestHandler]:
        """
        Get a handler for a specific request type.

        Args:
            request_type: The name of the request type.

        Returns:
            The handler function or None if it doesn't exist.
        """
        return self._handlers.get(request_type)

    def get_supported_request_types(self) -> List[str]:
        """
        Get a list of all supported request types.

        Returns:
            A list of strings, where each string is a registered request type.
        """
        return list(self._handlers.keys())

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a request using the appropriate registered handler.

        Args:
            request: The request dictionary, expected to contain a 'type' field.

        Returns:
            A dictionary representing the response from the handler or an error response.
        """
        if "type" not in request:
            logger.error("Request handling failed: Missing 'type' field in request.")
            return {"success": False, "error": "Missing request type"}

        request_type = request["type"]
        handler = self.get_handler(request_type)

        if not handler:
            logger.error(f"Request handling failed: Unknown request type '{request_type}'.")
            return {"success": False, "error": f"Unknown request type: {request_type}"}

        try:
            logger.debug(f"Executing handler for request type '{request_type}'. Request ID: {request.get('id', 'N/A')}")
            response = await handler(request)
            logger.debug(f"Handler for request type '{request_type}' completed. Request ID: {request.get('id', 'N/A')}")
            return response
        except Exception as e:
            logger.error(
                f"Error handling request type '{request_type}': {str(e)}. Request ID: {request.get('id', 'N/A')}",
                exc_info=True,
            )
            return {"success": False, "error": f"Error handling request: {str(e)}"}
