"""
Handler Registry for the Aider MCP Server.

This module handles registration and management of operation handlers.
Extracted from ApplicationCoordinator to improve modularity and maintainability.
"""

import asyncio
import logging
import typing
from typing import Any, Callable, Coroutine, Dict, Optional, Tuple

from aider_mcp_server.mcp_types import (
    LoggerFactory,
    LoggerProtocol,
    OperationResult,
    RequestParameters,
)
from aider_mcp_server.security import Permissions, SecurityContext

# Initialize the logger factory
get_logger_func: LoggerFactory

try:
    from aider_mcp_server.atoms.logging import get_logger as custom_get_logger

    get_logger_func = typing.cast(LoggerFactory, custom_get_logger)
except ImportError:

    def fallback_get_logger(
        name: str, *args: typing.Any, **kwargs: typing.Any
    ) -> LoggerProtocol:
        logger = logging.getLogger(name)
        if not logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            if logger.level == logging.NOTSET:
                logger.setLevel(logging.INFO)
        return logger

    get_logger_func = fallback_get_logger

logger = get_logger_func(__name__)

# Type alias for handler functions
HandlerFunc = Callable[
    [str, str, RequestParameters, SecurityContext, bool, bool],
    Coroutine[Any, Any, OperationResult],
]


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
        self._handlers: Dict[str, Tuple[HandlerFunc, Optional[Permissions]]] = {}
        self._handlers_lock = asyncio.Lock()

        logger.info("HandlerRegistry initialized")

    async def register_handler(
        self,
        operation_name: str,
        handler: HandlerFunc,
        required_permission: Optional[Permissions] = None,
    ) -> None:
        """
        Registers a handler function for a specific operation.

        Args:
            operation_name: The name of the operation
            handler: The handler function to register
            required_permission: Optional permission required to execute the operation
        """
        async with self._handlers_lock:
            if operation_name in self._handlers:
                logger.warning(
                    f"Handler for operation '{operation_name}' already registered. Overwriting."
                )
            self._handlers[operation_name] = (handler, required_permission)
            logger.info(f"Handler registered for operation: '{operation_name}'")

    async def unregister_handler(self, operation_name: str) -> None:
        """
        Unregisters a handler function.

        Args:
            operation_name: The name of the operation
        """
        async with self._handlers_lock:
            if operation_name in self._handlers:
                del self._handlers[operation_name]
                logger.info(f"Handler unregistered for operation: '{operation_name}'")
            else:
                logger.warning(
                    f"Attempted to unregister non-existent handler: '{operation_name}'"
                )

    async def get_handler(self, operation_name: str) -> Optional[HandlerFunc]:
        """
        Gets the handler function for a specific operation name.

        Args:
            operation_name: The name of the operation

        Returns:
            The handler function or None if it doesn't exist
        """
        handler_info = await self.get_handler_info(operation_name)
        return handler_info[0] if handler_info else None

    async def get_required_permission(
        self, operation_name: str
    ) -> Optional[Permissions]:
        """
        Gets the required permission for a specific operation name.

        Args:
            operation_name: The name of the operation

        Returns:
            The required permission or None if none is required or the operation doesn't exist
        """
        handler_info = await self.get_handler_info(operation_name)
        return handler_info[1] if handler_info else None

    async def get_handler_info(
        self, operation_name: str
    ) -> Optional[Tuple[HandlerFunc, Optional[Permissions]]]:
        """
        Gets handler function and required permission by operation name.

        Args:
            operation_name: The name of the operation

        Returns:
            A tuple of (handler_function, required_permission) or None if the operation doesn't exist
        """
        async with self._handlers_lock:
            return self._handlers.get(operation_name)

    async def has_handler(self, operation_name: str) -> bool:
        """
        Checks if a handler exists for the specified operation.

        Args:
            operation_name: The name of the operation

        Returns:
            True if a handler exists, False otherwise
        """
        async with self._handlers_lock:
            return operation_name in self._handlers

    async def get_all_operations(self) -> Dict[str, Optional[Permissions]]:
        """
        Gets all registered operations and their required permissions.

        Returns:
            Dictionary mapping operation names to their required permissions
        """
        async with self._handlers_lock:
            return {name: info[1] for name, info in self._handlers.items()}

    async def clear(self) -> None:
        """Clears all registered handlers."""
        async with self._handlers_lock:
            self._handlers.clear()
        logger.info("HandlerRegistry cleared")
