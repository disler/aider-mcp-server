"""
Defines the protocol for Error Handling.
"""

from typing import Optional, Protocol

from typing_extensions import runtime_checkable

from aider_mcp_server.atoms.types.internal_types import ErrorContext


@runtime_checkable
class IErrorHandler(Protocol):
    """
    Protocol for components that handle errors.
    """

    async def handle_error(self, error: Exception, context: Optional[ErrorContext] = None) -> None:
        """
        Handles an error, potentially logging it or taking other actions.

        Args:
            error: The exception that occurred.
            context: Optional context about the error.
        """
        ...
