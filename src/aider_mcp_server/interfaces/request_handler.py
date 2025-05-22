"""
Defines the protocol for components that handle internal requests.
"""

from typing import Protocol

from typing_extensions import runtime_checkable

from aider_mcp_server.atoms.internal_types import InternalRequest, InternalResponse


@runtime_checkable
class IRequestHandler(Protocol):
    """
    Protocol for components that process internal requests.
    """

    async def handle_request(self, request: InternalRequest) -> InternalResponse:
        """
        Process an incoming internal request.

        Args:
            request: The request to handle.

        Returns:
            The response to the request.
        """
        ...
