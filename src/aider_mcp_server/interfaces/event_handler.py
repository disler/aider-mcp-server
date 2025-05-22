"""
Defines the protocol for components that handle internal events.
"""

from typing import Optional, Protocol

from typing_extensions import runtime_checkable

from aider_mcp_server.atoms.internal_types import InternalEvent


@runtime_checkable
class IEventHandler(Protocol):
    """
    Protocol for components that process internal events.
    """

    async def handle_event(self, event: InternalEvent) -> Optional[InternalEvent]:
        """
        Process an incoming internal event.

        Args:
            event: The event to handle.

        Returns:
            An optional event to be published as a result of handling this event,
            or None if no further event needs to be published.
        """
        ...
