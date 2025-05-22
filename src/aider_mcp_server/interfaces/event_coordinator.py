"""
Defines the protocol for the EventCoordinator.
"""

from typing import Protocol

from typing_extensions import runtime_checkable

from aider_mcp_server.atoms.event_types import EventTypes
from aider_mcp_server.atoms.internal_types import InternalEvent
from aider_mcp_server.interfaces.event_handler import IEventHandler


@runtime_checkable
class IEventCoordinator(Protocol):
    """
    Protocol for the event coordinator.
    Manages event subscriptions and publishing of internal events.
    """

    async def startup(self) -> None:
        """Initializes and starts the event coordinator."""
        ...

    async def shutdown(self) -> None:
        """Shuts down the event coordinator and cleans up resources."""
        ...

    async def subscribe(self, event_type: EventTypes, handler: IEventHandler) -> None:
        """
        Subscribes an event handler to a specific event type.

        Args:
            event_type: The type of event to subscribe to.
            handler: The event handler to invoke for the event type.
        """
        ...

    async def unsubscribe(self, event_type: EventTypes, handler: IEventHandler) -> None:
        """
        Unsubscribes an event handler from a specific event type.

        Args:
            event_type: The type of event to unsubscribe from.
            handler: The event handler to remove.
        """
        ...

    async def publish_event(self, event: InternalEvent) -> None:
        """
        Publishes an internal event to all subscribed handlers.

        Args:
            event: The internal event to publish.
        """
        ...
