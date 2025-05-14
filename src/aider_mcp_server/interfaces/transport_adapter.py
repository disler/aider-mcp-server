"""
Transport adapter interfaces for the Aider MCP Server.

This module contains interfaces and base classes for transport adapters, providing
a standardized way to connect different transport mechanisms (SSE, Stdio, WebSocket, etc.)
to the application coordinator.
"""

import abc
from typing import TYPE_CHECKING, Any, Dict, Optional, Protocol, Set, runtime_checkable

from aider_mcp_server.atoms.event_types import EventTypes

# Use type aliases directly to avoid circular imports
# These match the definitions in mcp_types.py
EventData = Dict[str, Any]
RequestParameters = Dict[str, Any]

# Forward reference to avoid circular imports
if TYPE_CHECKING:
    from aider_mcp_server.security import SecurityContext
else:
    # Runtime placeholder that will be replaced by the real type during runtime
    SecurityContext = Any


@runtime_checkable
class ITransportAdapter(Protocol):
    """
    Protocol defining the interface for transport adapters.

    Any class implementing a transport mechanism (like SSE, Stdio, WebSocket)
    that interacts with the ApplicationCoordinator should adhere to this protocol.
    This interface defines the minimal set of attributes and methods required.
    """

    # Core transport identification properties - defined as methods to avoid Protocol limitations
    def get_transport_id(self) -> str:
        """Get the unique identifier for this transport instance."""
        ...

    def get_transport_type(self) -> str:
        """Get the type of transport (e.g., 'sse', 'stdio')."""
        ...

    # Core lifecycle methods
    async def initialize(self) -> None:
        """
        Asynchronously initializes the transport adapter.

        This method should handle setup specific to the transport,
        registering with the coordinator, and starting any necessary background tasks.
        """
        ...

    async def shutdown(self) -> None:
        """
        Asynchronously shuts down the transport adapter.

        This method should handle cleanup specific to the transport,
        unregistering from the coordinator, and stopping any background tasks.
        """
        ...

    # Communication methods
    async def send_event(self, event_type: EventTypes, data: EventData) -> None:
        """
        Asynchronously sends an event with associated data to the client
        connected via this transport.

        Args:
            event_type: The event type (e.g., EventTypes.PROGRESS).
            data: A dictionary containing the event payload.
        """
        ...

    # Capability and security methods
    def get_capabilities(self) -> Set[EventTypes]:
        """
        Returns a set of event types that this transport adapter is capable
        of sending or receiving.

        This informs the ApplicationCoordinator which events can be routed
        to this transport.
        """
        ...

    def validate_request_security(
        self, request_data: RequestParameters
    ) -> SecurityContext:
        """
        Validates security information provided in the incoming request data
        and returns the SecurityContext applicable to this specific request.

        Args:
            request_data: The data from the incoming request.

        Returns:
            A SecurityContext representing the security validation result.

        Raises:
            ValueError: If security validation fails (e.g., invalid token).
            PermissionError: If the request lacks necessary permissions.
        """
        ...

    # Optional extended methods for more advanced transports
    async def start_listening(self) -> None:
        """
        Starts listening for incoming connections or messages.

        For transports that need explicit activation of listeners
        (e.g., WebSocket server, Stdio reader).
        """
        ...

    def should_receive_event(
        self,
        event_type: EventTypes,
        data: EventData,
        request_details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Determines if this transport should receive a specific event.

        This method allows transports to implement custom filtering logic
        beyond the basic subscription mechanism.

        Args:
            event_type: The type of event.
            data: The event data payload.
            request_details: Optional original request parameters for context.

        Returns:
            True if the transport should receive the event, False otherwise.
        """
        return True  # Default implementation always receives subscribed events


class TransportAdapterBase(abc.ABC, ITransportAdapter):
    """
    Abstract base class for transport adapters that implements common functionality.

    This class provides a foundation for concrete transport implementations,
    handling common tasks like registration, heartbeat management, and basic
    state tracking. It should be extended by specific transport mechanisms.
    """

    def __init__(self, transport_id: str, transport_type: str, **kwargs: Any):
        """
        Initialize the base transport adapter with essential properties.

        Args:
            transport_id: Unique identifier for this transport instance
            transport_type: Type identifier for the transport (e.g., "sse", "stdio")
            **kwargs: Additional arguments to pass to specialized subclasses
        """
        self._transport_id = transport_id
        self._transport_type = transport_type

    def get_transport_id(self) -> str:
        """Get the unique identifier for this transport instance."""
        return self._transport_id

    def get_transport_type(self) -> str:
        """Get the type of transport (e.g., 'sse', 'stdio')."""
        return self._transport_type

    @classmethod
    def get_default_capabilities(cls) -> Set[EventTypes]:
        """
        Get the default capabilities for this transport adapter class.

        This class method allows the TransportAdapterRegistry to determine
        capabilities without instantiating the adapter.

        Returns:
            A set of event types that this adapter supports by default.
        """
        # Default implementation returns an empty set
        # Subclasses should override this with their specific capabilities
        return set()

    # Additional common functionality will be implemented here
