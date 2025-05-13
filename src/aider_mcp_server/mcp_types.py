"""
Centralized type definitions for the Aider MCP Server.

This module contains all shared protocol definitions, type aliases, and utility
types used throughout the codebase. Centralizing these definitions helps avoid
duplication, ensures consistency, and resolves circular dependency issues.
"""

import asyncio
import enum

# Import SecurityContext for type hinting, but only during type checking
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Set,
    TypeVar,
    Union,
    runtime_checkable,
)

from pydantic import BaseModel

# Re-export the EventTypes enum to avoid circular imports
from aider_mcp_server.atoms.event_types import EventTypes

if TYPE_CHECKING:
    from aider_mcp_server.security import SecurityContext


# Generic Type Variables
T = TypeVar("T")
ResponseT = TypeVar("ResponseT")
DataT = TypeVar("DataT")

# ====== Protocol Definitions ======


@runtime_checkable
class LoggerProtocol(Protocol):
    """Protocol defining the required methods for logger instances."""

    def debug(self, message: str, **kwargs: Any) -> None: ...
    def info(self, message: str, **kwargs: Any) -> None: ...
    def warning(self, message: str, **kwargs: Any) -> None: ...
    def error(self, message: str, **kwargs: Any) -> None: ...
    def critical(self, message: str, **kwargs: Any) -> None: ...
    def exception(self, message: str, **kwargs: Any) -> None: ...


@runtime_checkable
class TransportInterface(Protocol):
    """
    Protocol defining the interface for transport adapters.

    Any class implementing a transport mechanism (like SSE, Stdio, WebSocket)
    that interacts with the ApplicationCoordinator should adhere to this protocol.
    """

    transport_id: str
    transport_type: str

    async def initialize(self) -> None:
        """
        Asynchronously initializes the transport adapter.
        This method should handle setup specific to the transport,
        registering with the coordinator, and starting any necessary background tasks (like heartbeat).
        """
        ...

    async def shutdown(self) -> None:
        """
        Asynchronously shuts down the transport adapter.
        This method should handle cleanup specific to the transport,
        unregistering from the coordinator, and stopping any background tasks.
        """
        ...

    async def send_event(self, event: EventTypes, data: Dict[str, Any]) -> None:
        """
        Asynchronously sends an event with associated data to the client
        connected via this transport.

        Args:
            event: The event type (e.g., EventTypes.PROGRESS).
            data: A dictionary containing the event payload.
        """
        ...

    def get_capabilities(self) -> Set[EventTypes]:
        """
        Returns a set of event types that this transport adapter is capable
        of sending or receiving.

        This informs the ApplicationCoordinator which events can be routed
        to this transport.
        """
        ...

    def validate_request_security(
        self, request_data: Dict[str, Any]
    ) -> "SecurityContext":
        """
        Validates security information provided in the incoming request data
        and returns the SecurityContext applicable to this specific request.
        This method is called by the transport itself before processing a request.
        """
        ...


@runtime_checkable
class Shutdownable(Protocol):
    """Protocol defining a minimal interface for objects that can be shut down."""

    transport_id: str

    async def shutdown(self) -> None: ...
    async def initialize(self) -> None: ...
    async def start_listening(self) -> None: ...


@runtime_checkable
class ShutdownContextProtocol(Protocol):
    """Protocol defining only the members needed by shutdown context managers."""

    transport_id: str

    async def shutdown(self) -> None: ...


# ====== Common Type Aliases ======

# Type alias for the function that gets a logger
LoggerFactory = Callable[..., LoggerProtocol]

# Event data type
EventData = Dict[str, Any]

# Request parameters type
RequestParameters = Dict[str, Any]

# Operation result type
OperationResult = Dict[str, Any]

# Type for asyncio tasks with specific result types
AsyncTask = asyncio.Task

# Common response types
ResponseType = Union[Dict[str, Any], List[Any], str, None]

# Callback type for operation complete events
OperationCallback = Callable[[str, bool, Optional[Dict[str, Any]]], None]

# ====== MCP Protocol Base Types ======


class MCPRequest(BaseModel, Generic[T]):
    """Base class for MCP protocol requests."""

    name: str
    parameters: T
    request_id: Optional[str] = None


class MCPResponse(BaseModel):
    """Base class for MCP protocol responses."""

    pass


class MCPErrorResponse(MCPResponse):
    """Error response for MCP protocol."""

    error: str
    details: Optional[str] = None


# Union type for all possible MCP responses
MCPToolResponse = Union[MCPResponse, MCPErrorResponse]

# ====== Utility Types ======


class ResultStatus(enum.Enum):
    """Status codes for operation results."""

    SUCCESS = "success"
    FAILURE = "failure"
    PENDING = "pending"
    CANCELLED = "cancelled"


class ProgressUpdate(BaseModel):
    """Model for progress updates during long-running operations."""

    operation_id: str
    step: int
    total_steps: int
    description: str
    percentage: float
    status: ResultStatus = ResultStatus.PENDING
    additional_info: Optional[Dict[str, Any]] = None
