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
    TypeVar,
    Union,
    runtime_checkable,
)

from pydantic import BaseModel

# Re-export the EventTypes enum to avoid circular imports

if TYPE_CHECKING:
    from aider_mcp_server.interfaces.transport_adapter import ITransportAdapter


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
    def verbose(self, message: str, **kwargs: Any) -> None: ...


# Import the TransportInterface from the interfaces package instead of defining it here
# Use a type alias to maintain backward compatibility
# Transport adapter classes should use ITransportAdapter from interfaces.transport_adapter
# This alias helps with the transition and existing code
if TYPE_CHECKING:
    TransportInterface = "ITransportAdapter"
else:
    from aider_mcp_server.interfaces.transport_adapter import ITransportAdapter

    TransportInterface = ITransportAdapter


@runtime_checkable
class Shutdownable(Protocol):
    """Protocol defining a minimal interface for objects that can be shut down."""

    def get_transport_id(self) -> str: ...
    def get_transport_type(self) -> str: ...

    async def shutdown(self) -> None: ...
    async def initialize(self) -> None: ...
    async def start_listening(self) -> None: ...


# ShutdownContextProtocol has been moved to interfaces/shutdown_context.py to resolve circular imports

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
