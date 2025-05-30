import time
from asyncio.subprocess import Process as AsyncioProcess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union

from pydantic import BaseModel, Field

from aider_mcp_server.atoms.errors.application_errors import BaseApplicationError

T = TypeVar("T")


# MCP Protocol Base Types
class MCPRequest(BaseModel, Generic[T]):
    """Base class for MCP protocol requests."""

    name: str
    parameters: T


class MCPResponse(BaseModel):
    """Base class for MCP protocol responses."""

    pass


class MCPErrorResponse(MCPResponse):
    """
    Error response for MCP protocol.

    Attributes:
        error_code (str): A unique code identifying the error.
        message (str): A message describing the error, suitable for end users.
        details (Optional[Dict[str, Any]]): Additional error details for debugging.
    """

    error_code: str
    error: str
    details: Optional[Dict[str, Any]] = None

    @property
    def message(self) -> str:
        return self.error

    @classmethod
    def from_exception(cls, exc: Exception) -> "MCPErrorResponse":
        """
        Create an MCPErrorResponse from an exception.

        Args:
            exc (Exception): The exception to convert.

        Returns:
            MCPErrorResponse: The error response created from the exception.
        """
        if isinstance(exc, BaseApplicationError):
            return cls(
                error_code=exc.error_code,
                error=exc.user_friendly_message,
                details=exc.details,
            )
        else:
            return cls(
                error_code="internal_server_error",
                error=str(exc),
                details={},
            )


# Tool-specific request parameter models
class AICodeParams(BaseModel):
    """Parameters for the aider_ai_code tool."""

    ai_coding_prompt: str
    relative_editable_files: List[str]
    relative_readonly_files: List[str] = Field(default_factory=list)


class ListModelsParams(BaseModel):
    """Parameters for the list_models tool."""

    substring: str = ""


# Tool-specific response models
class AICodeResponse(MCPResponse):
    """Response for the aider_ai_code tool."""

    status: str  # 'success' or 'failure'
    message: Optional[str] = None


class ListModelsResponse(MCPResponse):
    """Response for the list_models tool."""

    models: List[str]


# Specific request types
class AICodeRequest(MCPRequest[AICodeParams]):
    """Request for the aider_ai_code tool."""

    name: str = "aider_ai_code"


class ListModelsRequest(MCPRequest[ListModelsParams]):
    """Request for the list_models tool."""

    name: str = "list_models"


# Union type for all possible MCP responses
MCPToolResponse = Union[AICodeResponse, ListModelsResponse, MCPErrorResponse]


# Multi-client HTTP server data types
class ClientRequest(BaseModel):
    """Request from a client to create or manage a session."""

    client_id: str
    workspace_id: Optional[str] = None
    request_data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)


class SessionInfo(BaseModel):
    """Information about an active client session."""

    session_id: str
    client_id: str
    workspace_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    last_activity: datetime = Field(default_factory=datetime.now)
    status: str = "active"  # "active", "idle", "disconnected"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ServerInfo(BaseModel):
    """Information about a running HTTP server instance."""

    server_id: str
    host: str
    port: int
    actual_port: Optional[int] = None
    status: str = "starting"  # "starting", "running", "stopping", "stopped"
    workspace_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    active_clients: int = 0
    transport_adapter_id: Optional[str] = None  # Reference to adapter instance


# Process Manager data types
class ProcessInfo(BaseModel):
    """Information about a managed server process."""

    process_id: str
    client_id: Optional[str] = None
    port: int
    workspace_path: Path
    process: AsyncioProcess  # The actual asyncio.subprocess.Process object
    status: str  # e.g., "starting", "running", "stopping", "stopped", "failed"
    command: List[str]  # The command used to start the process

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_health_check: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    restart_count: int = 0

    class Config:
        arbitrary_types_allowed = True
