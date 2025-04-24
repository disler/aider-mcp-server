from typing import Generic, List, Optional, TypeVar, Union

from pydantic import BaseModel, Field

T = TypeVar("T")


# MCP Protocol Base Types
class MCPRequest(Generic[T], BaseModel):
    """Base class for MCP protocol requests."""

    name: str
    parameters: T


class MCPResponse(BaseModel):
    """Base class for MCP protocol responses."""

    pass


class MCPErrorResponse(MCPResponse):
    """Error response for MCP protocol."""

    error: str


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
