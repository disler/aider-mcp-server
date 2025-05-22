"""
Defines internal data types used within the Aider MCP Server,
distinct from the MCP protocol types used for external communication.
"""

import time
from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel, Field

from aider_mcp_server.atoms.event_types import EventTypes


class ErrorContext(BaseModel):
    """
    Provides context for errors, such as request ID or operation name.
    """

    request_id: Optional[str] = None
    operation_name: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class InternalEvent(BaseModel):
    """
    Represents an event within the internal system.
    """

    event_type: EventTypes
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    timestamp: float = Field(default_factory=time.time)


class InternalErrorEvent(InternalEvent):
    """
    Represents an error event within the internal system.
    """

    error_message: str
    error_details: Optional[Dict[str, Any]] = None
    context: Optional[ErrorContext] = None


class InternalRequest(BaseModel):
    """
    Represents a request within the internal system.
    """

    request_id: str
    request_type: str
    data: Dict[str, Any]
    client_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class InternalResponse(BaseModel):
    """
    Represents a response within the internal system.
    """

    request_id: str
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# Conversion utilities
def to_internal_event(
    event_type: EventTypes, data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None
) -> InternalEvent:
    """Converts an EventTypes enum and data dict to an InternalEvent object."""
    return InternalEvent(event_type=event_type, data=data, metadata=metadata)


def from_internal_event(
    internal_event: InternalEvent,
) -> Tuple[EventTypes, Dict[str, Any], Optional[Dict[str, Any]]]:
    """Converts an InternalEvent object back to EventTypes, data, and metadata."""
    return internal_event.event_type, internal_event.data, internal_event.metadata
