"""
Custom type aliases for complex types to improve code readability and type safety.

This module provides type aliases for common complex types used throughout the codebase,
enhancing readability and type safety while reducing duplication.
"""

from typing import Any, Dict, Optional, TypedDict, Union

# ====== Basic Type Aliases ======

# Handler result type
HandlerResult = Dict[str, Any]

# Event data type (re-exported from mcp_types for backwards compatibility)
EventData = Dict[str, Any]

# Request payload type
RequestPayload = Dict[str, Any]

# Response payload type
ResponsePayload = Dict[str, Any]

# Transport message type
TransportMessage = Dict[str, Any]

# ====== Typed Dictionaries for more specific type definitions ======


class ErrorResponseDict(TypedDict):
    """Type definition for standardized error responses."""

    success: bool  # Always False for errors
    error: str
    details: Optional[str]


class SuccessResponseDict(TypedDict, total=False):
    """Type definition for standardized success responses."""

    success: bool  # Always True for success
    data: Any
    message: Optional[str]


# Combined response type
ResponseDict = Union[ErrorResponseDict, SuccessResponseDict]


# File operation result type
class FileOperationResult(TypedDict, total=False):
    """Type definition for file operation results."""

    success: bool
    path: str
    error: Optional[str]
    content: Optional[str]


# Configuration settings type
ConfigSettings = Dict[str, Any]

# Session data type
SessionData = Dict[str, Any]
