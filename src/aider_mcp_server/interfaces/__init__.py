"""
Interface definitions for the Aider MCP Server.

This package contains protocol and interface definitions that define
standardized contracts for various components of the system.
"""

from aider_mcp_server.interfaces.application_coordinator import (
    IApplicationCoordinator,
)
from aider_mcp_server.interfaces.authentication_provider import (
    AuthToken,
    IAuthenticationProvider,
    UserInfo,
)
from aider_mcp_server.interfaces.dependency_container import (
    IDependencyContainer,
    Scope,
)
from aider_mcp_server.interfaces.error_handler import IErrorHandler
from aider_mcp_server.interfaces.event_coordinator import IEventCoordinator
from aider_mcp_server.interfaces.event_handler import IEventHandler
from aider_mcp_server.interfaces.request_handler import IRequestHandler
from aider_mcp_server.interfaces.security_service import ISecurityService
from aider_mcp_server.interfaces.transport_adapter import (
    ITransportAdapter,
    TransportAdapterBase,
)

__all__ = [
    "ITransportAdapter",
    "TransportAdapterBase",
    "IEventHandler",
    "IRequestHandler",
    "IApplicationCoordinator",
    "IEventCoordinator",
    "IErrorHandler",
    "ISecurityService",
    "IAuthenticationProvider",
    "AuthToken",
    "UserInfo",
    "IDependencyContainer",
    "Scope",
]
