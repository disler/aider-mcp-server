"""
Security atoms for the Aider MCP Server.

Provides atomic security components including permissions and security context.
"""

from aider_mcp_server.atoms.security.context import (
    ANONYMOUS_SECURITY_CONTEXT,
    SecurityContext,
)
from aider_mcp_server.atoms.security.permissions import Permissions

__all__ = [
    "Permissions",
    "SecurityContext", 
    "ANONYMOUS_SECURITY_CONTEXT",
]