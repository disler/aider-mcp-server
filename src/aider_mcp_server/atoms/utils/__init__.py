"""
Utility atoms for the Aider MCP Server.

Provides atomic utility components including configuration constants,
diff caching, and fallback configurations.
"""

from aider_mcp_server.atoms.utils.config_constants import (
    DEFAULT_EDITOR_MODEL,
    DEFAULT_WS_HOST,
    DEFAULT_WS_PORT,
)

__all__ = [
    "DEFAULT_EDITOR_MODEL",
    "DEFAULT_WS_HOST",
    "DEFAULT_WS_PORT",
]
