"""
Configuration constants for the Aider MCP Server.

Provides default values for common configuration parameters.
This is an atomic component containing only immutable constants.
"""

# Default AI model for aider operations
DEFAULT_EDITOR_MODEL = "openai/gpt-4.1"

# Default port for SSE server
DEFAULT_WS_PORT = 8765

# Default host for SSE server
DEFAULT_WS_HOST = "localhost"

# Default port for HTTP server
DEFAULT_HTTP_PORT = 8766

# Default host for HTTP server
DEFAULT_HTTP_HOST = "127.0.0.1"

# Multi-client HTTP server configuration
MULTI_CLIENT_PORT_RANGE_START = 8000
MULTI_CLIENT_PORT_RANGE_END = 9000
MAX_CONCURRENT_CLIENTS = 10
CLIENT_SESSION_TIMEOUT = 3600  # 1 hour in seconds
WORKSPACE_BASE_DIR = "~/.local/share/aider-mcp-server/workspaces"  # User-specific workspace directory
DEFAULT_MANAGER_PORT = 7999  # Manager discovery port
