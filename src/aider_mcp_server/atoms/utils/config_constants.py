"""
Configuration constants for the Aider MCP Server.

Provides default values for common configuration parameters.
This is an atomic component containing only immutable constants.
"""

# Default AI model for aider operations
DEFAULT_EDITOR_MODEL = "gemini-2.5-flash-preview-04-17"

# Default port for SSE server
DEFAULT_WS_PORT = 8765

# Default host for SSE server
DEFAULT_WS_HOST = "localhost"

# Default port for HTTP server
DEFAULT_HTTP_PORT = 8766

# Default host for HTTP server
DEFAULT_HTTP_HOST = "127.0.0.1"
