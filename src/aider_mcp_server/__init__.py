"""
Aider MCP Server Package

This package provides the core functionality for the Aider Model Context Protocol (MCP) server,
supporting multiple transport mechanisms like Standard I/O (stdio) and Server-Sent Events (SSE).
"""

import importlib.metadata
import logging

from .interfaces.transport_adapter import ITransportAdapter, TransportAdapterBase
from .molecules.transport.base_adapter import AbstractTransportAdapter
from .organisms.transports.sse.sse_transport_adapter import SSETransportAdapter
from .organisms.transports.stdio.stdio_transport_adapter import StdioTransportAdapter
from .pages.application.coordinator import ApplicationCoordinator

# Import main from cli.py instead of __main__.py to avoid circular imports
from .templates.initialization.cli import main  # Import main function for entry point
from .templates.servers.multi_transport_server import serve_multi_transport
from .templates.servers.server import AIDER_AI_CODE_TOOL, LIST_MODELS_TOOL  # Expose tool definitions
from .templates.servers.sse_server import serve_sse

# Get the package version dynamically from installed package metadata
try:
    __version__ = importlib.metadata.version("aider-mcp-server")  # Use package name
except importlib.metadata.PackageNotFoundError:
    # Handle case where package is not installed (e.g., during development)
    __version__ = "0.0.0-dev"
    logging.getLogger(__name__).warning(
        "Could not determine package version from metadata. Defaulting to %s",
        __version__,
    )


# Define the public API of the package
__all__ = [
    "main",  # Re-added to support entry points in setup.py
    "serve_sse",
    "serve_multi_transport",
    "ApplicationCoordinator",
    "SSETransportAdapter",
    "StdioTransportAdapter",
    "AbstractTransportAdapter",
    "ITransportAdapter",
    "TransportAdapterBase",
    "AIDER_AI_CODE_TOOL",
    "LIST_MODELS_TOOL",
    "__version__",
]
