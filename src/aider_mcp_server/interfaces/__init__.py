"""
Interface definitions for the Aider MCP Server.

This package contains protocol and interface definitions that define
standardized contracts for various components of the system.
"""

from aider_mcp_server.interfaces.transport_adapter import (
    ITransportAdapter,
    TransportAdapterBase,
)

__all__ = ["ITransportAdapter", "TransportAdapterBase"]
