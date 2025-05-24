"""
Enhanced Transport Adapter Registry for MCP Protocol 2025-03-26 Compliance.

This enhanced registry provides:
- Transport prioritization (HTTP Streamable > Modernized SSE > Legacy SSE)
- MCP protocol version support and capability detection
- Transport recommendation logic based on deprecation status
- Health monitoring and fallback mechanisms
- Support for latest MCP 2025-03-26 features including authorization framework
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import pkgutil
import time
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Type

from aider_mcp_server.atoms.logging import get_logger
from aider_mcp_server.interfaces.transport_adapter import ITransportAdapter
from aider_mcp_server.transport_adapter import AbstractTransportAdapter

# No TYPE_CHECKING imports needed


class TransportProtocolVersion(Enum):
    """MCP Protocol versions supported by transports."""
    MCP_2024_11_05 = "2024-11-05"
    MCP_2025_03_26 = "2025-03-26"  # Latest with HTTP Streamable and auth framework


class TransportStatus(Enum):
    """Transport lifecycle status."""
    RECOMMENDED = "recommended"      # HTTP Streamable Transport
    SUPPORTED = "supported"          # Modernized SSE Transport  
    DEPRECATED = "deprecated"        # Legacy SSE Transport
    UNAVAILABLE = "unavailable"      # Failed to initialize


@dataclass
class TransportCapabilities:
    """Describes the capabilities of a transport adapter."""
    protocol_version: TransportProtocolVersion
    supports_authorization: bool
    supports_bidirectional: bool
    supports_resumability: bool
    supports_batching: bool
    deprecated: bool
    deprecation_message: Optional[str] = None


@dataclass
class TransportMetadata:
    """Metadata about a discovered transport adapter."""
    transport_type: str
    adapter_class: Type[AbstractTransportAdapter]
    capabilities: TransportCapabilities
    priority: int
    status: TransportStatus
    discovered_at: float


class EnhancedTransportAdapterRegistry:
    """
    Enhanced Transport Adapter Registry with MCP 2025-03-26 support.
    
    Features:
    - Transport prioritization based on MCP protocol compliance
    - Capability detection and recommendation logic
    - Health monitoring and fallback mechanisms
    - Support for authorization framework and latest features
    """

    def __init__(self) -> None:
        """Initialize the enhanced transport adapter registry."""
        self._logger = get_logger(__name__)
        
        # Transport discovery and metadata
        self._transport_metadata: Dict[str, TransportMetadata] = {}
        self._adapter_instances: Dict[str, ITransportAdapter] = {}
        
        # Priority and recommendation logic
        self._transport_priorities = {
            "http_streamable": 100,     # Highest priority - recommended
            "sse_modernized": 50,       # Medium priority - supported but deprecated
            "sse": 25,                  # Low priority - legacy deprecated
            "stdio": 10,                # Lowest priority - basic fallback
        }
        
        # Synchronization
        self._lock = asyncio.Lock()
        
        # Health monitoring
        self._health_stats: Dict[str, Any] = {
            "total_transports_discovered": 0,
            "recommended_transports": 0,
            "deprecated_transports": 0,
            "initialization_failures": 0,
            "last_discovery": None
        }
        
        self._logger.info("Enhanced Transport Adapter Registry initialized with MCP 2025-03-26 support")

    def discover_adapters(self, package_name: str) -> None:
        """
        Discover transport adapters with enhanced capability detection.
        
        Args:
            package_name: The Python package to search for transport adapters
        """
        self._logger.info(f"Discovering transport adapters in package: {package_name}")
        
        try:
            package = importlib.import_module(package_name)
        except ImportError:
            self._logger.error(f"Failed to import package: {package_name}")
            return

        if not hasattr(package, "__path__"):
            self._logger.warning(f"Package {package_name} has no __path__ attribute")
            if hasattr(package, "__file__"):
                self._discover_adapters_from_module(package)
            return

        prefix = package.__name__ + "."
        
        for _importer, modname, _ispkg in pkgutil.iter_modules(package.__path__, prefix):
            try:
                module = importlib.import_module(modname)
                self._discover_adapters_from_module(module)
            except Exception as e:
                self._logger.error(f"Error discovering adapters in module {modname}: {e}", exc_info=True)

        # Update discovery statistics
        self._health_stats["last_discovery"] = time.time()
        self._health_stats["total_transports_discovered"] = len(self._transport_metadata)
        
        recommended = sum(1 for meta in self._transport_metadata.values() 
                         if meta.status == TransportStatus.RECOMMENDED)
        deprecated = sum(1 for meta in self._transport_metadata.values() 
                        if meta.status == TransportStatus.DEPRECATED)
        
        self._health_stats["recommended_transports"] = recommended
        self._health_stats["deprecated_transports"] = deprecated
        
        self._logger.info(
            f"Transport discovery complete. Found {len(self._transport_metadata)} adapters "
            f"({recommended} recommended, {deprecated} deprecated)"
        )

    def _discover_adapters_from_module(self, module: Any) -> None:
        """Discover and analyze transport adapters from a module."""
        for _name, cls in inspect.getmembers(module, inspect.isclass):
            if not (issubclass(cls, AbstractTransportAdapter) and cls is not AbstractTransportAdapter):
                continue
                
            transport_type = getattr(cls, "TRANSPORT_TYPE_NAME", None)
            if not isinstance(transport_type, str) or not transport_type:
                continue
            
            # Analyze transport capabilities
            capabilities = self._analyze_transport_capabilities(cls)
            metadata = TransportMetadata(
                transport_type=transport_type,
                adapter_class=cls,
                capabilities=capabilities,
                priority=self._transport_priorities.get(transport_type, 0),
                status=self._determine_transport_status(capabilities),
                discovered_at=time.time()
            )
            
            # Register the transport
            if transport_type in self._transport_metadata:
                self._logger.warning(
                    f"Duplicate transport type '{transport_type}' found. "
                    f"Class {cls.__name__} will overwrite existing registration."
                )
            
            self._transport_metadata[transport_type] = metadata
            
            # Log discovery with status information
            status_info = f"({metadata.status.value})"
            if capabilities.deprecated:
                status_info += f" - {capabilities.deprecation_message}"
            
            self._logger.info(f"Discovered transport: {cls.__name__} [{transport_type}] {status_info}")

    def _analyze_transport_capabilities(self, adapter_class: Type[AbstractTransportAdapter]) -> TransportCapabilities:
        """Analyze the capabilities of a transport adapter class."""
        # Default capabilities for unknown transports
        capabilities = TransportCapabilities(
            protocol_version=TransportProtocolVersion.MCP_2024_11_05,
            supports_authorization=False,
            supports_bidirectional=False,
            supports_resumability=False,
            supports_batching=False,
            deprecated=False
        )
        
        class_name = adapter_class.__name__.lower()
        
        # HTTP Streamable Transport (recommended)
        if "http" in class_name and "streamable" in class_name:
            capabilities.protocol_version = TransportProtocolVersion.MCP_2025_03_26
            capabilities.supports_authorization = True
            capabilities.supports_bidirectional = True
            capabilities.supports_resumability = True
            capabilities.supports_batching = True
            capabilities.deprecated = False
            
        # Modernized SSE Transport (deprecated but compliant)
        elif "sse" in class_name and "modernized" in class_name:
            capabilities.protocol_version = TransportProtocolVersion.MCP_2025_03_26
            capabilities.supports_authorization = True
            capabilities.supports_bidirectional = False
            capabilities.supports_resumability = False
            capabilities.supports_batching = True
            capabilities.deprecated = True
            capabilities.deprecation_message = "SSE Transport deprecated in favor of HTTP Streamable"
            
        # Legacy SSE Transport (deprecated)
        elif "sse" in class_name:
            capabilities.protocol_version = TransportProtocolVersion.MCP_2024_11_05
            capabilities.supports_authorization = False
            capabilities.supports_bidirectional = False
            capabilities.supports_resumability = False
            capabilities.supports_batching = False
            capabilities.deprecated = True
            capabilities.deprecation_message = "Legacy SSE Transport - migrate to HTTP Streamable"
            
        # Check for explicit capability attributes on the class
        if hasattr(adapter_class, "MCP_PROTOCOL_VERSION"):
            version_str = getattr(adapter_class, "MCP_PROTOCOL_VERSION")
            if version_str == "2025-03-26":
                capabilities.protocol_version = TransportProtocolVersion.MCP_2025_03_26
                
        if hasattr(adapter_class, "SUPPORTS_AUTHORIZATION"):
            capabilities.supports_authorization = bool(getattr(adapter_class, "SUPPORTS_AUTHORIZATION"))
            
        if hasattr(adapter_class, "DEPRECATED"):
            capabilities.deprecated = bool(getattr(adapter_class, "DEPRECATED"))
            
        return capabilities

    def _determine_transport_status(self, capabilities: TransportCapabilities) -> TransportStatus:
        """Determine the status of a transport based on its capabilities."""
        if capabilities.deprecated:
            return TransportStatus.DEPRECATED
        
        if (capabilities.protocol_version == TransportProtocolVersion.MCP_2025_03_26 and
            capabilities.supports_authorization and 
            capabilities.supports_bidirectional):
            return TransportStatus.RECOMMENDED
            
        return TransportStatus.SUPPORTED

    async def get_recommended_transport_types(self) -> List[str]:
        """
        Get transport types ordered by recommendation (best first).
        
        Returns:
            List of transport types ordered by priority and capabilities
        """
        async with self._lock:
            # Sort by priority (highest first) and then by protocol version
            sorted_transports = sorted(
                self._transport_metadata.values(),
                key=lambda meta: (
                    meta.priority,
                    meta.capabilities.protocol_version == TransportProtocolVersion.MCP_2025_03_26,
                    not meta.capabilities.deprecated
                ),
                reverse=True
            )
            
            return [meta.transport_type for meta in sorted_transports]

    async def get_transport_recommendations(self) -> Dict[str, Any]:
        """
        Get detailed transport recommendations with rationale.
        
        Returns:
            Dictionary with recommended, supported, and deprecated transports
        """
        async with self._lock:
            recommendations: Dict[str, Any] = {
                "recommended": [],
                "supported": [],
                "deprecated": [],
                "rationale": {}
            }
            
            for transport_type, metadata in self._transport_metadata.items():
                transport_info = {
                    "type": transport_type,
                    "class": metadata.adapter_class.__name__,
                    "protocol_version": metadata.capabilities.protocol_version.value,
                    "features": {
                        "authorization": metadata.capabilities.supports_authorization,
                        "bidirectional": metadata.capabilities.supports_bidirectional,
                        "resumability": metadata.capabilities.supports_resumability,
                        "batching": metadata.capabilities.supports_batching
                    }
                }
                
                if metadata.capabilities.deprecated:
                    transport_info["deprecation_message"] = metadata.capabilities.deprecation_message or "Transport is deprecated"
                
                # Categorize transport
                if metadata.status == TransportStatus.RECOMMENDED:
                    recommendations["recommended"].append(transport_info)
                elif metadata.status == TransportStatus.DEPRECATED:
                    recommendations["deprecated"].append(transport_info)
                else:
                    recommendations["supported"].append(transport_info)
                
                # Add rationale
                rationale: List[str] = []
                if metadata.capabilities.protocol_version == TransportProtocolVersion.MCP_2025_03_26:
                    rationale.append("Latest MCP Protocol 2025-03-26 support")
                if metadata.capabilities.supports_authorization:
                    rationale.append("Authorization framework support")
                if metadata.capabilities.supports_bidirectional:
                    rationale.append("Bidirectional communication")
                if metadata.capabilities.deprecated:
                    msg = metadata.capabilities.deprecation_message or "No specific reason provided"
                    rationale.append(f"Deprecated: {msg}")
                    
                recommendations["rationale"][transport_type] = rationale
            
            return recommendations

    async def initialize_adapter(
        self,
        transport_type: str,
        coordinator: Any,  # ApplicationCoordinator
        config: Optional[Dict[str, Any]] = None
    ) -> Optional[ITransportAdapter]:
        """
        Initialize a transport adapter with enhanced error handling and deprecation warnings.
        
        Args:
            transport_type: Type of transport to initialize
            coordinator: ApplicationCoordinator instance  
            config: Configuration for the adapter
            
        Returns:
            Initialized transport adapter or None if failed
        """
        async with self._lock:
            metadata = self._transport_metadata.get(transport_type)
            
        if not metadata:
            self._logger.error(f"Unknown transport type: {transport_type}")
            self._health_stats["initialization_failures"] += 1
            return None
        
        # Issue deprecation warning if applicable
        if metadata.capabilities.deprecated:
            warning_msg = (
                f"Transport '{transport_type}' is deprecated. "
                f"{metadata.capabilities.deprecation_message or 'Consider migrating to recommended transport.'}"
            )
            self._logger.warning(warning_msg)
            warnings.warn(warning_msg, DeprecationWarning, stacklevel=2)
        
        try:
            self._logger.info(f"Initializing {metadata.status.value} transport: {transport_type}")
            
            # Initialize the adapter
            adapter_instance = metadata.adapter_class(coordinator=coordinator, **(config or {}))
            await adapter_instance.initialize()
            
            # Register the instance
            transport_id = adapter_instance.get_transport_id()
            async with self._lock:
                self._adapter_instances[transport_id] = adapter_instance
            
            self._logger.info(f"Successfully initialized {transport_type} adapter: {transport_id}")
            return adapter_instance
            
        except Exception as e:
            self._logger.error(f"Failed to initialize {transport_type} adapter: {e}", exc_info=True)
            self._health_stats["initialization_failures"] += 1
            return None

    def get_adapter(self, transport_id: str) -> Optional[ITransportAdapter]:
        """Get an initialized transport adapter by ID."""
        adapter = self._adapter_instances.get(transport_id)
        if adapter:
            self._logger.debug(f"Retrieved adapter: {transport_id}")
        else:
            self._logger.debug(f"Adapter not found: {transport_id}")
        return adapter

    async def get_transport_capabilities(self, transport_type: str) -> Optional[TransportCapabilities]:
        """Get the capabilities of a specific transport type."""
        async with self._lock:
            metadata = self._transport_metadata.get(transport_type)
            return metadata.capabilities if metadata else None

    async def get_health_status(self) -> Dict[str, Any]:
        """Get health and statistics information about the registry."""
        async with self._lock:
            return {
                "registry_status": "healthy",
                "statistics": self._health_stats.copy(),
                "active_adapters": len(self._adapter_instances),
                "discovered_transports": len(self._transport_metadata),
                "transport_breakdown": {
                    status.value: sum(1 for meta in self._transport_metadata.values() 
                                    if meta.status == status)
                    for status in TransportStatus
                }
            }

    async def shutdown_all(self) -> None:
        """Shutdown all transport adapters and clear the registry."""
        self._logger.info("Shutting down all transport adapters...")
        
        # Get list of adapters to shutdown
        adapters_to_shutdown: List[ITransportAdapter]
        async with self._lock:
            adapters_to_shutdown = list(self._adapter_instances.values())
            self._adapter_instances.clear()
        
        # Shutdown each adapter
        for adapter in adapters_to_shutdown:
            try:
                transport_id = adapter.get_transport_id()
                self._logger.info(f"Shutting down adapter: {transport_id}")
                await adapter.shutdown()
            except Exception as e:
                self._logger.error(f"Error shutting down adapter: {e}", exc_info=True)
        
        # Clear metadata
        async with self._lock:
            self._transport_metadata.clear()
            self._health_stats["total_transports_discovered"] = 0
            self._health_stats["recommended_transports"] = 0
            self._health_stats["deprecated_transports"] = 0
        
        self._logger.info("All transport adapters shut down and registry cleared")