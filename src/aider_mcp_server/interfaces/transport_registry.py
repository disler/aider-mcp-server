"""
Transport adapter registry for the Aider MCP Server.

This module provides a registry for transport adapters, allowing for
dynamic loading of adapters from plugins or modules.
"""

import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type, cast

from aider_mcp_server.atoms.types.event_types import EventTypes
from aider_mcp_server.atoms.logging.logger import get_logger
from aider_mcp_server.interfaces.transport_adapter import ITransportAdapter

logger = get_logger("transport_registry")


class TransportAdapterRegistry:
    """
    Registry for transport adapters.

    This class is responsible for:
    1. Discovering available transport adapters
    2. Registering adapter classes
    3. Instantiating adapter instances when requested
    4. Providing metadata about available adapters
    """

    _instance: Optional["TransportAdapterRegistry"] = None
    _adapter_classes: Dict[str, Type[ITransportAdapter]] = {}
    _adapter_cache: Dict[str, ITransportAdapter] = {}
    _initialized: bool = False

    def __new__(cls) -> "TransportAdapterRegistry":
        """Implement the singleton pattern."""
        if cls._instance is None:
            cls._instance = super(TransportAdapterRegistry, cls).__new__(cls)
        return cls._instance

    @classmethod
    async def get_instance(cls) -> "TransportAdapterRegistry":
        """Get or create a singleton instance of the registry."""
        if cls._instance is None:
            cls._instance = cls()
        if not cls._instance._initialized:
            await cls._instance.initialize()
        return cls._instance

    async def initialize(self) -> None:
        """Initialize the registry and discover built-in adapters."""
        if self._initialized:
            return

        # Discover built-in adapters
        self._discover_built_in_adapters()

        # Optional: Scan for plugin adapters in a designated directory
        # await self._discover_plugin_adapters()

        self._initialized = True
        logger.info(f"Transport adapter registry initialized with {len(self._adapter_classes)} adapter classes")

    def _discover_built_in_adapters(self) -> None:
        """Discover built-in transport adapters."""
        from aider_mcp_server.sse_transport_adapter import SSETransportAdapter
        from aider_mcp_server.stdio_transport_adapter import StdioTransportAdapter

        # Register known adapter implementations
        # We use cast to handle the type compatibility with Protocol
        self.register_adapter_class("sse", cast(Type[ITransportAdapter], SSETransportAdapter))
        self.register_adapter_class("stdio", cast(Type[ITransportAdapter], StdioTransportAdapter))

    async def _discover_plugin_adapters(self, plugin_dir: Optional[Path] = None) -> None:
        """
        Discover plugin transport adapters.

        This is a placeholder for future implementation of a plugin system.
        It would scan a plugins directory and load adapter classes dynamically.
        """
        # This is a stub for future implementation
        pass

    def register_adapter_class(self, adapter_type: str, adapter_class: Type[ITransportAdapter]) -> None:
        """
        Register a transport adapter class.

        Args:
            adapter_type: The type identifier for this adapter class
            adapter_class: The adapter class to register
        """
        if not inspect.isclass(adapter_class):
            raise TypeError(f"Expected a class, got {type(adapter_class)}")

        # Verify that the class implements ITransportAdapter
        if not issubclass(adapter_class, ITransportAdapter):
            raise TypeError(f"Class {adapter_class.__name__} does not implement ITransportAdapter")

        self._adapter_classes[adapter_type] = adapter_class
        logger.debug(f"Registered adapter class '{adapter_class.__name__}' for type '{adapter_type}'")

    def unregister_adapter_class(self, adapter_type: str) -> None:
        """
        Unregister a transport adapter class.

        Args:
            adapter_type: The type identifier for the adapter class to unregister
        """
        if adapter_type in self._adapter_classes:
            del self._adapter_classes[adapter_type]
            logger.debug(f"Unregistered adapter class for type '{adapter_type}'")

    def get_adapter_class(self, adapter_type: str) -> Optional[Type[ITransportAdapter]]:
        """
        Get a transport adapter class by type.

        Args:
            adapter_type: The type identifier for the adapter class

        Returns:
            The adapter class if found, None otherwise
        """
        return self._adapter_classes.get(adapter_type)

    def list_adapter_types(self) -> List[str]:
        """
        List all registered adapter types.

        Returns:
            A list of adapter type identifiers
        """
        return list(self._adapter_classes.keys())

    def get_adapter_capabilities(self, adapter_type: str) -> Optional[Set[EventTypes]]:
        """
        Get the capabilities of an adapter type without instantiating it.

        Some adapters may have class-level capability methods that can be called
        without instantiation. If not available, returns None.

        Args:
            adapter_type: The type identifier for the adapter

        Returns:
            The set of event types the adapter can handle, or None if unavailable
        """
        adapter_class = self.get_adapter_class(adapter_type)
        if adapter_class is None:
            return None

        # Try to get capabilities from class method if available
        if hasattr(adapter_class, "get_default_capabilities") and callable(adapter_class.get_default_capabilities):
            capabilities: Set[EventTypes] = adapter_class.get_default_capabilities()
            return capabilities

        return None

    async def create_adapter(
        self, adapter_type: str, transport_id: Optional[str] = None, **kwargs: Any
    ) -> Optional[ITransportAdapter]:
        """
        Create a new instance of a transport adapter.

        Args:
            adapter_type: The type identifier for the adapter to create
            transport_id: Optional unique ID for this transport instance
            **kwargs: Additional keyword arguments to pass to the adapter constructor

        Returns:
            A new adapter instance, or None if the adapter type is not registered
        """
        adapter_class = self.get_adapter_class(adapter_type)
        if adapter_class is None:
            logger.error(f"No adapter class registered for type '{adapter_type}'")
            return None

        try:
            # Create new adapter instance with transport_id if provided
            adapter_kwargs = kwargs.copy()
            if transport_id:
                adapter_kwargs["transport_id"] = transport_id

            adapter_instance = adapter_class(**adapter_kwargs)

            # Cache the instance if needed for future reference
            instance_id = adapter_instance.get_transport_id()
            cache_key = f"{adapter_type}:{instance_id}"
            self._adapter_cache[cache_key] = adapter_instance

            logger.info(f"Created adapter instance of type '{adapter_type}' with ID '{instance_id}'")
            return adapter_instance
        except Exception as e:
            logger.error(f"Failed to create adapter of type '{adapter_type}': {e}", exc_info=True)
            return None

    def get_cached_adapter(self, adapter_type: str, transport_id: str) -> Optional[ITransportAdapter]:
        """
        Get a cached adapter instance by type and ID.

        Args:
            adapter_type: The type identifier for the adapter
            transport_id: The unique ID for the transport instance

        Returns:
            The cached adapter instance if found, None otherwise
        """
        cache_key = f"{adapter_type}:{transport_id}"
        return self._adapter_cache.get(cache_key)

    def remove_cached_adapter(self, adapter_type: str, transport_id: str) -> None:
        """
        Remove a cached adapter instance.

        Args:
            adapter_type: The type identifier for the adapter
            transport_id: The unique ID for the transport instance
        """
        cache_key = f"{adapter_type}:{transport_id}"
        if cache_key in self._adapter_cache:
            del self._adapter_cache[cache_key]
            logger.debug(f"Removed cached adapter instance '{cache_key}'")
