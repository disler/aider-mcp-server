"""
Transport Adapter Registry for Aider MCP Server.

This module provides a registry for discovering, instantiating, and managing
transport adapters that conform to the ITransportAdapter interface.
"""

import importlib
import inspect
import pkgutil
from typing import Any, Dict, List, Optional, Type

from aider_mcp_server.atoms.logging import get_logger
from aider_mcp_server.interfaces import ITransportAdapter, TransportAdapterBase

# Logger for the module
logger = get_logger(__name__)


class TransportAdapterRegistry:
    """
    Manages discovery, instantiation, and lifecycle of transport adapters.
    """

    def __init__(self) -> None:
        """Initialize the TransportAdapterRegistry."""
        self._adapter_classes: Dict[str, Type[ITransportAdapter]] = {}
        self._active_adapters: Dict[str, ITransportAdapter] = {}
        self.logger = logger  # Use the module-level logger instance

    async def discover_adapters(self, package_name: str = "aider_mcp_server.transports") -> None:
        """
        Discover transport adapters in the specified package.

        Args:
            package_name: The Python package to scan for adapters (e.g., "aider_mcp_server.transports").
        """
        self.logger.info(f"Discovering transport adapters in package: {package_name}")
        try:
            package = importlib.import_module(package_name)
        except ImportError:
            self.logger.error(f"Failed to import package for adapter discovery: {package_name}")
            return

        if not hasattr(package, "__path__"):
            self.logger.warning(f"Package {package_name} has no __path__ attribute. Cannot discover modules within it.")
            return

        discovered_count = 0
        for _, name, is_pkg in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
            if not is_pkg:
                try:
                    module = importlib.import_module(name)
                    for member_name, obj_class in inspect.getmembers(module, inspect.isclass):
                        if (
                            issubclass(obj_class, ITransportAdapter)
                            and obj_class is not ITransportAdapter
                            and obj_class is not TransportAdapterBase
                        ):
                            # Use the lowercase class name as the key, as per spec.
                            # e.g., SSETransportAdapter -> "ssetransportadapter"
                            adapter_key = obj_class.__name__.lower()

                            if adapter_key in self._adapter_classes:
                                self.logger.warning(
                                    f"Duplicate adapter key '{adapter_key}' found. "
                                    f"Existing: {self._adapter_classes[adapter_key].__name__}, New: {obj_class.__name__}. Skipping new."
                                )
                                continue

                            self._adapter_classes[adapter_key] = obj_class
                            self.logger.info(f"Discovered transport adapter: '{adapter_key}' ({obj_class.__name__})")
                            discovered_count += 1
                except Exception as e:
                    self.logger.error(f"Error importing or inspecting module {name}: {e}", exc_info=True)
        self.logger.info(f"Adapter discovery complete. Found {discovered_count} adapter classes.")

    async def initialize_adapter(self, adapter_name: str, **kwargs: Any) -> ITransportAdapter:
        """
        Initialize a specific adapter by name.

        Args:
            adapter_name: The name of the adapter to initialize (e.g., "ssetransportadapter").
            **kwargs: Arguments to pass to the adapter's constructor.

        Returns:
            The initialized transport adapter instance.

        Raises:
            ValueError: If the adapter_name is unknown.
            RuntimeError: If adapter initialization fails.
        """
        self.logger.info(f"Request to initialize adapter: '{adapter_name}'")
        adapter_instance = None  # Define here for potential use in finally/except

        if adapter_name not in self._adapter_classes:
            self.logger.error(f"Unknown adapter type: '{adapter_name}'")
            raise ValueError(f"Unknown adapter: {adapter_name}")

        if adapter_name in self._active_adapters:
            self.logger.warning(
                f"Adapter '{adapter_name}' is already initialized and active. Returning existing instance."
            )
            return self._active_adapters[adapter_name]

        adapter_class = self._adapter_classes[adapter_name]
        self.logger.info(f"Instantiating adapter '{adapter_name}' from class {adapter_class.__name__}")

        try:
            adapter_instance = adapter_class(**kwargs)
            await adapter_instance.initialize()

            self._active_adapters[adapter_name] = adapter_instance
            self.logger.info(
                f"Adapter '{adapter_name}' (ID: {adapter_instance.get_transport_id()}) initialized and activated successfully."
            )
            return adapter_instance
        except Exception as e:
            self.logger.error(f"Failed to initialize adapter '{adapter_name}': {e}", exc_info=True)
            # Ensure it's not left in active_adapters if initialization failed partway
            # and adapter_instance was successfully created before the error.
            if (
                adapter_instance is not None
                and adapter_name in self._active_adapters
                and self._active_adapters.get(adapter_name) is adapter_instance
            ):
                del self._active_adapters[adapter_name]
            raise RuntimeError(f"Failed to initialize adapter {adapter_name}") from e

    def get_adapter(self, adapter_name: str) -> ITransportAdapter:
        """
        Get an active adapter by its registration name.

        Args:
            adapter_name: The registration name of the adapter (e.g., "ssetransportadapter").

        Returns:
            The active transport adapter instance.

        Raises:
            ValueError: If the adapter is not initialized or not found.
        """
        adapter = self._active_adapters.get(adapter_name)
        if not adapter:
            self.logger.debug(f"Adapter '{adapter_name}' not found in active adapters.")
            raise ValueError(f"Adapter not initialized: {adapter_name}")
        return adapter

    def get_active_adapter_by_id(self, transport_id: str) -> Optional[ITransportAdapter]:
        """
        Get an active adapter by its unique transport_id.

        Args:
            transport_id: The unique ID of the transport instance.

        Returns:
            The active adapter instance, or None if not found.
        """
        for adapter in self._active_adapters.values():
            if adapter.get_transport_id() == transport_id:
                return adapter
        self.logger.debug(f"No active adapter found with transport_id: {transport_id}")
        return None

    def get_all_active_adapters(self) -> List[ITransportAdapter]:
        """
        Get a list of all currently active adapters.

        Returns:
            A list of active ITransportAdapter instances.
        """
        return list(self._active_adapters.values())

    async def shutdown_adapter(self, adapter_name: str) -> None:
        """
        Shutdown a specific adapter by its registration name.

        Args:
            adapter_name: The registration name of the adapter to shut down.
        """
        self.logger.info(f"Request to shut down adapter: '{adapter_name}'")
        adapter = self._active_adapters.pop(adapter_name, None)

        if adapter:
            transport_id = "unknown"
            try:
                # Attempt to get transport_id, but handle if adapter state is inconsistent
                transport_id = adapter.get_transport_id()
            except Exception:
                self.logger.warning(f"Could not retrieve transport_id for adapter '{adapter_name}' during shutdown.")

            try:
                await adapter.shutdown()
                self.logger.info(f"Adapter '{adapter_name}' (ID: {transport_id}) shut down successfully.")
            except Exception as e:
                self.logger.error(
                    f"Error shutting down adapter '{adapter_name}' (ID: {transport_id}): {e}",
                    exc_info=True,
                )
        else:
            self.logger.warning(f"Adapter '{adapter_name}' not found or not active, cannot shut down.")

    async def shutdown_all(self) -> None:
        """Shutdown all active transport adapters."""
        self.logger.info("Shutting down all active transport adapters...")

        adapter_names = list(self._active_adapters.keys())

        for adapter_name in adapter_names:
            adapter = self._active_adapters.pop(adapter_name, None)
            if adapter:
                transport_id = "unknown"
                try:
                    transport_id = adapter.get_transport_id()
                except Exception:
                    self.logger.warning(
                        f"Could not retrieve transport_id for adapter '{adapter_name}' during shutdown_all."
                    )

                self.logger.info(f"Shutting down adapter '{adapter_name}' (ID: {transport_id})...")
                try:
                    await adapter.shutdown()
                    self.logger.info(f"Adapter '{adapter_name}' (ID: {transport_id}) shut down successfully.")
                except Exception as e:
                    self.logger.error(
                        f"Error during shutdown of adapter '{adapter_name}' (ID: {transport_id}): {e}",
                        exc_info=True,
                    )

        self._active_adapters.clear()
        self.logger.info("All active transport adapters have been processed for shutdown.")
