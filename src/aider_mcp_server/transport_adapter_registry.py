import asyncio
import importlib
import inspect
import logging
import pkgutil
import typing
from typing import Any, Dict, List, Optional, Type

from aider_mcp_server.interfaces.transport_adapter import ITransportAdapter
from aider_mcp_server.mcp_types import LoggerFactory, LoggerProtocol
from aider_mcp_server.transport_adapter import AbstractTransportAdapter

if typing.TYPE_CHECKING:
    from aider_mcp_server.transport_coordinator import ApplicationCoordinator


# Initialize the logger factory (copied from event_system.py)
get_logger_func: LoggerFactory

try:
    # Attempt to import the custom logger from the project structure
    from aider_mcp_server.atoms.logging import get_logger as custom_get_logger

    get_logger_func = typing.cast(LoggerFactory, custom_get_logger)
except ImportError:
    # Fallback to standard logging if custom logger is not found
    def fallback_get_logger(name: str, *args: typing.Any, **kwargs: typing.Any) -> LoggerProtocol:
        logger_instance = logging.getLogger(name)
        if not logger_instance.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger_instance.addHandler(handler)
            # Set a default level if not configured
            if logger_instance.level == logging.NOTSET:
                logger_instance.setLevel(logging.INFO)

        # Wrapper class to satisfy LoggerProtocol
        class CustomLogger(LoggerProtocol):
            def debug(self, message: str, **kwargs: typing.Any) -> None:
                logger_instance.debug(message, **kwargs)  # type: ignore

            def info(self, message: str, **kwargs: typing.Any) -> None:
                logger_instance.info(message, **kwargs)  # type: ignore

            def warning(self, message: str, **kwargs: typing.Any) -> None:
                logger_instance.warning(message, **kwargs)  # type: ignore

            def error(self, message: str, **kwargs: typing.Any) -> None:
                logger_instance.error(message, **kwargs)  # type: ignore

            def critical(self, message: str, **kwargs: typing.Any) -> None:
                logger_instance.critical(message, **kwargs)  # type: ignore

            def exception(self, message: str, **kwargs: typing.Any) -> None:
                logger_instance.exception(message, **kwargs)  # type: ignore

            def verbose(self, message: str, **kwargs: typing.Any) -> None:
                logger_instance.debug(message, **kwargs)  # type: ignore

        return CustomLogger()

    get_logger_func = fallback_get_logger


class TransportAdapterRegistry:
    """
    Manages the discovery, initialization, and lifecycle of transport adapters.
    """

    def __init__(self, logger_factory: Optional[LoggerFactory] = None):
        """
        Initialize the TransportAdapterRegistry.

        Args:
            logger_factory: Optional logger factory function. If None, uses global get_logger_func.
        """
        if logger_factory:
            self.logger: LoggerProtocol = logger_factory(__name__)
        else:
            self.logger = get_logger_func(__name__)

        self._adapter_classes: Dict[str, Type[AbstractTransportAdapter]] = {}
        self._adapters: Dict[str, ITransportAdapter] = {}
        self._lock = asyncio.Lock()
        self.logger.info("TransportAdapterRegistry initialized")

    def discover_adapters(self, package_name: str) -> None:
        """
        Discovers available transport adapter classes within a given package.

        It iterates through modules in the package, identifies classes that are
        subclasses of AbstractTransportAdapter (and not AbstractTransportAdapter itself),
        and stores them in an internal dictionary, keyed by their TRANSPORT_TYPE_NAME
        class attribute.

        Args:
            package_name: The Python dot-notation package name (e.g., "aider_mcp_server.adapters").
        """
        self.logger.info(f"Discovering transport adapters in package: {package_name}")
        try:
            package = importlib.import_module(package_name)
        except ImportError:
            self.logger.error(f"Failed to import package: {package_name}")
            return

        prefix = package.__name__ + "."

        # Ensure package.__path__ is available and iterable
        if not hasattr(package, "__path__"):
            self.logger.warning(f"Package {package_name} has no __path__ attribute. Cannot discover modules.")
            # Attempt to load from the package module itself if it's a single file acting as a namespace
            if hasattr(package, "__file__"):
                self._discover_adapters_from_module(package)
            return

        for importer, modname, ispkg in pkgutil.iter_modules(package.__path__, prefix):
            try:
                module = importlib.import_module(modname)
                self._discover_adapters_from_module(module)
            except Exception as e:
                self.logger.error(f"Error discovering adapters in module {modname}: {e}", exc_info=True)

        self.logger.info(f"Adapter discovery complete. Found {len(self._adapter_classes)} adapter classes.")

    def _discover_adapters_from_module(self, module: Any) -> None:
        """Helper to discover adapters from a single module."""
        for name, cls in inspect.getmembers(module, inspect.isclass):
            if issubclass(cls, AbstractTransportAdapter) and cls is not AbstractTransportAdapter:
                transport_type_name = getattr(cls, "TRANSPORT_TYPE_NAME", None)
                if isinstance(transport_type_name, str) and transport_type_name:
                    # Run _register_adapter_class in a way that doesn't block discover_adapters
                    # If discover_adapters is called from sync code, this needs careful handling.
                    # For now, assuming discover_adapters might be part of an async setup or called from sync context
                    # where blocking briefly for lock acquisition is acceptable.
                    # If discover_adapters itself needs to be async, this changes.
                    # The prompt implies discover_adapters is sync, but _register_adapter_class is async.
                    # This is a conflict. Let's make _register_adapter_class sync for now or use asyncio.run()
                    # as a bridge if absolutely necessary and if discover_adapters is truly sync.
                    # Given the user's example, they used asyncio.run()
                    try:
                        asyncio.get_running_loop()
                        # If a loop is running, schedule it. This is tricky if discover_adapters is purely sync.
                        # A better pattern might be to make discover_adapters async if it uses async helpers.
                        # For now, sticking to the user's asyncio.run() pattern.
                        asyncio.ensure_future(self._register_adapter_class(transport_type_name, cls))
                    except RuntimeError:  # No running event loop
                        asyncio.run(self._register_adapter_class(transport_type_name, cls))
                else:
                    self.logger.warning(
                        f"Adapter class {cls.__name__} in module {module.__name__} "
                        f"does not define a valid TRANSPORT_TYPE_NAME string attribute. Skipping."
                    )

    async def _register_adapter_class(
        self, transport_type_name: str, adapter_class: Type[AbstractTransportAdapter]
    ) -> None:
        """Registers an adapter class. Made async to use self._lock."""
        async with self._lock:
            if transport_type_name in self._adapter_classes:
                self.logger.warning(
                    f"Duplicate transport type '{transport_type_name}' found. "
                    f"Class {adapter_class.__name__} will overwrite "
                    f"{self._adapter_classes[transport_type_name].__name__}."
                )
            self._adapter_classes[transport_type_name] = adapter_class
        self.logger.info(f"Discovered transport adapter: {adapter_class.__name__} for type '{transport_type_name}'")

    async def initialize_adapter(
        self,
        transport_type: str,
        coordinator: "ApplicationCoordinator",
        config: Optional[Dict[str, Any]] = None,
    ) -> Optional[ITransportAdapter]:
        """
        Initializes a transport adapter of the given type.

        Args:
            transport_type: The type of transport adapter to initialize (e.g., "stdio", "sse").
            coordinator: The ApplicationCoordinator instance.
            config: Configuration dictionary for the adapter.

        Returns:
            An initialized ITransportAdapter instance, or None if initialization fails.
        """
        self.logger.info(f"Initializing transport adapter of type: {transport_type}")
        adapter_class: Optional[Type[AbstractTransportAdapter]] = None
        async with self._lock:  # Protect access to _adapter_classes
            adapter_class = self._adapter_classes.get(transport_type)

        if not adapter_class:
            self.logger.error(f"No adapter class found for transport type: {transport_type}")
            return None

        try:
            # Pass coordinator and unpack config. Adapters should accept 'coordinator'.
            # Specific config keys are handled by individual adapter constructors.
            adapter_instance = adapter_class(coordinator=coordinator, **(config or {}))

            # Call the adapter's own initialize method
            await adapter_instance.initialize()

            async with self._lock:  # Protect access to _adapters
                self._adapters[adapter_instance.get_transport_id()] = adapter_instance

            self.logger.info(
                f"Successfully initialized adapter {adapter_instance.get_transport_id()} of type {transport_type}"
            )
            return adapter_instance
        except Exception as e:
            self.logger.error(f"Failed to initialize adapter of type {transport_type}: {e}", exc_info=True)
            return None

    def get_adapter(self, transport_id: str) -> Optional[ITransportAdapter]:
        """
        Retrieves an initialized adapter by its transport ID.

        Args:
            transport_id: The unique ID of the transport adapter.

        Returns:
            The ITransportAdapter instance, or None if not found.
        """
        # Reading from _adapters might not strictly need a lock if writes are protected,
        # but for consistency and safety if iterators are involved elsewhere:
        # async with self._lock: # Not strictly necessary for dict.get()
        adapter = self._adapters.get(transport_id)
        if adapter:
            self.logger.debug(f"Retrieved adapter: {transport_id}")
        else:
            self.logger.debug(f"Adapter not found: {transport_id}")
        return adapter

    async def shutdown_all(self) -> None:
        """
        Shuts down all initialized transport adapters and clears the registry.
        """
        self.logger.info("Shutting down all transport adapters...")

        adapters_to_shutdown: List[ITransportAdapter]
        async with self._lock:
            adapters_to_shutdown = list(self._adapters.values())
            self._adapters.clear()  # Clear early

        for adapter in adapters_to_shutdown:
            try:
                self.logger.info(f"Shutting down adapter {adapter.get_transport_id()}")
                await adapter.shutdown()
            except Exception as e:
                self.logger.error(f"Error shutting down adapter {adapter.get_transport_id()}: {e}", exc_info=True)

        async with self._lock:
            self._adapter_classes.clear()

        self.logger.info("All transport adapters shut down and registry cleared.")
