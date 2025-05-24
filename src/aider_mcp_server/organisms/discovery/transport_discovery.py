import asyncio
import uuid
from typing import Any, Awaitable, Callable, Dict, List, Optional

from aider_mcp_server.pages.application.coordinator import ApplicationCoordinator
from aider_mcp_server.atoms.logging.logger import Logger, get_logger
from aider_mcp_server.organisms.processors.error_handling import ErrorHandler, TransportError


class DiscoveryService:
    """
    A discovery service that allows transports to discover and connect to the coordinator.

    This singleton service manages transport discovery, registration callbacks,
    and provides information about coordinator availability and transport status.
    """

    _instance: Optional["DiscoveryService"] = None
    _initialized: bool = False

    def __new__(cls) -> "DiscoveryService":
        if cls._instance is None:
            cls._instance = super(DiscoveryService, cls).__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self._coordinator: Optional[ApplicationCoordinator] = None
        self._registered_callbacks: Dict[str, Callable[[], Awaitable[None]]] = {}
        self._discovery_lock = asyncio.Lock()
        self._transport_info: Dict[str, Dict[str, Any]] = {}

        # Initialize logger using the project's logging system
        self._logger: Logger = get_logger("transport_discovery")
        self._initialized = True

        self._logger.info("Transport discovery service initialized")

    def set_coordinator(self, coordinator: ApplicationCoordinator) -> None:
        """
        Set the ApplicationCoordinator instance for this discovery service.

        Args:
            coordinator: The ApplicationCoordinator instance
        """
        self._coordinator = coordinator
        self._logger.info("ApplicationCoordinator set for discovery service")

    async def register_discovery_callback(self, callback: Callable[[], Awaitable[None]]) -> str:
        """
        Register a callback to be called when a transport is discovered.

        Args:
            callback: Async callback function to be called on transport discovery

        Returns:
            Callback ID that can be used to unregister the callback
        """
        callback_id = str(uuid.uuid4())

        try:
            async with self._discovery_lock:
                self._registered_callbacks[callback_id] = callback
                self._logger.debug(f"Registered discovery callback with ID: {callback_id}")

            return callback_id

        except Exception as e:
            error_msg = f"Failed to register discovery callback: {e}"
            self._logger.error(error_msg)
            raise TransportError(error_msg) from e

    async def unregister_discovery_callback(self, callback_id: str) -> None:
        """
        Unregister a discovery callback.

        Args:
            callback_id: The ID of the callback to unregister
        """
        try:
            async with self._discovery_lock:
                if callback_id in self._registered_callbacks:
                    del self._registered_callbacks[callback_id]
                    self._logger.debug(f"Unregistered discovery callback with ID: {callback_id}")
                else:
                    self._logger.warning(f"Attempted to unregister unknown callback ID: {callback_id}")

        except Exception as e:
            error_msg = f"Failed to unregister discovery callback {callback_id}: {e}"
            self._logger.error(error_msg)
            raise TransportError(error_msg) from e

    async def notify_transport_available(self, transport_name: str, transport_info: Dict[str, Any]) -> None:
        """
        Notify that a transport is available.

        Args:
            transport_name: Name of the transport that became available
            transport_info: Information about the transport
        """
        try:
            # Store transport information
            async with self._discovery_lock:
                self._transport_info[transport_name] = {
                    **transport_info,
                    "status": "available",
                    "discovered_at": asyncio.get_event_loop().time(),
                }

            self._logger.info(f"Transport '{transport_name}' is now available")

            # Broadcast event to coordinator if available
            if self._coordinator and self._coordinator._initialized:
                try:
                    await self._coordinator.broadcast_event(
                        "transport_available", {"name": transport_name, "info": transport_info}
                    )
                    self._logger.debug(f"Broadcasted transport_available event for '{transport_name}'")
                except Exception as e:
                    self._logger.error(f"Failed to broadcast transport_available event: {e}")

            # Call registered callbacks
            async with self._discovery_lock:
                callbacks = list(self._registered_callbacks.values())

            for callback in callbacks:
                try:
                    await callback()
                except Exception as e:
                    # Log error but continue with other callbacks
                    ErrorHandler.log_exception(
                        e, context=f"discovery callback for transport '{transport_name}'", logger_instance=self._logger
                    )

        except Exception as e:
            error_msg = f"Failed to notify transport '{transport_name}' availability: {e}"
            self._logger.error(error_msg)
            raise TransportError(error_msg) from e

    async def notify_transport_unavailable(self, transport_name: str) -> None:
        """
        Notify that a transport is no longer available.

        Args:
            transport_name: Name of the transport that became unavailable
        """
        try:
            async with self._discovery_lock:
                if transport_name in self._transport_info:
                    self._transport_info[transport_name]["status"] = "unavailable"
                    self._transport_info[transport_name]["disconnected_at"] = asyncio.get_event_loop().time()

            self._logger.info(f"Transport '{transport_name}' is now unavailable")

            # Broadcast event to coordinator if available
            if self._coordinator and self._coordinator._initialized:
                try:
                    await self._coordinator.broadcast_event("transport_unavailable", {"name": transport_name})
                    self._logger.debug(f"Broadcasted transport_unavailable event for '{transport_name}'")
                except Exception as e:
                    self._logger.error(f"Failed to broadcast transport_unavailable event: {e}")

        except Exception as e:
            error_msg = f"Failed to notify transport '{transport_name}' unavailability: {e}"
            self._logger.error(error_msg)
            raise TransportError(error_msg) from e

    async def check_coordinator_available(self) -> bool:
        """
        Check if the coordinator is available.

        Returns:
            True if the coordinator is available and initialized, False otherwise
        """
        try:
            if self._coordinator is None:
                self._logger.debug("Coordinator availability check: No coordinator set")
                return False

            # Check if coordinator is initialized
            available = bool(self._coordinator._initialized)
            self._logger.debug(f"Coordinator availability check: {available}")
            return available

        except Exception as e:
            self._logger.error(f"Error checking coordinator availability: {e}")
            return False

    async def get_available_transports(self) -> Dict[str, Any]:
        """
        Get information about available transports.

        Returns:
            Dictionary containing information about available transports
        """
        try:
            async with self._discovery_lock:
                # Filter to only include available transports
                available_transports = {
                    name: info for name, info in self._transport_info.items() if info.get("status") == "available"
                }

            self._logger.debug(f"Retrieved {len(available_transports)} available transports")
            return available_transports

        except Exception as e:
            error_msg = f"Failed to get available transports: {e}"
            self._logger.error(error_msg)
            raise TransportError(error_msg) from e

    async def get_transport_info(self, transport_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific transport.

        Args:
            transport_name: Name of the transport

        Returns:
            Transport information if available, None otherwise
        """
        try:
            async with self._discovery_lock:
                transport_info = self._transport_info.get(transport_name)

            if transport_info:
                self._logger.debug(f"Retrieved info for transport '{transport_name}'")
            else:
                self._logger.debug(f"No info available for transport '{transport_name}'")

            return transport_info

        except Exception as e:
            error_msg = f"Failed to get transport info for '{transport_name}': {e}"
            self._logger.error(error_msg)
            raise TransportError(error_msg) from e

    async def discover_transports(self) -> List[str]:
        """
        Discover available transports by querying the coordinator.

        Returns:
            List of available transport names
        """
        try:
            if not await self.check_coordinator_available():
                self._logger.warning("Cannot discover transports: coordinator not available")
                return []

            # Get transport registry from coordinator
            if self._coordinator is not None and hasattr(self._coordinator, "_transport_registry"):
                registry = self._coordinator._transport_registry

                # Get active adapters from registry
                if hasattr(registry, "_active_adapters"):
                    active_transports = list(registry._active_adapters.keys())
                    self._logger.info(f"Discovered {len(active_transports)} active transports: {active_transports}")
                    return active_transports

            self._logger.debug("No active transports found in coordinator")
            return []

        except Exception as e:
            error_msg = f"Failed to discover transports: {e}"
            self._logger.error(error_msg)
            raise TransportError(error_msg) from e

    async def register_transport_with_coordinator(self, transport_name: str, **transport_config: Any) -> bool:
        """
        Register a transport with the coordinator.

        Args:
            transport_name: Name of the transport to register
            **transport_config: Configuration parameters for the transport

        Returns:
            True if registration was successful, False otherwise
        """
        try:
            if not await self.check_coordinator_available():
                self._logger.error(f"Cannot register transport '{transport_name}': coordinator not available")
                return False

            # Register transport with coordinator
            if self._coordinator is not None:
                await self._coordinator.register_transport(transport_name, **transport_config)
            else:
                raise TransportError("Coordinator not available for transport registration")

            # Update our transport info
            await self.notify_transport_available(transport_name, transport_config)

            self._logger.info(f"Successfully registered transport '{transport_name}' with coordinator")
            return True

        except Exception as e:
            error_msg = f"Failed to register transport '{transport_name}' with coordinator: {e}"
            ErrorHandler.log_exception(e, context=error_msg, logger_instance=self._logger)
            return False

    async def shutdown(self) -> None:
        """
        Shutdown the discovery service and clean up resources.
        """
        try:
            async with self._discovery_lock:
                # Clear all callbacks
                callback_count = len(self._registered_callbacks)
                self._registered_callbacks.clear()

                # Clear transport info
                transport_count = len(self._transport_info)
                self._transport_info.clear()

            self._coordinator = None

            self._logger.info(
                f"Discovery service shutdown complete: "
                f"cleared {callback_count} callbacks and {transport_count} transport records"
            )

        except Exception as e:
            ErrorHandler.log_exception(e, context="discovery service shutdown", logger_instance=self._logger)


# Global discovery service instance
def get_discovery_service() -> DiscoveryService:
    """
    Get the global discovery service instance.

    Returns:
        DiscoveryService singleton instance
    """
    return DiscoveryService()
