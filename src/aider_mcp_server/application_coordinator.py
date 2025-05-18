from typing import Any, Dict, Optional, Type, Union

from atoms.event_types import EventTypes

from aider_mcp_server.component_initializer import ComponentInitializer, Components
from aider_mcp_server.interfaces.transport_adapter import ITransportAdapter

# TransportAdapterRegistry is now accessed via components
from aider_mcp_server.mcp_types import LoggerFactory
from aider_mcp_server.security import Permissions
from aider_mcp_server.singleton_manager import SingletonManager

# Other components (EventCoordinator, HandlerRegistry, RequestProcessor, ResponseFormatter, SessionManager)
# are now accessed via the 'components' object passed during initialization.


class ApplicationCoordinator:
    """
    Acts as a central facade for the MCP server, coordinating various components.
    It handles incoming requests, manages transport adapters, and orchestrates
    event broadcasting and handling.

    This class is a singleton, accessible via `ApplicationCoordinator.getInstance()`.
    """

    # Singleton instance is managed by SingletonManager

    def __init__(self, logger_factory: LoggerFactory, components: Components):
        """
        Initializes the ApplicationCoordinator with pre-initialized components.
        This constructor is typically called by the SingletonManager during instance creation.

        Args:
            logger_factory: Factory function to create loggers.
            components: A Components object containing all initialized core services.
        """
        self.logger = logger_factory("ApplicationCoordinator")
        self.logger.verbose("ApplicationCoordinator initializing with pre-initialized components...")

        # Store references to the initialized components
        self._transport_registry = components.transport_registry
        self._session_manager = components.session_manager
        self._handler_registry = components.handler_registry
        self._response_formatter = components.response_formatter
        self._event_coordinator = components.event_coordinator
        self._request_processor = components.request_processor

        self.logger.info("ApplicationCoordinator instance configured with components.")

    @classmethod
    async def getInstance(cls, logger_factory: LoggerFactory) -> "ApplicationCoordinator":
        """
        Provides access to the singleton instance of ApplicationCoordinator.
        If the instance doesn't exist, it's created and initialized asynchronously.

        Args:
            logger_factory: Factory function to create loggers.

        Returns:
            The singleton instance of ApplicationCoordinator.

        Raises:
            RuntimeError: If initialization of the ApplicationCoordinator or its components fails.
        """
        return await SingletonManager.get_instance(
            cls,
            async_init_func=cls._create_and_initialize,
            logger_factory=logger_factory,  # Pass logger_factory to _create_and_initialize
        )

    @classmethod
    async def _create_and_initialize(cls, logger_factory: LoggerFactory) -> "ApplicationCoordinator":
        """
        Internal factory method called by SingletonManager to create and initialize
        the ApplicationCoordinator instance along with all its dependent components.

        Args:
            logger_factory: Factory function to create loggers.

        Returns:
            A new, fully initialized ApplicationCoordinator instance.

        Raises:
            RuntimeError: If component initialization fails.
        """
        # Use a specific logger for this critical initialization phase
        init_logger = logger_factory("ApplicationCoordinator.Initializer")
        init_logger.verbose("Starting ApplicationCoordinator and component initialization...")

        try:
            initializer = ComponentInitializer(logger_factory)
            components = await initializer.initialize_components()
            instance = cls(logger_factory, components)  # Pass components to __init__
            init_logger.info("ApplicationCoordinator instance created and initialized successfully.")
            return instance
        except Exception as e:
            init_logger.error(f"Fatal error during ApplicationCoordinator initialization: {e}", exc_info=True)
            # Re-raise to ensure SingletonManager and the caller are aware of the failure.
            raise RuntimeError(f"Failed to initialize ApplicationCoordinator: {e}") from e

    async def register_transport(self, transport_id: str, transport: Type[ITransportAdapter]) -> None:
        self.logger.verbose(f"Registering transport: {transport_id}")
        # This is a placeholder since TransportAdapterRegistry doesn't have register_transport method
        # Instead, it uses register_adapter_class, but that expects different parameters
        # For now, just log that this method was called
        # To implement fully, this would likely interact with self._transport_registry
        pass

    async def unregister_transport(self, transport_id: str) -> None:
        self.logger.verbose(f"Unregistering transport: {transport_id}")
        # This is a placeholder since TransportAdapterRegistry doesn't have unregister_transport method
        # For now, just log that this method was called
        # To implement fully, this would likely interact with self._transport_registry
        pass

    async def register_handler(
        self,
        operation_name: str,
        handler: Type[Any],  # Note: HandlerRegistry expects HandlerFunc, not Type[Any]
        required_permission: Optional[Permissions] = None,
    ) -> None:
        self.logger.verbose(f"Registering handler for operation: {operation_name}")
        await self._handler_registry.register_handler(operation_name, handler, required_permission)
        self.logger.verbose(f"Handler for operation {operation_name} registered.")

    async def unregister_handler(self, operation_name: str) -> None:
        self.logger.verbose(f"Unregistering handler for operation: {operation_name}")
        await self._handler_registry.unregister_handler(operation_name)
        self.logger.verbose(f"Handler for operation {operation_name} unregistered.")

    async def start_request(
        self,
        request_id: str,
        transport_id: str,
        operation_name: str,
        request_data: Dict[str, Any],
    ) -> None:
        self.logger.verbose(
            f"Starting request {request_id} for operation {operation_name} from transport {transport_id}. Data: {request_data}"
        )
        if self._request_processor:
            await self._request_processor.process_request(request_id, transport_id, operation_name, request_data)
            self.logger.verbose(f"Request {request_id} processing initiated.")
        else:
            self.logger.error(f"RequestProcessor not available to start request {request_id}")

    async def fail_request(
        self,
        request_id: str,
        operation_name: str,
        error: str,
        error_details: Union[str, Dict[str, Any]],
        originating_transport_id: Optional[str] = None,
        request_details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.logger.verbose(
            f"Failing request {request_id} for operation {operation_name}. Error: {error}, Details: {error_details}"
        )
        if self._request_processor:
            await self._request_processor.fail_request(
                request_id,
                operation_name,
                error,
                error_details,
                originating_transport_id,
                request_details,
            )
            self.logger.verbose(f"Request {request_id} failure processing initiated.")
        else:
            self.logger.error(f"RequestProcessor not available to fail request {request_id}")

    async def broadcast_event(
        self,
        event_type: EventTypes,
        data: Dict[str, Any],
        exclude_transport_id: Optional[str] = None,
    ) -> None:
        if self._event_coordinator:
            await self._event_coordinator.broadcast_event(event_type, data, exclude_transport_id)

    async def send_event_to_transport(self, transport_id: str, event_type: EventTypes, data: Dict[str, Any]) -> None:
        if self._event_coordinator:
            await self._event_coordinator.send_event_to_transport(transport_id, event_type, data)

    async def subscribe_to_event_type(self, transport_id: str, event_type: EventTypes) -> None:
        if self._event_coordinator:
            await self._event_coordinator.subscribe_to_event_type(transport_id, event_type)

    async def unsubscribe_from_event_type(self, transport_id: str, event_type: EventTypes) -> None:
        if self._event_coordinator:
            await self._event_coordinator.unsubscribe_from_event_type(transport_id, event_type)

    async def get_transport(self, transport_id: str) -> Optional[Type[ITransportAdapter]]:
        if self._transport_registry:
            # Transport registry may not have get_transport implemented yet
            # Return None for now
            return None
        return None

    async def transport_exists(self, transport_id: str) -> bool:
        if self._transport_registry:
            # Transport registry may not have transport_exists implemented yet
            # Return False for now
            return False
        return False

    async def update_request(
        self,
        request_id: str,
        status: str,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self._event_coordinator:
            # Event coordinator may not have update_request implemented yet
            # Use broadcast_event as a workaround
            data = {
                "request_id": request_id,
                "status": status,
                "message": message or "",
                "details": details or {},
            }
            await self._event_coordinator.broadcast_event(EventTypes.STATUS, data)
