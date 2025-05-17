import asyncio
from typing import Any, Dict, Optional, Type, Union

from atoms.event_types import EventTypes

from aider_mcp_server.event_coordinator import EventCoordinator
from aider_mcp_server.handler_registry import HandlerRegistry
from aider_mcp_server.interfaces.transport_adapter import ITransportAdapter
from aider_mcp_server.interfaces.transport_registry import TransportAdapterRegistry
from aider_mcp_server.mcp_types import LoggerFactory
from aider_mcp_server.request_processor import RequestProcessor
from aider_mcp_server.response_formatter import ResponseFormatter
from aider_mcp_server.security import Permissions
from aider_mcp_server.session_manager import SessionManager


class ApplicationCoordinator:
    _instance: Optional["ApplicationCoordinator"] = None
    _creation_lock = asyncio.Lock()

    def __init__(self, logger_factory: LoggerFactory) -> None:
        self.logger = logger_factory("ApplicationCoordinator")
        self.logger.verbose("ApplicationCoordinator initializing...")
        # Initialize components that don't have circular dependencies
        self._transport_registry: Optional[TransportAdapterRegistry] = None
        self._session_manager = SessionManager()
        self._handler_registry = HandlerRegistry()
        self._response_formatter = ResponseFormatter(logger_factory)

        # These will be initialized properly in getInstance after TransportRegistry is available
        self._event_coordinator: Optional[EventCoordinator] = None
        self._request_processor: Optional[RequestProcessor] = None

    @classmethod
    async def getInstance(cls, logger_factory: LoggerFactory) -> "ApplicationCoordinator":
        if cls._instance is None:
            # Temporary logger for getInstance scope if instance isn't created yet
            temp_logger = logger_factory("ApplicationCoordinator.getInstance")
            temp_logger.verbose("Attempting to get ApplicationCoordinator instance.")
            async with cls._creation_lock:
                temp_logger.verbose("Acquired creation lock.")
                if cls._instance is None:
                    temp_logger.verbose("ApplicationCoordinator instance is None, creating new instance.")
                    try:
                        # Create instance but don't initialize transport registry yet
                        instance = cls(logger_factory)
                        instance.logger.verbose("Initial ApplicationCoordinator object created.")

                        # Now get the TransportAdapterRegistry instance asynchronously with timeout
                        instance.logger.verbose("Initializing TransportAdapterRegistry...")
                        try:
                            # Add timeout to avoid indefinite waiting
                            transport_registry = await asyncio.wait_for(
                                TransportAdapterRegistry.get_instance(),
                                timeout=10.0,  # 10 second timeout
                            )
                            instance.logger.verbose("TransportAdapterRegistry.get_instance() successful.")
                        except asyncio.TimeoutError as e:
                            instance.logger.error("Timeout while initializing TransportAdapterRegistry.")
                            raise RuntimeError("Timeout while initializing TransportAdapterRegistry") from e
                        except Exception as e:
                            instance.logger.error(f"Failed to initialize TransportAdapterRegistry: {e}")
                            raise RuntimeError(f"Failed to initialize TransportAdapterRegistry: {e}") from e

                        if transport_registry is None:
                            instance.logger.error("TransportAdapterRegistry initialization returned None.")
                            raise RuntimeError("TransportAdapterRegistry initialization returned None")

                        # Set the transport registry and initialize dependent components
                        instance._transport_registry = transport_registry
                        instance.logger.verbose("TransportAdapterRegistry initialized and set.")

                        try:
                            instance.logger.verbose("Initializing EventCoordinator...")
                            # Initialize EventCoordinator with error handling
                            instance._event_coordinator = EventCoordinator(transport_registry, logger_factory)
                            instance.logger.verbose("EventCoordinator initialized.")
                        except Exception as e:
                            instance.logger.error(f"Failed to initialize EventCoordinator: {e}")
                            raise RuntimeError(f"Failed to initialize EventCoordinator: {e}") from e

                        try:
                            instance.logger.verbose("Initializing RequestProcessor...")
                            # Re-initialize the request processor with all dependencies
                            instance._request_processor = RequestProcessor(
                                instance._event_coordinator,
                                instance._session_manager,
                                logger_factory,
                                instance._handler_registry,
                                instance._response_formatter,
                            )
                            instance.logger.verbose("RequestProcessor initialized.")
                        except Exception as e:
                            instance.logger.error(f"Failed to initialize RequestProcessor: {e}")
                            raise RuntimeError(f"Failed to initialize RequestProcessor: {e}") from e

                        # Only set the instance if all initialization steps completed successfully
                        cls._instance = instance
                        instance.logger.info("ApplicationCoordinator instance created and initialized successfully.")
                    except Exception as e:
                        # Log the initialization error using the logger from the partially created instance if available
                        # otherwise use the temp_logger
                        logger_to_use = getattr(cls._instance, 'logger', temp_logger) if cls._instance else temp_logger
                        logger_to_use.error(f"Failed to initialize ApplicationCoordinator: {e}")
                        # Re-raise the exception to inform the caller
                        raise e
                else:
                    temp_logger.verbose("ApplicationCoordinator instance already exists.")
        return cls._instance

    async def register_transport(self, transport_id: str, transport: Type[ITransportAdapter]) -> None:
        self.logger.verbose(f"Registering transport: {transport_id}")
        # This is a placeholder since TransportAdapterRegistry doesn't have register_transport method
        # Instead, it uses register_adapter_class, but that expects different parameters
        # For now, just log that this method was called
        pass

    async def unregister_transport(self, transport_id: str) -> None:
        self.logger.verbose(f"Unregistering transport: {transport_id}")
        # This is a placeholder since TransportAdapterRegistry doesn't have unregister_transport method
        # For now, just log that this method was called
        pass

    async def register_handler(
        self,
        operation_name: str,
        handler: Type[Any],
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
        self.logger.verbose(f"Starting request {request_id} for operation {operation_name} from transport {transport_id}. Data: {request_data}")
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
        self.logger.verbose(f"Failing request {request_id} for operation {operation_name}. Error: {error}, Details: {error_details}")
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
