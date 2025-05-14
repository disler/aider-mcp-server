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
        # Initialize components that don't have circular dependencies
        self._transport_registry: Optional[TransportAdapterRegistry] = None
        self._session_manager = SessionManager()
        self._handler_registry = HandlerRegistry()
        self._response_formatter = ResponseFormatter(logger_factory)

        # These will be initialized properly in getInstance after TransportRegistry is available
        self._event_coordinator: Optional[EventCoordinator] = None
        self._request_processor: Optional[RequestProcessor] = None

    @classmethod
    async def getInstance(
        cls, logger_factory: LoggerFactory
    ) -> "ApplicationCoordinator":
        if cls._instance is None:
            async with cls._creation_lock:
                if cls._instance is None:
                    try:
                        # Create instance but don't initialize transport registry yet
                        instance = cls(logger_factory)

                        # Now get the TransportAdapterRegistry instance asynchronously with timeout
                        try:
                            # Add timeout to avoid indefinite waiting
                            transport_registry = await asyncio.wait_for(
                                TransportAdapterRegistry.get_instance(),
                                timeout=10.0,  # 10 second timeout
                            )
                        except asyncio.TimeoutError:
                            raise RuntimeError(
                                "Timeout while initializing TransportAdapterRegistry"
                            ) from None
                        except Exception as e:
                            raise RuntimeError(
                                f"Failed to initialize TransportAdapterRegistry: {e}"
                            ) from e

                        if transport_registry is None:
                            raise RuntimeError(
                                "TransportAdapterRegistry initialization returned None"
                            )

                        # Set the transport registry and initialize dependent components
                        instance._transport_registry = transport_registry

                        try:
                            # Initialize EventCoordinator with error handling
                            instance._event_coordinator = EventCoordinator(
                                transport_registry, logger_factory
                            )
                        except Exception as e:
                            raise RuntimeError(
                                f"Failed to initialize EventCoordinator: {e}"
                            ) from e

                        try:
                            # Re-initialize the request processor with all dependencies
                            instance._request_processor = RequestProcessor(
                                instance._event_coordinator,
                                instance._session_manager,
                                logger_factory,
                                instance._handler_registry,
                                instance._response_formatter,
                            )
                        except Exception as e:
                            raise RuntimeError(
                                f"Failed to initialize RequestProcessor: {e}"
                            ) from e

                        # Only set the instance if all initialization steps completed successfully
                        cls._instance = instance
                    except Exception as e:
                        # Log the initialization error
                        logger = logger_factory("ApplicationCoordinator")
                        logger.error(
                            f"Failed to initialize ApplicationCoordinator: {e}"
                        )
                        # Re-raise the exception to inform the caller
                        raise e

        return cls._instance

    async def register_transport(
        self, transport_id: str, transport: Type[ITransportAdapter]
    ) -> None:
        # This is a placeholder since TransportAdapterRegistry doesn't have register_transport method
        # Instead, it uses register_adapter_class, but that expects different parameters
        # For now, just log that this method was called
        pass

    async def unregister_transport(self, transport_id: str) -> None:
        # This is a placeholder since TransportAdapterRegistry doesn't have unregister_transport method
        # For now, just log that this method was called
        pass

    async def register_handler(
        self,
        operation_name: str,
        handler: Type[Any],
        required_permission: Optional[Permissions] = None,
    ) -> None:
        await self._handler_registry.register_handler(
            operation_name, handler, required_permission
        )

    async def unregister_handler(self, operation_name: str) -> None:
        await self._handler_registry.unregister_handler(operation_name)

    async def start_request(
        self,
        request_id: str,
        transport_id: str,
        operation_name: str,
        request_data: Dict[str, Any],
    ) -> None:
        if self._request_processor:
            await self._request_processor.process_request(
                request_id, transport_id, operation_name, request_data
            )

    async def fail_request(
        self,
        request_id: str,
        operation_name: str,
        error: str,
        error_details: Union[str, Dict[str, Any]],
        originating_transport_id: Optional[str] = None,
        request_details: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self._request_processor:
            await self._request_processor.fail_request(
                request_id,
                operation_name,
                error,
                error_details,
                originating_transport_id,
                request_details,
            )

    async def broadcast_event(
        self,
        event_type: EventTypes,
        data: Dict[str, Any],
        exclude_transport_id: Optional[str] = None,
    ) -> None:
        if self._event_coordinator:
            await self._event_coordinator.broadcast_event(
                event_type, data, exclude_transport_id
            )

    async def send_event_to_transport(
        self, transport_id: str, event_type: EventTypes, data: Dict[str, Any]
    ) -> None:
        if self._event_coordinator:
            await self._event_coordinator.send_event_to_transport(
                transport_id, event_type, data
            )

    async def subscribe_to_event_type(
        self, transport_id: str, event_type: EventTypes
    ) -> None:
        if self._event_coordinator:
            await self._event_coordinator.subscribe_to_event_type(
                transport_id, event_type
            )

    async def unsubscribe_from_event_type(
        self, transport_id: str, event_type: EventTypes
    ) -> None:
        if self._event_coordinator:
            await self._event_coordinator.unsubscribe_from_event_type(
                transport_id, event_type
            )

    async def get_transport(
        self, transport_id: str
    ) -> Optional[Type[ITransportAdapter]]:
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
