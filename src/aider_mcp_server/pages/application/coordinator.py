import asyncio
from typing import Any, Dict, Optional, Type, Union

from aider_mcp_server.atoms.types.event_types import EventTypes

from aider_mcp_server.atoms.logging.logger import get_logger
from aider_mcp_server.interfaces.transport_adapter import ITransportAdapter
from aider_mcp_server.molecules.events.event_system import EventSystem
from aider_mcp_server.molecules.monitoring.health_monitor import HealthMonitor, HealthStatus
from aider_mcp_server.organisms.coordinators.event_coordinator import EventCoordinator
from aider_mcp_server.organisms.processors.request_processor import RequestProcessor
from aider_mcp_server.organisms.registries.handler_registry import HandlerRegistry, RequestHandler
from aider_mcp_server.organisms.registries.transport_adapter_registry_enhanced import EnhancedTransportAdapterRegistry


class ApplicationCoordinator:
    """
    Manages transports, handlers, and request processing.
    Serves as the central singleton for the application.
    Coordinates event distribution.
    Handles initialization and shutdown.
    """

    _instance: Optional["ApplicationCoordinator"] = None
    _initialized: bool = False

    def __new__(cls) -> "ApplicationCoordinator":
        if cls._instance is None:
            cls._instance = super(ApplicationCoordinator, cls).__new__(cls)
        return cls._instance

    @classmethod
    async def getInstance(cls, logger_factory: Optional[Any] = None) -> "ApplicationCoordinator":
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        """
        Initialize the ApplicationCoordinator.
        This constructor is called only once due to the singleton pattern.
        It sets up all core components.
        """
        # Only initialize once
        if ApplicationCoordinator._initialized:
            return

        self._logger = get_logger(__name__)  # Added as per project logging convention

        self._event_system = EventSystem()
        # As per Task 4 spec, EventCoordinator takes logger_factory and EventSystem
        self._event_coordinator = EventCoordinator(get_logger, self._event_system)
        self._request_processor = RequestProcessor()
        # As per Task 9 spec, TransportAdapterRegistry is initialized without arguments.
        # This assumes TransportAdapterRegistry's __init__ matches Task 3 spec: TransportAdapterRegistry()
        self._transport_registry = EnhancedTransportAdapterRegistry()
        self._handler_registry = HandlerRegistry()
        self._initialization_lock = asyncio.Lock()

        # Health monitoring component (Phase 3.2)
        self._health_monitor: Optional[HealthMonitor] = None

        # For backward compatibility with tests
        self._transports: Dict[str, ITransportAdapter] = {}

        ApplicationCoordinator._initialized = True
        self._logger.info("ApplicationCoordinator components instantiated.")

    async def initialize(self) -> None:
        """Initialize the application coordinator and all components."""
        async with self._initialization_lock:
            self._logger.info("Initializing ApplicationCoordinator components...")
            # Discover available transport adapters
            # Task 3 spec for discover_adapters defaults package_name to 'transports'.
            # The provided TransportAdapterRegistry.discover_adapters requires package_name.
            self._transport_registry.discover_adapters(package_name="transports")
            self._logger.info("Transport adapters discovery initiated.")

            # Set up request processor with handler registry
            supported_types = self._handler_registry.get_supported_request_types()
            self._logger.info(f"Found {len(supported_types)} supported request types from HandlerRegistry.")
            for request_type in supported_types:
                handler = self._handler_registry.get_handler(request_type)
                if handler:
                    self._request_processor.register_handler(request_type, handler)
                    self._logger.debug(f"Registered handler for '{request_type}' with RequestProcessor.")

            # Initialize health monitoring (Phase 3.2)
            self._health_monitor = HealthMonitor(self, metrics_retention_minutes=60, health_check_interval=30.0)  # type: ignore[arg-type]
            await self._health_monitor.start_monitoring()
            self._logger.info("Health monitoring started.")

            self._logger.info("ApplicationCoordinator initialization complete.")

    async def register_transport(
        self, transport_name: str, **kwargs: Any
    ) -> Optional[ITransportAdapter]:  # Return type changed to Optional[ITransportAdapter]
        """Register and initialize a transport adapter."""
        self._logger.info(f"Registering transport: {transport_name} with config: {kwargs}")
        # Task 9 spec calls initialize_adapter(transport_name, **kwargs)
        # Task 3 spec for TAR.initialize_adapter is (adapter_name, **kwargs)
        # Provided TAR.initialize_adapter is (transport_type, coordinator, config)
        # Adhering to Task 9 spec for the call:
        transport = await self._transport_registry.initialize_adapter(transport_name, self, {})

        if transport:
            # EventCoordinator in chat has register_transport_adapter
            # Task 4 spec for EventCoordinator has register_transport
            # Assuming register_transport_adapter is the correct method on the provided EventCoordinator
            await self._event_coordinator.register_transport_adapter(transport)
            self._logger.info(f"Transport '{transport_name}' registered and initialized successfully.")
            return transport
        else:
            self._logger.error(f"Failed to initialize transport '{transport_name}'.")
            return None

    async def register_transport_adapter(self, transport_adapter: ITransportAdapter) -> None:
        """Register an already-instantiated transport adapter with the EventCoordinator."""
        transport_id = transport_adapter.get_transport_id()
        transport_type = transport_adapter.get_transport_type()
        self._logger.info(f"Registering instantiated transport adapter: {transport_id} ({transport_type})")

        # Add to _transports for backward compatibility
        self._transports[transport_id] = transport_adapter

        await self._event_coordinator.register_transport_adapter(transport_adapter)
        self._logger.info(f"Transport adapter '{transport_id}' registered with EventCoordinator.")

    def register_handler(self, request_type: str, handler: RequestHandler) -> None:
        """Register a request handler."""
        self._logger.debug(f"Registering handler for request type: {request_type}")
        self._handler_registry.register_handler(request_type, handler)
        self._request_processor.register_handler(request_type, handler)
        self._logger.info(f"Handler for request type '{request_type}' registered.")

    def register_handler_class(self, handler_class: Type[Any]) -> None:
        """Register all handler methods from a class."""
        self._logger.info(f"Registering handler class: {handler_class.__name__}")
        self._handler_registry.register_handler_class(handler_class)
        # Re-register all handlers with RequestProcessor after class registration
        supported_types = self._handler_registry.get_supported_request_types()
        self._logger.debug(f"Updating RequestProcessor with handlers from {handler_class.__name__}.")
        for request_type in supported_types:
            handler = self._handler_registry.get_handler(request_type)
            if handler:
                # This might re-register, which is fine for RequestProcessor/HandlerRegistry
                self._request_processor.register_handler(request_type, handler)
        self._logger.info(f"Handlers from class '{handler_class.__name__}' registered.")

    async def process_request(self, request: Dict[str, Any]) -> Any:
        """Process an incoming request."""
        self._logger.debug(f"Processing request: {request.get('type', 'Unknown type')}")
        return await self._request_processor.process_request(request)

    async def broadcast_event(
        self,
        event_type: Union[str, EventTypes],
        event_data: Optional[Dict[str, Any]] = None,
        client_id: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        exclude_transport_id: Optional[str] = None,
    ) -> None:
        """Broadcast an event to all registered transports."""
        # Handle both 'event_data' and 'data' parameters for backward compatibility
        actual_data = event_data or data or {}
        self._logger.debug(f"Broadcasting event: Type='{event_type}', ClientID='{client_id}', Data='{actual_data}'")

        # Handle both EventTypes enum and string input
        if isinstance(event_type, EventTypes):
            event_type_enum = event_type
        elif isinstance(event_type, str):
            # Try to convert string to EventTypes enum
            try:
                event_type_enum = EventTypes(event_type)
            except ValueError:
                self._logger.warning(f"Unknown event type: {event_type}, using as-is")
                event_type_enum = event_type  # type: ignore
        else:
            # Fallback for any other type
            event_type_enum = event_type  # type: ignore

        # Use the EventCoordinator's broadcast_event method (without client_id)
        await self._event_coordinator.broadcast_event(event_type_enum, actual_data)

    async def subscribe_to_event_type(self, transport_id: str, event_type: Union[str, EventTypes]) -> None:
        """Subscribe a transport to an event type."""
        self._logger.debug(f"Subscribing transport {transport_id} to event type {event_type}")
        # Convert EventTypes enum to string or handle string input
        if isinstance(event_type, EventTypes):
            event_type_enum = event_type
        elif isinstance(event_type, str):
            try:
                event_type_enum = EventTypes(event_type)
            except ValueError:
                self._logger.warning(f"Unknown event type: {event_type}")
                return
        else:
            # Fallback for any other type - should not happen with proper typing
            event_type_enum = event_type  # type: ignore

        # Delegate to event coordinator
        await self._event_coordinator.subscribe_to_event_type(transport_id, event_type_enum)

    async def unregister_transport(self, transport_id: str) -> None:
        """Unregister a transport."""
        self._logger.info(f"Unregistering transport: {transport_id}")
        # Delegate to event coordinator
        await self._event_coordinator.unregister_transport_adapter(transport_id)

    async def start_request(
        self,
        request_id: str,
        transport_id: Optional[str] = None,
        operation_name: Optional[str] = None,
        request_data: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Start tracking a request."""
        # Build context from all provided data
        full_context = context or {}
        if transport_id:
            full_context["transport_id"] = transport_id
        if operation_name:
            full_context["operation_name"] = operation_name
        if request_data:
            full_context["request_data"] = request_data
        await self.record_request_start(request_id, full_context)

    async def fail_request(
        self,
        request_id: str,
        operation_name: Optional[str] = None,
        error: Optional[str] = None,
        error_details: Optional[str] = None,
        originating_transport_id: Optional[str] = None,
        error_type: Optional[str] = None,
    ) -> None:
        """Mark a request as failed."""
        # Use error or error_type as the error type for health monitoring
        final_error_type = error_type or error or "unknown_error"
        await self.record_request_completion(request_id, success=False, error_type=final_error_type)

    @property
    def is_shutting_down(self) -> bool:
        """Check if the coordinator is shutting down."""
        # For now, always return False - could be enhanced with a shutdown flag
        return False

    async def _initialize_coordinator(
        self, host: Optional[str] = None, port: Optional[int] = None, register_in_discovery: bool = False, **kwargs: Any
    ) -> None:
        """Initialize coordinator (alias for initialize method)."""
        # Log the initialization parameters for debugging
        self._logger.debug(
            f"Initializing coordinator with host={host}, port={port}, register_in_discovery={register_in_discovery}"
        )
        await self.initialize()

    async def shutdown(self) -> None:
        """Shut down the application coordinator and all components."""
        async with self._initialization_lock:
            self._logger.info("Shutting down ApplicationCoordinator...")

            # Stop health monitoring
            if self._health_monitor:
                await self._health_monitor.stop_monitoring()
                self._logger.info("Health monitoring stopped.")

            if hasattr(self._transport_registry, "shutdown_all"):
                await self._transport_registry.shutdown_all()
            else:
                self._logger.warning("TransportAdapterRegistry does not have shutdown_all method.")

            # Reset initialization flag to allow reinitialization if needed (e.g. in tests)
            ApplicationCoordinator._initialized = False
            # ApplicationCoordinator._instance = None # To allow full re-creation in tests
            self._logger.info("ApplicationCoordinator shutdown complete.")

    # Health monitoring methods (Phase 3.2)

    async def get_health_status(self) -> Optional[HealthStatus]:
        """Get comprehensive system health status."""
        if self._health_monitor:
            return await self._health_monitor.get_health_status()
        return None

    async def get_health_metrics_summary(self, minutes_back: int = 30) -> Optional[Dict[str, Any]]:
        """Get health metrics summary for the specified time period."""
        if self._health_monitor:
            return await self._health_monitor.get_metrics_summary(minutes_back)
        return None

    async def record_request_start(self, request_id: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Record the start of a request for health monitoring."""
        if self._health_monitor:
            await self._health_monitor.record_request_start(request_id, context)

    async def record_request_completion(self, request_id: str, success: bool, error_type: Optional[str] = None) -> None:
        """Record the completion of a request for health monitoring."""
        if self._health_monitor:
            await self._health_monitor.record_request_completion(request_id, success, error_type)

    async def record_throttling_event(self, request_id: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Record a throttling event for health monitoring."""
        if self._health_monitor:
            await self._health_monitor.record_throttling_event(request_id, context)

    async def register_streaming_client(self, client_id: str, client_info: Dict[str, Any]) -> None:
        """Register a streaming client for health monitoring."""
        if self._health_monitor:
            await self._health_monitor.register_streaming_client(client_id, client_info)

    async def unregister_streaming_client(self, client_id: str) -> None:
        """Unregister a streaming client from health monitoring."""
        if self._health_monitor:
            await self._health_monitor.unregister_streaming_client(client_id)
