import asyncio
from typing import Any, Dict, Optional, Type

from aider_mcp_server.atoms.logging.logger import get_logger
from aider_mcp_server.interfaces.transport_adapter import ITransportAdapter
from aider_mcp_server.molecules.events.event_system import EventSystem
from aider_mcp_server.organisms.coordinators.event_coordinator import EventCoordinator
from aider_mcp_server.organisms.processors.request_processor import RequestProcessor
from aider_mcp_server.organisms.registries.handler_registry import HandlerRegistry, RequestHandler
from aider_mcp_server.organisms.registries.transport_adapter_registry import TransportAdapterRegistry


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
        self._transport_registry = TransportAdapterRegistry()
        self._handler_registry = HandlerRegistry()
        self._initialization_lock = asyncio.Lock()

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
        transport = await self._transport_registry.initialize_adapter(transport_name, self, {})  # type: ignore[arg-type]

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
        self, event_type: str, event_data: Dict[str, Any], client_id: Optional[str] = None
    ) -> None:
        """Broadcast an event to all registered transports."""
        self._logger.debug(f"Broadcasting event: Type='{event_type}', ClientID='{client_id}', Data='{event_data}'")
        # Task 9 spec implies EventCoordinator has:
        # broadcast_event(event_type: str, event_data: Dict[str, Any], client_id: Optional[str])
        # Task 4 spec for EventCoordinator is compatible if priority defaults.
        # Provided EventCoordinator has a deprecated broadcast_event with different signature.
        # Sticking to Task 9 spec for the call:
        await self._event_coordinator.broadcast_event(event_type, event_data, client_id=client_id)  # type: ignore

    async def shutdown(self) -> None:
        """Shut down the application coordinator and all components."""
        async with self._initialization_lock:
            self._logger.info("Shutting down ApplicationCoordinator...")
            if hasattr(self._transport_registry, "shutdown_all"):
                await self._transport_registry.shutdown_all()
            else:
                self._logger.warning("TransportAdapterRegistry does not have shutdown_all method.")

            # Reset initialization flag to allow reinitialization if needed (e.g. in tests)
            ApplicationCoordinator._initialized = False
            # ApplicationCoordinator._instance = None # To allow full re-creation in tests
            self._logger.info("ApplicationCoordinator shutdown complete.")
