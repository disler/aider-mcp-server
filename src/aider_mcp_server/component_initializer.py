import asyncio

from aider_mcp_server.default_authentication_provider import DefaultAuthenticationProvider
from aider_mcp_server.error_formatter import ErrorResponseFormatter
from aider_mcp_server.event_coordinator import EventCoordinator
from aider_mcp_server.event_mediator import EventMediator
from aider_mcp_server.handler_registry import HandlerRegistry
from aider_mcp_server.interfaces.authentication_provider import IAuthenticationProvider
from aider_mcp_server.interfaces.security_service import ISecurityService
from aider_mcp_server.interfaces.transport_registry import TransportAdapterRegistry
from aider_mcp_server.mcp_types import LoggerFactory
from aider_mcp_server.request_processor import RequestProcessor
from aider_mcp_server.response_formatter import ResponseFormatter
from aider_mcp_server.security_service import SecurityService
from aider_mcp_server.session_manager import SessionManager

# MODIFIED: Import EventSystem from event_system.py
from aider_mcp_server.event_system import EventSystem


class Components:
    """
    A container for all major initialized components of the application.
    This structure helps in passing around the initialized components easily.
    """

    def __init__(
        self,
        logger_factory: LoggerFactory,
        transport_registry: TransportAdapterRegistry,
        session_manager: SessionManager,
        handler_registry: HandlerRegistry,
        response_formatter: ResponseFormatter,
        event_coordinator: EventCoordinator,
        request_processor: RequestProcessor,
        security_service: ISecurityService,
        auth_provider: IAuthenticationProvider,
    ):
        self.logger_factory = logger_factory
        self.transport_registry = transport_registry
        self.session_manager = session_manager
        self.handler_registry = handler_registry
        self.response_formatter = response_formatter
        self.event_coordinator = event_coordinator
        self.request_processor = request_processor
        self.security_service = security_service
        self.auth_provider = auth_provider


class ComponentInitializer:
    """
    Responsible for initializing and wiring all major components of the application.
    """

    def __init__(self, logger_factory: LoggerFactory):
        self.logger = logger_factory(__name__)  # Logger for ComponentInitializer itself
        self.logger_factory = logger_factory  # To be passed to components that need it

    async def initialize_components(self) -> Components:
        """
        Initializes all core components and returns them in a Components object.

        Raises:
            RuntimeError: If any critical component fails to initialize.
        """
        self.logger.verbose("Component initialization started.")

        # Initialize components without complex dependencies first
        session_manager = SessionManager()
        self.logger.verbose("SessionManager initialized.")

        handler_registry = HandlerRegistry()  # Uses its own logger or fallback
        self.logger.verbose("HandlerRegistry initialized.")

        error_formatter = ErrorResponseFormatter(self.logger_factory)
        self.logger.verbose("ErrorResponseFormatter initialized.")

        response_formatter = ResponseFormatter(self.logger_factory, error_formatter)
        self.logger.verbose("ResponseFormatter initialized.")

        # Initialize AuthenticationProvider
        try:
            self.logger.verbose("Initializing AuthenticationProvider...")
            auth_provider = DefaultAuthenticationProvider(self.logger_factory)
            self.logger.verbose("AuthenticationProvider initialized.")
        except Exception as e:
            self.logger.error(f"Failed to initialize AuthenticationProvider: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize AuthenticationProvider: {e}") from e

        # Initialize SecurityService with AuthenticationProvider
        try:
            self.logger.verbose("Initializing SecurityService...")
            security_service = SecurityService(self.logger_factory, auth_provider)
            self.logger.verbose("SecurityService initialized.")
        except Exception as e:
            self.logger.error(f"Failed to initialize SecurityService: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize SecurityService: {e}") from e

        # Initialize TransportAdapterRegistry
        self.logger.verbose("Initializing TransportAdapterRegistry...")
        try:
            transport_registry = await asyncio.wait_for(
                TransportAdapterRegistry.get_instance(),
                timeout=10.0,
            )
            self.logger.verbose("TransportAdapterRegistry.get_instance() successful.")
        except asyncio.TimeoutError as e:
            self.logger.error("Timeout while initializing TransportAdapterRegistry.")
            raise RuntimeError("Timeout while initializing TransportAdapterRegistry") from e
        except Exception as e:
            self.logger.error(f"Failed to initialize TransportAdapterRegistry: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize TransportAdapterRegistry: {e}") from e

        # TransportAdapterRegistry.get_instance() is guaranteed to return a valid instance or raise an exception
        self.logger.verbose("TransportAdapterRegistry initialized.")

        # Initialize EventSystem
        try:
            self.logger.verbose("Initializing EventSystem...")
            # MODIFIED: Instantiate EventSystem
            event_system = EventSystem() # EventSystem from event_system.py takes no arguments
            self.logger.verbose("EventSystem initialized.")
        except Exception as e:
            self.logger.error(f"Failed to initialize EventSystem: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize EventSystem: {e}") from e

        # Initialize EventMediator
        try:
            self.logger.verbose("Initializing EventMediator...")
            # EventMediator now receives the EventSystem instance
            event_mediator = EventMediator(self.logger_factory, event_system)
            self.logger.verbose("EventMediator initialized.")
        except Exception as e:
            self.logger.error(f"Failed to initialize EventMediator: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize EventMediator: {e}") from e

        # Initialize EventCoordinator
        try:
            self.logger.verbose("Initializing EventCoordinator...")
            # EventCoordinator now takes logger_factory and event_mediator
            event_coordinator = EventCoordinator(self.logger_factory, event_mediator)
            self.logger.verbose("EventCoordinator initialized.")
        except Exception as e:
            self.logger.error(f"Failed to initialize EventCoordinator: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize EventCoordinator: {e}") from e

        # Initialize RequestProcessor with SecurityService
        try:
            self.logger.verbose("Initializing RequestProcessor...")
            request_processor = RequestProcessor(
                event_coordinator,
                session_manager,
                self.logger_factory,
                handler_registry,
                response_formatter,
                security_service,
            )
            self.logger.verbose("RequestProcessor initialized.")
        except Exception as e:
            self.logger.error(f"Failed to initialize RequestProcessor: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize RequestProcessor: {e}") from e

        self.logger.info("All components initialized successfully.")
        return Components(
            logger_factory=self.logger_factory,
            transport_registry=transport_registry,
            session_manager=session_manager,
            handler_registry=handler_registry,
            response_formatter=response_formatter,
            error_formatter=error_formatter,
            event_coordinator=event_coordinator,
            request_processor=request_processor,
            security_service=security_service,
            auth_provider=auth_provider,
        )
