"""
Example refactoring of ApplicationCoordinator to use the dependency container.

This file demonstrates how to refactor the existing code to use the DependencyContainer.
It is not meant to be used directly, but serves as a guide for the actual refactoring.
"""

import asyncio

from aider_mcp_server.default_authentication_provider import DefaultAuthenticationProvider
from aider_mcp_server.dependency_container import DependencyContainer
from aider_mcp_server.event_coordinator import EventCoordinator
from aider_mcp_server.event_mediator import EventMediator
from aider_mcp_server.event_system import EventSystem
from aider_mcp_server.handler_registry import HandlerRegistry
from aider_mcp_server.interfaces.authentication_provider import IAuthenticationProvider
from aider_mcp_server.interfaces.security_service import ISecurityService
from aider_mcp_server.interfaces.transport_registry import TransportAdapterRegistry
from aider_mcp_server.mcp_types import LoggerFactory, LoggerProtocol
from aider_mcp_server.request_processor import RequestProcessor
from aider_mcp_server.response_formatter import ResponseFormatter
from aider_mcp_server.security_service import SecurityService
from aider_mcp_server.session_manager import SessionManager


async def initialize_container(logger_factory: LoggerFactory) -> DependencyContainer:
    """
    Initialize the dependency container with all required services.

    This replaces the ComponentInitializer.initialize_components method.

    Args:
        logger_factory: Factory function to create loggers.

    Returns:
        A fully initialized dependency container.
    """
    # Create container
    container = DependencyContainer(logger_factory=logger_factory)
    await container.__aenter__()  # Enter the async context

    # Register the logger factory itself
    await container.register_singleton(LoggerFactory, instance=logger_factory)

    # Register core services
    await container.register_singleton(SessionManager, implementation_type=SessionManager)
    await container.register_singleton(HandlerRegistry, implementation_type=HandlerRegistry)
    await container.register_singleton(ResponseFormatter, implementation_type=ResponseFormatter)

    # Register authentication and security
    await container.register_singleton(IAuthenticationProvider, implementation_type=DefaultAuthenticationProvider)
    await container.register_singleton(ISecurityService, implementation_type=SecurityService)

    # Register transport registry
    async def create_transport_registry() -> TransportAdapterRegistry:
        return await TransportAdapterRegistry.get_instance()

    await container.register_singleton(TransportAdapterRegistry, factory=create_transport_registry)

    # Register event system and mediator
    await container.register_singleton(EventSystem, implementation_type=EventSystem)
    await container.register_singleton(EventMediator, implementation_type=EventMediator)
    await container.register_singleton(EventCoordinator, implementation_type=EventCoordinator)

    # Register request processor
    await container.register_singleton(RequestProcessor, implementation_type=RequestProcessor)

    return container


async def refactored_server_startup():
    """
    Example of refactored server startup code using the dependency container.

    This would replace the current server startup code in multi_transport_server.py.
    """

    # Create logger factory
    def logger_factory(name: str) -> "LoggerProtocol":
        # Actual implementation goes here
        pass

    # Initialize container with all services
    container = await initialize_container(logger_factory)

    try:
        # Resolve and start required services
        # transport_registry = await container.resolve(TransportAdapterRegistry) # Unused
        # event_coordinator = await container.resolve(EventCoordinator) # Unused
        # request_processor = await container.resolve(RequestProcessor) # Unused

        # Start the server with these services
        # ...

        # For handling requests, create a new scope
        async def handle_request(request_data: dict):
            async with await container.create_scope() as request_scope:
                # Resolve request-specific services
                request_processor = await request_scope.resolve(RequestProcessor)

                # Process the request
                await request_processor.process_request(
                    request_data["request_id"],
                    request_data["transport_id"],
                    request_data["operation_name"],
                    request_data,
                )

        # Wait for server to complete
        # ...

    finally:
        # Clean up resources
        await container.__aexit__(None, None, None)


# Example of refactored ApplicationCoordinator
class RefactoredApplicationCoordinator:
    """
    Refactored ApplicationCoordinator that uses dependency injection.

    This class shows how ApplicationCoordinator could be refactored to use the
    dependency container instead of directly managing components.
    """

    def __init__(
        self,
        logger_factory: LoggerFactory,
        event_coordinator: EventCoordinator,
        request_processor: RequestProcessor,
        transport_registry: TransportAdapterRegistry,
        session_manager: SessionManager,
        handler_registry: HandlerRegistry,
    ):
        """
        Initialize with injected dependencies.

        Instead of receiving a Components object, this version receives
        individual dependencies through constructor injection.
        """
        self.logger = logger_factory("ApplicationCoordinator")
        self.logger.verbose("ApplicationCoordinator initializing with injected dependencies...")

        # Store references to injected dependencies
        self._event_coordinator = event_coordinator
        self._request_processor = request_processor
        self._transport_registry = transport_registry
        self._session_manager = session_manager
        self._handler_registry = handler_registry

        self.logger.info("ApplicationCoordinator instance configured with injected dependencies.")

    @classmethod
    async def create(cls, container: DependencyContainer) -> "RefactoredApplicationCoordinator":
        """
        Factory method to create an ApplicationCoordinator instance.

        Args:
            container: The dependency container to resolve dependencies from.

        Returns:
            A new RefactoredApplicationCoordinator instance.
        """
        # Resolve dependencies from container
        logger_factory = await container.resolve(LoggerFactory)
        event_coordinator = await container.resolve(EventCoordinator)
        request_processor = await container.resolve(RequestProcessor)
        transport_registry = await container.resolve(TransportAdapterRegistry)
        session_manager = await container.resolve(SessionManager)
        handler_registry = await container.resolve(HandlerRegistry)

        # Create and return instance
        return cls(
            logger_factory=logger_factory,
            event_coordinator=event_coordinator,
            request_processor=request_processor,
            transport_registry=transport_registry,
            session_manager=session_manager,
            handler_registry=handler_registry,
        )

    # Rest of the class implementation remains similar
    # Just use the injected dependencies instead of accessing them through a components object


# Example of refactored component with async initialization
class RefactoredTransportAdapter:
    """Example of refactoring a transport adapter to support async initialization."""

    def __init__(
        self,
        transport_id: str,
        event_coordinator: EventCoordinator,
        logger_factory: LoggerFactory,
    ):
        """Initialize with injected dependencies."""
        self.transport_id = transport_id
        self._event_coordinator = event_coordinator
        self.logger = logger_factory(f"TransportAdapter.{transport_id}")
        self._initialized = False

    async def initialize(self) -> None:
        """Async initialization method."""
        # Perform async initialization tasks
        await asyncio.sleep(0.1)  # Simulated async work
        self._initialized = True
        self.logger.verbose(f"TransportAdapter {self.transport_id} initialized")

    @classmethod
    async def create(
        cls,
        transport_id: str,
        event_coordinator: EventCoordinator,
        logger_factory: LoggerFactory,
    ) -> "RefactoredTransportAdapter":
        """
        Factory method for creating and initializing the adapter.

        This pattern is useful when a class needs async initialization.
        """
        adapter = cls(
            transport_id=transport_id,
            event_coordinator=event_coordinator,
            logger_factory=logger_factory,
        )
        await adapter.initialize()
        return adapter
