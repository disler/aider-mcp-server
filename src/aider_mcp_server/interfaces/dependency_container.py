"""
Dependency injection container interface for managing component lifecycles and dependencies.

This module defines the abstract dependency container interface that allows
for dependency registration, resolution, and lifecycle management.
"""

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Awaitable, Callable, Optional, Type, TypeVar, Union

T = TypeVar("T")
TFactory = Callable[..., Union[T, Awaitable[T]]]
TAsyncFactory = Callable[..., Awaitable[T]]


class Scope(Enum):
    """Scope of dependency instances."""

    SINGLETON = auto()  # One instance for the entire application
    TRANSIENT = auto()  # New instance created every time
    SCOPED = auto()  # One instance per scope (e.g., request)


class IDependencyContainer(ABC):
    """Interface for dependency container."""

    @abstractmethod
    async def __aenter__(self) -> "IDependencyContainer":
        """Enter async context."""
        pass

    @abstractmethod
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context."""
        pass

    @abstractmethod
    async def register_singleton(
        self,
        service_type: Type[T],
        implementation_type: Optional[Type[T]] = None,
        factory: Optional[TFactory[T]] = None,
        instance: Optional[T] = None,
    ) -> None:
        """
        Register a singleton service.

        Args:
            service_type: The type to register, typically an interface.
            implementation_type: Concrete implementation type to create.
            factory: Factory function to create the instance.
            instance: Existing instance to use.

        Notes:
            At most one of implementation_type, factory, or instance must be provided.
        """
        pass

    @abstractmethod
    async def register_transient(
        self,
        service_type: Type[T],
        implementation_type: Optional[Type[T]] = None,
        factory: Optional[TFactory[T]] = None,
    ) -> None:
        """
        Register a transient service (new instance each time).

        Args:
            service_type: The type to register, typically an interface.
            implementation_type: Concrete implementation type to create.
            factory: Factory function to create the instance.

        Notes:
            At most one of implementation_type or factory must be provided.
        """
        pass

    @abstractmethod
    async def register_scoped(
        self,
        service_type: Type[T],
        implementation_type: Optional[Type[T]] = None,
        factory: Optional[TFactory[T]] = None,
    ) -> None:
        """
        Register a scoped service (one instance per scope).

        Args:
            service_type: The type to register, typically an interface.
            implementation_type: Concrete implementation type to create.
            factory: Factory function to create the instance.

        Notes:
            At most one of implementation_type or factory must be provided.
        """
        pass

    @abstractmethod
    async def resolve(self, service_type: Type[T]) -> T:
        """
        Resolve a service from the container.

        Args:
            service_type: The type to resolve.

        Returns:
            An instance of the requested service.

        Raises:
            DependencyResolutionError: If the service cannot be resolved.
        """
        pass

    @abstractmethod
    async def create_scope(self) -> "IDependencyContainer":
        """
        Create a new scoped container inheriting registrations from this container.

        Returns:
            A new container with a separate scope.
        """
        pass

    @abstractmethod
    async def has_registration(self, service_type: Type[T]) -> bool:
        """
        Check if a service type is registered.

        Args:
            service_type: The type to check.

        Returns:
            True if the service is registered, False otherwise.
        """
        pass


class DependencyRegistrationError(Exception):
    """Error raised when there's an issue with dependency registration."""

    pass


class DependencyResolutionError(Exception):
    """Error raised when a dependency cannot be resolved."""

    pass


class CircularDependencyError(DependencyResolutionError):
    """Error raised when a circular dependency is detected."""

    pass
