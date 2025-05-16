"""
Transport adapter base implementation for Aider MCP Server.

This module provides a base transport adapter implementation that handles common
functionality like registration, heartbeat management, and coordinator integration.
Specific transport mechanisms should inherit from AbstractTransportAdapter.
"""

import abc
import asyncio
import logging
import typing
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Optional,
    Set,
)

# Use absolute imports from the package root
from aider_mcp_server.atoms.event_types import EventTypes
from aider_mcp_server.interfaces.transport_adapter import TransportAdapterBase
from aider_mcp_server.mcp_types import (
    EventData,
    LoggerFactory,
    LoggerProtocol,
    RequestParameters,
)
from aider_mcp_server.security import SecurityContext

# Import ApplicationCoordinator from transport_coordinator only during type checking
if TYPE_CHECKING:
    from aider_mcp_server.transport_coordinator import ApplicationCoordinator


# Initialize the logger factory
get_logger_func: LoggerFactory

try:
    # Attempt to import the custom logger using absolute path
    from aider_mcp_server.atoms.logging import get_logger as custom_get_logger

    # Cast the custom logger function to the protocol type
    get_logger_func = typing.cast(LoggerFactory, custom_get_logger)
except ImportError:
    # Fallback to standard logging if custom logger is not available
    def fallback_get_logger(name: str, *args: Any, **kwargs: Any) -> LoggerProtocol:
        logger = logging.getLogger(name)
        return typing.cast(LoggerProtocol, logger)

    get_logger_func = fallback_get_logger
    if not logging.root.handlers:
        logging.basicConfig(level=logging.INFO)
    temp_logger = logging.getLogger(__name__)
    temp_logger.warning(
        "Could not import custom logger from aider_mcp_server.atoms.logging. Using standard logging fallback."
    )


# Get the module-level logger using the determined function
logger: LoggerProtocol = get_logger_func(__name__)


# Define the abstract base class for transport adapters
class AbstractTransportAdapter(TransportAdapterBase):
    """
    Abstract base class for transport adapters.

    Transport adapters are responsible for bridging specific transport mechanisms
    (like SSE, Stdio, WebSocket) with the ApplicationCoordinator. They handle
    receiving requests from their transport, validating security, and sending
    events/results back to their connected clients.

    This class provides common functionality like registration with the coordinator,
    heartbeat management, and default capability reporting. Concrete transport
    implementations should inherit from this class and implement the abstract methods.
    """

    _transport_id: str
    _transport_type: str
    _coordinator: Optional["ApplicationCoordinator"]
    _heartbeat_interval: Optional[float]
    _heartbeat_task: Optional[asyncio.Task[None]]
    logger: LoggerProtocol  # Define logger attribute for type checking

    def __init__(
        self,
        transport_id: str,
        transport_type: str,
        coordinator: Optional["ApplicationCoordinator"] = None,
        heartbeat_interval: Optional[float] = None,
    ):
        """
        Initializes the abstract transport adapter.

        Args:
            transport_id: A unique identifier for this transport instance.
            transport_type: A string identifying the type of transport (e.g., "sse", "stdio").
            coordinator: The ApplicationCoordinator instance.
            heartbeat_interval: Interval in seconds for sending heartbeat events, or None to disable.
        """
        super().__init__(transport_id, transport_type)
        self._coordinator = coordinator
        self._heartbeat_interval = heartbeat_interval
        self._heartbeat_task = None

        # Get instance-specific logger
        self.logger = get_logger_func(f"{__name__}.{self.__class__.__name__}.{self.get_transport_id()}")
        self.logger.debug(
            f"AbstractTransportAdapter initialized with ID: {self.get_transport_id()}, Type: {self.get_transport_type()}"
        )

    async def initialize(self) -> None:
        """
        Initializes the transport adapter. Registers with the coordinator and starts heartbeat.
        Should be called after the adapter is created.
        """
        self.logger.debug(f"Initializing transport {self.get_transport_id()} ({self.get_transport_type()})...")
        if self._coordinator:
            # Pass self, which conforms to TransportInterface
            await self._coordinator.register_transport(self.get_transport_id(), self)
            self.logger.debug(f"Transport {self.get_transport_id()} registered with coordinator.")

            capabilities = self.get_capabilities()
            for event_type in capabilities:
                # Handle both async and sync subscribe_to_event_type methods
                if asyncio.iscoroutinefunction(self._coordinator.subscribe_to_event_type):
                    await self._coordinator.subscribe_to_event_type(self.get_transport_id(), event_type)
                else:
                    # Non-awaitable result, but we need to ensure it's used to avoid warnings
                    result = self._coordinator.subscribe_to_event_type(self.get_transport_id(), event_type)
                    # If it's a coroutine (but detected incorrectly), await it
                    if asyncio.iscoroutine(result):
                        await result
            self.logger.debug(f"Transport {self.get_transport_id()} subscribed to default events: {capabilities}")

            if self._heartbeat_interval is not None and self._heartbeat_interval > 0:
                self.logger.debug(
                    f"Starting heartbeat task for transport {self.get_transport_id()} with interval {self._heartbeat_interval}s."
                )
                self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
                self._heartbeat_task.add_done_callback(self._heartbeat_task_done_callback)
            else:
                self.logger.debug(f"Heartbeat disabled for transport {self.get_transport_id()}.")
        else:
            self.logger.warning(
                f"No coordinator provided for transport {self.get_transport_id()}. Registration and heartbeat skipped."
            )
        self.logger.debug(f"Transport {self.get_transport_id()} initialization complete.")

    async def shutdown(self) -> None:
        """
        Shuts down the transport adapter. Unregisters from the coordinator and stops heartbeat.
        Should be called before the adapter is discarded.
        """
        self.logger.info(f"Shutting down transport {self.get_transport_id()} ({self.get_transport_type()})...")

        if self._heartbeat_task and not self._heartbeat_task.done():
            self.logger.debug(f"Cancelling heartbeat task for transport {self.get_transport_id()}.")
            self._heartbeat_task.cancel()
            try:
                await asyncio.wait_for(self._heartbeat_task, timeout=5.0)
            except asyncio.CancelledError:
                self.logger.debug(f"Heartbeat task for {self.get_transport_id()} cancelled.")
            except asyncio.TimeoutError:
                self.logger.warning(f"Timeout waiting for heartbeat task {self.get_transport_id()} to cancel.")
            except Exception as e:
                self.logger.error(f"Error cancelling heartbeat task for {self.get_transport_id()}: {e}")
            finally:
                self._heartbeat_task = None

        if self._coordinator:
            await self._coordinator.unregister_transport(self.get_transport_id())
            self.logger.info(f"Transport {self.get_transport_id()} unregistered from coordinator.")
        else:
            self.logger.warning(
                f"No coordinator available for transport {self.get_transport_id()}. Unregistration skipped."
            )
        self.logger.info(f"Transport {self.get_transport_id()} shutdown complete.")

    async def _heartbeat_loop(self) -> None:
        """Internal loop to send heartbeat events."""
        if self._heartbeat_interval is None or self._heartbeat_interval <= 0:
            self.logger.error(
                f"Heartbeat loop started for transport {self.get_transport_id()} with invalid interval: {self._heartbeat_interval}. Exiting."
            )
            return

        self.logger.debug(
            f"Heartbeat loop started for transport {self.get_transport_id()} with interval {self._heartbeat_interval}s."
        )
        try:
            while True:
                await asyncio.sleep(self._heartbeat_interval)
                if self._coordinator:
                    # Use time.time() instead of asyncio.get_event_loop().time() to avoid
                    # "no running event loop" errors
                    import time

                    await self._coordinator.broadcast_event(
                        event_type=EventTypes.HEARTBEAT,
                        data={
                            "transport_id": self.get_transport_id(),
                            "timestamp": time.time(),
                        },
                        exclude_transport_id=None,
                    )
                else:
                    self.logger.warning(
                        f"Coordinator not available in heartbeat loop for transport {self.get_transport_id()}."
                    )
        except asyncio.CancelledError:
            self.logger.debug(f"Heartbeat loop for transport {self.get_transport_id()} cancelled.")
            raise
        except Exception as e:
            self.logger.error(f"Error in heartbeat loop for transport {self.get_transport_id()}: {e}")
        finally:
            self.logger.debug(f"Heartbeat loop for transport {self.get_transport_id()} stopped.")

    def _heartbeat_task_done_callback(self, task: asyncio.Task[Any]) -> None:
        """Callback executed when the heartbeat task finishes."""
        try:
            task.result()
            self.logger.debug(f"Heartbeat task for transport {self.get_transport_id()} finished normally.")
        except asyncio.CancelledError:
            self.logger.debug(f"Heartbeat task for transport {self.get_transport_id()} was cancelled.")
        except Exception as e:
            self.logger.error(f"Heartbeat task for transport {self.get_transport_id()} finished with exception: {e}")
        finally:
            if self._heartbeat_task is task:
                self._heartbeat_task = None

    @abc.abstractmethod
    async def send_event(self, event_type: EventTypes, data: EventData) -> None:
        """
        Asynchronously sends an event with associated data to the client
        connected via this transport.

        Args:
            event_type: The event type (e.g., EventTypes.PROGRESS).
            data: A dictionary containing the event payload.
        """
        pass

    def get_capabilities(self) -> Set[EventTypes]:
        """
        Returns a set of event types that this transport adapter is capable
        of sending or receiving.

        Default implementation returns a standard set of events.
        """
        return {
            EventTypes.STATUS,
            EventTypes.PROGRESS,
            EventTypes.TOOL_RESULT,
            EventTypes.HEARTBEAT,
        }

    @abc.abstractmethod
    def validate_request_security(self, request_data: RequestParameters) -> SecurityContext:
        """
        Validates security information provided in the incoming request data
        and returns the SecurityContext applicable to this specific request.
        This method is called by the transport itself before processing a request.

        Args:
            request_data: The data from the incoming request.

        Returns:
            A SecurityContext representing the security validation result.

        Raises:
            ValueError: If security validation fails (e.g., invalid token).
            PermissionError: If the request lacks necessary permissions.
        """
        pass

    async def start_listening(self) -> None:
        """
        Starts listening for incoming connections or messages.

        Default implementation is a no-op. Override in transport implementations
        that need to actively listen for connections (e.g., WebSocket server).
        """
        self.logger.debug(f"start_listening called for {self.get_transport_id()} (no-op)")
        pass

    def get_transport_id(self) -> str:
        """Get the unique identifier for this transport instance."""
        return self._transport_id

    def get_transport_type(self) -> str:
        """Get the type of transport (e.g., 'sse', 'stdio')."""
        return self._transport_type

    def should_receive_event(
        self,
        event_type: EventTypes,
        data: EventData,
        request_details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Determines if this transport should receive a specific event.

        Default implementation always returns True for events the transport
        is subscribed to. Override to implement custom filtering logic.

        Args:
            event_type: The type of event.
            data: The event data payload.
            request_details: Optional original request parameters for context.

        Returns:
            True if the transport should receive the event, False otherwise.
        """
        return True  # Default implementation always receives subscribed events
