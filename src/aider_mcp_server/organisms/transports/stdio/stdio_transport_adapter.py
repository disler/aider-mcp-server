"""
Stdio Transport Adapter for Aider MCP Server.

This module implements an adapter for the stdio transport that interfaces
with the ApplicationCoordinator, handling communication over standard input/output.
"""

import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Set, TextIO, Union

from aider_mcp_server.atoms.security.context import ANONYMOUS_SECURITY_CONTEXT, SecurityContext

# Use absolute imports from the package root
from aider_mcp_server.atoms.types.event_types import EventTypes
from aider_mcp_server.atoms.types.mcp_types import (
    AsyncTask,
    EventData,
    RequestParameters,
)
from aider_mcp_server.molecules.transport.base_adapter import AbstractTransportAdapter
from aider_mcp_server.molecules.transport.discovery import CoordinatorDiscovery, CoordinatorInfo

if TYPE_CHECKING:
    from aider_mcp_server.organisms.coordinators.transport_coordinator import ApplicationCoordinator


# Logger is inherited from AbstractTransportAdapter


class StdioTransportAdapter(AbstractTransportAdapter):
    """
    Adapter that bridges stdio transport with the ApplicationCoordinator.

    This class handles:
    1. Reading JSON messages from stdin
    2. Processing tool call requests
    3. Writing JSON responses to stdout
    4. Finding and connecting to existing coordinators via discovery
    """

    @property
    def transport_id(self) -> str:
        """
        Property for accessing the transport ID.
        Used for compatibility with code that expects a transport_id attribute.
        """
        return self._transport_id

    @classmethod
    def get_default_capabilities(cls) -> Set[EventTypes]:
        """
        Get the default capabilities for this transport adapter class without instantiation.

        This allows the TransportAdapterRegistry to determine capabilities
        without instantiating the adapter.

        Returns:
            A set of event types that this adapter supports by default.
        """
        # Stdio typically doesn't need heartbeats, but can receive other events
        return {
            EventTypes.STATUS,
            EventTypes.PROGRESS,
            EventTypes.TOOL_RESULT,
            # Exclude HEARTBEAT unless specifically needed/handled
        }

    _read_task: Optional[AsyncTask[None]]
    _input: TextIO
    _output: TextIO
    _stop_reading: bool
    _discovery: Optional[CoordinatorDiscovery]

    def __init__(
        self,
        coordinator: Optional["ApplicationCoordinator"] = None,
        input_stream: Optional[TextIO] = None,
        output_stream: Optional[TextIO] = None,
        heartbeat_interval: Optional[float] = None,
        discovery_file: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Initialize the stdio transport adapter.

        Args:
            coordinator: Optional ApplicationCoordinator instance
            input_stream: Input stream to read from (defaults to sys.stdin)
            output_stream: Output stream to write to (defaults to sys.stdout)
            heartbeat_interval: Time between heartbeat messages (defaults to None for stdio)
            discovery_file: Optional path to the coordinator discovery file
        """
        transport_id = f"stdio_{uuid.uuid4()}"
        super().__init__(
            transport_id=transport_id,
            transport_type="stdio",
            coordinator=coordinator,
            heartbeat_interval=heartbeat_interval,
        )
        self._input = input_stream if input_stream is not None else sys.stdin
        self._output = output_stream if output_stream is not None else sys.stdout
        self._read_task = None
        self._stop_reading = False
        self._discovery = None
        self._discovery_file = Path(discovery_file) if discovery_file else None
        self.logger.info(f"StdioTransportAdapter created with ID: {self.transport_id}")

    @classmethod
    async def find_and_connect(
        cls,
        discovery_file: Optional[Union[str, Path]] = None,
        input_stream: Optional[TextIO] = None,
        output_stream: Optional[TextIO] = None,
        heartbeat_interval: Optional[float] = None,
    ) -> Optional["StdioTransportAdapter"]:
        """
        Factory method to find an existing coordinator and create a connected adapter.

        Args:
            discovery_file: Path to the coordinator discovery file
            input_stream: Input stream to read from
            output_stream: Output stream to write to
            heartbeat_interval: Time between heartbeat messages

        Returns:
            A connected StdioTransportAdapter or None if no coordinator found
        """
        from aider_mcp_server.organisms.coordinators.transport_coordinator import ApplicationCoordinator

        # Convert string path to Path if provided
        discovery_path = Path(discovery_file) if discovery_file else None

        try:
            # Find existing coordinator using discovery
            coordinator_info = await ApplicationCoordinator.find_existing_coordinator(discovery_file=discovery_path)

            if not coordinator_info:
                return None

            # Create a new instance with discovery file configuration
            adapter = cls(
                coordinator=None,  # We'll set this after initialization
                input_stream=input_stream,
                output_stream=output_stream,
                heartbeat_interval=heartbeat_interval,
                discovery_file=discovery_path,
            )

            # Connect to the found coordinator
            success = await adapter.connect_to_coordinator(coordinator_info)
            return adapter if success else None

        except Exception as e:
            print(f"Error finding and connecting to coordinator: {e}", file=sys.stderr)
            return None

    async def _auto_discover_coordinator(self) -> None:
        """
        Automatically discover and connect to available coordinators.
        Creates a new coordinator if none found.
        """
        try:
            from aider_mcp_server.organisms.coordinators.transport_coordinator import ApplicationCoordinator
            from aider_mcp_server.atoms.logging.logger import get_logger

            self.logger.info(f"Auto-discovering coordinator for STDIO transport {self.transport_id}...")

            # Try to find existing coordinator using discovery
            coordinator_info = await ApplicationCoordinator.find_existing_coordinator(discovery_file=self._discovery_file)
            
            if coordinator_info:
                self.logger.info(
                    f"Found existing coordinator {coordinator_info.coordinator_id} "
                    f"at {coordinator_info.host}:{coordinator_info.port}"
                )
                success = await self.connect_to_coordinator(coordinator_info)
                if success:
                    self.logger.info(f"Successfully connected to existing coordinator")
                    return
                else:
                    self.logger.warning(f"Failed to connect to existing coordinator, will create new one")

            # No existing coordinator found or connection failed, create/get singleton
            self.logger.info(f"Creating new coordinator instance for STDIO transport {self.transport_id}")
            self._coordinator = await ApplicationCoordinator.getInstance(get_logger)
            self.logger.info(f"Successfully created/connected to coordinator instance")

        except Exception as e:
            self.logger.error(f"Error during coordinator auto-discovery: {e}")
            # Continue without coordinator - degraded functionality but non-fatal
            self.logger.warning(f"STDIO transport {self.transport_id} will continue without coordinator (degraded functionality)")

    async def _subscribe_to_aider_events(self) -> None:
        """
        Subscribe to AIDER events for cross-transport communication.
        This enables the STDIO transport to relay events to other transports.
        """
        if not self._coordinator:
            self.logger.warning(f"No coordinator available for event subscription")
            return

        try:
            # Define AIDER event types to subscribe to
            aider_events = [
                EventTypes.AIDER_SESSION_STARTED,
                EventTypes.AIDER_SESSION_PROGRESS,
                EventTypes.AIDER_SESSION_COMPLETED,
                EventTypes.AIDER_RATE_LIMIT_DETECTED,
                EventTypes.AIDER_THROTTLING_DETECTED,
                EventTypes.AIDER_ERROR_OCCURRED
            ]

            self.logger.info(f"Subscribing STDIO transport {self.transport_id} to AIDER events: {aider_events}")

            # Subscribe to each AIDER event type
            for event_type in aider_events:
                try:
                    # Check if subscribe_to_event_type method is awaitable
                    subscribe_method = getattr(self._coordinator, "subscribe_to_event_type", None)
                    if subscribe_method:
                        result = subscribe_method(self.transport_id, event_type)
                        if asyncio.iscoroutine(result):
                            await result
                        self.logger.debug(f"Subscribed to event type: {event_type}")
                    else:
                        self.logger.warning(f"Coordinator does not support event subscription")
                        break
                except Exception as e:
                    self.logger.warning(f"Failed to subscribe to event {event_type}: {e}")

            self.logger.info(f"STDIO transport {self.transport_id} event subscription completed")

        except Exception as e:
            self.logger.error(f"Error subscribing to AIDER events: {e}")

    async def connect_to_coordinator(self, coordinator_info: CoordinatorInfo) -> bool:
        """
        Connect to a specific coordinator using the provided information.

        Args:
            coordinator_info: Information about the coordinator to connect to

        Returns:
            True if connection successful, False otherwise
        """
        try:
            from aider_mcp_server.organisms.coordinators.transport_coordinator import ApplicationCoordinator

            self.logger.info(
                f"Connecting to coordinator {coordinator_info.coordinator_id} "
                f"at {coordinator_info.host}:{coordinator_info.port}"
            )

            # Here you would establish a connection to the coordinator
            # For now, we'll use the singleton ApplicationCoordinator instance
            # In a networked implementation, this would involve creating a client connection

            # Get logger factory from the transport coordinator module
            from aider_mcp_server.atoms.logging.logger import get_logger

            self._coordinator = await ApplicationCoordinator.getInstance(get_logger)
            await self.initialize()

            return True
        except Exception as e:
            self.logger.error(f"Error connecting to coordinator: {e}")
            return False

    async def initialize(self) -> None:
        """
        Initialize the stdio transport and register with coordinator.
        Automatically discovers and connects to coordinators if none provided.
        Does NOT start the read task automatically. Call start_listening() separately.
        """
        self.logger.info(f"Initializing stdio transport {self.transport_id}...")
        
        # If no coordinator provided, attempt auto-discovery
        if not self._coordinator:
            await self._auto_discover_coordinator()
        
        await super().initialize()
        
        # Subscribe to AIDER events for cross-transport communication
        if self._coordinator:
            await self._subscribe_to_aider_events()
        
        self.logger.info(f"Stdio transport {self.transport_id} initialized.")

    async def shutdown(self) -> None:
        """
        Shut down the stdio transport and unregister from coordinator.
        Stops the task reading from stdin and cleans up discovery resources.
        """
        self.logger.info(f"Shutting down stdio transport {self.transport_id}...")
        self._stop_reading = True
        if self._read_task and not self._read_task.done():
            self.logger.debug(f"Cancelling stdin read task for {self.transport_id}.")
            self._read_task.cancel()
            try:
                await asyncio.wait_for(self._read_task, timeout=1.0)  # Shorter timeout for stdio
            except asyncio.CancelledError:
                self.logger.debug(f"Stdin read task for {self.transport_id} cancelled.")
            except asyncio.TimeoutError:
                self.logger.warning(f"Timeout waiting for stdin read task {self.transport_id} to cancel.")
            except Exception as e:
                self.logger.error(f"Error cancelling stdin read task for {self.transport_id}: {e}")
            finally:
                self._read_task = None

        # Clean up discovery if it was created
        if self._discovery:
            try:
                await self._discovery.shutdown()
                self.logger.info(f"Shutdown discovery service for {self.transport_id}.")
            except Exception as e:
                self.logger.error(f"Error shutting down discovery service: {e}")
            self._discovery = None

        await super().shutdown()
        self.logger.info(f"Stdio transport {self.transport_id} shut down.")

    def get_capabilities(self) -> Set[EventTypes]:
        """Declare the event types this Stdio transport adapter supports."""
        # Stdio typically doesn't need heartbeats, but can receive other events
        return {
            EventTypes.STATUS,
            EventTypes.PROGRESS,
            EventTypes.TOOL_RESULT,
            # Exclude HEARTBEAT unless specifically needed/handled
        }

    async def send_event(self, event: EventTypes, data: EventData) -> None:
        """
        Send an event to stdout as JSON.

        Args:
            event: The event type (e.g., EventTypes.PROGRESS)
            data: The event payload
        """
        try:
            message = {
                "event": event.value,
                "data": data,
            }
            json_str = json.dumps(message)
            # Use async write via executor to avoid blocking
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lambda: print(json_str, file=self._output, flush=True))
            # self.logger.debug(f"Sent {event.value} event to stdout")
        except Exception as e:
            self.logger.error(f"Error sending {event.value} event to stdout: {e}")

    async def start_listening(self) -> None:
        """
        Starts the asynchronous task to read JSON messages from stdin.
        This method should be called after initialization.
        """
        if self._read_task is not None and not self._read_task.done():
            self.logger.warning(f"Stdin read task for {self.transport_id} is already running.")
            return

        self.logger.info(f"Starting stdin read task for {self.transport_id}...")
        self._stop_reading = False
        self._read_task = asyncio.create_task(self._read_stdin_loop())
        self._read_task.add_done_callback(self._read_task_done_callback)
        self.logger.info(f"Stdin read task for {self.transport_id} started.")

    async def _read_stdin_loop(self) -> None:
        """Continuously read JSON messages from stdin until stopped."""
        self.logger.debug(f"Starting stdin read loop for {self.transport_id}")
        loop = asyncio.get_running_loop()
        try:
            while not self._stop_reading:
                try:
                    line: str = await asyncio.wait_for(
                        loop.run_in_executor(None, self._input.readline),
                        timeout=0.5,
                    )
                except asyncio.TimeoutError:
                    continue

                if not line:
                    self.logger.info(f"Stdin for {self.transport_id} appears closed or empty. Stopping read loop.")
                    break

                line = line.strip()
                if not line:  # Skip empty lines after stripping
                    continue

                try:
                    message: RequestParameters = json.loads(line)
                    asyncio.create_task(self._handle_stdin_message(message))
                except json.JSONDecodeError:
                    self.logger.error(f"Invalid JSON received on stdin: {line[:100]}...")
                except Exception as e:
                    self.logger.error(f"Error scheduling stdin message handling: {e}")

        except asyncio.CancelledError:
            self.logger.debug(f"Stdin read loop for {self.transport_id} received cancellation.")
            # Do not re-raise here, let finally block execute
        except Exception as e:
            self.logger.error(f"Error in stdin read loop for {self.transport_id}: {e}")
        finally:
            self.logger.debug(f"Stdin read loop for {self.transport_id} stopped.")
            # Clear task reference only if it's still pointing to this task
            # Check if current_task exists before comparing
            current_task = asyncio.current_task()
            if current_task and self._read_task is current_task:
                self._read_task = None

    def _read_task_done_callback(self, task: AsyncTask[None]) -> None:
        """Callback executed when the stdin read task finishes."""
        try:
            task.result()
            self.logger.debug(f"Stdin read task for {self.transport_id} finished normally.")
        except asyncio.CancelledError:
            self.logger.debug(f"Stdin read task for {self.transport_id} was cancelled.")
        except Exception as e:
            self.logger.error(f"Stdin read task for {self.transport_id} finished with exception: {e}")
        finally:
            if self._read_task is task:
                self._read_task = None

    async def _handle_stdin_message(self, message: Any) -> None:
        """Process a message received from stdin."""
        if not isinstance(message, dict):
            self.logger.error(f"Invalid message format received on stdin: {message}")
            return

        raw_request_id = message.get("request_id")
        if isinstance(raw_request_id, str) and raw_request_id:
            request_id: str = raw_request_id
        else:
            request_id = str(uuid.uuid4())
            log_level = self.logger.warning if raw_request_id is not None else self.logger.debug
            log_level(
                f"Missing or invalid 'request_id' in stdin message. Using generated ID: {request_id}. Original value: {raw_request_id!r}"
            )
            message["request_id"] = request_id  # Add generated ID back

        self.logger.info(f"Processing stdin message with ID: {request_id}")

        if not self._coordinator:
            error_msg = f"No coordinator available for transport {self.transport_id}."
            self.logger.error(error_msg)
            return

        raw_operation_name = message.get("name")
        if not isinstance(raw_operation_name, str) or not raw_operation_name:
            error_msg = "Missing or invalid 'name' field in stdin message. Expected non-empty string."
            self.logger.error(error_msg)
            await self._coordinator.fail_request(
                request_id=request_id,
                operation_name="unknown",
                error="Invalid request",
                error_details=error_msg,
                originating_transport_id=self.transport_id,
            )
            return
        operation_name: str = raw_operation_name

        try:
            # Handle parameters - the `.get()` might return a non-dict which we need to validate
            parameters_raw = message.get("parameters", {})
            if not isinstance(parameters_raw, dict):
                error_msg = "Invalid 'parameters' field in stdin message. Expected dictionary."
                self.logger.error(error_msg)
                await self._coordinator.fail_request(
                    request_id=request_id,
                    operation_name="unknown",
                    error="Invalid request",
                    error_details=error_msg,
                    originating_transport_id=self.transport_id,
                )
                return

            # Start the request with the coordinator.
            # The coordinator handles security validation via validate_request_security.
            await self._coordinator.start_request(
                request_id=request_id,
                transport_id=self.transport_id,
                operation_name=operation_name,
                request_data=message,
            )

        except Exception as e:
            error_msg = f"Error processing stdin message {request_id}: {str(e)}"
            self.logger.exception(error_msg)
            if self._coordinator:
                await self._coordinator.fail_request(
                    request_id=request_id,
                    operation_name=operation_name,
                    error="Internal error",
                    error_details=error_msg,
                    originating_transport_id=self.transport_id,
                )
            else:
                self.logger.error(f"Coordinator not available to report error for request {request_id}.")

    def validate_request_security(self, request_data: RequestParameters) -> SecurityContext:
        """
        Validates security for stdio requests.
        Returns the predefined anonymous security context.

        Args:
            request_data: The incoming request data (unused).

        Returns:
            The ANONYMOUS_SECURITY_CONTEXT instance.
        """
        self.logger.debug("Using anonymous security context for stdio request.")
        # Stdio runs locally, assume anonymous/full permissions via this context
        return ANONYMOUS_SECURITY_CONTEXT
