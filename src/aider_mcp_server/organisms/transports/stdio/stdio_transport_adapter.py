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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, TextIO, Union

import aiohttp
import aiohttp.client_exceptions

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


# Define AIDER event types that should be relayed to streaming transports
AIDER_EVENT_TYPES_TO_RELAY = {
    EventTypes.AIDER_SESSION_STARTED,
    EventTypes.AIDER_SESSION_PROGRESS,
    EventTypes.AIDER_SESSION_COMPLETED,
    EventTypes.AIDER_RATE_LIMIT_DETECTED,
    EventTypes.AIDER_THROTTLING_DETECTED,
    EventTypes.AIDER_ERROR_OCCURRED,
    EventTypes.AIDER_CHANGES_SUMMARY,
    EventTypes.AIDER_FILE_PROGRESS,
    EventTypes.AIDER_OPERATION_STATUS,
}


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

    _read_task: Optional[AsyncTask[None]] = None
    _input: TextIO
    _output: TextIO
    _stop_reading: bool = False
    _discovery: Optional[CoordinatorDiscovery] = None
    _discovery_file: Optional[Path] = None
    _streaming_coordinators: List[CoordinatorInfo] = []
    _client_session: Optional[aiohttp.ClientSession] = None

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
        # Initialize discovery instance if discovery_file is provided
        if discovery_file:
            self._discovery_file = Path(discovery_file)
            self._discovery = CoordinatorDiscovery(discovery_file=self._discovery_file)
        else:
            self._discovery_file = None
            self._discovery = None

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
        # Convert string path to Path if provided
        discovery_path = Path(discovery_file) if discovery_file else None

        try:
            # Create a discovery instance for this factory method
            discovery = CoordinatorDiscovery(discovery_file=discovery_path)
            async with discovery:  # Use async context manager for shutdown
                # Find existing streaming coordinators using discovery
                streaming_coordinators = await discovery.find_streaming_coordinators()

                if not streaming_coordinators:
                    cls.logger.info("No active streaming coordinators found.")
                    return None

                # For find_and_connect, we assume we want to connect to *one* coordinator
                # to potentially send requests *to*. This method isn't ideal for the
                # relaying scenario, which is handled by the instance's _auto_discover_coordinator.
                # However, if we interpret this as finding *any* coordinator to connect to
                # for request processing, we can pick one. Let's pick the first streaming one found.
                coordinator_info = streaming_coordinators[0]
                cls.logger.info(
                    f"Found streaming coordinator {coordinator_info.coordinator_id} "
                    f"at {coordinator_info.host}:{coordinator_info.port}. Attempting to connect."
                )

                # Create a new instance with discovery file configuration
                # Note: This instance will *also* run _auto_discover_coordinator during initialize,
                # which will find streaming coordinators again for relaying.
                adapter = cls(
                    coordinator=None,  # Will be set during initialize
                    input_stream=input_stream,
                    output_stream=output_stream,
                    heartbeat_interval=heartbeat_interval,
                    discovery_file=discovery_path,
                )

                # Initialize the adapter. This will trigger its own discovery and setup.
                await adapter.initialize()

                # After initialization, the adapter should have its _coordinator set
                # and potentially discovered streaming coordinators for relaying.
                # This factory method's purpose is slightly ambiguous in the context
                # of the relaying feature vs. being a client for requests.
                # Assuming it's primarily for setting up the STDIO adapter to *use*
                # the MCP system, we return the initialized adapter.
                # The actual connection logic for *sending* requests to a remote
                # coordinator isn't implemented here, only the relaying *from*
                # the local coordinator *to* remote streaming ones.

                # If the adapter successfully initialized and has a coordinator, return it.
                if adapter._coordinator:
                    cls.logger.info(
                        "Successfully initialized adapter connected to local coordinator for request processing."
                    )
                    return adapter
                else:
                    cls.logger.warning("Adapter initialized but failed to connect to a local coordinator.")
                    return None

        except Exception as e:
            cls.logger.error(f"Error finding and connecting to coordinator: {e}")
            return None

    async def _auto_discover_coordinator(self) -> None:
        """
        Automatically discover and connect to available coordinators.
        Creates a new coordinator if none found.
        Also discovers streaming coordinators for event relay.
        """
        try:
            from aider_mcp_server.atoms.logging.logger import get_logger
            from aider_mcp_server.organisms.coordinators.transport_coordinator import ApplicationCoordinator

            self.logger.info(f"Auto-discovering coordinators for STDIO transport {self.transport_id}...")

            # Always get/create the local ApplicationCoordinator singleton
            # This is needed to handle incoming requests from stdin and receive local events
            if not self._coordinator:
                self.logger.info(f"Getting/Creating local coordinator instance for STDIO transport {self.transport_id}")
                self._coordinator = await ApplicationCoordinator.getInstance(get_logger)
                self.logger.info("Successfully got/connected to local coordinator instance")

            # Discover streaming coordinators for event relay if discovery is enabled
            if self._discovery:
                self.logger.info("Discovering streaming coordinators...")
                self._streaming_coordinators = await self._discovery.find_streaming_coordinators()
                self.logger.info(f"Found {len(self._streaming_coordinators)} active streaming coordinators.")
                for coord in self._streaming_coordinators:
                    self.logger.info(
                        f"  - {coord.coordinator_id} at {coord.host}:{coord.port} ({coord.transport_type})"
                    )
                    if not coord.streaming_capabilities:
                        self.logger.warning(
                            f"    Coordinator {coord.coordinator_id} reported no streaming capabilities."
                        )
                    else:
                        self.logger.debug(f"    Capabilities: {coord.streaming_capabilities}")
            else:
                self.logger.info("Coordinator discovery is not enabled (no discovery_file provided).")

        except Exception as e:
            self.logger.error(f"Error during coordinator auto-discovery: {e}")
            # Continue without coordinator or streaming relay - degraded functionality but non-fatal
            if not self._coordinator:
                self.logger.warning(
                    f"STDIO transport {self.transport_id} failed to connect to local coordinator (degraded functionality)"
                )
            self.logger.warning(
                f"STDIO transport {self.transport_id} failed to discover streaming coordinators (no event relay)"
            )

    async def _subscribe_to_aider_events(self) -> None:
        """
        Subscribe to AIDER events on the local coordinator.
        This enables the STDIO transport to receive these events and potentially relay them.
        """
        if not self._coordinator:
            self.logger.warning("No local coordinator available for event subscription")
            return

        try:
            # Define AIDER event types to subscribe to
            # These are defined in AIDER_EVENT_TYPES_TO_RELAY at the top
            aider_events = list(AIDER_EVENT_TYPES_TO_RELAY)

            self.logger.info(
                f"Subscribing STDIO transport {self.transport_id} to AIDER events on local coordinator: {aider_events}"
            )

            # Subscribe to each AIDER event type
            for event_type in aider_events:
                try:
                    # Check if subscribe_to_event_type method is awaitable
                    subscribe_method = getattr(self._coordinator, "subscribe_to_event_type", None)
                    if subscribe_method:
                        result = subscribe_method(self.transport_id, event_type)
                        if asyncio.iscoroutine(result):
                            await result
                        self.logger.debug(f"Subscribed to event type: {event_type.value}")
                    else:
                        self.logger.warning("Local Coordinator does not support event subscription")
                        break  # Exit loop if subscription method is missing
                except Exception as e:
                    self.logger.warning(f"Failed to subscribe to event {event_type.value}: {e}")

            self.logger.info(f"STDIO transport {self.transport_id} event subscription completed on local coordinator")

        except Exception as e:
            self.logger.error(f"Error subscribing to AIDER events: {e}")

    async def connect_to_coordinator(self, coordinator_info: CoordinatorInfo) -> bool:
        """
        Connect to a specific coordinator using the provided information.
        NOTE: This method's primary purpose seems to be for the STDIO adapter
        to act as a *client* receiving requests/events from a remote coordinator.
        The event relaying feature works by the STDIO adapter receiving events
        from the *local* coordinator and sending them *to* remote streaming endpoints.
        This method is kept for potential future use or existing call sites,
        but is not directly used for the event relaying implementation described.

        Args:
            coordinator_info: Information about the coordinator to connect to

        Returns:
            True if connection successful, False otherwise
        """
        self.logger.warning(
            "StdioTransportAdapter.connect_to_coordinator called. "
            "This method is not used for the event relaying feature."
        )
        # The logic below seems to just get the local singleton anyway.
        # Keeping it for compatibility but noting its limited scope here.
        try:
            from aider_mcp_server.atoms.logging.logger import get_logger
            from aider_mcp_server.organisms.coordinators.transport_coordinator import ApplicationCoordinator

            self.logger.info(
                f"Attempting to connect STDIO adapter {self.transport_id} to coordinator {coordinator_info.coordinator_id} "
                f"at {coordinator_info.host}:{coordinator_info.port} (via local singleton access)"
            )

            # Access the local singleton instance
            self._coordinator = await ApplicationCoordinator.getInstance(get_logger)
            self.logger.info("Successfully accessed local coordinator singleton.")

            # Note: This does NOT establish a network connection to the specified host/port.
            # It merely ensures the adapter has a reference to the local coordinator.
            # If a true network client connection were needed, this method would require
            # significant changes and potentially new libraries (like aiohttp client).

            # Initialize the adapter (this will trigger auto-discovery and event subscription)
            await self.initialize()

            return True
        except Exception as e:
            self.logger.error(f"Error connecting to coordinator (via local singleton access): {e}")
            return False

    async def initialize(self) -> None:
        """
        Initialize the stdio transport and register with coordinator.
        Automatically discovers local and streaming coordinators.
        Does NOT start the read task automatically. Call start_listening() separately.
        """
        self.logger.info(f"Initializing stdio transport {self.transport_id}...")

        # Auto-discover local and streaming coordinators
        await self._auto_discover_coordinator()

        # Register this adapter with the local coordinator (if available)
        # This allows the local coordinator to send events *to* this adapter
        await super().initialize()

        # Subscribe to AIDER events on the local coordinator (if available)
        # This ensures send_event is called for AIDER events
        if self._coordinator:
            await self._subscribe_to_aider_events()

        # Create a client session for relaying events if streaming coordinators were found
        if self._streaming_coordinators and self._client_session is None:
            self.logger.info("Creating aiohttp ClientSession for event relaying.")
            self._client_session = aiohttp.ClientSession()

        self.logger.info(f"Stdio transport {self.transport_id} initialized.")

    async def shutdown(self) -> None:
        """
        Shut down the stdio transport and unregister from coordinator.
        Stops the task reading from stdin, cleans up discovery resources,
        and closes the aiohttp client session.
        """
        self.logger.info(f"Shutting down stdio transport {self.transport_id}...")
        self._stop_reading = True
        if self._read_task and not self._read_task.done():
            self.logger.debug(f"Cancelling stdin read task for {self.transport_id}.")
            self._read_task.cancel()
            try:
                # Give the task a moment to cancel gracefully
                await asyncio.wait_for(self._read_task, timeout=0.1)  # Shorter timeout
            except asyncio.CancelledError:
                self.logger.debug(f"Stdin read task for {self.transport_id} cancelled.")
            except asyncio.TimeoutError:
                self.logger.warning(f"Timeout waiting for stdin read task {self.transport_id} to cancel.")
            except Exception as e:
                self.logger.error(f"Error cancelling stdin read task for {self.transport_id}: {e}")
            finally:
                self._read_task = None  # Ensure task reference is cleared

        # Close the aiohttp client session if it exists
        if self._client_session:
            self.logger.info("Closing aiohttp ClientSession.")
            await self._client_session.close()
            self._client_session = None

        # Clean up discovery if it was created
        if self._discovery:
            try:
                await self._discovery.shutdown()
                self.logger.info(f"Shutdown discovery service for {self.transport_id}.")
            except Exception as e:
                self.logger.error(f"Error shutting down discovery service: {e}")
            self._discovery = None

        # Perform base adapter shutdown (unregister from local coordinator)
        await super().shutdown()
        self.logger.info(f"Stdio transport {self.transport_id} shut down.")

    def get_capabilities(self) -> Set[EventTypes]:
        """Declare the event types this Stdio transport adapter supports."""
        # Stdio typically doesn't need heartbeats, but can receive other events
        # It supports receiving all AIDER events from the local coordinator
        # and sending basic status/progress/tool_result events to stdout.
        # The relaying capability is internal and not declared here.
        return {
            EventTypes.STATUS,
            EventTypes.PROGRESS,
            EventTypes.TOOL_RESULT,
            # Include all AIDER events it subscribes to, as it *receives* them
            *AIDER_EVENT_TYPES_TO_RELAY,
        }

    async def _relay_event_to_remote(
        self,
        coord_info: CoordinatorInfo,
        endpoint: Dict[str, Any],
        event: EventTypes,
        data: EventData,
    ) -> None:
        """Helper to send an event to a single remote streaming endpoint."""
        if not self._client_session:
            self.logger.error(f"Cannot relay event {event.value}: aiohttp client session is not initialized.")
            return

        endpoint_path = endpoint.get("path")
        if not endpoint_path:
            self.logger.warning(f"Streaming endpoint for {coord_info.coordinator_id} is missing 'path': {endpoint}")
            return

        url = f"http://{coord_info.host}:{coord_info.port}{endpoint_path}"
        payload = {"event": event.value, "data": data}

        try:
            # Use POST to send the event data
            async with self._client_session.post(url, json=payload) as response:
                response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                self.logger.debug(
                    f"Successfully relayed event {event.value} to "
                    f"{coord_info.coordinator_id} at {url} (Status: {response.status})"
                )
        except aiohttp.client_exceptions.ClientConnectorError as e:
            self.logger.warning(
                f"Failed to connect to streaming endpoint for {coord_info.coordinator_id} "
                f"at {url} to relay event {event.value}: {e}"
            )
        except aiohttp.client_exceptions.ClientResponseError as e:
            self.logger.warning(
                f"Received error response from streaming endpoint for {coord_info.coordinator_id} "
                f"at {url} while relaying event {event.value}: {e.status} - {e.message}"
            )
        except Exception as e:
            self.logger.error(
                f"Unexpected error relaying event {event.value} to {coord_info.coordinator_id} at {url}: {e}"
            )

    async def send_event(self, event: EventTypes, data: EventData) -> None:
        """
        Send an event to stdout as JSON and relay AIDER events to discovered streaming transports.

        Args:
            event: The event type (e.g., EventTypes.PROGRESS)
            data: The event payload
        """
        # 1. Send event to stdout (original functionality)
        try:
            message = {
                "event": event.value,
                "data": data,
            }
            json_str = json.dumps(message)
            # Use async write via executor to avoid blocking
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lambda: print(json_str, file=self._output, flush=True))
            # self.logger.debug(f"Sent {event.value} event to stdout") # Keep debug level low for stdout
        except Exception as e:
            self.logger.error(f"Error sending {event.value} event to stdout: {e}")

        # 2. Relay AIDER events to discovered streaming transports
        if event in AIDER_EVENT_TYPES_TO_RELAY:
            if self._client_session and self._streaming_coordinators:
                self.logger.debug(
                    f"Relaying AIDER event {event.value} to {len(self._streaming_coordinators)} streaming coordinators."
                )
                for coord in self._streaming_coordinators:
                    # Find the specific streaming endpoint for AIDER events
                    # Look in sse_endpoints for aider_events
                    sse_endpoints = coord.streaming_capabilities.get("sse_endpoints", {})
                    aider_endpoint_path = sse_endpoints.get("aider_events")
                    if aider_endpoint_path:
                        # Create a task to relay the event to this coordinator
                        # Create endpoint dict with path
                        endpoint_config = {"path": aider_endpoint_path}
                        asyncio.create_task(self._relay_event_to_remote(coord, endpoint_config, event, data))
                    else:
                        self.logger.warning(
                            f"Streaming coordinator {coord.coordinator_id} "
                            f"does not have an 'aider_events' endpoint in capabilities: {coord.streaming_capabilities}"
                        )
            else:
                self.logger.debug(
                    f"Not relaying AIDER event {event.value}: "
                    f"client session initialized: {self._client_session is not None}, "
                    f"streaming coordinators found: {len(self._streaming_coordinators)}"
                )

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
        self._read_task = asyncio.create_task(self._read_stdin_loop(), name=f"stdio_read_loop_{self.transport_id}")
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
