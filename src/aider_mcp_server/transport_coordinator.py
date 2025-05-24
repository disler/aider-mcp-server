import asyncio
import contextlib
import logging
import typing
import uuid
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

# Use absolute imports from the package root
from aider_mcp_server.atoms.types.event_types import EventTypes
from aider_mcp_server.molecules.transport.discovery import CoordinatorDiscovery, CoordinatorInfo

# Import the interface directly for runtime
from aider_mcp_server.interfaces.transport_adapter import ITransportAdapter as TransportInterface
from aider_mcp_server.atoms.types.mcp_types import (
    AsyncTask,
    LoggerFactory,
    LoggerProtocol,
    OperationResult,
    RequestParameters,
)
from aider_mcp_server.atoms.security.context import Permissions, SecurityContext

# Import only during type checking
if TYPE_CHECKING:
    from aider_mcp_server.progress_reporter import ProgressReporter


# Initialize the logger factory
get_logger_func: LoggerFactory

try:
    # Use absolute import path
    from aider_mcp_server.atoms.logging.logger import get_logger as custom_get_logger

    get_logger_func = typing.cast(LoggerFactory, custom_get_logger)
except ImportError:

    def fallback_get_logger(name: str, *args: Any, **kwargs: Any) -> LoggerProtocol:
        logger = logging.getLogger(name)
        # Ensure logger has handlers if none are configured (basic setup)
        if not logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            # Set a default level if not configured
            if logger.level == logging.NOTSET:
                logger.setLevel(logging.INFO)
        return logger  # type: ignore[return-value]

    get_logger_func = fallback_get_logger


logger = get_logger_func(__name__)

# Type alias for handler functions
HandlerFunc = Callable[
    [str, str, RequestParameters, SecurityContext, bool, bool],
    Coroutine[Any, Any, OperationResult],
]

# Constants
SHUTDOWN_TIMEOUT = 10.0  # Seconds to wait for tasks/transports during shutdown


class ApplicationCoordinator:
    """
    Central coordinator for managing transports, handlers, and requests.

    This class acts as a singleton, ensuring only one instance manages the
    application state. It routes requests from different transports to the
    appropriate handlers and broadcasts events back to relevant transports.

    It also supports async context management (`async with`). Uses asyncio.Lock
    for internal state synchronization.

    The coordinator can be discovered by other processes using the CoordinatorDiscovery
    mechanism, which allows stdio and other transports to find and connect to an
    existing coordinator.
    """

    _instance: Optional["ApplicationCoordinator"] = None
    _creation_lock = asyncio.Lock()  # Async lock for singleton creation
    _initialized = False  # Flag to prevent re-initialization

    def __init__(self) -> None:
        """
        Initializes the ApplicationCoordinator. Should only be called via getInstance.
        """
        if ApplicationCoordinator._initialized:
            logger.warning("ApplicationCoordinator already initialized. Skipping re-initialization.")
            return

        # Only set up basic attributes here
        # Full initialization happens in _initialize_coordinator
        self._initialized_event = asyncio.Event()  # Event to signal initialization completion
        self._shutdown_event = asyncio.Event()  # Event to signal shutdown
        self._coordinator_id: Optional[str] = None
        self._discovery: Optional[CoordinatorDiscovery] = None
        self.logger = get_logger_func(f"{__name__}.ApplicationCoordinator")

    async def _initialize_coordinator(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        register_in_discovery: bool = False,
        discovery_file: Optional[Path] = None,
    ) -> None:
        """
        Internal method to initialize the ApplicationCoordinator state.
        Called from __aenter__ to avoid directly calling __init__.

        Args:
            host: Host where the coordinator is running (for discovery registration)
            port: Port where the coordinator is listening (for discovery registration)
            register_in_discovery: Whether to register this coordinator in the discovery system
            discovery_file: Custom path to the discovery file
        """
        self._transports: Dict[str, TransportInterface] = {}
        self._handlers: Dict[str, Tuple[HandlerFunc, Optional[Permissions]]] = {}
        self._active_requests: Dict[str, RequestParameters] = {}
        self._transport_capabilities: Dict[str, Set[EventTypes]] = {}
        self._transport_subscriptions: Dict[str, Set[EventTypes]] = {}

        # Locks for async safety
        self._transports_lock = asyncio.Lock()
        self._handlers_lock = asyncio.Lock()  # Use sync lock for sync handler registration
        self._active_requests_lock = asyncio.Lock()
        self._transport_capabilities_lock = asyncio.Lock()
        self._transport_subscriptions_lock = asyncio.Lock()

        # Event loop management
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            # This should generally not happen if initialized within an async context
            logger.error("No running event loop found during ApplicationCoordinator initialization!")
            # We'll try to get the current event loop or create one, but we won't set it globally
            # as that can cause issues in tests and other environments
            self._loop = asyncio.get_event_loop_policy().get_event_loop()

        # Initialize coordinator discovery if requested
        if register_in_discovery and host is not None and port is not None:
            try:
                self._discovery = CoordinatorDiscovery(discovery_file=discovery_file)
                self._coordinator_id = await self._discovery.register_coordinator(
                    host=host,
                    port=port,
                    transport_type="sse",  # Default to SSE for now
                    metadata={"version": "1.0.0"},  # Add metadata as needed
                )
                logger.info(f"Registered coordinator {self._coordinator_id} in discovery system")
            except Exception as e:
                logger.error(f"Failed to register coordinator in discovery: {e}")
                self._discovery = None

        # Mark as initialized *after* setup is complete
        ApplicationCoordinator._initialized = True
        self._initialized_event.set()  # Signal that initialization is done

    @classmethod
    async def find_existing_coordinator(cls, discovery_file: Optional[Path] = None) -> Optional[CoordinatorInfo]:
        """
        Attempts to find an existing coordinator in the discovery system.

        Args:
            discovery_file: Optional path to the discovery file

        Returns:
            Information about the best coordinator found, or None if none are available
        """
        try:
            discovery = CoordinatorDiscovery(discovery_file=discovery_file)
            return await discovery.find_best_coordinator()
        except Exception as e:
            logger.error(f"Error finding existing coordinator: {e}")
            return None

    @classmethod
    async def getInstance(cls, logger_factory: Optional[LoggerFactory] = None) -> "ApplicationCoordinator":
        """
        Gets the singleton instance of the ApplicationCoordinator.

        Uses double-checked locking with asyncio.Lock for async-safe initialization.

        Args:
            logger_factory: Factory function to create loggers. If None, uses default.

        Returns:
            ApplicationCoordinator: The singleton instance.
        """
        if cls._instance is None:
            async with cls._creation_lock:
                # Double-check inside the lock
                if cls._instance is None:
                    # Use provided logger_factory or default to get_logger_func (already defined)
                    cls._instance = cls()
        return cls._instance

    async def wait_for_initialization(self, timeout: Optional[float] = 30.0) -> None:
        """
        Waits until the coordinator's asyncio components are fully initialized.

        Args:
            timeout: Maximum time (in seconds) to wait for initialization.
                   Set to None for no timeout. Defaults to 30 seconds.

        Raises:
            asyncio.TimeoutError: If initialization doesn't complete within the timeout period.
            RuntimeError: If coordinator is shutting down during initialization wait.
        """
        if self._shutdown_event.is_set():
            raise RuntimeError("Cannot wait for initialization while coordinator is shutting down")

        if timeout is not None:
            # Use wait_for with timeout
            try:
                await asyncio.wait_for(self._initialized_event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                logger.error(f"Coordinator initialization timed out after {timeout} seconds")
                raise
        else:
            # Wait indefinitely
            await self._initialized_event.wait()

    def is_shutting_down(self) -> bool:
        """Checks if the coordinator shutdown process has been initiated."""
        return self._shutdown_event.is_set()

    # --- Async Context Management ---

    async def __aenter__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        register_in_discovery: bool = False,
        discovery_file: Optional[Path] = None,
    ) -> "ApplicationCoordinator":
        """
        Enter the async context, ensuring initialization.

        Args:
            host: Host where the coordinator is running (for discovery registration)
            port: Port where the coordinator is listening (for discovery registration)
            register_in_discovery: Whether to register this coordinator in the discovery system
            discovery_file: Custom path to the discovery file
        """
        # In case getInstance wasn't awaited, ensure initialization happens.
        # Note: getInstance should ideally be the entry point.
        if not ApplicationCoordinator._initialized:
            async with ApplicationCoordinator._creation_lock:
                if not ApplicationCoordinator._initialized:
                    ApplicationCoordinator._instance = self  # Assign self if called directly
                    # Initialize explicitly instead of calling __init__ directly
                    await self._initialize_coordinator(
                        host=host,
                        port=port,
                        register_in_discovery=register_in_discovery,
                        discovery_file=discovery_file,
                    )

        await self.wait_for_initialization()
        # Context entry logged only at DEBUG level
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],  # Traceback type is complex, use Any
    ) -> None:
        """Exit the async context, triggering shutdown."""
        # Context exit logged only at DEBUG level
        await self.shutdown()

    # --- Transport Management ---

    async def register_transport(self, transport_id: str, transport: "TransportInterface") -> None:
        """Registers a new transport adapter."""
        async with self._transports_lock:
            if transport_id in self._transports:
                logger.debug(f"Transport {transport_id} already registered. Overwriting.")
            self._transports[transport_id] = transport
            if hasattr(self.logger, "verbose") and callable(self.logger.verbose):
                self.logger.verbose(f"Transport {transport_id} registered and connected.")
            else:
                self.logger.debug(f"(Verbose) Transport {transport_id} registered and connected.")
            # Transport registration logged only at DEBUG level
        # Update capabilities and default subscriptions (outside transports_lock)
        await self.update_transport_capabilities(transport_id, transport.get_capabilities())

    async def unregister_transport(self, transport_id: str) -> None:
        """Unregisters a transport adapter."""
        transport_exists = False
        async with self._transports_lock:
            if transport_id in self._transports:
                del self._transports[transport_id]
                transport_exists = True
                logger.debug(f"Transport unregistered: {transport_id}")
                if hasattr(self.logger, "verbose") and callable(self.logger.verbose):
                    self.logger.verbose(f"Transport {transport_id} disconnected and unregistered.")
                else:
                    self.logger.debug(f"(Verbose) Transport {transport_id} disconnected and unregistered.")
            else:
                if hasattr(self.logger, "verbose") and callable(self.logger.verbose):
                    self.logger.verbose(f"Verbose: Attempt to unregister non-existent transport: {transport_id}.")
                else:
                    self.logger.debug(
                        f"(Verbose) Verbose: Attempt to unregister non-existent transport: {transport_id}."
                    )
                logger.warning(f"Attempted to unregister non-existent transport: {transport_id}")

        if transport_exists:
            # Clean up capabilities and subscriptions (outside transports_lock)
            async with self._transport_capabilities_lock:
                if transport_id in self._transport_capabilities:
                    del self._transport_capabilities[transport_id]
            async with self._transport_subscriptions_lock:
                if transport_id in self._transport_subscriptions:
                    del self._transport_subscriptions[transport_id]

    async def update_transport_capabilities(self, transport_id: str, capabilities: Set[EventTypes]) -> None:
        """Updates the capabilities of a registered transport."""
        async with self._transport_capabilities_lock:
            self._transport_capabilities[transport_id] = capabilities
            # Capability updates logged only at DEBUG level
        # By default, subscribe to all capabilities when capabilities are updated
        await self.update_transport_subscriptions(transport_id, capabilities)

    async def update_transport_subscriptions(self, transport_id: str, subscriptions: Set[EventTypes]) -> None:
        """Updates the event types a transport is subscribed to (replaces existing)."""
        # Check if transport exists first (read lock)
        transport_exists = await self._transport_exists(transport_id)
        if not transport_exists:
            logger.warning(f"Attempted to update subscriptions for non-existent transport: {transport_id}")
            return

        async with self._transport_subscriptions_lock:
            # Validate that subscriptions are a subset of capabilities? Optional.
            self._transport_subscriptions[transport_id] = subscriptions
            if hasattr(self.logger, "verbose") and callable(self.logger.verbose):
                self.logger.verbose(
                    f"Transport {transport_id} subscriptions updated to: {[s.value for s in subscriptions]}."
                )
            else:
                self.logger.debug(
                    f"(Verbose) Transport {transport_id} subscriptions updated to: {[s.value for s in subscriptions]}."
                )
            # Subscription updates logged only at DEBUG level

    # --- Subscription Management (for test compatibility) ---

    async def subscribe_to_event_type(self, transport_id: str, event_type: EventTypes) -> None:
        """Subscribes a transport to a specific event type."""
        transport_exists = await self._transport_exists(transport_id)
        if not transport_exists:
            logger.warning(f"Attempted to subscribe non-existent transport {transport_id} to {event_type.value}")
            return

        async with self._transport_subscriptions_lock:
            if transport_id not in self._transport_subscriptions:
                self._transport_subscriptions[transport_id] = set()
            self._transport_subscriptions[transport_id].add(event_type)
            logger.debug(f"Transport {transport_id} subscribed to {event_type.value}")
            if hasattr(self.logger, "verbose") and callable(self.logger.verbose):
                self.logger.verbose(
                    f"Transport {transport_id} successfully subscribed to event type {event_type.value}."
                )
            else:
                self.logger.debug(
                    f"(Verbose) Transport {transport_id} successfully subscribed to event type {event_type.value}."
                )

    async def unsubscribe_from_event_type(self, transport_id: str, event_type: EventTypes) -> None:
        """Unsubscribes a transport from a specific event type."""
        async with self._transport_subscriptions_lock:
            if transport_id in self._transport_subscriptions:
                self._transport_subscriptions[transport_id].discard(event_type)
                logger.debug(f"Transport {transport_id} unsubscribed from {event_type.value}")
                if hasattr(self.logger, "verbose") and callable(self.logger.verbose):
                    self.logger.verbose(
                        f"Transport {transport_id} successfully unsubscribed from event type {event_type.value}."
                    )
                else:
                    self.logger.debug(
                        f"(Verbose) Transport {transport_id} successfully unsubscribed from event type {event_type.value}."
                    )
            # else: No warning needed if transport exists but wasn't subscribed

    async def is_subscribed(self, transport_id: str, event_type: EventTypes) -> bool:
        """Checks if a transport is subscribed to a specific event type."""
        async with self._transport_subscriptions_lock:
            subscriptions = self._transport_subscriptions.get(transport_id, set())
            return event_type in subscriptions

    # --- Handler Management ---

    async def register_handler(
        self,
        operation_name: str,
        handler: HandlerFunc,
        required_permission: Optional[Permissions] = None,
    ) -> None:
        """Registers a handler function for a specific operation."""
        async with self._handlers_lock:
            if operation_name in self._handlers:
                logger.warning(f"Handler for operation '{operation_name}' already registered. Overwriting.")
            self._handlers[operation_name] = (handler, required_permission)
            logger.info(f"Handler registered for operation: '{operation_name}'")

    async def unregister_handler(self, operation_name: str) -> None:
        """Unregisters a handler function."""
        async with self._handlers_lock:
            if operation_name in self._handlers:
                del self._handlers[operation_name]
                logger.info(f"Handler unregistered for operation: '{operation_name}'")
            else:
                logger.warning(f"Attempted to unregister non-existent handler: '{operation_name}'")

    # --- Handler Retrieval (for test compatibility) ---

    async def get_handler(self, operation_name: str) -> Optional[HandlerFunc]:
        """Gets the handler function for a specific operation name."""
        handler_info = await self._get_handler_info(operation_name)
        return handler_info[0] if handler_info else None

    async def get_required_permission(self, operation_name: str) -> Optional[Permissions]:
        """Gets the required permission for a specific operation name."""
        handler_info = await self._get_handler_info(operation_name)
        return handler_info[1] if handler_info else None

    # --- Request Lifecycle Management ---

    async def start_request(  # noqa: C901
        self,
        request_id: str,
        transport_id: str,
        operation_name: str,
        request_data: RequestParameters,
    ) -> None:
        """
        Starts processing a new request received from a transport.

        Validates security, finds the handler, and runs it in the background.
        """
        await self.wait_for_initialization()  # Ensure coordinator is ready
        if self.is_shutting_down():
            logger.warning(f"Coordinator is shutting down. Rejecting request {request_id}.")
            # Optionally inform the transport if possible/needed
            return

        logger.info(f"Starting request {request_id} for operation '{operation_name}' from transport {transport_id}")

        transport = await self._get_transport(transport_id)
        if not transport:
            logger.error(f"Cannot start request {request_id}: Transport {transport_id} not found.")
            # Cannot send error back as transport is gone.
            return

        # Extract diff cache settings from request data
        use_diff_cache = request_data.get("use_diff_cache", True)
        clear_cached_for_unchanged = request_data.get("clear_cached_for_unchanged", True)

        # 1. Validate Security Context (outside locks)
        try:
            # Assuming validate_request_security is synchronous or handled by transport
            # If it needs to be async, transport interface and implementation must change.
            security_context = transport.validate_request_security(request_data)
            logger.debug(
                f"Request {request_id} security context validated: User '{security_context.user_id}', Permissions: {security_context.permissions}"
            )
            if hasattr(self.logger, "verbose") and callable(self.logger.verbose):
                self.logger.verbose(
                    f"Verbose: Security context for request {request_id} (user: {security_context.user_id}) validated successfully."
                )
            else:
                self.logger.debug(
                    f"(Verbose) Verbose: Security context for request {request_id} (user: {security_context.user_id}) validated successfully."
                )
        except Exception as e:
            if hasattr(self.logger, "verbose") and callable(self.logger.verbose):
                self.logger.verbose(
                    f"Verbose: Security validation failed for request {request_id} from {transport_id}. Error: {e}"
                )
            else:
                self.logger.debug(
                    f"(Verbose) Verbose: Security validation failed for request {request_id} from {transport_id}. Error: {e}"
                )
            logger.error(
                f"Security validation failed for request {request_id} from {transport_id}: {e}",
                exc_info=True,
            )
            error_result = {
                "success": False,
                "error": "Security validation failed",
                "details": str(e),
            }
            # Send error result directly back to the originating transport
            await self.send_event_to_transport(
                transport_id,
                EventTypes.TOOL_RESULT,
                {
                    "type": EventTypes.TOOL_RESULT.value,
                    "request_id": request_id,
                    "tool_name": operation_name,
                    "result": error_result,
                },
            )
            return

        # 2. Find Handler and Check Permissions (read handler lock)
        handler_info = await self._get_handler_info(operation_name)
        if not handler_info:
            if hasattr(self.logger, "verbose") and callable(self.logger.verbose):
                self.logger.verbose(
                    f"Verbose: No handler found for operation '{operation_name}' (request {request_id}). Failing request."
                )
            else:
                self.logger.debug(
                    f"(Verbose) Verbose: No handler found for operation '{operation_name}' (request {request_id}). Failing request."
                )
            logger.warning(f"No handler found for operation '{operation_name}' (request {request_id}).")
            # Fail request needs original params
            request_params = request_data.get("parameters", {})
            await self.fail_request(
                request_id,
                operation_name,
                "Operation not supported",
                f"No handler registered for operation '{operation_name}'.",
                originating_transport_id=transport_id,  # Ensure error goes back
                request_details=request_params,
            )
            return

        handler, required_permission = handler_info
        if required_permission and not security_context.has_permission(required_permission):
            if hasattr(self.logger, "verbose") and callable(self.logger.verbose):
                self.logger.verbose(
                    f"Verbose: Permission denied for operation '{operation_name}' (request {request_id}). User '{security_context.user_id}' lacks permission '{required_permission.name}'."
                )
            else:
                self.logger.debug(
                    f"(Verbose) Verbose: Permission denied for operation '{operation_name}' (request {request_id}). User '{security_context.user_id}' lacks permission '{required_permission.name}'."
                )
            logger.warning(
                f"Permission denied for operation '{operation_name}' (request {request_id}). User '{security_context.user_id}' lacks permission '{required_permission.name}'."
            )
            # Include parameters in the error details sent back
            request_params = request_data.get("parameters", {})
            error_result = {
                "success": False,
                "error": "Permission denied",
                "details": {
                    "message": f"User does not have the required permission '{required_permission.name}' for operation '{operation_name}'.",
                    "parameters": request_params,  # Include original parameters
                },
            }
            await self.send_event_to_transport(
                transport_id,
                EventTypes.TOOL_RESULT,
                {
                    "type": EventTypes.TOOL_RESULT.value,
                    "request_id": request_id,
                    "tool_name": operation_name,
                    "result": error_result,
                },
            )
            return  # Do not proceed with the request

        # 3. Store Active Request State (write active requests lock)
        request_params = request_data.get("parameters", {})
        async with self._active_requests_lock:
            if request_id in self._active_requests:
                logger.warning(f"Request ID {request_id} already active. Overwriting state.")
                # Consider cancelling the previous task if it exists?
                existing_task = self._active_requests[request_id].get("task")
                if existing_task and not existing_task.done():
                    logger.warning(f"Cancelling previous task for duplicate request ID {request_id}.")
                    existing_task.cancel()

            # Create the task first
            task = self._loop.create_task(
                self._run_handler(
                    request_id,
                    transport_id,
                    operation_name,
                    handler,
                    request_params,
                    security_context,
                    use_diff_cache,
                    clear_cached_for_unchanged,
                ),
                name=f"handler-{operation_name}-{request_id}",
            )

            self._active_requests[request_id] = {
                "operation": operation_name,
                "transport_id": transport_id,
                "status": "starting",
                "task": task,  # Store the created task
                "details": {"parameters": request_params},  # Store original parameters
            }
            logger.debug(f"Request {request_id} state initialized and task created.")

        # 4. Send 'starting' status update (outside active requests lock)
        # This now happens *after* the task is created and stored
        await self.update_request(
            request_id,
            "starting",
            f"Operation '{operation_name}' starting.",
            # Pass details containing parameters for the initial status message
            details={"parameters": request_params},
        )

        # Task is already running

    async def _run_handler(
        self,
        request_id: str,
        transport_id: str,
        operation_name: str,
        handler: HandlerFunc,
        parameters: RequestParameters,
        security_context: SecurityContext,
        use_diff_cache: bool,
        clear_cached_for_unchanged: bool,
    ) -> None:
        """Wrapper to run the handler, process result/errors, and clean up."""
        result_data: Optional[OperationResult] = None
        handler_completed_normally = False
        try:
            # Execute the actual handler coroutine
            result_data = await handler(
                request_id,
                transport_id,
                parameters,
                security_context,
                use_diff_cache,
                clear_cached_for_unchanged,
            )
            handler_completed_normally = True  # Mark success before potential result processing issues
            logger.info(f"Handler for '{operation_name}' (request {request_id}) completed successfully.")

            # Log whether the diff was cached
            if isinstance(result_data, dict) and result_data.get("is_cached_diff", False):
                logger.info(f"Received cached diff for request {request_id}.")
            else:
                logger.info(f"Received full diff for request {request_id}.")

            # Process result_data based on type
            if isinstance(result_data, dict):
                # Handle dict result
                if "success" not in result_data:
                    result_data["success"] = True  # Assume success if key missing
            else:
                # Handle None or non-dict result
                # mypy thinks this branch is unreachable, but we're keeping it for type safety
                # and to handle potential future changes in handler implementations
                logger.warning(  # type: ignore[unreachable]
                    f"Handler for '{operation_name}' (request {request_id}) returned {result_data} of type {type(result_data)}. Wrapping."
                )
                result_data = {"success": True, "result": result_data}  # Basic wrapping

        except asyncio.CancelledError:
            logger.warning(f"Handler task for '{operation_name}' (request {request_id}) was cancelled.")
            # Don't call fail_request here, cancellation is often initiated externally (e.g., shutdown)
            # Cleanup will happen in the finally block. If cancellation needs a specific error sent,
            # the cancelling code should handle it or call fail_request before cancelling.
            # Let's send a specific "cancelled" status via TOOL_RESULT for clarity.
            await self._send_event_to_transports(
                EventTypes.TOOL_RESULT,
                {
                    "type": EventTypes.TOOL_RESULT.value,
                    "request_id": request_id,
                    "tool_name": operation_name,
                    "result": {
                        "success": False,
                        "error": "Operation cancelled",
                        "details": {
                            "message": "The operation was cancelled.",
                            "parameters": parameters,  # Include parameters
                        },
                    },
                },
                originating_transport_id=transport_id,
                request_details=parameters,
            )
            # No return here, let finally block handle cleanup

        except Exception as e:
            if hasattr(self.logger, "verbose") and callable(self.logger.verbose):
                self.logger.verbose(
                    f"Verbose: Exception in handler for '{operation_name}' (request {request_id}): {type(e).__name__} - {e}"
                )
            else:
                self.logger.debug(
                    f"(Verbose) Verbose: Exception in handler for '{operation_name}' (request {request_id}): {type(e).__name__} - {e}"
                )
            logger.error(
                f"Handler for '{operation_name}' (request {request_id}) raised an exception: {e}",
                exc_info=True,
            )
            error_type = type(e).__name__
            # fail_request sends the event AND cleans up state
            await self.fail_request(
                request_id,
                operation_name,
                f"Operation failed: {error_type}",
                # Pass exception details in a structured way
                {"message": str(e), "exception_type": error_type},
                originating_transport_id=transport_id,
                request_details=parameters,
            )
            return  # fail_request handles cleanup, so return here

        finally:
            # Send final result ONLY if handler completed normally without exceptions
            if handler_completed_normally and result_data is not None:
                await self._send_event_to_transports(
                    EventTypes.TOOL_RESULT,
                    {
                        "type": EventTypes.TOOL_RESULT.value,
                        "request_id": request_id,
                        "tool_name": operation_name,
                        "result": result_data,
                    },
                    originating_transport_id=transport_id,
                    request_details=parameters,  # Include original params for context
                )

            # Clean up active request state unless fail_request already did
            # Check if the request still exists before cleaning up
            request_info = await self._get_request_info(request_id)
            if request_info:
                await self._cleanup_request(request_id)

    async def update_request(
        self,
        request_id: str,
        status: str,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Sends a progress or status update for an ongoing request.
        Ensures original request parameters are included in the 'details' field.
        """
        if self.is_shutting_down():
            return  # Don't send updates during shutdown

        await self.wait_for_initialization()

        # Get request info without holding lock for long
        request_info = await self._get_request_info(request_id)
        if not request_info:
            # This can happen normally if the request finishes/fails between the handler
            # yielding and the update call, or if update is called after cleanup.
            logger.debug(f"Cannot update request {request_id}: Not found or already completed.")
            return

        operation_name = request_info.get("operation", "unknown_operation")
        originating_transport_id = request_info.get("transport_id")
        # Ensure original parameters are always present in the details sent
        # Perform merge outside the lock
        original_params = request_info.get("details", {}).get("parameters", {})
        # Start with original parameters
        merged_details = {"parameters": original_params.copy()}
        # Update with any new details provided
        if details:
            # Avoid overwriting 'parameters' key if present in 'details'
            new_details_filtered = {k: v for k, v in details.items() if k != "parameters"}
            merged_details.update(new_details_filtered)

        # Determine event type based on status (PROGRESS or STATUS)
        # Use PROGRESS for 'starting', 'in_progress', 'completed', 'error' reported via reporter
        # Use STATUS for initial 'starting' from coordinator, maybe others?
        # Let's align with test_multi_transport: PROGRESS for reporter updates ('starting', 'in_progress', 'completed', 'error')
        # and STATUS for the initial coordinator 'starting' message.
        # This method is called by both coordinator (initial start) and reporter.
        # We need a way to distinguish or decide based on status.
        # Let's assume:
        # - status 'starting' called from start_request -> EventTypes.STATUS
        # - status 'starting', 'in_progress', 'completed', 'error' called from ProgressReporter -> EventTypes.PROGRESS
        # How to know the caller? We can't easily.
        # Let's simplify: If status is 'in_progress', it's PROGRESS. Otherwise, it's STATUS.
        # Re-evaluating based on test_progress_request_routing:
        # - Coordinator sends initial STATUS 'starting'
        # - Reporter sends PROGRESS 'starting'
        # - Reporter sends PROGRESS 'in_progress'
        # - Reporter sends PROGRESS 'completed'
        # This implies the event type should perhaps be passed in or determined more robustly.
        # Let's stick to the logic from the test: PROGRESS includes 'starting', 'in_progress', 'completed', 'error' states
        # when reported via update_request (typically by ProgressReporter).
        # The initial STATUS 'starting' is sent explicitly in start_request.
        # So, update_request should generally send PROGRESS, unless a specific need for STATUS arises.
        # Let's default to PROGRESS here, as it covers more states reported during execution.
        # The initial STATUS is handled separately in start_request.

        # Correction: The initial call from start_request *does* call update_request.
        # Let's make the event type depend on the status argument more explicitly.
        if (
            status in ["starting", "completed", "error"]
            and message
            and "Operation" in message
            and ("started." in message or "completed" in message or "failed" in message)
        ):
            # Likely called from ProgressReporter __aenter__ or __aexit__
            event_type = EventTypes.PROGRESS
        elif status == "in_progress":
            event_type = EventTypes.PROGRESS
        elif status == "starting":
            # Likely the initial call from start_request
            event_type = EventTypes.STATUS
        else:
            # Default or other statuses might be STATUS
            event_type = EventTypes.STATUS

        event_data = {
            "type": event_type.value,
            "request_id": request_id,
            "operation": operation_name,
            "status": status,
            # Ensure message is not None, provide a default if needed
            "message": message if message is not None else f"Status updated to {status}",
            "details": merged_details,  # Send merged details including parameters
        }

        logger.debug(f"Sending update for request {request_id}: Status={status}, Event={event_type.value}")
        await self._send_event_to_transports(
            event_type,
            event_data,
            originating_transport_id=originating_transport_id,
            request_details=original_params,  # Pass original params for filtering
        )

        # Update internal state (briefly acquire lock)
        async with self._active_requests_lock:
            # Check again if request exists, as it might have been cleaned up
            # between the _get_request_info call and acquiring the lock here.
            if request_id in self._active_requests:
                self._active_requests[request_id]["status"] = status
                # Avoid merging details into the stored state unless necessary
                # Only update if new details (beyond params) were provided
                # if details:
                #    self._active_requests[request_id].setdefault("details", {}).update(new_details_filtered)

    async def fail_request(
        self,
        request_id: str,
        operation_name: str,
        error: str,
        error_details: Optional[Union[str, Dict[str, Any]]] = None,  # Allow dict for structured errors
        originating_transport_id: Optional[str] = None,
        request_details: Optional[Dict[str, Any]] = None,  # Original parameters
    ) -> None:
        """
        Marks a request as failed, sends an error result event (TOOL_RESULT),
        and cleans up state. Includes original parameters in the result details.
        """
        if self.is_shutting_down():
            return  # Avoid sending failures during shutdown chaos

        await self.wait_for_initialization()
        logger.error(f"Failing request {request_id} (Operation: {operation_name}): {error} - {error_details}")

        # Check if request exists before sending failure event
        request_info = await self._get_request_info(request_id)
        if not request_info:
            logger.warning(f"Attempted to fail already cleaned up request: {request_id}")
            return

        # If originating_transport_id wasn't provided, try to get it from request_info
        origin_tid = originating_transport_id or request_info.get("transport_id")
        # If request_details (params) weren't provided, try to get them from stored info
        req_params = request_details or request_info.get("details", {}).get("parameters", {})

        # Structure the details field for the error result
        structured_error_details = {"parameters": req_params}
        if isinstance(error_details, dict):
            # Merge dict details, ensuring 'parameters' isn't overwritten if present
            details_to_merge = {k: v for k, v in error_details.items() if k != "parameters"}
            structured_error_details.update(details_to_merge)
        elif isinstance(error_details, str):
            structured_error_details["message"] = error_details
        else:
            # Handle None or any other type (convert to string)
            structured_error_details["original_error"] = (
                str(error_details) if error_details is not None else "No details provided"
            )

        result_data = {
            "success": False,
            "error": error,
            "details": structured_error_details,
        }

        await self._send_event_to_transports(
            EventTypes.TOOL_RESULT,
            {
                "type": EventTypes.TOOL_RESULT.value,
                "request_id": request_id,
                "tool_name": operation_name,
                "result": result_data,
            },
            originating_transport_id=origin_tid,
            request_details=req_params,  # Pass original params for context/filtering
        )

        # Clean up the request state immediately after reporting failure
        await self._cleanup_request(request_id)

    async def _cleanup_request(self, request_id: str) -> None:
        """Removes a request from the active requests list and cancels its task."""
        task_to_cancel: Optional[AsyncTask[Any]] = None
        async with self._active_requests_lock:
            if request_id in self._active_requests:
                request_info = self._active_requests.pop(request_id)  # Remove and get info
                task_to_cancel = request_info.get("task")
                logger.info(f"Cleaned up state for request {request_id}")
            else:
                logger.debug(f"Attempted to clean up already removed request: {request_id}")
                return  # Nothing more to do

        # Cancel the task outside the lock
        if task_to_cancel and not task_to_cancel.done():
            task_to_cancel.cancel()
            logger.debug(f"Cancelled task for request {request_id} during cleanup.")
            # Optionally wait for cancellation briefly to allow cleanup within the task
            with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
                await asyncio.wait_for(task_to_cancel, timeout=0.1)

    # --- Event Broadcasting ---

    async def broadcast_event(
        self,
        event_type: EventTypes,
        data: Dict[str, Any],
        exclude_transport_id: Optional[str] = None,
    ) -> None:
        """
        Broadcasts an event to all subscribed transports, optionally excluding one.
        """
        if self.is_shutting_down():
            return
        await self.wait_for_initialization()
        logger.debug(f"Broadcasting event {event_type.value} (excluding {exclude_transport_id}): {data}")
        # Extract potential request details (parameters) if available in data for filtering
        request_params = data.get("details", {}).get("parameters")
        await self._send_event_to_transports(
            event_type,
            data,
            exclude_transport_id=exclude_transport_id,
            request_details=request_params,  # Pass params if available
        )

    async def send_event_to_transport(
        self,
        transport_id: str,
        event_type: EventTypes,
        data: Dict[str, Any],
    ) -> None:
        """
        Sends a single event directly to a specific transport, if it exists.
        Does not check subscriptions. Primarily used for direct responses like errors.
        """
        if self.is_shutting_down():
            return
        await self.wait_for_initialization()

        transport = await self._get_transport(transport_id)
        if transport:
            logger.debug(
                f"Sending direct event {event_type.value} to transport {transport_id} (Request: {data.get('request_id', 'N/A')})"
            )
            try:
                # Run send_event, but don't block coordinator if it takes time
                # Create task, but don't necessarily await it here unless confirmation is needed
                self._loop.create_task(
                    transport.send_event(event_type, data),
                    name=f"direct-send-{event_type.value}-{transport_id}-{data.get('request_id', uuid.uuid4())}",
                )
            except Exception as e:
                if hasattr(self.logger, "verbose") and callable(self.logger.verbose):
                    self.logger.verbose(
                        f"Verbose: Error creating task for direct event {event_type.value} to transport {transport_id}. Exception: {e}"
                    )
                else:
                    self.logger.debug(
                        f"(Verbose) Verbose: Error creating task for direct event {event_type.value} to transport {transport_id}. Exception: {e}"
                    )
                # This catch block might not be effective if send_event is async and raises later
                logger.error(
                    f"Error creating task to send direct event {event_type.value} to transport {transport_id}: {e}",
                    exc_info=True,
                )
        else:
            if hasattr(self.logger, "verbose") and callable(self.logger.verbose):
                self.logger.verbose(
                    f"Verbose: Attempted to send direct event {event_type.value} to non-existent transport {transport_id}, but transport not found."
                )
            else:
                self.logger.debug(
                    f"(Verbose) Verbose: Attempted to send direct event {event_type.value} to non-existent transport {transport_id}, but transport not found."
                )
            logger.warning(
                f"Attempted to send direct event {event_type.value} to non-existent transport {transport_id}"
            )

    async def _should_send_to_transport(
        self,
        transport_id: str,
        transport: "TransportInterface",
        event_type: EventTypes,
        data: Dict[str, Any],
        originating_transport_id: Optional[str],
        subscriptions: Dict[str, Set[EventTypes]],
        request_details: Optional[Dict[str, Any]],
    ) -> bool:
        """
        Determine if an event should be sent to a specific transport based on
        subscription status, origin rules, and transport-specific filtering.
        """
        # Check subscription
        is_subscribed = event_type in subscriptions.get(transport_id, set())
        if not is_subscribed:
            return False

        # Special handling for STATUS events
        if event_type == EventTypes.STATUS and originating_transport_id is not None:
            if transport_id != originating_transport_id:
                return False

        # Check transport-specific filtering
        if hasattr(transport, "should_receive_event") and callable(transport.should_receive_event):
            try:
                # Pass original request parameters if available
                if not transport.should_receive_event(event_type, data, request_details):
                    logger.debug(
                        f"Transport {transport_id} filtered out event {event_type.value} for request {data.get('request_id', 'N/A')}"
                    )
                    return False
            except Exception as e:
                logger.error(
                    f"Error calling should_receive_event for transport {transport_id}: {e}",
                    exc_info=True,
                )
                return False

        return True

    def _create_send_event_task(
        self,
        transport: "TransportInterface",
        transport_id: str,
        event_type: EventTypes,
        data: Dict[str, Any],
    ) -> AsyncTask[None]:
        """
        Create a task to send an event to a transport.
        """
        request_id = data.get("request_id", uuid.uuid4())
        logger.debug(f"Queueing event {event_type.value} for transport {transport_id} (Request: {request_id})")
        return self._loop.create_task(
            transport.send_event(event_type, data),
            name=f"send-{event_type.value}-{transport_id}-{request_id}",
        )

    async def _log_originating_transport_status(
        self,
        event_type: EventTypes,
        data: Dict[str, Any],
        originating_transport_id: str,
        sent_to: Set[str],
        transports_to_notify: List[Tuple[str, "TransportInterface"]],
        subscriptions: Dict[str, Set[EventTypes]],
    ) -> None:
        """
        Log warnings if the originating transport didn't receive an event it might have expected.
        """
        request_id = data.get("request_id", "N/A")
        origin_subscribed = event_type in subscriptions.get(originating_transport_id, set())
        origin_exists = any(t_id == originating_transport_id for t_id, _ in transports_to_notify)

        if origin_exists and not origin_subscribed:
            logger.warning(
                f"Event {event_type.value} for request {request_id} was not sent to originating transport {originating_transport_id} because it was not subscribed."
            )
        elif not origin_exists:
            logger.warning(
                f"Event {event_type.value} for request {request_id} could not be sent to originating transport {originating_transport_id} because it was not found (likely unregistered)."
            )

    async def _handle_task_results(
        self,
        tasks: List[AsyncTask[Any]],
        results: List[Any],
    ) -> None:
        """
        Process the results of event sending tasks and log any errors.
        """
        for i, result in enumerate(results):
            if not isinstance(result, Exception):
                continue

            task_name = tasks[i].get_name()
            log_transport_id = "unknown"
            try:
                parts = task_name.split("-")
                if len(parts) >= 4 and parts[0] == "send":
                    log_transport_id = parts[2]
            except Exception:
                logger.warning(f"Failed to parse transport ID from task name {task_name}")

            # Avoid logging CancelledError stack traces unless debugging needed
            log_exc_info = result if not isinstance(result, asyncio.CancelledError) else None
            if hasattr(self.logger, "verbose") and callable(self.logger.verbose):
                self.logger.verbose(
                    f"Verbose: Error sending event via task {task_name} (Transport: {log_transport_id}). Exception: {result}"
                )
            else:
                self.logger.debug(
                    f"(Verbose) Verbose: Error sending event via task {task_name} (Transport: {log_transport_id}). Exception: {result}"
                )
            logger.error(
                f"Error sending event via task {task_name} (Transport: {log_transport_id}): {result}",
                exc_info=log_exc_info,
            )

    async def _send_event_to_transports(
        self,
        event_type: EventTypes,
        data: Dict[str, Any],
        originating_transport_id: Optional[str] = None,
        exclude_transport_id: Optional[str] = None,
        request_details: Optional[Dict[str, Any]] = None,  # Original request params for filtering
    ) -> None:
        """
        Internal helper to send an event to relevant transports based on subscriptions.
        Includes logic to check transport's should_receive_event method if available.
        """
        if self.is_shutting_down():
            return
        await self.wait_for_initialization()

        tasks = []
        sent_to = set()

        # Acquire locks briefly to get copies of state
        async with self._transports_lock:
            transports_to_notify = list(self._transports.items())
        async with self._transport_subscriptions_lock:
            subscriptions_copy = self._transport_subscriptions.copy()

        # Process transports outside the locks
        for transport_id, transport in transports_to_notify:
            if transport_id == exclude_transport_id:
                continue

            # Check if we should send to this transport
            should_send = await self._should_send_to_transport(
                transport_id,
                transport,
                event_type,
                data,
                originating_transport_id,
                subscriptions_copy,
                request_details,
            )

            if should_send:
                tasks.append(self._create_send_event_task(transport, transport_id, event_type, data))
                sent_to.add(transport_id)

        # Check if the originating transport should have received it but didn't
        if (
            originating_transport_id
            and originating_transport_id not in sent_to
            and originating_transport_id != exclude_transport_id
        ):
            await self._log_originating_transport_status(
                event_type,
                data,
                originating_transport_id,
                sent_to,
                transports_to_notify,
                subscriptions_copy,
            )

        # Log if no transports received the event
        if not sent_to:
            # Avoid logging warning if it was a STATUS event only meant for origin and origin wasn't subscribed/found
            is_status_for_origin_only = event_type == EventTypes.STATUS and originating_transport_id is not None
            if not is_status_for_origin_only:
                logger.debug(
                    f"No transports subscribed or eligible to receive event {event_type.value} (Request: {data.get('request_id', 'N/A')})"
                )

        # Wait for all send tasks to complete
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            await self._handle_task_results(tasks, results)

    # --- Utility Methods ---

    async def _get_transport(self, transport_id: str) -> Optional["TransportInterface"]:
        """Safely gets a transport adapter by ID."""
        async with self._transports_lock:
            return self._transports.get(transport_id)

    async def _transport_exists(self, transport_id: str) -> bool:
        """Safely checks if a transport ID exists."""
        async with self._transports_lock:
            return transport_id in self._transports

    async def _get_handler_info(self, operation_name: str) -> Optional[Tuple[HandlerFunc, Optional[Permissions]]]:
        """Safely gets handler function and required permission by operation name."""
        async with self._handlers_lock:
            return self._handlers.get(operation_name)

    async def _get_request_info(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Safely gets active request information by ID."""
        async with self._active_requests_lock:
            # Return a copy to prevent modification outside the lock
            request_data = self._active_requests.get(request_id)
            return request_data.copy() if request_data else None

    # --- Shutdown ---

    async def _cancel_active_request_handlers(self) -> None:
        """Cancel all active request handlers during shutdown."""
        logger.info(f"Cancelling active request handlers (timeout: {SHUTDOWN_TIMEOUT}s)...")
        tasks_to_cancel: List[AsyncTask[Any]] = []

        # Get tasks to cancel
        async with self._active_requests_lock:
            active_request_ids = list(self._active_requests.keys())
            for request_id in active_request_ids:
                # Use .get() defensively
                task = self._active_requests.get(request_id, {}).get("task")
                if task and not task.done():
                    tasks_to_cancel.append(task)
                    logger.debug(f"Marking task for request {request_id} for cancellation.")
            self._active_requests.clear()  # Clear requests immediately

        # Cancel tasks and track results
        cancelled_count = 0
        error_count = 0
        if tasks_to_cancel:
            for task in tasks_to_cancel:
                task.cancel()
            # Wait for cancellations with timeout
            results = await asyncio.gather(*tasks_to_cancel, return_exceptions=True)

            for i, result in enumerate(results):
                if isinstance(result, asyncio.CancelledError):
                    cancelled_count += 1
                elif isinstance(result, Exception):
                    error_count += 1
                    task_name = tasks_to_cancel[i].get_name()
                    logger.error(
                        f"Error during handler task cancellation ({task_name}): {result}",
                        exc_info=result,
                    )
                # else: Task completed normally before cancellation took effect

            logger.info(
                f"Attempted to cancel {len(tasks_to_cancel)} active handlers. Confirmed cancelled: {cancelled_count}, Errors: {error_count}."
            )
        else:
            logger.info("No active handlers to cancel.")

    async def _shutdown_transports(self) -> None:
        """Shutdown all registered transports during coordinator shutdown."""
        logger.info(f"Shutting down registered transports (timeout: {SHUTDOWN_TIMEOUT}s)...")
        transport_shutdown_tasks = []
        transports_to_shutdown: List["TransportInterface"] = []

        # Get transports to shutdown
        async with self._transports_lock:
            transports_to_shutdown = list(self._transports.values())
            self._transports.clear()  # Clear transports immediately

        if not transports_to_shutdown:
            logger.info("No transports registered to shut down.")
            return

        # Create shutdown tasks
        for transport in transports_to_shutdown:
            logger.debug(f"Initiating shutdown for transport {transport.get_transport_id()}...")
            transport_shutdown_tasks.append(
                self._loop.create_task(
                    transport.shutdown(),
                    name=f"shutdown-{transport.get_transport_id()}",
                )
            )

        # Wait for all transport shutdowns to complete with timeout
        done, pending = await asyncio.wait(
            transport_shutdown_tasks,
            timeout=SHUTDOWN_TIMEOUT,
            return_when=asyncio.ALL_COMPLETED,
        )

        # Handle pending (timed out) tasks
        for task in pending:
            task_name = task.get_name()
            transport_id = task_name.replace("shutdown-", "") if task_name.startswith("shutdown-") else "unknown"
            logger.warning(f"Transport {transport_id} shutdown timed out after {SHUTDOWN_TIMEOUT}s. Cancelling task.")
            task.cancel()
            # Await cancellation briefly
            with contextlib.suppress(asyncio.CancelledError):
                await task

        # Check for exceptions in completed tasks
        exceptions = []
        for task in done:
            try:
                task.result()
            except Exception as e:
                task_name = task.get_name()
                transport_id = task_name.replace("shutdown-", "") if task_name.startswith("shutdown-") else "unknown"
                logger.error(f"Error shutting down transport {transport_id}: {e}", exc_info=e)
                exceptions.append(e)

        logger.info(
            f"Finished shutting down {len(transports_to_shutdown)} transports ({len(pending)} timed out, {len(exceptions)} errors)."
        )

    async def _clear_coordinator_state(self) -> None:
        """Clear all internal state during shutdown."""
        # Clear handlers and other state
        async with self._handlers_lock:
            self._handlers.clear()
        async with self._transport_capabilities_lock:
            self._transport_capabilities.clear()
        async with self._transport_subscriptions_lock:
            self._transport_subscriptions.clear()

    async def shutdown(self) -> None:
        """Gracefully shuts down the coordinator and all transports."""
        if self._shutdown_event.is_set():
            logger.info("Shutdown already in progress or completed.")
            return

        logger.warning("Initiating ApplicationCoordinator shutdown...")  # Changed to warning
        self._shutdown_event.set()  # Signal shutdown early

        # 1. Cancel all active request handlers
        await self._cancel_active_request_handlers()

        # 2. Shutdown all registered transports
        await self._shutdown_transports()

        # 3. Shutdown discovery service if active
        if self._discovery:
            try:
                await self._discovery.shutdown()
                logger.info("Shutdown coordinator discovery service.")
            except Exception as e:
                logger.error(f"Error shutting down discovery service: {e}")
            self._discovery = None
            self._coordinator_id = None

        # 4. Clear handlers and other state
        await self._clear_coordinator_state()

        # Reset singleton state ONLY if necessary (e.g., for tests)
        # Avoid doing this in production unless specifically required.
        # logger.warning("Resetting ApplicationCoordinator singleton state.")
        # ApplicationCoordinator._instance = None
        # ApplicationCoordinator._initialized = False
        # self._initialized_event.clear()
        # self._shutdown_event.clear() # Keep shutdown signaled

        logger.warning("ApplicationCoordinator shutdown complete.")  # Changed to warning

    # --- Context Manager for Progress Reporting ---

    # Keep sync as it just creates the reporter instance
    def get_progress_reporter(
        self,
        request_id: str,
        operation_name: Optional[str] = None,
        # Add parameters argument to match test mock setup
        parameters: Optional[Dict[str, Any]] = None,
        initial_message: Optional[str] = None,
        initial_details: Optional[Dict[str, Any]] = None,
    ) -> "ProgressReporter":
        """
        Returns a ProgressReporter context manager for a given request.

        Passes the coordinator instance to the reporter. The reporter is responsible
        for fetching necessary info (like operation_name if None) asynchronously
        using the coordinator instance. Includes original parameters in initial details.
        """
        try:
            # Use absolute import path
            from aider_mcp_server.progress_reporter import ProgressReporter
        except ImportError as e:
            logger.error(
                "Failed to import ProgressReporter. Progress reporting unavailable.",
                exc_info=True,
            )
            raise ImportError("ProgressReporter class not found. Ensure it is installed correctly.") from e

        # Prepare initial details, ensuring parameters are included
        final_initial_details = {"parameters": parameters.copy() if parameters else {}}
        if initial_details:
            # Merge, avoiding overwriting 'parameters' key if present in initial_details
            details_to_merge = {k: v for k, v in initial_details.items() if k != "parameters"}
            final_initial_details.update(details_to_merge)

        # Operation name inference logic is removed; reporter should handle if needed.
        op_name = operation_name

        return ProgressReporter(
            coordinator=self,  # Pass the coordinator instance
            request_id=request_id,
            operation_name=op_name,  # Pass potentially None operation name
            initial_message=initial_message,
            # Pass the combined initial details including parameters
            initial_details=final_initial_details,
        )


# Ensure the ProgressReporter class is updated to handle the coordinator instance
# and potentially fetch the operation name asynchronously if needed.
