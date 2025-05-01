import asyncio
import logging
import typing
import uuid # Import uuid
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    NoReturn,
    Optional,
    Protocol,
    Set,
    Tuple,
    Type, # Import Type for __aexit__ type hints
    Union,
    TYPE_CHECKING,
)
import contextlib # Import contextlib for suppress

# Use absolute imports from the package root
from aider_mcp_server.atoms.event_types import EventTypes
from aider_mcp_server.security import SecurityContext, Permissions # Import Permissions

# Import TransportInterface only during type checking
if TYPE_CHECKING:
    # Use absolute import path
    from aider_mcp_server.transport_adapter import TransportInterface
    # Import ProgressReporter for type hint in get_progress_reporter
    from aider_mcp_server.progress_reporter import ProgressReporter


# Define the LoggerProtocol and get_logger_func setup locally
class LoggerProtocol(Protocol):
    def debug(self, message: str, **kwargs: Any) -> None: ...
    def info(self, message: str, **kwargs: Any) -> None: ...
    def warning(self, message: str, **kwargs: Any) -> None: ...
    def error(self, message: str, **kwargs: Any) -> None: ...
    def critical(self, message: str, **kwargs: Any) -> None: ...
    def exception(self, message: str, **kwargs: Any) -> None: ...

get_logger_func: Callable[..., LoggerProtocol]

try:
    # Use absolute import path
    from aider_mcp_server.atoms.logging import get_logger as custom_get_logger
    get_logger_func = typing.cast(Callable[..., LoggerProtocol], custom_get_logger)
except ImportError:
    def fallback_get_logger(name: str, *args: Any, **kwargs: Any) -> LoggerProtocol:
        logger = logging.getLogger(name)
        # Ensure logger has handlers if none are configured (basic setup)
        if not logger.hasHandlers():
             handler = logging.StreamHandler()
             formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
             handler.setFormatter(formatter)
             logger.addHandler(handler)
             # Set a default level if not configured
             if logger.level == logging.NOTSET:
                 logger.setLevel(logging.INFO)
        return logger # type: ignore[return-value]

    get_logger_func = fallback_get_logger


logger = get_logger_func(__name__)

# Type alias for handler functions
HandlerFunc = Callable[
    [str, str, Dict[str, Any], SecurityContext], Coroutine[Any, Any, Dict[str, Any]]
]

# Constants
SHUTDOWN_TIMEOUT = 10.0 # Seconds to wait for tasks/transports during shutdown

class ApplicationCoordinator:
    """
    Central coordinator for managing transports, handlers, and requests.

    This class acts as a singleton, ensuring only one instance manages the
    application state. It routes requests from different transports to the
    appropriate handlers and broadcasts events back to relevant transports.

    It also supports async context management (`async with`). Uses asyncio.Lock
    for internal state synchronization.
    """
    _instance: Optional["ApplicationCoordinator"] = None
    _creation_lock = asyncio.Lock() # Async lock for singleton creation
    _initialized = False # Flag to prevent re-initialization

    def __init__(self) -> None:
        """
        Initializes the ApplicationCoordinator. Should only be called via getInstance.
        """
        if ApplicationCoordinator._initialized:
             logger.warning("ApplicationCoordinator already initialized. Skipping re-initialization.")
             return

        logger.info("Initializing ApplicationCoordinator singleton...")
        self._transports: Dict[str, "TransportInterface"] = {}
        self._handlers: Dict[str, Tuple[HandlerFunc, Optional[Permissions]]] = {}
        self._active_requests: Dict[str, Dict[str, Any]] = {}
        self._transport_capabilities: Dict[str, Set[EventTypes]] = {}
        self._transport_subscriptions: Dict[str, Set[EventTypes]] = {}

        # Locks for async safety
        self._transports_lock = asyncio.Lock()
        self._handlers_lock = asyncio.Lock()
        self._active_requests_lock = asyncio.Lock()
        self._transport_capabilities_lock = asyncio.Lock()
        self._transport_subscriptions_lock = asyncio.Lock()

        # Event loop management
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            # This should generally not happen if initialized within an async context
            logger.error("No running event loop found during ApplicationCoordinator initialization!")
            # Depending on the application, this might be a fatal error.
            # For robustness, we might try creating one, but it's a sign of misuse.
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

        self._initialized_event = asyncio.Event() # Event to signal initialization completion
        self._shutdown_event = asyncio.Event() # Event to signal shutdown

        # Mark as initialized *after* setup is complete
        ApplicationCoordinator._initialized = True
        self._initialized_event.set() # Signal that initialization is done
        logger.info("ApplicationCoordinator initialized successfully.")


    @classmethod
    async def getInstance(cls) -> "ApplicationCoordinator":
        """
        Gets the singleton instance of the ApplicationCoordinator.

        Uses double-checked locking with asyncio.Lock for async-safe initialization.

        Returns:
            ApplicationCoordinator: The singleton instance.
        """
        if cls._instance is None:
            async with cls._creation_lock:
                # Double-check inside the lock
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    async def wait_for_initialization(self) -> None:
        """Waits until the coordinator's asyncio components are fully initialized."""
        await self._initialized_event.wait()

    def is_shutting_down(self) -> bool:
        """Checks if the coordinator shutdown process has been initiated."""
        return self._shutdown_event.is_set()

    # --- Async Context Management ---

    async def __aenter__(self) -> "ApplicationCoordinator":
        """Enter the async context, ensuring initialization."""
        # In case getInstance wasn't awaited, ensure initialization happens.
        # Note: getInstance should ideally be the entry point.
        if not ApplicationCoordinator._initialized:
             async with ApplicationCoordinator._creation_lock:
                  if not ApplicationCoordinator._initialized:
                       ApplicationCoordinator._instance = self # Assign self if called directly
                       self.__init__() # Run init logic if not already done

        await self.wait_for_initialization()
        logger.debug("ApplicationCoordinator context entered.")
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any], # Traceback type is complex, use Any
    ) -> None:
        """Exit the async context, triggering shutdown."""
        logger.debug(f"ApplicationCoordinator context exiting (exception: {exc_type}).")
        await self.shutdown()


    # --- Transport Management ---

    async def register_transport(self, transport_id: str, transport: "TransportInterface") -> None:
        """Registers a new transport adapter."""
        async with self._transports_lock:
            if transport_id in self._transports:
                logger.warning(f"Transport {transport_id} already registered. Overwriting.")
            self._transports[transport_id] = transport
            logger.info(f"Transport registered: {transport_id} ({transport.transport_type})")
        # Update capabilities and default subscriptions (outside transports_lock)
        await self.update_transport_capabilities(transport_id, transport.get_capabilities())

    async def unregister_transport(self, transport_id: str) -> None:
        """Unregisters a transport adapter."""
        transport_exists = False
        async with self._transports_lock:
            if transport_id in self._transports:
                del self._transports[transport_id]
                transport_exists = True
                logger.info(f"Transport unregistered: {transport_id}")
            else:
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
            logger.debug(f"Updated capabilities for {transport_id}: {capabilities}")
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
            logger.debug(f"Updated subscriptions for {transport_id}: {subscriptions}")

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

    async def unsubscribe_from_event_type(self, transport_id: str, event_type: EventTypes) -> None:
        """Unsubscribes a transport from a specific event type."""
        async with self._transport_subscriptions_lock:
            if transport_id in self._transport_subscriptions:
                self._transport_subscriptions[transport_id].discard(event_type)
                logger.debug(f"Transport {transport_id} unsubscribed from {event_type.value}")
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

    async def start_request(
        self,
        request_id: str,
        transport_id: str,
        operation_name: str,
        request_data: Dict[str, Any],
    ) -> None:
        """
        Starts processing a new request received from a transport.

        Validates security, finds the handler, and runs it in the background.
        """
        await self.wait_for_initialization() # Ensure coordinator is ready
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

        # 1. Validate Security Context (outside locks)
        try:
            # Assuming validate_request_security is synchronous or handled by transport
            security_context = transport.validate_request_security(request_data)
            logger.debug(f"Request {request_id} security context validated: User '{security_context.user_id}', Permissions: {security_context.permissions}")
        except Exception as e:
            logger.error(f"Security validation failed for request {request_id} from {transport_id}: {e}", exc_info=True)
            error_result = { "success": False, "error": "Security validation failed", "details": str(e) }
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
            logger.warning(f"No handler found for operation '{operation_name}' (request {request_id}).")
            # Fail request needs original params
            request_params = request_data.get("parameters", {})
            await self.fail_request(
                request_id,
                operation_name,
                "Operation not supported",
                f"No handler registered for operation '{operation_name}'.",
                originating_transport_id=transport_id, # Ensure error goes back
                request_details=request_params,
            )
            return

        handler, required_permission = handler_info
        if required_permission and not security_context.has_permission(required_permission):
            logger.warning(f"Permission denied for operation '{operation_name}' (request {request_id}). User '{security_context.user_id}' lacks permission '{required_permission.name}'.")
            error_result = {
                "success": False,
                "error": "Permission denied",
                "details": f"User does not have the required permission '{required_permission.name}' for operation '{operation_name}'.",
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
            return # Do not proceed with the request

        # 3. Store Active Request State (write active requests lock)
        request_params = request_data.get("parameters", {})
        task: Optional[asyncio.Task] = None
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
                self._run_handler(request_id, transport_id, operation_name, handler, request_params, security_context),
                name=f"handler-{operation_name}-{request_id}"
            )

            self._active_requests[request_id] = {
                "operation": operation_name,
                "transport_id": transport_id,
                "status": "starting",
                "task": task, # Store the created task
                "details": {"parameters": request_params}, # Store original parameters
            }
            logger.debug(f"Request {request_id} state initialized and task created.")


        # 4. Send 'starting' status update (outside active requests lock)
        # This now happens *after* the task is created and stored
        await self.update_request(
            request_id,
            "starting",
            f"Operation '{operation_name}' starting.",
            # No additional details needed here initially beyond params
        )

        # Task is already running


    async def _run_handler(
        self,
        request_id: str,
        transport_id: str,
        operation_name: str,
        handler: HandlerFunc,
        parameters: Dict[str, Any],
        security_context: SecurityContext,
    ) -> None:
        """Wrapper to run the handler, process result/errors, and clean up."""
        result_data: Optional[Dict[str, Any]] = None
        handler_completed_normally = False
        try:
            # Execute the actual handler coroutine
            result_data = await handler(request_id, transport_id, parameters, security_context)
            handler_completed_normally = True # Mark success before potential result processing issues
            logger.info(f"Handler for '{operation_name}' (request {request_id}) completed successfully.")

            # Result data validation/wrapping
            if not isinstance(result_data, dict):
                 logger.warning(f"Handler for '{operation_name}' (request {request_id}) returned non-dict result: {type(result_data)}. Wrapping.")
                 result_data = {"success": True, "result": result_data} # Basic wrapping
            elif "success" not in result_data:
                 result_data["success"] = True # Assume success if key missing

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
                     "result": {"success": False, "error": "Operation cancelled", "details": "The operation was cancelled."},
                 },
                 originating_transport_id=transport_id,
                 request_details=parameters,
            )
            return # Skip normal result sending

        except Exception as e:
            logger.error(f"Handler for '{operation_name}' (request {request_id}) raised an exception: {e}", exc_info=True)
            error_type = type(e).__name__
            # fail_request sends the event AND cleans up state
            await self.fail_request(request_id, operation_name, f"Operation failed: {error_type}", str(e), originating_transport_id=transport_id, request_details=parameters)
            return # fail_request handles cleanup, so return here

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
                     request_details=parameters, # Include original params for context
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
        """
        if self.is_shutting_down(): return # Don't send updates during shutdown

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
        # Merge original parameters with new details for context
        # Perform merge outside the lock
        merged_details = request_info.get("details", {}).copy()
        if details:
            merged_details.update(details)

        # Determine event type based on status
        event_type = EventTypes.PROGRESS if status == "in_progress" else EventTypes.STATUS

        event_data = {
            "type": event_type.value,
            "request_id": request_id,
            "operation": operation_name,
            "status": status,
            "message": message or f"Status updated to {status}",
            "details": merged_details, # Send merged details
        }

        logger.debug(f"Sending update for request {request_id}: Status={status}, Event={event_type.value}")
        await self._send_event_to_transports(
            event_type,
            event_data,
            originating_transport_id=originating_transport_id,
            request_details=merged_details.get("parameters", {}), # Pass original params for filtering
        )

        # Update internal state (briefly acquire lock)
        async with self._active_requests_lock:
            # Check again if request exists, as it might have been cleaned up
            # between the _get_request_info call and acquiring the lock here.
            if request_id in self._active_requests:
                self._active_requests[request_id]["status"] = status
                # Avoid merging details into the stored state unless necessary
                # self._active_requests[request_id]["details"].update(details or {})


    async def fail_request(
        self,
        request_id: str,
        operation_name: str,
        error: str,
        error_details: Optional[Union[str, Dict[str, Any]]] = None, # Allow dict for structured errors
        originating_transport_id: Optional[str] = None,
        request_details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Marks a request as failed, sends an error result event, and cleans up state.
        """
        if self.is_shutting_down(): return # Avoid sending failures during shutdown chaos

        await self.wait_for_initialization()
        logger.error(f"Failing request {request_id} (Operation: {operation_name}): {error} - {error_details}")

        # Check if request exists before sending failure event
        request_info = await self._get_request_info(request_id)
        if not request_info:
            logger.warning(f"Attempted to fail already cleaned up request: {request_id}")
            return

        # If originating_transport_id wasn't provided, try to get it from request_info
        origin_tid = originating_transport_id or request_info.get("transport_id")
        # If request_details weren't provided, try to get them
        req_details = request_details or request_info.get("details", {}).get("parameters")


        details_content = error_details if error_details is not None else "No additional details provided."
        result_data = { "success": False, "error": error, "details": details_content }

        await self._send_event_to_transports(
            EventTypes.TOOL_RESULT,
            {
                "type": EventTypes.TOOL_RESULT.value,
                "request_id": request_id,
                "tool_name": operation_name,
                "result": result_data,
            },
            originating_transport_id=origin_tid,
            request_details=req_details, # Pass original params for context
        )

        # Clean up the request state immediately after reporting failure
        await self._cleanup_request(request_id)


    async def _cleanup_request(self, request_id: str) -> None:
        """Removes a request from the active requests list and cancels its task."""
        task_to_cancel: Optional[asyncio.Task] = None
        async with self._active_requests_lock:
            if request_id in self._active_requests:
                request_info = self._active_requests.pop(request_id) # Remove and get info
                task_to_cancel = request_info.get("task")
                logger.info(f"Cleaned up state for request {request_id}")
            else:
                logger.debug(f"Attempted to clean up already removed request: {request_id}")
                return # Nothing more to do

        # Cancel the task outside the lock
        if task_to_cancel and not task_to_cancel.done():
            task_to_cancel.cancel()
            logger.debug(f"Cancelled task for request {request_id} during cleanup.")
            # Optionally wait for cancellation briefly to allow cleanup within the task
            with contextlib.suppress(asyncio.CancelledError):
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
        if self.is_shutting_down(): return
        await self.wait_for_initialization()
        logger.debug(f"Broadcasting event {event_type.value} (excluding {exclude_transport_id}): {data}")
        await self._send_event_to_transports(
            event_type,
            data,
            exclude_transport_id=exclude_transport_id
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
        if self.is_shutting_down(): return
        await self.wait_for_initialization()

        transport = await self._get_transport(transport_id)
        if transport:
            logger.debug(f"Sending direct event {event_type.value} to transport {transport_id} (Request: {data.get('request_id', 'N/A')})")
            try:
                # Run send_event, but don't block coordinator if it takes time
                # Create task, but don't necessarily await it here unless confirmation is needed
                self._loop.create_task(
                    transport.send_event(event_type, data),
                    name=f"direct-send-{event_type.value}-{transport_id}-{data.get('request_id', uuid.uuid4())}"
                )
            except Exception as e:
                # This catch block might not be effective if send_event is async and raises later
                logger.error(f"Error creating task to send direct event {event_type.value} to transport {transport_id}: {e}", exc_info=True)
        else:
            logger.warning(f"Attempted to send direct event {event_type.value} to non-existent transport {transport_id}")


    async def _send_event_to_transports(
        self,
        event_type: EventTypes,
        data: Dict[str, Any],
        originating_transport_id: Optional[str] = None,
        exclude_transport_id: Optional[str] = None,
        request_details: Optional[Dict[str, Any]] = None, # Original request params for filtering
    ) -> None:
        """
        Internal helper to send an event to relevant transports based on subscriptions.
        """
        if self.is_shutting_down(): return
        await self.wait_for_initialization()

        tasks = []
        sent_to = set()
        transports_to_notify: List[Tuple[str, "TransportInterface"]] = []
        subscriptions_copy: Dict[str, Set[EventTypes]] = {}

        # Acquire locks briefly to get copies of state
        async with self._transports_lock:
            transports_to_notify = list(self._transports.items())
        async with self._transport_subscriptions_lock:
            subscriptions_copy = self._transport_subscriptions.copy()

        # Process transports outside the locks
        for transport_id, transport in transports_to_notify:
            if transport_id == exclude_transport_id:
                continue

            # Check subscription using the copied state
            if event_type in subscriptions_copy.get(transport_id, set()):
                # Check transport-specific filtering (assuming sync method or handled internally)
                should_receive = True
                if hasattr(transport, 'should_receive_event') and callable(transport.should_receive_event):
                     # This call should ideally be quick and non-blocking
                     try:
                          should_receive = transport.should_receive_event(event_type, data, request_details)
                     except Exception as e:
                          logger.error(f"Error calling should_receive_event for transport {transport_id}: {e}", exc_info=True)
                          should_receive = False # Don't send if filter fails

                if not should_receive:
                     logger.debug(f"Transport {transport_id} filtered out event {event_type.value} for request {data.get('request_id', 'N/A')}")
                     continue # Skip sending

                logger.debug(f"Queueing event {event_type.value} for transport {transport_id} (Request: {data.get('request_id', 'N/A')})")
                tasks.append(
                    self._loop.create_task(
                        transport.send_event(event_type, data),
                        name=f"send-{event_type.value}-{transport_id}-{data.get('request_id', uuid.uuid4())}"
                    )
                )
                sent_to.add(transport_id)

        if not sent_to and originating_transport_id:
             logger.warning(f"Event {event_type.value} for request {data.get('request_id', 'N/A')} was intended for {originating_transport_id}, but it was not subscribed, found, or eligible.")
        elif not sent_to:
             logger.debug(f"No transports subscribed or eligible to receive event {event_type.value} (Request: {data.get('request_id', 'N/A')})")

        # Wait for all send tasks to complete (optional, consider fire-and-forget)
        if tasks:
            # Use return_exceptions=True to handle errors without stopping gather
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    task_name = tasks[i].get_name()
                    log_transport_id = "unknown"
                    try: # Best effort parsing of transport ID from task name
                        parts = task_name.split('-')
                        if len(parts) >= 4 and parts[0] == 'send': log_transport_id = parts[2]
                    except Exception: pass
                    logger.error(f"Error sending event via task {task_name} (Transport: {log_transport_id}): {result}", exc_info=result if not isinstance(result, asyncio.CancelledError) else None)


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
            return self._active_requests.get(request_id, {}).copy()


    # --- Shutdown ---

    async def shutdown(self) -> None:
        """Gracefully shuts down the coordinator and all transports."""
        if self._shutdown_event.is_set():
            logger.info("Shutdown already in progress or completed.")
            return

        logger.warning("Initiating ApplicationCoordinator shutdown...") # Changed to warning
        self._shutdown_event.set() # Signal shutdown early

        # 1. Cancel all active request handlers
        logger.info(f"Cancelling active request handlers (timeout: {SHUTDOWN_TIMEOUT}s)...")
        tasks_to_cancel: List[asyncio.Task] = []
        async with self._active_requests_lock:
            active_request_ids = list(self._active_requests.keys())
            for request_id in active_request_ids:
                task = self._active_requests[request_id].get("task")
                if task and not task.done():
                    tasks_to_cancel.append(task)
                    logger.debug(f"Marking task for request {request_id} for cancellation.")
            self._active_requests.clear() # Clear requests immediately

        cancelled_count = 0
        error_count = 0
        if tasks_to_cancel:
            for task in tasks_to_cancel:
                task.cancel()
            # Wait for cancellations with timeout
            results = await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
            # Allow a very short time for cancellation exceptions to propagate if needed
            # await asyncio.sleep(0.01) # Removed, gather should be sufficient

            for i, result in enumerate(results):
                if isinstance(result, asyncio.CancelledError):
                    cancelled_count += 1
                elif isinstance(result, Exception):
                    error_count += 1
                    task_name = tasks_to_cancel[i].get_name()
                    logger.error(f"Error during handler task cancellation ({task_name}): {result}", exc_info=result)
                # else: Task completed normally before cancellation took effect

            logger.info(f"Attempted to cancel {len(tasks_to_cancel)} active handlers. Confirmed cancelled: {cancelled_count}, Errors: {error_count}.")
        else:
            logger.info("No active handlers to cancel.")


        # 2. Shutdown all registered transports
        logger.info(f"Shutting down registered transports (timeout: {SHUTDOWN_TIMEOUT}s)...")
        transport_shutdown_tasks = []
        transports_to_shutdown: List["TransportInterface"] = []
        async with self._transports_lock:
            transports_to_shutdown = list(self._transports.values())
            self._transports.clear() # Clear transports immediately

        if transports_to_shutdown:
            for transport in transports_to_shutdown:
                logger.debug(f"Initiating shutdown for transport {transport.transport_id}...")
                transport_shutdown_tasks.append(
                    self._loop.create_task(transport.shutdown(), name=f"shutdown-{transport.transport_id}")
                )

            # Wait for all transport shutdowns to complete with timeout
            results = await asyncio.gather(*transport_shutdown_tasks, return_exceptions=True)
            for i, result in enumerate(results):
                 if isinstance(result, Exception):
                      transport_id = transports_to_shutdown[i].transport_id
                      logger.error(f"Error shutting down transport {transport_id}: {result}", exc_info=result)
            logger.info(f"Finished shutting down {len(transports_to_shutdown)} transports.")
        else:
             logger.info("No transports registered to shut down.")


        # 3. Clear handlers and other state
        async with self._handlers_lock: self._handlers.clear()
        async with self._transport_capabilities_lock: self._transport_capabilities.clear()
        async with self._transport_subscriptions_lock: self._transport_subscriptions.clear()

        # Reset singleton state ONLY if necessary (e.g., for tests)
        # Avoid doing this in production unless specifically required.
        # logger.warning("Resetting ApplicationCoordinator singleton state.")
        # ApplicationCoordinator._instance = None
        # ApplicationCoordinator._initialized = False
        # self._initialized_event.clear()
        # self._shutdown_event.clear() # Keep shutdown signaled

        logger.warning("ApplicationCoordinator shutdown complete.") # Changed to warning

    # --- Context Manager for Progress Reporting ---

    def get_progress_reporter(
        self,
        request_id: str,
        operation_name: Optional[str] = None,
        initial_message: Optional[str] = None,
        initial_details: Optional[Dict[str, Any]] = None,
    ) -> "ProgressReporter":
        """
        Returns a ProgressReporter context manager for a given request.

        Note: This method remains synchronous as it primarily constructs the
        reporter object. The reporter itself uses async methods of the coordinator.
        However, it needs to access request info, which is now async.
        Let's make this async to fetch info safely.
        """
        # This method needs to be async now to call _get_request_info
        # However, ProgressReporter likely expects sync init.
        # Let's keep it sync for now and fetch info *inside* ProgressReporter's __aenter__
        # or pass the coordinator instance and let the reporter fetch info.

        # Option 1: Keep sync, pass coordinator instance
        # Option 2: Make async (breaks ProgressReporter init potentially)
        # Option 3: Fetch info synchronously (requires separate sync method + lock) - Avoid

        # Let's stick with Option 1: Pass coordinator instance. ProgressReporter needs update.
        # Assuming ProgressReporter is updated to accept coordinator and fetch info itself.

        try:
            # Use absolute import path
            from aider_mcp_server.progress_reporter import ProgressReporter
        except ImportError as e:
             logger.error("Failed to import ProgressReporter. Progress reporting unavailable.", exc_info=True)
             raise ImportError("ProgressReporter class not found. Ensure it is installed correctly.") from e

        # Operation name inference logic remains synchronous for now.
        # If op_name is None, the ProgressReporter will need to fetch it using the coordinator.
        op_name = operation_name
        # if not op_name:
        #     # Cannot call async _get_request_info here.
        #     # The reporter must handle fetching the operation name if needed.
        #     pass

        return ProgressReporter(
            coordinator=self, # Pass the coordinator instance
            request_id=request_id,
            operation_name=op_name, # Pass potentially None operation name
            initial_message=initial_message,
            initial_details=initial_details,
        )

# Ensure the ProgressReporter class is updated to handle the coordinator instance
# and potentially fetch the operation name asynchronously if needed.
