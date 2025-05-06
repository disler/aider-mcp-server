import asyncio
import logging
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Any, Dict, Optional, Type

# Use absolute imports from the package root
# No internal imports needed here currently

# Try importing WebSocket types safely for optional dependency
# Try importing SSE types safely for optional dependency
# Note: SSE transport is handled via ApplicationCoordinator now, not directly here.
# This class might become simpler or be removed if progress is solely managed via coordinator.
# Keeping it for now if direct WebSocket support is still intended alongside coordinator.

# Import ApplicationCoordinator for reporting progress
if TYPE_CHECKING:
    from aider_mcp_server.transport_coordinator import ApplicationCoordinator


# Use standard logging; configuration should be handled elsewhere
logger: logging.Logger = logging.getLogger(__name__)

HEARTBEAT_INTERVAL = 15 # Seconds for WebSocket progress heartbeat (if used directly)

# Added type parameter Self (or "ProgressReporter" for <3.11)
class ProgressReporter(AbstractAsyncContextManager["ProgressReporter"]):
    """
    Reports progress of a long-running operation via ApplicationCoordinator.

    Manages operation state and sends periodic heartbeats.
    Primarily intended for use within operation handlers that need fine-grained progress reporting.
    """

    def __init__(
        self,
        coordinator: "ApplicationCoordinator", # Coordinator is now the first argument and mandatory
        request_id: str,
        operation_name: Optional[str] = None, # Operation name can be optional if fetched later
        initial_message: Optional[str] = None, # Added initial_message
        initial_details: Optional[Dict[str, Any]] = None, # Added initial_details (contains parameters)
    ) -> None:
        """
        Initialize the ProgressReporter.

        Args:
            coordinator: ApplicationCoordinator instance for sending updates.
            request_id: Identifier for the request triggering this operation.
            operation_name: Optional name of the operation being tracked.
            initial_message: Optional initial message to send when the reporter starts.
            initial_details: Optional dictionary of initial details, expected to contain
                             original request parameters under the 'parameters' key.
                             Defaults to an empty dictionary if not provided.
        """
        # Coordinator is now mandatory and passed first
        self.coordinator = coordinator
        self.request_id = request_id
        self.operation_name = operation_name if operation_name else "unknown_operation" # Use a default if None
        # Store initial message and details
        self.initial_message = initial_message
        # Ensure initial_details is always a dict, defaulting to empty if None is passed
        self.initial_details = initial_details if initial_details is not None else {}

        # Remove self.parameters, rely on initial_details containing parameters

        # Added type parameter None
        self._heartbeat_task: Optional[asyncio.Task[None]] = None

        # Mode is always Coordinator now
        self.mode = "Coordinator"

        # Log parameters from initial_details if present
        log_params = self.initial_details.get("parameters", "Not Provided")
        logger.info(f"ProgressReporter initialized for '{self.operation_name}'. Request ID: {self.request_id}. Mode: {self.mode}. Initial Details (Params): {log_params}")


    async def _send_progress(self, status: str, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Sends a progress update message via Coordinator."""
        log_level = logging.ERROR if status == "error" else logging.INFO

        # Details passed here will be merged by coordinator.update_request
        # with the original request parameters stored by the coordinator.
        log_message = f"Progress [{self.operation_name} - {status}]: {message} {details or ''} (Req ID: {self.request_id})"

        # Always send via Coordinator
        try:
            # Coordinator handles routing and merging parameters
            await self.coordinator.update_request(
                request_id=self.request_id,
                status=status,
                message=message,
                details=details, # Pass provided details directly
            )
            logger.debug(f"Sent progress update via Coordinator: {status} - {message} (Req ID: {self.request_id})")
        except Exception as e:
            logger.error(f"Error sending progress update via Coordinator for '{self.operation_name}' (Req ID: {self.request_id}): {e}. Update lost.")
            # Log the original message intended to be sent
            logger.log(log_level, f"{log_message} [Reporting via Log Only - Coordinator Send Failed]")


    async def _heartbeat(self) -> None:
        """Periodically sends 'working_heartbeat' via Coordinator."""
        while True:
            try:
                await asyncio.sleep(HEARTBEAT_INTERVAL)
                # Coordinator Heartbeat (via update_request)
                try:
                    # Pass initial_details to ensure parameters are included by coordinator
                    await self.coordinator.update_request(
                        request_id=self.request_id,
                        status="working_heartbeat",
                        message="Operation in progress...",
                        details=self.initial_details # Pass initial details for context
                    )
                    logger.debug(f"Sent 'working_heartbeat' progress via Coordinator for '{self.operation_name}' (Req ID: {self.request_id})")
                except Exception as e:
                     logger.error(f"Error sending Coordinator heartbeat for '{self.operation_name}' (Req ID: {self.request_id}): {e}")
                     # Decide if coordinator should be considered inactive based on error type

            except asyncio.CancelledError:
                logger.info(f"Heartbeat task for '{self.operation_name}' (Req ID: {self.request_id}) cancelled.")
                break
            except Exception as e:
                logger.error(f"Unexpected error in heartbeat task for '{self.operation_name}' (Req ID: {self.request_id}): {e}")
                # Consider stopping the heartbeat if errors persist
                break


    async def start(self) -> None:
        """Signal the start of the operation and start heartbeat."""
        start_message = self.initial_message if self.initial_message is not None else f"Operation '{self.operation_name}' started."
        logger.info(f"Operation '{self.operation_name}' starting (Req ID: {self.request_id}). Message: '{start_message}'")
        # Send initial progress update using initial_details
        await self._send_progress("starting", start_message, self.initial_details)
        # Start coordinator-based heartbeat task
        if self._heartbeat_task is None:
             self._heartbeat_task = asyncio.create_task(self._heartbeat())
             logger.info(f"Coordinator heartbeat task started for '{self.operation_name}' (Req ID: {self.request_id}).")


    async def update(self, message: str, status: str = "in_progress", details: Optional[Dict[str, Any]] = None) -> None:
        """Send an intermediate progress update."""
        # Pass details directly; coordinator merges parameters.
        # If these details need to *include* the original parameters, the caller should merge them
        # or we should merge with self.initial_details here. Let's assume coordinator handles it.
        await self._send_progress(status, message, details)

    async def complete(self, message: str = "Operation completed successfully.", details: Optional[Dict[str, Any]] = None) -> None:
        """Signal the successful completion of the operation."""
        logger.info(f"Operation '{self.operation_name}' completed (Req ID: {self.request_id}).")
        await self._stop_heartbeat() # Stop heartbeat task
        # Send the final "completed" progress update.
        # Pass details directly; coordinator merges parameters.
        # Merge with initial_details to ensure final status includes original context?
        # Let's pass initial_details merged with specific completion details.
        final_details = self.initial_details.copy()
        if details:
            final_details.update(details)
        await self._send_progress("completed", message, final_details)

    async def error(self, error_message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Signal that the operation failed."""
        logger.error(f"Operation '{self.operation_name}' failed: {error_message} (Req ID: {self.request_id})")
        await self._stop_heartbeat() # Stop heartbeat task
        # Send the final "error" progress update.
        # Pass details directly; coordinator merges parameters.
        # Merge with initial_details to ensure error status includes original context.
        final_details = self.initial_details.copy()
        if details:
            final_details.update(details)
        await self._send_progress("error", error_message, final_details)

    async def _stop_heartbeat(self) -> None:
        """Stops the heartbeat task gracefully."""
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                logger.info(f"Heartbeat task for '{self.operation_name}' (Req ID: {self.request_id}) cancelled successfully.")
            except Exception as e:
                 logger.error(f"Error awaiting cancelled heartbeat task for '{self.operation_name}' (Req ID: {self.request_id}): {e}")
        self._heartbeat_task = None

    async def __aenter__(self) -> "ProgressReporter":
        """Enter the async context, starting the operation."""
        await self.start()
        return self

    # Added type hints for __aexit__ parameters and return type
    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any], # TracebackType not easily available without types module import
    ) -> Optional[bool]:
        """Exit the async context, reporting completion or error."""
        await self._stop_heartbeat() # Ensure heartbeat is stopped first

        if exc_type:
            # An exception occurred within the 'with' block
            error_msg = f"Operation '{self.operation_name}' failed due to unhandled exception: {exc_val}"
            # Use logger.exception for automatic traceback logging
            logger.exception(f"Unhandled exception in context manager for '{self.operation_name}' (Req ID: {self.request_id})")
            # Send the final error progress update, including exception type and initial details.
            error_details = {"exception_type": str(exc_type.__name__)}
            # self.error already merges with initial_details
            await self.error(error_msg, details=error_details)
            return False # Do not suppress the exception
        else:
            # No exception, operation presumed successful by the context manager.
            # Send the final completion progress update. self.complete already merges with initial_details.
            await self.complete()
            return True # Suppress exceptions if __aexit__ completes normally (standard behavior)

