import asyncio
import json
import logging
from contextlib import AbstractAsyncContextManager
from typing import Optional, Any, Dict, TYPE_CHECKING, Type, Union

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
    Reports progress of a long-running operation via ApplicationCoordinator or WebSocket.

    Manages operation state and sends periodic heartbeats over WebSocket if provided directly.
    Primarily intended for use within operation handlers that need fine-grained progress reporting.
    """

    def __init__(
        self,
        operation_name: str,
        request_id: str, # Request ID is now mandatory for coordinator interaction
        coordinator: Optional["ApplicationCoordinator"] = None, # Coordinator is preferred
    ) -> None:
        """
        Initialize the ProgressReporter.

        Args:
            operation_name: Name of the operation being tracked.
            request_id: Identifier for the request triggering this operation.
            coordinator: Optional ApplicationCoordinator instance for sending updates.
        """
        if not coordinator:
             raise ValueError("ProgressReporter requires either a coordinator or a websocket connection.")

        self.operation_name = operation_name
        self.request_id = request_id
        self.coordinator = coordinator
        # Ensure websocket is None if websockets library is not installed
        # Added type parameter None
        self._heartbeat_task: Optional[asyncio.Task[None]] = None

        # Determine mode based on available reporting mechanism
        if self.coordinator:
            self.mode = "Coordinator"
        else:
            # Should not happen due to initial check, but defensive
            self.mode = "LogOnly"
            logger.warning(f"ProgressReporter for '{operation_name}' (Req ID: {request_id}) has no active reporting mechanism (Coordinator or WebSocket). Updates will only be logged.")

        logger.info(f"ProgressReporter initialized for '{operation_name}'. Request ID: {request_id}. Mode: {self.mode}")


    async def _send_progress(self, status: str, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Sends a progress update message via Coordinator or WebSocket, or logs it."""
        log_level = logging.ERROR if status == "error" else logging.INFO
        log_message = f"Progress [{self.operation_name} - {status}]: {message} {details or ''} (Req ID: {self.request_id})"

        # Try sending via Coordinator first
        if self.coordinator:
            try:
                # Coordinator handles routing to the correct transport(s)
                await self.coordinator.update_request(
                    request_id=self.request_id,
                    status=status,
                    message=message,
                    details=details,
                )
                logger.debug(f"Sent progress update via Coordinator: {status} - {message} (Req ID: {self.request_id})")
                return # Success
            except Exception as e:
                logger.error(f"Error sending progress update via Coordinator for '{self.operation_name}' (Req ID: {self.request_id}): {e}. Falling back if possible.")
                # Don't disable coordinator reporting on transient errors unless necessary

        # Log if both Coordinator and WebSocket failed or were unavailable
        if not self.coordinator:
            logger.log(log_level, f"{log_message} [Reporting via Log Only]")


    async def _heartbeat(self) -> None:
        """Periodically sends WebSocket ping (if direct WS used) or 'working_heartbeat' via Coordinator."""
        while True:
            try:
                await asyncio.sleep(HEARTBEAT_INTERVAL)
                active_mechanism = False

                # Coordinator Heartbeat (via update_request)
                if self.coordinator:
                    active_mechanism = True
                    try:
                        await self.coordinator.update_request(
                            request_id=self.request_id,
                            status="working_heartbeat",
                            message="Operation in progress...",
                            details=None
                        )
                        logger.debug(f"Sent 'working_heartbeat' progress via Coordinator for '{self.operation_name}' (Req ID: {self.request_id})")
                    except Exception as e:
                         logger.error(f"Error sending Coordinator heartbeat for '{self.operation_name}' (Req ID: {self.request_id}): {e}")
                         # Decide if coordinator should be considered inactive based on error type

                # If no active mechanism remains, stop the heartbeat task
                if not active_mechanism:
                    logger.info(f"No active reporting mechanism, stopping heartbeat task for '{self.operation_name}' (Req ID: {self.request_id}).")
                    break

            except asyncio.CancelledError:
                logger.info(f"Heartbeat task for '{self.operation_name}' (Req ID: {self.request_id}) cancelled.")
                break
            except Exception as e:
                logger.error(f"Unexpected error in heartbeat task for '{self.operation_name}' (Req ID: {self.request_id}): {e}")
                break


    async def start(self) -> None:
        """Signal the start of the operation and start heartbeat if applicable."""
        logger.info(f"Operation '{self.operation_name}' starting (Req ID: {self.request_id}).")
        await self._send_progress("starting", f"Operation '{self.operation_name}' started.")

    async def update(self, message: str, status: str = "in_progress", details: Optional[Dict[str, Any]] = None) -> None:
        """Send an intermediate progress update."""
        await self._send_progress(status, message, details)

    async def complete(self, message: str = "Operation completed successfully.", details: Optional[Dict[str, Any]] = None) -> None:
        """Signal the successful completion of the operation."""
        logger.info(f"Operation '{self.operation_name}' completed (Req ID: {self.request_id}).")
        await self._stop_heartbeat() # Stop direct WS heartbeat if running
        # Note: Final result/completion status is typically sent by the coordinator
        # after the handler returns. Sending 'completed' here might be redundant
        # if using the coordinator pattern. Consider if this method is still needed.
        # If kept, it should likely just log, as the coordinator handles the final TOOL_RESULT.
        # await self._send_progress("completed", message, details) # Commented out - let coordinator handle final state

    async def error(self, error_message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Signal that the operation failed."""
        logger.error(f"Operation '{self.operation_name}' failed: {error_message} (Req ID: {self.request_id})")
        await self._stop_heartbeat() # Stop direct WS heartbeat if running
        # Similar to complete(), the final error state is typically sent by the coordinator.
        # This method might just log the error encountered within the handler.
        # await self._send_progress("error", error_message, details) # Commented out - let coordinator handle final state

    async def _stop_heartbeat(self) -> None:
        """Stops the direct WebSocket heartbeat task gracefully."""
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                logger.info(f"Direct WebSocket heartbeat task for '{self.operation_name}' (Req ID: {self.request_id}) cancelled successfully.")
            except Exception as e:
                 logger.error(f"Error awaiting cancelled direct WS heartbeat task for '{self.operation_name}' (Req ID: {self.request_id}): {e}")
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
        await self._stop_heartbeat() # Ensure direct WS heartbeat is stopped

        if exc_type:
            # An exception occurred within the 'with' block
            error_msg = f"Operation '{self.operation_name}' failed due to unhandled exception: {exc_val} (Req ID: {self.request_id})"
            # Use logger.exception for automatic traceback logging
            logger.exception(f"Unhandled exception in context manager for '{self.operation_name}' (Req ID: {self.request_id})")
            # Let the coordinator handle the final error state when the handler raises the exception.
            # await self.error(error_msg, details={"exception_type": str(exc_type.__name__)}) # Commented out
            return False # Do not suppress the exception
        else:
            # No exception, operation presumed successful by the context manager.
            # Let the coordinator handle the final completion state when the handler returns.
            # await self.complete() # Commented out
            return True # Suppress exceptions if __aexit__ completes normally (standard behavior)
