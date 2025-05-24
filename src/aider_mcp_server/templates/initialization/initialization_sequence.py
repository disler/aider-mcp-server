import asyncio
from typing import Any, Dict, List, Optional

from aider_mcp_server.application_coordinator import ApplicationCoordinator
from aider_mcp_server.atoms.logging.logger import get_logger


class InitializationSequence:
    """
    Manages the application lifecycle and coordinates initialization of all components.
    Wraps ApplicationCoordinator and handles transport configurations, error handling,
    timeouts, and logging.
    """

    def __init__(self) -> None:
        """
        Initialize the InitializationSequence.
        Sets up the ApplicationCoordinator, lock, logger, and initial state.
        """
        self._coordinator = ApplicationCoordinator()
        self._initialized: bool = False
        self._initialization_lock = asyncio.Lock()
        # Use project's logging pattern
        self._logger = get_logger(__name__)

    async def initialize(
        self,
        transport_configs: Optional[List[Dict[str, Any]]] = None,
        timeout: float = 30.0,
    ) -> None:
        """
        Initialize the application with the specified transport configurations.

        Args:
            transport_configs: A list of dictionaries, each configuring a transport.
                               Each dictionary should have a 'name' key for the transport
                               and other keys as specific transport configurations.
            timeout: Maximum time in seconds to wait for initialization steps.
        """
        async with self._initialization_lock:
            if self._initialized:
                self._logger.info("InitializationSequence already initialized.")
                return

            self._logger.info("Starting InitializationSequence initialization...")
            try:
                # Step 1: Initialize the coordinator itself
                self._logger.debug(f"Initializing ApplicationCoordinator with timeout {timeout}s...")
                await asyncio.wait_for(self._coordinator.initialize(), timeout=timeout)
                self._logger.info("ApplicationCoordinator initialized successfully.")

                # Step 2: Initialize transports based on configuration
                if transport_configs:
                    self._logger.info(f"Registering {len(transport_configs)} transport(s)...")
                    for config in transport_configs:
                        # Make a copy to avoid modifying the original list/dict
                        current_config = config.copy()
                        transport_name = current_config.pop("name", None)
                        if not transport_name:
                            self._logger.error("Transport configuration missing 'name' field. Skipping.")
                            continue

                        self._logger.debug(f"Registering transport '{transport_name}' with config: {current_config}")
                        try:
                            # Pass the remaining config items as kwargs
                            await self._coordinator.register_transport(transport_name, **current_config)
                            self._logger.info(f"Transport '{transport_name}' registered successfully.")
                        except Exception as e:
                            self._logger.error(
                                f"Failed to initialize transport '{transport_name}': {e}",
                                exc_info=True,
                            )
                            # As per spec, raise RuntimeError for transport init failure
                            raise RuntimeError(f"Transport initialization failed for '{transport_name}'") from e
                else:
                    self._logger.info("No transport configurations provided.")

                self._initialized = True
                self._logger.info("InitializationSequence initialization completed successfully.")

            except asyncio.TimeoutError as e:
                self._logger.error(
                    f"InitializationSequence initialization timed out after {timeout} seconds.",
                    exc_info=True,
                )
                # Attempt cleanup if initialization times out
                await self._attempt_cleanup_on_failure()
                raise RuntimeError("InitializationSequence sequence timed out") from e
            except Exception as e:
                self._logger.error(f"InitializationSequence initialization failed: {e}", exc_info=True)
                # Attempt cleanup if initialization fails
                await self._attempt_cleanup_on_failure()
                raise RuntimeError("InitializationSequence sequence failed") from e

    async def _attempt_cleanup_on_failure(self) -> None:
        """Helper method to attempt shutdown if initialization fails."""
        self._logger.info("Attempting cleanup after failed initialization...")
        try:
            # Ensure coordinator shutdown is called even if it wasn't fully initialized
            # The coordinator's shutdown should be idempotent or handle partial states.
            await self._coordinator.shutdown()
            self._logger.info("Cleanup after failed initialization completed.")
        except Exception as cleanup_error:
            self._logger.error(
                f"Cleanup after failed initialization also failed: {cleanup_error}",
                exc_info=True,
            )
        finally:
            # Mark as uninitialized regardless of cleanup success
            self._initialized = False

    async def shutdown(self) -> None:
        """
        Shut down the application and release resources.
        """
        async with self._initialization_lock:
            if not self._initialized:
                self._logger.info("InitializationSequence not initialized or already shut down.")
                return

            self._logger.info("Starting InitializationSequence shutdown...")
            try:
                await self._coordinator.shutdown()
                self._initialized = False
                self._logger.info("InitializationSequence shutdown completed successfully.")
            except Exception as e:
                self._logger.error(f"InitializationSequence shutdown failed: {e}", exc_info=True)
                # Mark as uninitialized even if shutdown fails to allow re-initialization attempt
                self._initialized = False
                raise RuntimeError("InitializationSequence shutdown sequence failed") from e
