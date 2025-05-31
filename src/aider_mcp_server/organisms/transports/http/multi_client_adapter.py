import uuid

# Type checking imports
from typing import TYPE_CHECKING, Any, Dict, Optional, Set

from aider_mcp_server.atoms.logging.logger import get_logger
from aider_mcp_server.atoms.security.context import SecurityContext
from aider_mcp_server.atoms.types.data_types import ClientRequest, ServerInfo
from aider_mcp_server.atoms.types.event_types import EventTypes
from aider_mcp_server.atoms.types.mcp_types import (
    EventData,
    LoggerProtocol,
    RequestParameters,
)

# For type hint, ITransportAdapter is not strictly needed if AbstractTransportAdapter is sufficient
# from aider_mcp_server.interfaces.transport_adapter import ITransportAdapter
from aider_mcp_server.managers.http_server_manager import HttpServerManager
from aider_mcp_server.molecules.transport.base_adapter import (
    AbstractTransportAdapter,
)
from aider_mcp_server.organisms.transports.http.http_streamable_transport_adapter import (
    HttpStreamableTransportAdapter,
)
from aider_mcp_server.utils.multi_client.port_pool import PortPool

if TYPE_CHECKING:
    from aider_mcp_server.pages.application.coordinator import ApplicationCoordinator


class MultiClientHttpAdapter(AbstractTransportAdapter):
    """
    Coordinates multiple HttpStreamableTransportAdapter instances for different clients.

    This adapter manages the lifecycle of individual HTTP streamable servers,
    allocating ports and integrating with session management. It does not
    directly handle client data streams but delegates to child adapters.
    """

    logger: LoggerProtocol
    _server_manager: HttpServerManager
    _port_pool: PortPool
    _client_adapters: Dict[str, HttpStreamableTransportAdapter]  # client_id -> adapter instance
    _default_child_adapter_config: Dict[str, Any]

    def __init__(
        self,
        coordinator: Optional["ApplicationCoordinator"],
        server_manager: HttpServerManager,
        port_pool: PortPool,
        transport_id: Optional[str] = None,
        default_child_adapter_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,  # Consumes heartbeat_interval for AbstractTransportAdapter
    ):
        """
        Initialize the MultiClientHttpAdapter.

        Args:
            coordinator: The ApplicationCoordinator instance.
            server_manager: Manages client sessions and server metadata.
            port_pool: Manages allocation of network ports.
            transport_id: Optional custom transport ID for this manager adapter.
            default_child_adapter_config: Default configuration for spawned
                                          HttpStreamableTransportAdapter instances.
            **kwargs: Additional arguments for AbstractTransportAdapter (e.g., heartbeat_interval).
        """
        effective_transport_id = transport_id or f"multi_http_adapter_{uuid.uuid4().hex[:8]}"
        # Extract heartbeat_interval for super's init, pass remaining kwargs if any (though usually none for AbstractTransportAdapter)
        heartbeat_interval = kwargs.pop("heartbeat_interval", None)
        super().__init__(
            transport_id=effective_transport_id,
            transport_type="multi_http",
            coordinator=coordinator,
            heartbeat_interval=heartbeat_interval,
            **kwargs,  # Pass any remaining kwargs to super
        )
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}.{self.get_transport_id()}")

        self._server_manager = server_manager
        self._port_pool = port_pool
        self._client_adapters = {}
        self._default_child_adapter_config = default_child_adapter_config or {}

        # Ensure common child adapter settings have defaults if not provided
        self._default_child_adapter_config.setdefault("host", "127.0.0.1")  # noqa: S104
        self._default_child_adapter_config.setdefault("editor_model", "")
        self._default_child_adapter_config.setdefault("current_working_dir", "")
        self._default_child_adapter_config.setdefault("heartbeat_interval", 30.0)
        self._default_child_adapter_config.setdefault("stream_queue_size", 100)

        self.logger.info(
            f"MultiClientHttpAdapter initialized with ID: {self.get_transport_id()} "
            f"and default child config: {self._default_child_adapter_config}"
        )

    async def initialize(self) -> None:
        """Initialize the adapter and register with the coordinator."""
        await super().initialize()
        self.logger.info(f"MultiClientHttpAdapter {self.get_transport_id()} initialized and registered.")

    async def start_listening(self) -> None:
        """
        Starts the adapter. For MultiClientHttpAdapter, this means it's ready
        to handle client connection requests and spawn child adapters.
        It does not start a listening server itself.
        """
        self.logger.info(
            f"MultiClientHttpAdapter {self.get_transport_id()} is active and ready to manage client connections."
        )
        # No direct server to start here; child adapters will have their own.

    async def shutdown(self) -> None:
        """Shutdown all child adapters and release resources."""
        self.logger.info(f"Shutting down MultiClientHttpAdapter {self.get_transport_id()}...")

        client_ids = list(self._client_adapters.keys())
        for client_id in client_ids:
            self.logger.debug(f"Shutting down adapter for client {client_id} during manager shutdown.")
            await self.cleanup_client_connection(client_id)

        if self._client_adapters:  # Should be empty if cleanup_client_connection works
            self.logger.warning(
                f"Not all client adapters were cleaned up: {list(self._client_adapters.keys())}. Forcing clear."
            )
            self._client_adapters.clear()

        await super().shutdown()
        self.logger.info(f"MultiClientHttpAdapter {self.get_transport_id()} shutdown complete.")

    async def _create_and_start_child_adapter(self, client_id: str, port: int) -> HttpStreamableTransportAdapter:
        """Creates, initializes, and starts a HttpStreamableTransportAdapter for a client."""
        child_transport_id = f"http_stream_{client_id}_{uuid.uuid4().hex[:8]}"
        child_config = self._default_child_adapter_config.copy()

        child_adapter_instance = HttpStreamableTransportAdapter(
            coordinator=self._coordinator,
            host=child_config["host"],
            port=port,
            stream_queue_size=child_config["stream_queue_size"],
            editor_model=child_config["editor_model"],
            current_working_dir=child_config["current_working_dir"],
            heartbeat_interval=child_config["heartbeat_interval"],
            transport_id=child_transport_id,
        )

        try:
            await child_adapter_instance.initialize()
            await child_adapter_instance.start_listening()
        except Exception as e:
            # Single exception handler for both initialize and start_listening failures
            await child_adapter_instance.shutdown()
            raise RuntimeError(f"Failed to initialize/start adapter for client {client_id}: {str(e)}") from e

        actual_port = child_adapter_instance.get_actual_port()
        if actual_port is None:
            await child_adapter_instance.shutdown()
            self.logger.error(
                f"Child adapter for {client_id} started but actual port is None. Configured port was {port}."
            )
            raise RuntimeError(f"Failed to get actual port for client {client_id}'s server.")

        self.logger.info(f"Child adapter for client {client_id} started on http://{child_config['host']}:{actual_port}")
        return child_adapter_instance

    async def _update_server_info_for_client(
        self, client_id: str, child_adapter: HttpStreamableTransportAdapter, requested_port: int
    ) -> ServerInfo:
        """Updates and returns the ServerInfo for a client after child adapter setup."""
        server_info_to_update = await self._server_manager.get_client_server_info(client_id)
        if not server_info_to_update:
            self.logger.error(
                f"ServerInfo not found for client {client_id} after session creation. This is unexpected."
            )
            raise RuntimeError(f"ServerInfo consistency error for client {client_id}")

        # Update the ServerInfo object managed by HttpServerManager
        child_config = self._default_child_adapter_config  # Assuming host is from default config
        server_info_to_update.host = child_config["host"]
        server_info_to_update.port = requested_port  # The port requested from pool
        server_info_to_update.actual_port = child_adapter.get_actual_port()  # The port it's actually running on
        server_info_to_update.status = "running"
        server_info_to_update.transport_adapter_id = child_adapter.get_transport_id()
        # Note: This approach relies on the ServerInfo object returned by
        # HttpServerManager being the actual instance stored by the manager,
        # allowing its attributes to be updated directly.
        return server_info_to_update

    async def handle_client_connection(
        self,
        client_id: str,
        workspace_id: Optional[str] = None,
        request_data: Optional[Dict[str, Any]] = None,
    ) -> ServerInfo:
        """
        Handles a new client connection request.

        Args:
            client_id: Unique identifier for the client.
            workspace_id: Optional workspace identifier for the client.
            request_data: Optional additional data from the client request.

        Returns:
            ServerInfo containing details of the client's dedicated server.

        Raises:
            RuntimeError: If port acquisition fails or server setup fails.
            ValueError: If session creation fails (e.g., client already has session, max clients).
        """
        self.logger.info(f"Handling connection request for client_id: {client_id}")

        existing_server_info = await self._server_manager.get_client_server_info(client_id)
        if existing_server_info and existing_server_info.status == "running":
            adapter = self._client_adapters.get(client_id)
            # Check if adapter exists and its actual port matches the stored one
            if (
                adapter
                and adapter.get_actual_port() is not None
                and adapter.get_actual_port() == existing_server_info.actual_port
            ):
                self.logger.info(
                    f"Client {client_id} already has an active server. Returning existing info: {existing_server_info}"
                )
                return existing_server_info
            else:
                self.logger.warning(
                    f"Found existing server info for {client_id} (status: {existing_server_info.status}, "
                    f"port: {existing_server_info.actual_port}), but no matching active adapter "
                    f"(adapter found: {bool(adapter)}, adapter port: {adapter.get_actual_port() if adapter else 'N/A'}). "
                    f"Attempting to clean up and create a new one."
                )
                await self.cleanup_client_connection(client_id)  # Cleanup stale entry

        client_req = ClientRequest(
            client_id=client_id,
            workspace_id=workspace_id,
            request_data=request_data or {},
        )

        # This will raise ValueError if client_id already has a session (handled by HttpServerManager)
        # or RuntimeError if max clients exceeded.
        session_info = await self._server_manager.create_client_session(client_req)
        self.logger.debug(f"Session created for client {client_id}: {session_info.session_id}")

        port = -1
        child_adapter: Optional[HttpStreamableTransportAdapter] = None
        try:
            port = await self._port_pool.acquire_port()
            self.logger.info(f"Acquired port {port} for client {client_id}")

            child_adapter = await self._create_and_start_child_adapter(client_id, port)
            server_info = await self._update_server_info_for_client(client_id, child_adapter, port)

            self._client_adapters[client_id] = child_adapter
            self.logger.info(
                f"Successfully created and started server for client {client_id}. ServerInfo: {server_info}"
            )
            return server_info

        except Exception as e:
            self.logger.error(f"Failed to handle client connection for {client_id}: {e}", exc_info=True)
            
            # Only attempt adapter shutdown if it exists AND we didn't fail during creation/start
            # (i.e., if we got past _create_and_start_child_adapter successfully)
            if child_adapter and client_id in self._client_adapters:
                try:
                    await child_adapter.shutdown()
                except Exception as e_shutdown:
                    self.logger.error(
                        f"Error shutting down partially created adapter for {client_id}: {e_shutdown}", exc_info=True
                    )
                    
            if port != -1:  # If port was acquired, release it
                await self._port_pool.release_port(port)

            # Session was created, so it must be destroyed if setup fails later
            await self._server_manager.destroy_client_session(client_id)

            # Ensure adapter is removed from tracking if it was added before an error
            if client_id in self._client_adapters:
                del self._client_adapters[client_id]

            if isinstance(e, (RuntimeError, ValueError)):  # Re-raise specific errors
                raise
            raise RuntimeError(f"Failed to set up server for client {client_id}: {str(e)}") from e

    async def cleanup_client_connection(self, client_id: str) -> None:
        """Cleans up resources associated with a client connection."""
        self.logger.info(f"Cleaning up connection for client_id: {client_id}")
        adapter = self._client_adapters.pop(client_id, None)
        server_info = await self._server_manager.get_client_server_info(client_id)

        if adapter:
            try:
                await adapter.shutdown()
                self.logger.debug(f"Child adapter for client {client_id} shut down.")
            except Exception as e:
                self.logger.error(f"Error during adapter shutdown for client {client_id}: {e}", exc_info=True)

        if server_info and server_info.port != 0:  # Port 0 is usually an unassigned/initial value
            try:
                await self._port_pool.release_port(server_info.port)
                self.logger.info(f"Port {server_info.port} released for client {client_id}.")
            except Exception as e:
                self.logger.error(f"Error releasing port {server_info.port} for client {client_id}: {e}", exc_info=True)
        elif server_info:
            self.logger.warning(
                f"Port for client {client_id} was {server_info.port}, not releasing. ServerInfo: {server_info}"
            )
        else:
            self.logger.debug(
                f"No server_info found for client {client_id} when trying to release port, or port was 0."
            )

        try:
            await self._server_manager.destroy_client_session(client_id)
            self.logger.debug(f"Session destroyed for client {client_id} in HttpServerManager.")
        except ValueError:  # Raised by destroy_client_session if session not found
            self.logger.debug(f"Session for client {client_id} already destroyed or not found in HttpServerManager.")
        except Exception as e:
            self.logger.error(
                f"Error destroying session for client {client_id} in HttpServerManager: {e}", exc_info=True
            )

    async def get_client_server_info(self, client_id: str) -> Optional[ServerInfo]:
        """Retrieves ServerInfo for a given client."""
        self.logger.debug(f"Fetching server info for client_id: {client_id}")
        return await self._server_manager.get_client_server_info(client_id)

    def get_capabilities(self) -> Set[EventTypes]:
        """Returns capabilities of the MultiClientHttpAdapter itself."""
        return {EventTypes.STATUS}

    async def send_event(self, event_type: EventTypes, data: EventData) -> None:
        """Sends an event related to the MultiClientHttpAdapter itself."""
        self.logger.debug(
            f"send_event called on MultiClientHttpAdapter ({self.get_transport_id()}) "
            f"for event {event_type}. This is for manager-level events or logging."
        )
        # This adapter does not directly communicate with end-clients via this method.
        # If it had its own management interface, event sending logic would go here.

    def should_receive_event(
        self,
        event_type: EventTypes,
        data: EventData,
        request_details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Determines if this MultiClientHttpAdapter instance should process an event."""
        origin_transport_info = data.get("transport_origin", {})
        origin_transport_id = origin_transport_info.get("transport_id")

        if origin_transport_id == self.get_transport_id():
            if event_type == EventTypes.HEARTBEAT and data.get("transport_id") == self.get_transport_id():
                return True  # Allow self-generated heartbeat
            self.logger.debug(
                f"MultiClientHttpAdapter ({self.get_transport_id()}) skipping event {event_type.value} "
                f"as it originated from self and is not a self-generated heartbeat."
            )
            return False

        if event_type in self.get_capabilities():  # e.g., STATUS
            self.logger.debug(
                f"MultiClientHttpAdapter ({self.get_transport_id()}) will process {event_type.value} event from {origin_transport_id}."
            )
            return True

        self.logger.debug(
            f"MultiClientHttpAdapter ({self.get_transport_id()}) deciding not to receive event {event_type.value} "
            f"from {origin_transport_id} as it's not in its capabilities. Child adapters handle client-specific events."
        )
        return False

    def validate_request_security(self, request_details: RequestParameters) -> SecurityContext:
        """Validates security for requests made directly to the MultiClientHttpAdapter (if any)."""
        self.logger.debug(
            f"Performing placeholder security validation for request to MultiClientHttpAdapter ({self.get_transport_id()})."
        )
        return SecurityContext(
            user_id=f"manager_{self.get_transport_id()}",
            permissions=set(),
            is_anonymous=False,
            transport_id=self.get_transport_id(),
        )

    async def handle_sse_request(self, request_details: Any) -> Any:
        self.logger.warning(
            f"handle_sse_request called on MultiClientHttpAdapter ({self.get_transport_id()}), which is not supported."
        )
        raise NotImplementedError("MultiClientHttpAdapter does not handle direct SSE requests.")

    async def handle_message_request(self, request_details: Any) -> Any:
        self.logger.warning(
            f"handle_message_request called on MultiClientHttpAdapter ({self.get_transport_id()}), which is not supported."
        )
        raise NotImplementedError("MultiClientHttpAdapter does not handle direct message requests.")
