import uuid
from typing import Optional  # Added Optional
from unittest import mock

import pytest
import pytest_asyncio

from aider_mcp_server.atoms.security.context import SecurityContext
from aider_mcp_server.atoms.types.data_types import ClientRequest, ServerInfo, SessionInfo
from aider_mcp_server.atoms.types.event_types import EventTypes
from aider_mcp_server.atoms.types.mcp_types import EventData, LoggerProtocol
from aider_mcp_server.managers.http_server_manager import HttpServerManager
from aider_mcp_server.organisms.transports.http.http_streamable_transport_adapter import (
    HttpStreamableTransportAdapter,
)
from aider_mcp_server.organisms.transports.http.multi_client_adapter import (
    MultiClientHttpAdapter,
)
from aider_mcp_server.pages.application.coordinator import ApplicationCoordinator
from aider_mcp_server.utils.multi_client.port_pool import PortPool


@pytest.fixture
def mock_logger_factory():
    mock_logger = mock.MagicMock(spec=LoggerProtocol)
    # Configure specific logger methods if they are checked for specific calls
    mock_logger.info = mock.MagicMock()
    mock_logger.debug = mock.MagicMock()
    mock_logger.warning = mock.MagicMock()
    mock_logger.error = mock.MagicMock()
    factory = mock.MagicMock(return_value=mock_logger)
    return factory, mock_logger


@pytest.fixture
def mock_coordinator():
    coordinator = mock.AsyncMock(spec=ApplicationCoordinator)
    coordinator.register_transport_adapter = mock.AsyncMock()
    coordinator.unregister_transport = mock.AsyncMock()
    coordinator.broadcast_event = mock.AsyncMock()
    coordinator.is_shutting_down = False
    return coordinator


@pytest.fixture
def mock_server_manager():
    manager = mock.AsyncMock(spec=HttpServerManager)
    manager._mock_server_infos = {}
    manager._mock_session_infos = {}

    async def _create_client_session(client_request: ClientRequest):
        if client_request.client_id in manager._mock_session_infos:  # pragma: no cover
            # This case is typically handled by HttpServerManager itself,
            # but good for mock robustness if MultiClientHttpAdapter relies on this check.
            # The adapter's logic seems to call get_client_server_info first,
            # so this specific ValueError might not be directly hit from adapter if get_client_server_info returns something.
            # However, HttpServerManager.create_client_session *does* have this check.
            raise ValueError(f"Client {client_request.client_id} already has an active session")

        session_id = f"session_{uuid.uuid4().hex[:8]}"
        session_info = SessionInfo(
            session_id=session_id,
            client_id=client_request.client_id,
            workspace_id=client_request.workspace_id,
            metadata=client_request.request_data,
            status="active",
        )
        manager._mock_session_infos[client_request.client_id] = session_info

        server_id = f"server_{uuid.uuid4().hex[:8]}"
        server_info = ServerInfo(
            server_id=server_id,
            host="127.0.0.1",  # noqa: S104 - Changed from "0.0.0.0"
            port=0,
            status="starting",  # HttpServerManager sets this initially
            workspace_id=client_request.workspace_id,
            active_clients=1,
        )
        manager._mock_server_infos[client_request.client_id] = server_info
        return session_info

    async def _get_client_server_info(client_id: str) -> Optional[ServerInfo]:
        return manager._mock_server_infos.get(client_id)

    async def _destroy_client_session(client_id: str):
        if (
            client_id not in manager._mock_session_infos and client_id not in manager._mock_server_infos
        ):  # pragma: no cover
            # HttpServerManager raises if no session found.
            raise ValueError(f"No active session found for client {client_id}")
        manager._mock_session_infos.pop(client_id, None)
        manager._mock_server_infos.pop(client_id, None)
        return None

    manager.create_client_session = mock.MagicMock(side_effect=_create_client_session)
    manager.get_client_server_info = mock.MagicMock(side_effect=_get_client_server_info)
    manager.destroy_client_session = mock.MagicMock(side_effect=_destroy_client_session)
    return manager


@pytest.fixture
def mock_port_pool():
    pool = mock.AsyncMock(spec=PortPool)
    pool.acquire_port = mock.AsyncMock(return_value=12345)
    pool.release_port = mock.AsyncMock()
    return pool


@pytest.fixture
def mock_child_adapter_instance():
    adapter_instance = mock.AsyncMock(spec=HttpStreamableTransportAdapter)
    adapter_instance.initialize = mock.AsyncMock()
    adapter_instance.start_listening = mock.AsyncMock()
    adapter_instance.shutdown = mock.AsyncMock()
    adapter_instance.get_actual_port = mock.MagicMock(return_value=12345)
    adapter_instance.get_transport_id = mock.MagicMock(return_value=f"http_stream_{uuid.uuid4().hex[:8]}")
    return adapter_instance


@pytest_asyncio.fixture
async def multi_client_adapter_fixture_tuple(
    mock_coordinator, mock_server_manager, mock_port_pool, mock_logger_factory
):
    logger_factory, test_logger = mock_logger_factory
    default_config = {
        "host": "127.0.0.1",
        "editor_model": "test_editor",
        "current_working_dir": "/test/cwd",
        "heartbeat_interval": 10.0,
        "stream_queue_size": 50,
    }
    # Patch get_logger specifically for the module where MultiClientHttpAdapter is defined
    with mock.patch(
        "aider_mcp_server.organisms.transports.http.multi_client_adapter.get_logger", logger_factory
    ) as patched_get_logger:
        adapter = MultiClientHttpAdapter(
            coordinator=mock_coordinator,
            server_manager=mock_server_manager,
            port_pool=mock_port_pool,
            default_child_adapter_config=default_config.copy(),
        )
        # Ensure the adapter's logger is the one from our factory for assertions
        adapter.logger = test_logger  # Directly assign for consistent mocking
        yield adapter, test_logger, patched_get_logger


@pytest.mark.asyncio
class TestMultiClientHttpAdapter:
    async def test_init(self, mock_coordinator, mock_server_manager, mock_port_pool, mock_logger_factory):
        logger_factory, test_logger = mock_logger_factory
        default_config = {"host": "192.168.1.1", "stream_queue_size": 200}

        with mock.patch("aider_mcp_server.organisms.transports.http.multi_client_adapter.get_logger", logger_factory):
            adapter = MultiClientHttpAdapter(
                coordinator=mock_coordinator,
                server_manager=mock_server_manager,
                port_pool=mock_port_pool,
                transport_id="custom_mca_id",
                default_child_adapter_config=default_config.copy(),
                heartbeat_interval=60.0,
            )
            adapter.logger = test_logger  # Ensure logger is our mock

        assert adapter.get_transport_id() == "custom_mca_id"
        assert adapter._default_child_adapter_config["host"] == "192.168.1.1"
        assert adapter._default_child_adapter_config["stream_queue_size"] == 200
        # Check that defaults are filled for unspecified items
        assert "editor_model" in adapter._default_child_adapter_config

        test_logger.info.assert_any_call(
            f"MultiClientHttpAdapter initialized with ID: custom_mca_id "
            f"and default child config: {adapter._default_child_adapter_config}"
        )

    async def test_initialize(self, multi_client_adapter_fixture_tuple, mock_coordinator):
        adapter, test_logger, _ = multi_client_adapter_fixture_tuple
        mock_coordinator.register_transport_adapter.reset_mock()  # Ensure clean state for this test
        await adapter.initialize()
        mock_coordinator.register_transport_adapter.assert_called_once_with(adapter)
        test_logger.info.assert_any_call(
            f"MultiClientHttpAdapter {adapter.get_transport_id()} initialized and registered."
        )

    async def test_start_listening(self, multi_client_adapter_fixture_tuple):
        adapter, test_logger, _ = multi_client_adapter_fixture_tuple
        await adapter.start_listening()
        test_logger.info.assert_any_call(
            f"MultiClientHttpAdapter {adapter.get_transport_id()} is active and ready to manage client connections."
        )

    @mock.patch("aider_mcp_server.organisms.transports.http.multi_client_adapter.HttpStreamableTransportAdapter")
    async def test_handle_client_connection_success(
        self,
        MockChildAdapterClass,
        multi_client_adapter_fixture_tuple,
        mock_server_manager,
        mock_port_pool,
        mock_child_adapter_instance,
    ):
        adapter, test_logger, _ = multi_client_adapter_fixture_tuple
        MockChildAdapterClass.return_value = mock_child_adapter_instance

        client_id = "test_client_conn"
        acquired_port = 12345
        actual_port = 54321
        mock_port_pool.acquire_port.return_value = acquired_port
        mock_child_adapter_instance.get_actual_port.return_value = actual_port
        # child_transport_id = mock_child_adapter_instance.get_transport_id() # Unused variable

        # Ensure manager state is clean for this client
        mock_server_manager._mock_server_infos.pop(client_id, None)
        mock_server_manager._mock_session_infos.pop(client_id, None)

        server_info = await adapter.handle_client_connection(client_id, "ws_id", {"data": "val"})

        mock_server_manager.create_client_session.assert_called_once()
        mock_port_pool.acquire_port.assert_called_once()
        MockChildAdapterClass.assert_called_once()
        mock_child_adapter_instance.initialize.assert_called_once()
        mock_child_adapter_instance.start_listening.assert_called_once()

        assert server_info.actual_port == actual_port
        assert server_info.status == "running"
        assert server_info.transport_adapter_id.startswith(f"http_stream_{client_id}_")
        assert client_id in adapter._client_adapters
        test_logger.info.assert_any_call(f"Acquired port {acquired_port} for client {client_id}")

    async def test_handle_client_connection_existing_running_adapter(
        self, multi_client_adapter_fixture_tuple, mock_server_manager, mock_child_adapter_instance
    ):
        adapter, test_logger, _ = multi_client_adapter_fixture_tuple
        client_id = "existing_client_ok"
        existing_port = 67890
        existing_server_info = ServerInfo(
            server_id="s_exist",
            host="1.2.3.4",
            port=existing_port,
            actual_port=existing_port,
            status="running",
            transport_adapter_id=mock_child_adapter_instance.get_transport_id(),
        )
        mock_server_manager._mock_server_infos[client_id] = existing_server_info
        adapter._client_adapters[client_id] = mock_child_adapter_instance
        mock_child_adapter_instance.get_actual_port.return_value = existing_port

        returned_info = await adapter.handle_client_connection(client_id)
        assert returned_info == existing_server_info
        mock_server_manager.create_client_session.assert_not_called()
        test_logger.info.assert_any_call(
            f"Client {client_id} already has an active server. Returning existing info: {existing_server_info}"
        )

    @mock.patch("aider_mcp_server.organisms.transports.http.multi_client_adapter.HttpStreamableTransportAdapter")
    async def test_handle_client_connection_existing_stale_adapter(
        self,
        MockChildAdapterClass,
        multi_client_adapter_fixture_tuple,
        mock_server_manager,
        mock_port_pool,
        mock_child_adapter_instance,
    ):
        adapter, test_logger, _ = multi_client_adapter_fixture_tuple
        MockChildAdapterClass.return_value = mock_child_adapter_instance

        client_id = "stale_client_data"
        stale_server_info = ServerInfo(server_id="s_stale", host="1.2.3.4", port=111, actual_port=111, status="running")
        mock_server_manager._mock_server_infos[client_id] = stale_server_info
        # No adapter in _client_adapters, or it's different / get_actual_port mismatch

        # Mock cleanup to be called
        adapter.cleanup_client_connection = mock.AsyncMock()
        # Simulate that cleanup allows new session creation
        mock_server_manager.destroy_client_session.side_effect = lambda cid: mock_server_manager._mock_server_infos.pop(
            cid, None
        )

        await adapter.handle_client_connection(client_id)

        adapter.cleanup_client_connection.assert_called_once_with(client_id)
        test_logger.warning.assert_any_call(mock.ANY)  # Stale entry warning
        mock_server_manager.create_client_session.assert_called_once()  # New session created
        MockChildAdapterClass.assert_called_once()  # New adapter created

    async def test_handle_client_connection_session_creation_fails(
        self, multi_client_adapter_fixture_tuple, mock_server_manager
    ):
        adapter, _, _ = multi_client_adapter_fixture_tuple
        mock_server_manager.create_client_session.side_effect = ValueError("Session creation failed")
        with pytest.raises(ValueError, match="Session creation failed"):
            await adapter.handle_client_connection("fail_sess_client")
        adapter._port_pool.acquire_port.assert_not_called()

    async def test_handle_client_connection_port_acquisition_fails(
        self, multi_client_adapter_fixture_tuple, mock_server_manager, mock_port_pool
    ):
        adapter, _, _ = multi_client_adapter_fixture_tuple
        client_id = "fail_port_acq_client"
        # Let create_client_session succeed by ensuring no pre-existing session for this client_id
        mock_server_manager._mock_session_infos.pop(client_id, None)
        mock_server_manager._mock_server_infos.pop(client_id, None)

        mock_port_pool.acquire_port.side_effect = RuntimeError("Port pool empty")
        with pytest.raises(RuntimeError, match="Port pool empty"):
            await adapter.handle_client_connection(client_id)
        mock_server_manager.destroy_client_session.assert_called_once_with(client_id)

    @mock.patch("aider_mcp_server.organisms.transports.http.multi_client_adapter.HttpStreamableTransportAdapter")
    async def test_handle_client_connection_child_adapter_init_fails(
        self,
        MockChildAdapterClass,
        multi_client_adapter_fixture_tuple,
        mock_server_manager,
        mock_port_pool,
        mock_child_adapter_instance,
    ):
        adapter, _, _ = multi_client_adapter_fixture_tuple
        MockChildAdapterClass.return_value = mock_child_adapter_instance
        mock_child_adapter_instance.initialize.side_effect = Exception("Child init error")
        client_id = "fail_child_init_client"
        acquired_port = 12345
        mock_port_pool.acquire_port.return_value = acquired_port

        # Ensure no pre-existing session for this client_id
        mock_server_manager._mock_session_infos.pop(client_id, None)
        mock_server_manager._mock_server_infos.pop(client_id, None)

        with pytest.raises(RuntimeError, match="Child init error"):
            await adapter.handle_client_connection(client_id)

        mock_child_adapter_instance.shutdown.assert_called_once()
        mock_port_pool.release_port.assert_called_once_with(acquired_port)
        mock_server_manager.destroy_client_session.assert_called_once_with(client_id)

    async def test_cleanup_client_connection(
        self, multi_client_adapter_fixture_tuple, mock_server_manager, mock_port_pool, mock_child_adapter_instance
    ):
        adapter, test_logger, _ = multi_client_adapter_fixture_tuple
        client_id = "client_to_cleanup"
        port_to_release = 7777

        adapter._client_adapters[client_id] = mock_child_adapter_instance
        server_info_mock = ServerInfo(server_id="s_cleanup", host="h", port=port_to_release, status="running")
        mock_server_manager._mock_server_infos[client_id] = server_info_mock
        mock_server_manager._mock_session_infos[client_id] = SessionInfo(session_id="sess_cleanup", client_id=client_id)

        await adapter.cleanup_client_connection(client_id)

        mock_child_adapter_instance.shutdown.assert_called_once()
        mock_port_pool.release_port.assert_called_once_with(port_to_release)
        mock_server_manager.destroy_client_session.assert_called_once_with(client_id)
        assert client_id not in adapter._client_adapters
        test_logger.info.assert_any_call(f"Cleaning up connection for client_id: {client_id}")

        # Test cleanup when port is 0 (should not release)
        # Use a new client_id for this scenario to ensure server_info exists
        client_id_port_zero = "client_port_zero_cleanup"
        port_zero_server_info_mock = ServerInfo(server_id="s_cleanup_port_zero", host="h", port=0, status="running")
        mock_server_manager._mock_server_infos[client_id_port_zero] = port_zero_server_info_mock
        # Simulate session info for destroy_client_session to not raise error for this new client
        mock_server_manager._mock_session_infos[client_id_port_zero] = SessionInfo(
            session_id="sess_cleanup_port_zero", client_id=client_id_port_zero
        )

        mock_port_pool.release_port.reset_mock()  # Reset for this specific check
        test_logger.warning.reset_mock()  # Reset warnings to check only for this specific case

        await adapter.cleanup_client_connection(client_id_port_zero)
        mock_port_pool.release_port.assert_not_called()

        print(f"Warning calls for port zero scenario: {test_logger.warning.call_args_list}")
        # Check that warning was called with a message starting with the expected prefix
        found_warning = False
        for call_args in test_logger.warning.call_args_list:
            # Ensure the message is for the correct client_id and port 0
            if call_args[0][0].startswith(f"Port for client {client_id_port_zero} was 0, not releasing."):
                found_warning = True
                break
        assert found_warning, f"Expected warning for client {client_id_port_zero} with port 0 was not logged."

    async def test_shutdown(self, multi_client_adapter_fixture_tuple, mock_coordinator):
        adapter, test_logger, _ = multi_client_adapter_fixture_tuple
        client_ids = ["client_s1", "client_s2"]
        for cid in client_ids:
            adapter._client_adapters[cid] = mock.AsyncMock(spec=HttpStreamableTransportAdapter)
            # Simulate server info and session for cleanup
            adapter._server_manager._mock_server_infos[cid] = ServerInfo(
                server_id=f"s_{cid}", host="h", port=1000 + int(cid[-1]), status="running"
            )
            adapter._server_manager._mock_session_infos[cid] = SessionInfo(session_id=f"sess_{cid}", client_id=cid)

        parent_shutdown_mock = mock.AsyncMock()
        with mock.patch.object(
            MultiClientHttpAdapter, "cleanup_client_connection", wraps=adapter.cleanup_client_connection
        ) as mock_cleanup:
            with mock.patch(
                "aider_mcp_server.molecules.transport.base_adapter.AbstractTransportAdapter.shutdown",
                parent_shutdown_mock,
            ):
                await adapter.shutdown()

        assert mock_cleanup.call_count == len(client_ids)
        assert not adapter._client_adapters
        parent_shutdown_mock.assert_called_once()
        test_logger.info.assert_any_call(f"Shutting down MultiClientHttpAdapter {adapter.get_transport_id()}...")

    async def test_get_client_server_info(self, multi_client_adapter_fixture_tuple, mock_server_manager):
        adapter, _, _ = multi_client_adapter_fixture_tuple
        client_id = "client_info_test"
        expected_info = ServerInfo(server_id="s_info", host="h", port=123, status="running")
        # Clear side_effect so return_value can take effect
        mock_server_manager.get_client_server_info.side_effect = None

        # For a MagicMock mocking an async method, return_value should be a coroutine/future.
        # We create an async function that returns the expected_info and assign its coroutine
        # to the mock's return_value.
        async def _get_expected_info_coroutine():
            return expected_info

        mock_server_manager.get_client_server_info.return_value = _get_expected_info_coroutine()

        info = await adapter.get_client_server_info(client_id)
        assert info == expected_info
        mock_server_manager.get_client_server_info.assert_called_once_with(client_id)

    # Removed @pytest.mark.asyncio as this is a synchronous test
    def test_get_capabilities(self, multi_client_adapter_fixture_tuple):
        adapter, _, _ = multi_client_adapter_fixture_tuple
        assert adapter.get_capabilities() == {EventTypes.STATUS}

    # Removed @pytest.mark.asyncio as this is a synchronous test
    def test_should_receive_event(self, multi_client_adapter_fixture_tuple):
        adapter, test_logger, _ = multi_client_adapter_fixture_tuple
        my_id = adapter.get_transport_id()

        # Event from other transport, matching capabilities (STATUS)
        assert adapter.should_receive_event(EventTypes.STATUS, {"transport_origin": {"transport_id": "other"}})
        test_logger.debug.assert_any_call(
            f"MultiClientHttpAdapter ({my_id}) will process {EventTypes.STATUS.value} event from other."
        )

        # Event from self (non-heartbeat) - should skip
        assert not adapter.should_receive_event(EventTypes.STATUS, {"transport_origin": {"transport_id": my_id}})
        test_logger.debug.assert_any_call(
            f"MultiClientHttpAdapter ({my_id}) skipping event {EventTypes.STATUS.value} "
            f"as it originated from self and is not a self-generated heartbeat."
        )

    # Removed @pytest.mark.asyncio as this is a synchronous test
    def test_validate_request_security(self, multi_client_adapter_fixture_tuple):
        adapter, _, _ = multi_client_adapter_fixture_tuple
        sec_context = adapter.validate_request_security({})
        assert isinstance(sec_context, SecurityContext)
        assert sec_context.user_id == f"manager_{adapter.get_transport_id()}"

    async def test_handle_sse_and_message_request_not_implemented(self, multi_client_adapter_fixture_tuple):
        adapter, _, _ = multi_client_adapter_fixture_tuple
        with pytest.raises(NotImplementedError):
            await adapter.handle_sse_request({})
        with pytest.raises(NotImplementedError):
            await adapter.handle_message_request({})

    @mock.patch("aider_mcp_server.organisms.transports.http.multi_client_adapter.HttpStreamableTransportAdapter")
    async def test_handle_client_connection_child_adapter_start_listening_fails(
        self,
        MockChildAdapterClass,
        multi_client_adapter_fixture_tuple,
        mock_server_manager,
        mock_port_pool,
        mock_child_adapter_instance,
    ):
        adapter, _, _ = multi_client_adapter_fixture_tuple
        MockChildAdapterClass.return_value = mock_child_adapter_instance
        mock_child_adapter_instance.initialize.side_effect = None  # Ensure init succeeds
        mock_child_adapter_instance.start_listening.side_effect = Exception("Child start_listening error")
        client_id = "fail_child_start_listening_client"
        acquired_port = 12345
        mock_port_pool.acquire_port.return_value = acquired_port

        # Ensure no pre-existing session for this client_id
        mock_server_manager._mock_session_infos.pop(client_id, None)
        mock_server_manager._mock_server_infos.pop(client_id, None)
        # Reset call counts for mocks that might have been called in other tests via fixtures
        mock_server_manager.create_client_session.reset_mock()
        mock_server_manager.destroy_client_session.reset_mock()
        mock_port_pool.release_port.reset_mock()
        mock_child_adapter_instance.shutdown.reset_mock()

        with pytest.raises(RuntimeError, match="Child start_listening error"):
            await adapter.handle_client_connection(client_id)

        mock_server_manager.create_client_session.assert_called_once()
        mock_child_adapter_instance.shutdown.assert_called_once()
        mock_port_pool.release_port.assert_called_once_with(acquired_port)
        mock_server_manager.destroy_client_session.assert_called_once_with(client_id)

    @mock.patch("aider_mcp_server.organisms.transports.http.multi_client_adapter.HttpStreamableTransportAdapter")
    async def test_handle_client_connection_get_actual_port_none(
        self,
        MockChildAdapterClass,
        multi_client_adapter_fixture_tuple,
        mock_server_manager,
        mock_port_pool,
        mock_child_adapter_instance,
    ):
        adapter, test_logger, _ = multi_client_adapter_fixture_tuple
        MockChildAdapterClass.return_value = mock_child_adapter_instance
        mock_child_adapter_instance.initialize.side_effect = None
        mock_child_adapter_instance.start_listening.side_effect = None
        mock_child_adapter_instance.get_actual_port.return_value = None  # Simulate failure to get port
        client_id = "fail_get_actual_port_client"
        acquired_port = 12345
        mock_port_pool.acquire_port.return_value = acquired_port

        # Ensure no pre-existing session for this client_id
        mock_server_manager._mock_session_infos.pop(client_id, None)
        mock_server_manager._mock_server_infos.pop(client_id, None)
        mock_server_manager.create_client_session.reset_mock()
        mock_server_manager.destroy_client_session.reset_mock()
        mock_port_pool.release_port.reset_mock()
        mock_child_adapter_instance.shutdown.reset_mock()

        with pytest.raises(RuntimeError, match=f"Failed to get actual port for client {client_id}'s server."):
            await adapter.handle_client_connection(client_id)

        test_logger.error.assert_any_call(
            f"Child adapter for {client_id} started but actual port is None. Configured port was {acquired_port}."
        )
        mock_server_manager.create_client_session.assert_called_once()
        mock_child_adapter_instance.shutdown.assert_called_once()
        mock_port_pool.release_port.assert_called_once_with(acquired_port)
        mock_server_manager.destroy_client_session.assert_called_once_with(client_id)

    @mock.patch("aider_mcp_server.organisms.transports.http.multi_client_adapter.HttpStreamableTransportAdapter")
    async def test_handle_client_connection_server_info_consistency_error(
        self,
        MockChildAdapterClass,
        multi_client_adapter_fixture_tuple,
        mock_server_manager,
        mock_port_pool,
        mock_child_adapter_instance,
    ):
        adapter, test_logger, _ = multi_client_adapter_fixture_tuple
        MockChildAdapterClass.return_value = mock_child_adapter_instance  # This will be used
        client_id = "consistency_error_client"

        # Ensure manager state is clean for this client
        mock_server_manager._mock_server_infos.pop(client_id, None)
        mock_server_manager._mock_session_infos.pop(client_id, None)
        mock_server_manager.create_client_session.reset_mock()
        mock_server_manager.destroy_client_session.reset_mock()
        mock_port_pool.release_port.reset_mock()
        mock_child_adapter_instance.shutdown.reset_mock()

        # First call to get_client_server_info (initial check) should return None
        # create_client_session mock will run and populate its internal _mock_server_infos
        # Second call to get_client_server_info (for update) should also return None to trigger error
        async def _get_none_coroutine():
            return None

        mock_server_manager.get_client_server_info.side_effect = [
            _get_none_coroutine(),  # For initial check for client_id
            _get_none_coroutine(),  # For the check after create_client_session for client_id
        ]

        with pytest.raises(RuntimeError, match=f"ServerInfo consistency error for client {client_id}"):
            await adapter.handle_client_connection(client_id)

        test_logger.error.assert_any_call(
            f"ServerInfo not found for client {client_id} after session creation. This is unexpected."
        )
        mock_server_manager.create_client_session.assert_called_once()
        # Depending on where the error is raised, child adapter might or might not be fully setup/shutdown
        # The error is raised before child_adapter_instance is stored in self._client_adapters
        # but after it's created and started. So shutdown should be called.
        mock_child_adapter_instance.shutdown.assert_called_once()
        mock_port_pool.release_port.assert_called_once()  # Port was acquired
        mock_server_manager.destroy_client_session.assert_called_once_with(client_id)
        # Restore default side_effect for other tests if necessary, though fixtures are per-test

    async def test_cleanup_client_connection_adapter_shutdown_fails(
        self, multi_client_adapter_fixture_tuple, mock_server_manager, mock_port_pool, mock_child_adapter_instance
    ):
        adapter, test_logger, _ = multi_client_adapter_fixture_tuple
        client_id = "cleanup_adapter_fail_client"
        port_to_release = 8888

        adapter._client_adapters[client_id] = mock_child_adapter_instance
        mock_child_adapter_instance.shutdown.side_effect = Exception("Adapter shutdown error")
        server_info_mock = ServerInfo(server_id="s_cleanup_fail", host="h", port=port_to_release, status="running")
        mock_server_manager._mock_server_infos[client_id] = server_info_mock
        mock_server_manager._mock_session_infos[client_id] = SessionInfo(
            session_id="sess_cleanup_fail", client_id=client_id
        )

        # Reset mocks for clean assertion
        mock_port_pool.release_port.reset_mock()
        mock_server_manager.destroy_client_session.reset_mock()

        await adapter.cleanup_client_connection(client_id)

        mock_child_adapter_instance.shutdown.assert_called_once()
        test_logger.error.assert_any_call(
            f"Error during adapter shutdown for client {client_id}: Adapter shutdown error", exc_info=True
        )
        mock_port_pool.release_port.assert_called_once_with(port_to_release)
        mock_server_manager.destroy_client_session.assert_called_once_with(client_id)
        assert client_id not in adapter._client_adapters  # Ensure it's removed despite error

    async def test_send_event_logs_call(self, multi_client_adapter_fixture_tuple):
        adapter, test_logger, _ = multi_client_adapter_fixture_tuple
        event_type = EventTypes.STATUS
        event_data: EventData = {"message": "test"}

        await adapter.send_event(event_type, event_data)

        test_logger.debug.assert_any_call(
            f"send_event called on MultiClientHttpAdapter ({adapter.get_transport_id()}) "
            f"for event {event_type}. This is for manager-level events or logging."
        )

    # Removed @pytest.mark.asyncio as this is a synchronous test
    def test_should_receive_event_self_heartbeat(self, multi_client_adapter_fixture_tuple):
        adapter, test_logger, _ = multi_client_adapter_fixture_tuple
        my_id = adapter.get_transport_id()
        event_data: EventData = {
            "transport_origin": {"transport_id": my_id},
            "transport_id": my_id,  # Crucial for self-generated heartbeat logic
        }

        assert adapter.should_receive_event(EventTypes.HEARTBEAT, event_data)
        # Check if a specific log message for this path exists, if any.
        # The current code path for this returns True before logging a specific "will process" message.
        # It logs "skipping event ... not a self-generated heartbeat" if it's self-originated but NOT a heartbeat.
        # So, no specific positive log for this exact branch, but the True return is key.

    @mock.patch("aider_mcp_server.organisms.transports.http.multi_client_adapter.HttpStreamableTransportAdapter")
    async def test_multiple_clients_lifecycle_integration(
        self, MockChildAdapterClass, multi_client_adapter_fixture_tuple, mock_server_manager, mock_port_pool
    ):
        adapter, _, _ = multi_client_adapter_fixture_tuple
        num_clients = 2
        client_mocks = []

        for i in range(num_clients):
            client_id = f"integ_client_{i}"
            mock_child = mock.AsyncMock(spec=HttpStreamableTransportAdapter)
            mock_child.get_actual_port.return_value = 20000 + i
            mock_child.get_transport_id.return_value = f"child_integ_{i}"
            client_mocks.append(mock_child)

            MockChildAdapterClass.return_value = mock_child
            mock_port_pool.acquire_port.return_value = 10000 + i

            # Ensure manager state is clean for this client
            mock_server_manager._mock_server_infos.pop(client_id, None)
            mock_server_manager._mock_session_infos.pop(client_id, None)
            mock_server_manager.create_client_session.reset_mock()
            mock_server_manager.get_client_server_info.reset_mock()  # Reset for update path

            await adapter.handle_client_connection(client_id, f"ws_integ_{i}")
            assert client_id in adapter._client_adapters

        assert len(adapter._client_adapters) == num_clients

        # Cleanup one client
        client_to_cleanup_id = "integ_client_0"
        # Ensure session/server info exists for cleanup in mock manager
        mock_server_manager._mock_server_infos[client_to_cleanup_id] = ServerInfo(
            server_id="s0", host="h", port=10000, status="running"
        )
        mock_server_manager._mock_session_infos[client_to_cleanup_id] = SessionInfo(
            session_id="sess0", client_id=client_to_cleanup_id
        )

        await adapter.cleanup_client_connection(client_to_cleanup_id)
        client_mocks[0].shutdown.assert_called_once()
        assert client_to_cleanup_id not in adapter._client_adapters
        assert len(adapter._client_adapters) == num_clients - 1

        # Shutdown (will cleanup remaining)
        parent_shutdown_mock = mock.AsyncMock()
        with mock.patch(
            "aider_mcp_server.molecules.transport.base_adapter.AbstractTransportAdapter.shutdown", parent_shutdown_mock
        ):
            await adapter.shutdown()
        client_mocks[1].shutdown.assert_called_once()  # Remaining client
        assert not adapter._client_adapters
