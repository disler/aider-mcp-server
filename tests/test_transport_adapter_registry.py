import asyncio
import os
import sys
from types import ModuleType
from typing import Any, Dict, List, Optional, Set, Type
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
import pytest_asyncio

# Adjust path to import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from aider_mcp_server.atoms.event_types import EventTypes
from aider_mcp_server.interfaces.transport_adapter import ITransportAdapter  # For type hinting
from aider_mcp_server.security import SecurityContext
from aider_mcp_server.transport_adapter import AbstractTransportAdapter
from aider_mcp_server.transport_adapter_registry import TransportAdapterRegistry


# --- Mock ApplicationCoordinator ---
class MockApplicationCoordinator:
    def __init__(self):
        self.register_transport = AsyncMock()
        self.unregister_transport = AsyncMock()
        self.subscribe_to_event_type = AsyncMock()
        self.broadcast_event = AsyncMock()


# --- Mock Adapter Classes ---
class MockAdapterBase(AbstractTransportAdapter):
    """Base for mock adapters to reduce boilerplate."""

    def __init__(self, coordinator: Any, transport_id: str, transport_type: str, **kwargs: Any):
        super().__init__(
            transport_id=transport_id,
            transport_type=transport_type,
            coordinator=coordinator,
            heartbeat_interval=kwargs.get("heartbeat_interval"),
        )
        self.config = kwargs
        self.init_called = False
        self.shutdown_called = False
        self._send_event_mock = AsyncMock()
        self._validate_request_security_mock = MagicMock(
            return_value=SecurityContext(user_id="test_user", permissions=set())
        )

    async def initialize(self) -> None:
        await super().initialize()
        self.init_called = True
        self.logger.info(f"{self.get_transport_type()} initialized with {self.config}")

    async def shutdown(self) -> None:
        await super().shutdown()
        self.shutdown_called = True
        self.logger.info(f"{self.get_transport_type()} shutdown")

    async def send_event(self, event_type: EventTypes, data: Dict[str, Any]) -> None:
        await self._send_event_mock(event_type, data)

    def validate_request_security(self, request_data: Dict[str, Any]) -> SecurityContext:
        return self._validate_request_security_mock(request_data)

    def get_capabilities(self) -> Set[EventTypes]:
        return {EventTypes.STATUS, EventTypes.HEARTBEAT}  # Simplified for tests


class MockAdapterOne(MockAdapterBase):
    TRANSPORT_TYPE_NAME = "mock_one"

    def __init__(self, coordinator: Any, transport_id: str = "mock_one_default_id", **kwargs: Any):
        super().__init__(coordinator, transport_id, self.TRANSPORT_TYPE_NAME, **kwargs)


class MockAdapterTwo(MockAdapterBase):
    TRANSPORT_TYPE_NAME = "mock_two"

    def __init__(self, coordinator: Any, transport_id: str = "mock_two_default_id", **kwargs: Any):
        super().__init__(coordinator, transport_id, self.TRANSPORT_TYPE_NAME, **kwargs)


class AdapterWithoutTypeName(MockAdapterBase):
    # No TRANSPORT_TYPE_NAME
    def __init__(self, coordinator: Any, transport_id: str = "no_type_default_id", **kwargs: Any):
        super().__init__(coordinator, transport_id, "no_type_actual", **kwargs)


class AdapterWithInvalidTypeName(MockAdapterBase):
    TRANSPORT_TYPE_NAME = 123  # Invalid

    def __init__(self, coordinator: Any, transport_id: str = "invalid_type_default_id", **kwargs: Any):
        super().__init__(coordinator, transport_id, "invalid_type_actual", **kwargs)


class FailingInitAdapter(MockAdapterBase):
    TRANSPORT_TYPE_NAME = "failing_init"

    def __init__(self, coordinator: Any, transport_id: str = "failing_init_default_id", **kwargs: Any):
        super().__init__(coordinator, transport_id, self.TRANSPORT_TYPE_NAME, **kwargs)

    async def initialize(self) -> None:
        # Not calling super().initialize() to ensure failure is isolated if super() itself fails
        self.init_called = True  # Mark as called before error
        raise RuntimeError("Simulated initialization failure")


class FailingShutdownAdapter(MockAdapterBase):
    TRANSPORT_TYPE_NAME = "failing_shutdown"

    def __init__(self, coordinator: Any, transport_id: str = "failing_shutdown_default_id", **kwargs: Any):
        super().__init__(coordinator, transport_id, self.TRANSPORT_TYPE_NAME, **kwargs)

    async def shutdown(self) -> None:
        # Not calling super().shutdown() for isolation
        self.shutdown_called = True  # Mark as called before error
        raise RuntimeError("Simulated shutdown failure")


# --- Fixtures ---
@pytest.fixture
def mock_logger_factory():
    logger_mocks = {}

    def factory(name: str, *args: Any, **kwargs: Any) -> MagicMock:
        if name not in logger_mocks:
            # Create a full MagicMock that can be used as a logger
            mock = MagicMock()
            mock.debug = MagicMock()
            mock.info = MagicMock()
            mock.warning = MagicMock()
            mock.error = MagicMock()
            mock.critical = MagicMock()
            mock.exception = MagicMock()
            logger_mocks[name] = mock
        return logger_mocks[name]

    factory.logger_mocks = logger_mocks  # type: ignore[attr-defined]
    return factory


@pytest_asyncio.fixture
async def registry(mock_logger_factory: Any):
    # Patch get_logger_func in both modules where it's imported/used
    with (
        patch("aider_mcp_server.transport_adapter_registry.get_logger_func", mock_logger_factory),
        patch("aider_mcp_server.transport_adapter.get_logger_func", mock_logger_factory),
    ):
        reg = TransportAdapterRegistry(logger_factory=mock_logger_factory)
        yield reg
        # Ensure cleanup for tests that might not call shutdown_all explicitly
        if reg._adapters or reg._adapter_classes:
            await reg.shutdown_all()


@pytest.fixture
def mock_coordinator() -> MockApplicationCoordinator:
    return MockApplicationCoordinator()


# --- Helper Functions ---
def create_mock_module(
    name: str, classes: Optional[Dict[str, Type[AbstractTransportAdapter]]] = None, has_path: bool = True
) -> ModuleType:
    module = ModuleType(name)
    module.__file__ = f"/fake/path/{name.replace('.', '/')}.py"
    if classes:
        for cls_name, cls_obj in classes.items():
            setattr(module, cls_name, cls_obj)
    if has_path:
        # Use a fake path that doesn't exist on the filesystem
        module.__path__ = [f"/fake/path/{name.replace('.', '/')}"]
    return module


# --- Test Cases ---


class TestRegistryInitialization:
    async def test_registry_initialization(self, registry: TransportAdapterRegistry, mock_logger_factory: Any):
        assert registry._adapter_classes == {}
        assert registry._adapters == {}
        assert registry._lock is not None
        # Check if the main registry logger was requested
        assert "aider_mcp_server.transport_adapter_registry" in mock_logger_factory.logger_mocks
        logger = mock_logger_factory.logger_mocks["aider_mcp_server.transport_adapter_registry"]
        logger.info.assert_any_call("TransportAdapterRegistry initialized")


@pytest.mark.asyncio
class TestDiscoverAdapters:
    async def test_discover_adapters_successfully(self, registry: TransportAdapterRegistry, mock_logger_factory: Any):
        # Test direct module discovery by adding classes manually
        # This is a simpler approach that avoids complex mocking
        mock_module = create_mock_module(
            "test_module",
            {
                "Adapter1": MockAdapterOne,
                "Adapter2": MockAdapterTwo,
                "NotAnAdapter": dict,  # Should be skipped
            },
        )

        # Directly test the discovery logic
        registry._discover_adapters_from_module(mock_module)

        # Verify adapters were discovered
        assert MockAdapterOne.TRANSPORT_TYPE_NAME in registry._adapter_classes
        assert registry._adapter_classes[MockAdapterOne.TRANSPORT_TYPE_NAME] == MockAdapterOne
        assert MockAdapterTwo.TRANSPORT_TYPE_NAME in registry._adapter_classes
        assert registry._adapter_classes[MockAdapterTwo.TRANSPORT_TYPE_NAME] == MockAdapterTwo
        assert len(registry._adapter_classes) == 2

        # Verify logging
        logger = mock_logger_factory.logger_mocks["aider_mcp_server.transport_adapter_registry"]
        logger.info.assert_any_call(
            f"Discovered transport adapter: MockAdapterOne for type '{MockAdapterOne.TRANSPORT_TYPE_NAME}'"
        )
        logger.info.assert_any_call(
            f"Discovered transport adapter: MockAdapterTwo for type '{MockAdapterTwo.TRANSPORT_TYPE_NAME}'"
        )

    async def test_discover_adapters_package_with_no_path(
        self, registry: TransportAdapterRegistry, mock_logger_factory: Any
    ):
        mock_pkg_name = "single_file_mock_pkg"
        # Simulate a module that is a single file, not a directory package
        pkg_module_single_file = create_mock_module(mock_pkg_name, {"Adapter1": MockAdapterOne}, has_path=False)

        with patch("importlib.import_module") as mock_import_module:
            mock_import_module.return_value = pkg_module_single_file

            registry.discover_adapters(mock_pkg_name)
            await asyncio.sleep(0)

            assert MockAdapterOne.TRANSPORT_TYPE_NAME in registry._adapter_classes
            assert registry._adapter_classes[MockAdapterOne.TRANSPORT_TYPE_NAME] == MockAdapterOne
            logger = mock_logger_factory.logger_mocks["aider_mcp_server.transport_adapter_registry"]
            logger.warning.assert_any_call(
                f"Package {mock_pkg_name} has no __path__ attribute. Cannot discover modules."
            )
            logger.info.assert_any_call(
                f"Discovered transport adapter: MockAdapterOne for type '{MockAdapterOne.TRANSPORT_TYPE_NAME}'"
            )

    async def test_discover_adapters_package_import_error(
        self, registry: TransportAdapterRegistry, mock_logger_factory: Any
    ):
        mock_pkg_name = "non_existent_pkg"
        with patch(
            "aider_mcp_server.transport_adapter_registry.importlib.import_module",
            side_effect=ImportError(f"No module named {mock_pkg_name}"),
        ):
            registry.discover_adapters(mock_pkg_name)

            assert not registry._adapter_classes
            logger = mock_logger_factory.logger_mocks["aider_mcp_server.transport_adapter_registry"]
            logger.error.assert_any_call(f"Failed to import package: {mock_pkg_name}")

    async def test_discover_adapters_module_import_error(
        self, registry: TransportAdapterRegistry, mock_logger_factory: Any
    ):
        # Test that discover_adapters handles import errors gracefully
        mock_pkg_name = "non_existent_package"

        # Try to discover adapters from a non-existent package
        registry.discover_adapters(mock_pkg_name)

        # Should log error and not crash
        assert not registry._adapter_classes
        logger = mock_logger_factory.logger_mocks["aider_mcp_server.transport_adapter_registry"]
        logger.error.assert_any_call(f"Failed to import package: {mock_pkg_name}")

    async def test_discover_adapters_skips_invalid_adapters(
        self, registry: TransportAdapterRegistry, mock_logger_factory: Any
    ):
        # Test that invalid adapters are skipped during discovery
        mock_module = create_mock_module(
            "test_module",
            {
                "AdapterNoName": AdapterWithoutTypeName,
                "AdapterInvalidName": AdapterWithInvalidTypeName,
                "ValidAdapter": MockAdapterOne,
            },
        )

        # Discover from the module
        registry._discover_adapters_from_module(mock_module)

        # Only valid adapter should be registered
        assert MockAdapterOne.TRANSPORT_TYPE_NAME in registry._adapter_classes
        assert len(registry._adapter_classes) == 1

        # Verify warnings were logged
        logger = mock_logger_factory.logger_mocks["aider_mcp_server.transport_adapter_registry"]
        logger.warning.assert_any_call(
            "Adapter class AdapterWithoutTypeName in module test_module "
            "does not define a valid TRANSPORT_TYPE_NAME string attribute. Skipping."
        )
        logger.warning.assert_any_call(
            "Adapter class AdapterWithInvalidTypeName in module test_module "
            "does not define a valid TRANSPORT_TYPE_NAME string attribute. Skipping."
        )

    async def test_discover_adapters_duplicate_type_name(
        self, registry: TransportAdapterRegistry, mock_logger_factory: Any
    ):
        class DuplicateAdapter(MockAdapterBase):  # Same TRANSPORT_TYPE_NAME as MockAdapterOne
            TRANSPORT_TYPE_NAME = "mock_one"

            def __init__(self, coordinator: Any, transport_id: str = "dup_default_id", **kwargs: Any):
                super().__init__(coordinator, transport_id, self.TRANSPORT_TYPE_NAME, **kwargs)

        # First add MockAdapterOne
        mock_module1 = create_mock_module("test_module1", {"AdapterOriginal": MockAdapterOne})
        registry._discover_adapters_from_module(mock_module1)

        # Then add duplicate
        mock_module2 = create_mock_module("test_module2", {"AdapterDuplicate": DuplicateAdapter})
        registry._discover_adapters_from_module(mock_module2)

        # Should have overwritten with DuplicateAdapter
        assert registry._adapter_classes[MockAdapterOne.TRANSPORT_TYPE_NAME] == DuplicateAdapter

        # Verify warning was logged
        logger = mock_logger_factory.logger_mocks["aider_mcp_server.transport_adapter_registry"]
        logger.warning.assert_any_call(
            "Duplicate transport type 'mock_one' found. Class DuplicateAdapter will overwrite MockAdapterOne."
        )


@pytest.mark.asyncio
class TestInitializeAdapter:
    async def test_initialize_adapter_successfully(
        self, registry: TransportAdapterRegistry, mock_coordinator: MockApplicationCoordinator, mock_logger_factory: Any
    ):
        # First, discover the adapter
        registry._adapter_classes[MockAdapterOne.TRANSPORT_TYPE_NAME] = MockAdapterOne

        config = {"custom_setting": "value", "transport_id": "test_adapter_001"}
        adapter = await registry.initialize_adapter(MockAdapterOne.TRANSPORT_TYPE_NAME, mock_coordinator, config)

        assert isinstance(adapter, MockAdapterOne)
        assert adapter.init_called
        assert adapter.get_transport_id() == "test_adapter_001"
        assert adapter.config.get("custom_setting") == "value"
        assert registry.get_adapter("test_adapter_001") == adapter

        logger = mock_logger_factory.logger_mocks["aider_mcp_server.transport_adapter_registry"]
        logger.info.assert_any_call(f"Initializing transport adapter of type: {MockAdapterOne.TRANSPORT_TYPE_NAME}")
        logger.info.assert_any_call(
            f"Successfully initialized adapter test_adapter_001 of type {MockAdapterOne.TRANSPORT_TYPE_NAME}"
        )

        # Check AbstractTransportAdapter interactions with coordinator
        mock_coordinator.register_transport.assert_called_once_with("test_adapter_001", adapter)
        expected_capabilities = {EventTypes.STATUS, EventTypes.HEARTBEAT}
        calls = [call("test_adapter_001", cap) for cap in expected_capabilities]
        mock_coordinator.subscribe_to_event_type.assert_has_calls(calls, any_order=True)
        assert mock_coordinator.subscribe_to_event_type.call_count == len(expected_capabilities)

        # Check adapter's own logger
        adapter_logger_name = f"aider_mcp_server.transport_adapter.MockAdapterOne.{adapter.get_transport_id()}"
        assert adapter_logger_name in mock_logger_factory.logger_mocks
        adapter_logger = mock_logger_factory.logger_mocks[adapter_logger_name]
        adapter_logger.info.assert_any_call(
            f"{adapter.get_transport_type()} initialized with {{'custom_setting': 'value'}}"
        )

    async def test_initialize_adapter_default_transport_id(
        self, registry: TransportAdapterRegistry, mock_coordinator: MockApplicationCoordinator
    ):
        registry._adapter_classes[MockAdapterOne.TRANSPORT_TYPE_NAME] = MockAdapterOne
        adapter = await registry.initialize_adapter(
            MockAdapterOne.TRANSPORT_TYPE_NAME, mock_coordinator, {}
        )  # No transport_id in config
        assert isinstance(adapter, MockAdapterOne)
        assert adapter.get_transport_id() == "mock_one_default_id"  # Default from MockAdapterOne
        assert registry.get_adapter("mock_one_default_id") == adapter

    async def test_initialize_adapter_unknown_type(
        self, registry: TransportAdapterRegistry, mock_coordinator: MockApplicationCoordinator, mock_logger_factory: Any
    ):
        adapter = await registry.initialize_adapter("unknown_type", mock_coordinator, {})
        assert adapter is None
        logger = mock_logger_factory.logger_mocks["aider_mcp_server.transport_adapter_registry"]
        logger.error.assert_any_call("No adapter class found for transport type: unknown_type")

    async def test_initialize_adapter_init_method_fails(
        self, registry: TransportAdapterRegistry, mock_coordinator: MockApplicationCoordinator, mock_logger_factory: Any
    ):
        registry._adapter_classes[FailingInitAdapter.TRANSPORT_TYPE_NAME] = FailingInitAdapter

        adapter = await registry.initialize_adapter(FailingInitAdapter.TRANSPORT_TYPE_NAME, mock_coordinator, {})
        assert adapter is None
        logger = mock_logger_factory.logger_mocks["aider_mcp_server.transport_adapter_registry"]
        logger.error.assert_any_call(
            f"Failed to initialize adapter of type {FailingInitAdapter.TRANSPORT_TYPE_NAME}: Simulated initialization failure",
            exc_info=True,
        )


@pytest.mark.asyncio
class TestGetAdapter:
    async def test_get_adapter_found_and_not_found(
        self, registry: TransportAdapterRegistry, mock_coordinator: MockApplicationCoordinator
    ):
        registry._adapter_classes[MockAdapterOne.TRANSPORT_TYPE_NAME] = MockAdapterOne
        initialized_adapter = await registry.initialize_adapter(
            MockAdapterOne.TRANSPORT_TYPE_NAME, mock_coordinator, {"transport_id": "found_id"}
        )

        assert registry.get_adapter("found_id") == initialized_adapter
        assert registry.get_adapter("not_found_id") is None


@pytest.mark.asyncio
class TestShutdownAllAdapters:
    async def test_shutdown_all_successfully(
        self, registry: TransportAdapterRegistry, mock_coordinator: MockApplicationCoordinator, mock_logger_factory: Any
    ):
        registry._adapter_classes[MockAdapterOne.TRANSPORT_TYPE_NAME] = MockAdapterOne
        registry._adapter_classes[MockAdapterTwo.TRANSPORT_TYPE_NAME] = MockAdapterTwo

        adapter1_config = {"transport_id": "shutdown_test_id1"}
        adapter2_config = {"transport_id": "shutdown_test_id2"}

        adapter1 = await registry.initialize_adapter(
            MockAdapterOne.TRANSPORT_TYPE_NAME, mock_coordinator, adapter1_config
        )
        adapter2 = await registry.initialize_adapter(
            MockAdapterTwo.TRANSPORT_TYPE_NAME, mock_coordinator, adapter2_config
        )

        assert adapter1 is not None and adapter1.init_called
        assert adapter2 is not None and adapter2.init_called
        assert len(registry._adapters) == 2

        await registry.shutdown_all()

        assert adapter1.shutdown_called
        assert adapter2.shutdown_called
        assert not registry._adapters  # Should be cleared
        assert not registry._adapter_classes  # Should be cleared

        mock_coordinator.unregister_transport.assert_any_call("shutdown_test_id1")
        mock_coordinator.unregister_transport.assert_any_call("shutdown_test_id2")
        assert mock_coordinator.unregister_transport.call_count == 2

        logger = mock_logger_factory.logger_mocks["aider_mcp_server.transport_adapter_registry"]
        logger.info.assert_any_call("Shutting down all transport adapters...")
        logger.info.assert_any_call(f"Shutting down adapter {adapter1.get_transport_id()}")
        logger.info.assert_any_call(f"Shutting down adapter {adapter2.get_transport_id()}")
        logger.info.assert_any_call("All transport adapters shut down and registry cleared.")

        # Check adapter's own logger for shutdown message
        adapter1_logger_name = f"aider_mcp_server.transport_adapter.MockAdapterOne.{adapter1.get_transport_id()}"
        adapter1_logger = mock_logger_factory.logger_mocks[adapter1_logger_name]
        adapter1_logger.info.assert_any_call(f"{adapter1.get_transport_type()} shutdown")

    async def test_shutdown_all_one_adapter_fails(
        self, registry: TransportAdapterRegistry, mock_coordinator: MockApplicationCoordinator, mock_logger_factory: Any
    ):
        registry._adapter_classes[MockAdapterOne.TRANSPORT_TYPE_NAME] = MockAdapterOne
        registry._adapter_classes[FailingShutdownAdapter.TRANSPORT_TYPE_NAME] = FailingShutdownAdapter

        adapter_good_config = {"transport_id": "good_shutdown_id"}
        adapter_fail_config = {"transport_id": "fail_shutdown_id"}

        adapter_good = await registry.initialize_adapter(
            MockAdapterOne.TRANSPORT_TYPE_NAME, mock_coordinator, adapter_good_config
        )
        adapter_fail = await registry.initialize_adapter(
            FailingShutdownAdapter.TRANSPORT_TYPE_NAME, mock_coordinator, adapter_fail_config
        )

        assert adapter_good is not None
        assert adapter_fail is not None

        await registry.shutdown_all()

        assert adapter_good.shutdown_called
        assert adapter_fail.shutdown_called  # Marked true even if it raises error after
        assert not registry._adapters
        assert not registry._adapter_classes

        mock_coordinator.unregister_transport.assert_any_call("good_shutdown_id")
        # FailingShutdownAdapter's super().shutdown() is not called, so unregister_transport might not be called for it
        # depending on where the error occurs. If it's in AbstractTransportAdapter's shutdown *before* unregister,
        # then it won't be called. If it's in the subclass's override after super(), it will.
        # My FailingShutdownAdapter does not call super().shutdown(), so no unregister for it.
        assert mock_coordinator.unregister_transport.call_count == 1

        logger = mock_logger_factory.logger_mocks["aider_mcp_server.transport_adapter_registry"]
        logger.error.assert_any_call(
            f"Error shutting down adapter {adapter_fail.get_transport_id()}: Simulated shutdown failure", exc_info=True
        )
        logger.info.assert_any_call(f"Shutting down adapter {adapter_good.get_transport_id()}")


@pytest.mark.asyncio
class TestAsyncLocking:
    async def test_concurrent_initialization(
        self, registry: TransportAdapterRegistry, mock_coordinator: MockApplicationCoordinator
    ):
        # Discover adapters first
        registry._adapter_classes[MockAdapterOne.TRANSPORT_TYPE_NAME] = MockAdapterOne
        registry._adapter_classes[MockAdapterTwo.TRANSPORT_TYPE_NAME] = MockAdapterTwo

        configs = [
            (MockAdapterOne.TRANSPORT_TYPE_NAME, {"transport_id": "concurrent_id_1"}),
            (MockAdapterTwo.TRANSPORT_TYPE_NAME, {"transport_id": "concurrent_id_2"}),
            (MockAdapterOne.TRANSPORT_TYPE_NAME, {"transport_id": "concurrent_id_3"}),  # Another of type One
        ]

        tasks = [registry.initialize_adapter(ttype, mock_coordinator, cfg) for ttype, cfg in configs]

        results: List[Optional[ITransportAdapter]] = await asyncio.gather(*tasks)

        assert len(results) == 3
        for adapter in results:
            assert adapter is not None
            assert adapter.init_called
            assert adapter.get_transport_id() in registry._adapters
            assert registry._adapters[adapter.get_transport_id()] == adapter

        assert len(registry._adapters) == 3
        # Check coordinator calls (sum over all initializations)
        assert mock_coordinator.register_transport.call_count == 3

        # Each adapter subscribes to 2 capabilities (STATUS, HEARTBEAT)
        assert mock_coordinator.subscribe_to_event_type.call_count == 3 * 2
