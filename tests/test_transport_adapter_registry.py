import asyncio
import importlib
import inspect
import sys
import os
from types import ModuleType
from typing import Any, Dict, Type, Optional, Set, List
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch, call

# Adjust path to import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from aider_mcp_server.transport_adapter import AbstractTransportAdapter
from aider_mcp_server.transport_adapter_registry import TransportAdapterRegistry
from aider_mcp_server.interfaces.transport_adapter import ITransportAdapter # For type hinting
from aider_mcp_server.atoms.event_types import EventTypes
from aider_mcp_server.security import SecurityContext


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
            heartbeat_interval=kwargs.get("heartbeat_interval") 
        )
        self.config = kwargs
        self.init_called = False
        self.shutdown_called = False
        self._send_event_mock = AsyncMock()
        self._validate_request_security_mock = MagicMock(return_value=SecurityContext(is_secure=True))

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
        return {EventTypes.STATUS, EventTypes.HEARTBEAT} # Simplified for tests


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
        self.init_called = True # Mark as called before error
        raise RuntimeError("Simulated initialization failure")


class FailingShutdownAdapter(MockAdapterBase):
    TRANSPORT_TYPE_NAME = "failing_shutdown"

    def __init__(self, coordinator: Any, transport_id: str = "failing_shutdown_default_id", **kwargs: Any):
        super().__init__(coordinator, transport_id, self.TRANSPORT_TYPE_NAME, **kwargs)

    async def shutdown(self) -> None:
        # Not calling super().shutdown() for isolation
        self.shutdown_called = True # Mark as called before error
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
    factory.logger_mocks = logger_mocks # type: ignore[attr-defined]
    return factory


@pytest_asyncio.fixture
async def registry(mock_logger_factory: Any):
    # Patch get_logger_func in both modules where it's imported/used
    with patch('aider_mcp_server.transport_adapter_registry.get_logger_func', mock_logger_factory), \
         patch('aider_mcp_server.transport_adapter.get_logger_func', mock_logger_factory):
        reg = TransportAdapterRegistry(logger_factory=mock_logger_factory)
        yield reg
        # Ensure cleanup for tests that might not call shutdown_all explicitly
        if reg._adapters or reg._adapter_classes:
             await reg.shutdown_all()


@pytest.fixture
def mock_coordinator() -> MockApplicationCoordinator:
    return MockApplicationCoordinator()


# --- Helper Functions ---
def create_mock_module(name: str, classes: Optional[Dict[str, Type[AbstractTransportAdapter]]] = None, has_path: bool = True) -> ModuleType:
    module = ModuleType(name)
    module.__file__ = f"{name.replace('.', '/')}.py"
    if classes:
        for cls_name, cls_obj in classes.items():
            setattr(module, cls_name, cls_obj)
    if has_path:
        module.__path__ = [os.path.dirname(module.__file__)] if module.__file__ else ["dummy_path"]
    return module


# --- Test Cases ---

class TestRegistryInitialization:
    async def test_registry_initialization(self, registry: TransportAdapterRegistry, mock_logger_factory: Any):
        assert registry._adapter_classes == {}
        assert registry._adapters == {}
        assert registry._lock is not None
        # Check if the main registry logger was requested
        assert 'aider_mcp_server.transport_adapter_registry' in mock_logger_factory.logger_mocks
        logger = mock_logger_factory.logger_mocks['aider_mcp_server.transport_adapter_registry']
        logger.info.assert_any_call("TransportAdapterRegistry initialized")


@pytest.mark.asyncio
class TestDiscoverAdapters:
    async def test_discover_adapters_successfully(self, registry: TransportAdapterRegistry, mock_logger_factory: Any):
        mock_pkg_name = "mock_adapters_pkg"
        
        module1 = create_mock_module(f"{mock_pkg_name}.module1", {"Adapter1": MockAdapterOne})
        module2 = create_mock_module(f"{mock_pkg_name}.module2", {"Adapter2": MockAdapterTwo})
        pkg_module = create_mock_module(mock_pkg_name)

        with patch('importlib.import_module') as mock_import_module, \
             patch('pkgutil.iter_modules') as mock_iter_modules:

            def import_module_side_effect(name):
                if name == mock_pkg_name: return pkg_module
                if name == f"{mock_pkg_name}.module1": return module1
                if name == f"{mock_pkg_name}.module2": return module2
                raise ImportError(f"No module named {name}")
            mock_import_module.side_effect = import_module_side_effect
            
            mock_iter_modules.return_value = [
                (None, f"{mock_pkg_name}.module1", False),
                (None, f"{mock_pkg_name}.module2", False),
            ]

            registry.discover_adapters(mock_pkg_name)
            await asyncio.sleep(0)  # Allow ensure_future tasks to complete

            assert MockAdapterOne.TRANSPORT_TYPE_NAME in registry._adapter_classes
            assert registry._adapter_classes[MockAdapterOne.TRANSPORT_TYPE_NAME] == MockAdapterOne
            assert MockAdapterTwo.TRANSPORT_TYPE_NAME in registry._adapter_classes
            assert registry._adapter_classes[MockAdapterTwo.TRANSPORT_TYPE_NAME] == MockAdapterTwo
            
            logger = mock_logger_factory.logger_mocks['aider_mcp_server.transport_adapter_registry']
            logger.info.assert_any_call(f"Discovering transport adapters in package: {mock_pkg_name}")
            logger.info.assert_any_call(f"Discovered transport adapter: MockAdapterOne for type '{MockAdapterOne.TRANSPORT_TYPE_NAME}'")
            logger.info.assert_any_call(f"Discovered transport adapter: MockAdapterTwo for type '{MockAdapterTwo.TRANSPORT_TYPE_NAME}'")
            logger.info.assert_any_call(f"Adapter discovery complete. Found 2 adapter classes.")

    async def test_discover_adapters_package_with_no_path(self, registry: TransportAdapterRegistry, mock_logger_factory: Any):
        mock_pkg_name = "single_file_mock_pkg"
        # Simulate a module that is a single file, not a directory package
        pkg_module_single_file = create_mock_module(mock_pkg_name, {"Adapter1": MockAdapterOne}, has_path=False)
        
        with patch('importlib.import_module') as mock_import_module:
            mock_import_module.return_value = pkg_module_single_file

            registry.discover_adapters(mock_pkg_name)
            await asyncio.sleep(0)

            assert MockAdapterOne.TRANSPORT_TYPE_NAME in registry._adapter_classes
            assert registry._adapter_classes[MockAdapterOne.TRANSPORT_TYPE_NAME] == MockAdapterOne
            logger = mock_logger_factory.logger_mocks['aider_mcp_server.transport_adapter_registry']
            logger.warning.assert_any_call(f"Package {mock_pkg_name} has no __path__ attribute. Cannot discover modules.")
            logger.info.assert_any_call(f"Discovered transport adapter: MockAdapterOne for type '{MockAdapterOne.TRANSPORT_TYPE_NAME}'")

    async def test_discover_adapters_package_import_error(self, registry: TransportAdapterRegistry, mock_logger_factory: Any):
        mock_pkg_name = "non_existent_pkg"
        with patch('importlib.import_module', side_effect=ImportError(f"No module named {mock_pkg_name}")):
            registry.discover_adapters(mock_pkg_name)
            await asyncio.sleep(0)

            assert not registry._adapter_classes
            logger = mock_logger_factory.logger_mocks['aider_mcp_server.transport_adapter_registry']
            logger.error.assert_any_call(f"Failed to import package: {mock_pkg_name}")

    async def test_discover_adapters_module_import_error(self, registry: TransportAdapterRegistry, mock_logger_factory: Any):
        mock_pkg_name = "pkg_with_bad_module"
        pkg_module = create_mock_module(mock_pkg_name)

        with patch('importlib.import_module') as mock_import_module, \
             patch('pkgutil.iter_modules') as mock_iter_modules:
            
            mock_import_module.side_effect = lambda name: pkg_module if name == mock_pkg_name else (_ for _ in ()).throw(ImportError("Module load failed"))
            mock_iter_modules.return_value = [(None, f"{mock_pkg_name}.bad_module", False)]

            registry.discover_adapters(mock_pkg_name)
            await asyncio.sleep(0)
            
            assert not registry._adapter_classes
            logger = mock_logger_factory.logger_mocks['aider_mcp_server.transport_adapter_registry']
            logger.error.assert_any_call(f"Error discovering adapters in module {mock_pkg_name}.bad_module: Module load failed", exc_info=True)

    async def test_discover_adapters_skips_invalid_adapters(self, registry: TransportAdapterRegistry, mock_logger_factory: Any):
        mock_pkg_name = "pkg_with_invalid_adapters"
        module_with_invalid = create_mock_module(
            f"{mock_pkg_name}.invalid_module",
            {
                "AdapterNoName": AdapterWithoutTypeName,
                "AdapterInvalidName": AdapterWithInvalidTypeName,
                "ValidAdapter": MockAdapterOne
            }
        )
        pkg_module = create_mock_module(mock_pkg_name)

        with patch('importlib.import_module') as mock_import_module, \
             patch('pkgutil.iter_modules') as mock_iter_modules:
            
            mock_import_module.side_effect = lambda name: pkg_module if name == mock_pkg_name else module_with_invalid
            mock_iter_modules.return_value = [(None, f"{mock_pkg_name}.invalid_module", False)]

            registry.discover_adapters(mock_pkg_name)
            await asyncio.sleep(0)

            assert MockAdapterOne.TRANSPORT_TYPE_NAME in registry._adapter_classes
            assert len(registry._adapter_classes) == 1 # Only MockAdapterOne should be registered

            logger = mock_logger_factory.logger_mocks['aider_mcp_server.transport_adapter_registry']
            logger.warning.assert_any_call(
                f"Adapter class AdapterWithoutTypeName in module {mock_pkg_name}.invalid_module "
                f"does not define a valid TRANSPORT_TYPE_NAME string attribute. Skipping."
            )
            logger.warning.assert_any_call(
                f"Adapter class AdapterWithInvalidTypeName in module {mock_pkg_name}.invalid_module "
                f"does not define a valid TRANSPORT_TYPE_NAME string attribute. Skipping."
            )
            logger.info.assert_any_call(f"Discovered transport adapter: ValidAdapter for type '{MockAdapterOne.TRANSPORT_TYPE_NAME}'")

    async def test_discover_adapters_duplicate_type_name(self, registry: TransportAdapterRegistry, mock_logger_factory: Any):
        class DuplicateAdapter(MockAdapterBase): # Same TRANSPORT_TYPE_NAME as MockAdapterOne
            TRANSPORT_TYPE_NAME = "mock_one" 
            def __init__(self, coordinator: Any, transport_id: str = "dup_default_id", **kwargs: Any):
                super().__init__(coordinator, transport_id, self.TRANSPORT_TYPE_NAME, **kwargs)

        mock_pkg_name = "pkg_with_duplicates"
        module_dups = create_mock_module(
            f"{mock_pkg_name}.module_dups",
            {"AdapterOriginal": MockAdapterOne, "AdapterDuplicate": DuplicateAdapter}
        )
        pkg_module = create_mock_module(mock_pkg_name)

        with patch('importlib.import_module') as mock_import_module, \
             patch('pkgutil.iter_modules') as mock_iter_modules:
            
            mock_import_module.side_effect = lambda name: pkg_module if name == mock_pkg_name else module_dups
            mock_iter_modules.return_value = [(None, f"{mock_pkg_name}.module_dups", False)]
            
            # Mock inspect.getmembers to control order of discovery for predictability
            original_getmembers = inspect.getmembers
            def ordered_getmembers(obj, pred):
                if obj is module_dups:
                    return [("AdapterOriginal", MockAdapterOne), ("AdapterDuplicate", DuplicateAdapter)]
                return original_getmembers(obj, pred)
            
            with patch('inspect.getmembers', side_effect=ordered_getmembers):
                registry.discover_adapters(mock_pkg_name)
                await asyncio.sleep(0)

            assert registry._adapter_classes[MockAdapterOne.TRANSPORT_TYPE_NAME] == DuplicateAdapter # Overwritten
            logger = mock_logger_factory.logger_mocks['aider_mcp_server.transport_adapter_registry']
            logger.warning.assert_any_call(
                f"Duplicate transport type '{MockAdapterOne.TRANSPORT_TYPE_NAME}' found. "
                f"Class DuplicateAdapter will overwrite AdapterOriginal."
            )
            logger.info.assert_any_call(f"Discovered transport adapter: AdapterOriginal for type '{MockAdapterOne.TRANSPORT_TYPE_NAME}'")
            logger.info.assert_any_call(f"Discovered transport adapter: AdapterDuplicate for type '{DuplicateAdapter.TRANSPORT_TYPE_NAME}'")


@pytest.mark.asyncio
class TestInitializeAdapter:
    async def test_initialize_adapter_successfully(self, registry: TransportAdapterRegistry, mock_coordinator: MockApplicationCoordinator, mock_logger_factory: Any):
        # First, discover the adapter
        registry._adapter_classes[MockAdapterOne.TRANSPORT_TYPE_NAME] = MockAdapterOne
        
        config = {"custom_setting": "value", "transport_id": "test_adapter_001"}
        adapter = await registry.initialize_adapter(MockAdapterOne.TRANSPORT_TYPE_NAME, mock_coordinator, config)

        assert isinstance(adapter, MockAdapterOne)
        assert adapter.init_called
        assert adapter.get_transport_id() == "test_adapter_001"
        assert adapter.config.get("custom_setting") == "value"
        assert registry.get_adapter("test_adapter_001") == adapter

        logger = mock_logger_factory.logger_mocks['aider_mcp_server.transport_adapter_registry']
        logger.info.assert_any_call(f"Initializing transport adapter of type: {MockAdapterOne.TRANSPORT_TYPE_NAME}")
        logger.info.assert_any_call(f"Successfully initialized adapter test_adapter_001 of type {MockAdapterOne.TRANSPORT_TYPE_NAME}")

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
        adapter_logger.info.assert_any_call(f"{adapter.get_transport_type()} initialized with {{'custom_setting': 'value'}}")


    async def test_initialize_adapter_default_transport_id(self, registry: TransportAdapterRegistry, mock_coordinator: MockApplicationCoordinator):
        registry._adapter_classes[MockAdapterOne.TRANSPORT_TYPE_NAME] = MockAdapterOne
        adapter = await registry.initialize_adapter(MockAdapterOne.TRANSPORT_TYPE_NAME, mock_coordinator, {}) # No transport_id in config
        assert isinstance(adapter, MockAdapterOne)
        assert adapter.get_transport_id() == "mock_one_default_id" # Default from MockAdapterOne
        assert registry.get_adapter("mock_one_default_id") == adapter

    async def test_initialize_adapter_unknown_type(self, registry: TransportAdapterRegistry, mock_coordinator: MockApplicationCoordinator, mock_logger_factory: Any):
        adapter = await registry.initialize_adapter("unknown_type", mock_coordinator, {})
        assert adapter is None
        logger = mock_logger_factory.logger_mocks['aider_mcp_server.transport_adapter_registry']
        logger.error.assert_any_call("No adapter class found for transport type: unknown_type")

    async def test_initialize_adapter_init_method_fails(self, registry: TransportAdapterRegistry, mock_coordinator: MockApplicationCoordinator, mock_logger_factory: Any):
        registry._adapter_classes[FailingInitAdapter.TRANSPORT_TYPE_NAME] = FailingInitAdapter
        
        adapter = await registry.initialize_adapter(FailingInitAdapter.TRANSPORT_TYPE_NAME, mock_coordinator, {})
        assert adapter is None
        logger = mock_logger_factory.logger_mocks['aider_mcp_server.transport_adapter_registry']
        logger.error.assert_any_call(
            f"Failed to initialize adapter of type {FailingInitAdapter.TRANSPORT_TYPE_NAME}: Simulated initialization failure",
            exc_info=True
        )


@pytest.mark.asyncio
class TestGetAdapter:
    async def test_get_adapter_found_and_not_found(self, registry: TransportAdapterRegistry, mock_coordinator: MockApplicationCoordinator):
        registry._adapter_classes[MockAdapterOne.TRANSPORT_TYPE_NAME] = MockAdapterOne
        initialized_adapter = await registry.initialize_adapter(
            MockAdapterOne.TRANSPORT_TYPE_NAME, 
            mock_coordinator, 
            {"transport_id": "found_id"}
        )
        
        assert registry.get_adapter("found_id") == initialized_adapter
        assert registry.get_adapter("not_found_id") is None


@pytest.mark.asyncio
class TestShutdownAllAdapters:
    async def test_shutdown_all_successfully(self, registry: TransportAdapterRegistry, mock_coordinator: MockApplicationCoordinator, mock_logger_factory: Any):
        registry._adapter_classes[MockAdapterOne.TRANSPORT_TYPE_NAME] = MockAdapterOne
        registry._adapter_classes[MockAdapterTwo.TRANSPORT_TYPE_NAME] = MockAdapterTwo

        adapter1_config = {"transport_id": "shutdown_test_id1"}
        adapter2_config = {"transport_id": "shutdown_test_id2"}
        
        adapter1 = await registry.initialize_adapter(MockAdapterOne.TRANSPORT_TYPE_NAME, mock_coordinator, adapter1_config)
        adapter2 = await registry.initialize_adapter(MockAdapterTwo.TRANSPORT_TYPE_NAME, mock_coordinator, adapter2_config)

        assert adapter1 is not None and adapter1.init_called
        assert adapter2 is not None and adapter2.init_called
        assert len(registry._adapters) == 2

        await registry.shutdown_all()

        assert adapter1.shutdown_called
        assert adapter2.shutdown_called
        assert not registry._adapters  # Should be cleared
        assert not registry._adapter_classes # Should be cleared

        mock_coordinator.unregister_transport.assert_any_call("shutdown_test_id1")
        mock_coordinator.unregister_transport.assert_any_call("shutdown_test_id2")
        assert mock_coordinator.unregister_transport.call_count == 2
        
        logger = mock_logger_factory.logger_mocks['aider_mcp_server.transport_adapter_registry']
        logger.info.assert_any_call("Shutting down all transport adapters...")
        logger.info.assert_any_call(f"Shutting down adapter {adapter1.get_transport_id()}")
        logger.info.assert_any_call(f"Shutting down adapter {adapter2.get_transport_id()}")
        logger.info.assert_any_call("All transport adapters shut down and registry cleared.")

        # Check adapter's own logger for shutdown message
        adapter1_logger_name = f"aider_mcp_server.transport_adapter.MockAdapterOne.{adapter1.get_transport_id()}"
        adapter1_logger = mock_logger_factory.logger_mocks[adapter1_logger_name]
        adapter1_logger.info.assert_any_call(f"{adapter1.get_transport_type()} shutdown")


    async def test_shutdown_all_one_adapter_fails(self, registry: TransportAdapterRegistry, mock_coordinator: MockApplicationCoordinator, mock_logger_factory: Any):
        registry._adapter_classes[MockAdapterOne.TRANSPORT_TYPE_NAME] = MockAdapterOne
        registry._adapter_classes[FailingShutdownAdapter.TRANSPORT_TYPE_NAME] = FailingShutdownAdapter

        adapter_good_config = {"transport_id": "good_shutdown_id"}
        adapter_fail_config = {"transport_id": "fail_shutdown_id"}

        adapter_good = await registry.initialize_adapter(MockAdapterOne.TRANSPORT_TYPE_NAME, mock_coordinator, adapter_good_config)
        adapter_fail = await registry.initialize_adapter(FailingShutdownAdapter.TRANSPORT_TYPE_NAME, mock_coordinator, adapter_fail_config)

        assert adapter_good is not None
        assert adapter_fail is not None

        await registry.shutdown_all()

        assert adapter_good.shutdown_called
        assert adapter_fail.shutdown_called # Marked true even if it raises error after
        assert not registry._adapters
        assert not registry._adapter_classes

        mock_coordinator.unregister_transport.assert_any_call("good_shutdown_id")
        # FailingShutdownAdapter's super().shutdown() is not called, so unregister_transport might not be called for it
        # depending on where the error occurs. If it's in AbstractTransportAdapter's shutdown *before* unregister,
        # then it won't be called. If it's in the subclass's override after super(), it will.
        # My FailingShutdownAdapter does not call super().shutdown(), so no unregister for it.
        assert mock_coordinator.unregister_transport.call_count == 1 


        logger = mock_logger_factory.logger_mocks['aider_mcp_server.transport_adapter_registry']
        logger.error.assert_any_call(
            f"Error shutting down adapter {adapter_fail.get_transport_id()}: Simulated shutdown failure",
            exc_info=True
        )
        logger.info.assert_any_call(f"Shutting down adapter {adapter_good.get_transport_id()}")


@pytest.mark.asyncio
class TestAsyncLocking:
    async def test_concurrent_initialization(self, registry: TransportAdapterRegistry, mock_coordinator: MockApplicationCoordinator):
        # Discover adapters first
        registry._adapter_classes[MockAdapterOne.TRANSPORT_TYPE_NAME] = MockAdapterOne
        registry._adapter_classes[MockAdapterTwo.TRANSPORT_TYPE_NAME] = MockAdapterTwo

        configs = [
            (MockAdapterOne.TRANSPORT_TYPE_NAME, {"transport_id": "concurrent_id_1"}),
            (MockAdapterTwo.TRANSPORT_TYPE_NAME, {"transport_id": "concurrent_id_2"}),
            (MockAdapterOne.TRANSPORT_TYPE_NAME, {"transport_id": "concurrent_id_3"}), # Another of type One
        ]

        tasks = [
            registry.initialize_adapter(ttype, mock_coordinator, cfg)
            for ttype, cfg in configs
        ]
        
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
import asyncio
import importlib
import inspect
import sys
import os
import typing
from types import ModuleType
from typing import Any, Dict, Type, Optional, Set, List
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch, call

# Adjust path to import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from aider_mcp_server.transport_adapter import AbstractTransportAdapter
from aider_mcp_server.transport_adapter_registry import TransportAdapterRegistry
from aider_mcp_server.interfaces.transport_adapter import ITransportAdapter
from aider_mcp_server.atoms.event_types import EventTypes
from aider_mcp_server.security import SecurityContext
from aider_mcp_server.mcp_types import EventData


# --- Mock ApplicationCoordinator ---
class MockApplicationCoordinator:
    def __init__(self):
        self.register_transport = AsyncMock()
        self.unregister_transport = AsyncMock()
        self.subscribe_to_event_type = AsyncMock()
        self.broadcast_event = AsyncMock()


# --- Mock Adapter Classes ---
class MockAdapter(AbstractTransportAdapter):
    TRANSPORT_TYPE_NAME = "mock_adapter"  # Used by registry for discovery key

    def __init__(self, coordinator: MockApplicationCoordinator, **config: Any):
        actual_transport_id = config.get("transport_id", "mock_default_id")
        super().__init__(
            transport_id=actual_transport_id,
            transport_type=self.TRANSPORT_TYPE_NAME,
            coordinator=coordinator,
            heartbeat_interval=config.get("heartbeat_interval")
        )
        self.config = config
        self.init_called_flag = False
        self.shutdown_called_flag = False
        self.send_event_mock = AsyncMock()
        self.validate_request_security_mock = MagicMock(return_value=SecurityContext(is_secure=True))
        # Logger is initialized in AbstractTransportAdapter

    async def initialize(self) -> None:
        await super().initialize()
        self.init_called_flag = True
        self.logger.info(f"MockAdapter {self.get_transport_id()} initialized with config: {self.config}")

    async def shutdown(self) -> None:
        await super().shutdown()
        self.shutdown_called_flag = True
        self.logger.info(f"MockAdapter {self.get_transport_id()} shutdown")

    async def send_event(self, event_type: EventTypes, data: EventData) -> None:
        await self.send_event_mock(event_type, data)

    def validate_request_security(self, request_data: Dict[str, Any]) -> SecurityContext:
        return self.validate_request_security_mock(request_data)

    def get_capabilities(self) -> Set[EventTypes]:
        return {EventTypes.STATUS, EventTypes.HEARTBEAT}


class MockAdapterTwo(MockAdapter): # Inherits from MockAdapter for simplicity
    TRANSPORT_TYPE_NAME = "mock_adapter_two"

    def __init__(self, coordinator: MockApplicationCoordinator, **config: Any):
        # Ensure a different default transport_id if not provided in config
        default_id = "mock_two_default_id"
        current_config = {**config} # Make a mutable copy
        if "transport_id" not in current_config:
            current_config["transport_id"] = default_id
        super().__init__(coordinator, **current_config)


class FailingInitAdapter(MockAdapter):
    TRANSPORT_TYPE_NAME = "failing_init_adapter"

    def __init__(self, coordinator: MockApplicationCoordinator, **config: Any):
        super().__init__(coordinator, **config)
        # Ensure transport_id is unique if not provided
        if "transport_id" not in self.config:
             self._transport_id = "failing_init_default_id" # Manually override if needed for logging

    async def initialize(self) -> None:
        # Do not call super().initialize() to prevent registration if that's part of the test
        self.init_called_flag = True # Mark as called before error
        self.logger.error(f"Simulating initialization failure for {self.get_transport_id()}")
        raise RuntimeError("Simulated initialization failure in FailingInitAdapter")


class FailingShutdownAdapter(MockAdapter):
    TRANSPORT_TYPE_NAME = "failing_shutdown_adapter"

    def __init__(self, coordinator: MockApplicationCoordinator, **config: Any):
        super().__init__(coordinator, **config)
        if "transport_id" not in self.config:
            self._transport_id = "failing_shutdown_default_id"

    async def shutdown(self) -> None:
        # To test registry's error handling, this adapter's shutdown itself fails.
        # If we wanted to test if super().shutdown() (e.g. unregister) happens before failure,
        # we would call await super().shutdown() here.
        self.shutdown_called_flag = True # Mark as called before error
        self.logger.error(f"Simulating shutdown failure for {self.get_transport_id()}")
        raise RuntimeError("Simulated shutdown failure in FailingShutdownAdapter")


class AdapterWithoutTypeName(AbstractTransportAdapter):
    # No TRANSPORT_TYPE_NAME class attribute defined here
    def __init__(self, coordinator: MockApplicationCoordinator, **config: Any):
        super().__init__(
            transport_id=config.get("transport_id", "no_type_default_id"),
            transport_type="actual_no_type_name", # Internal type for super()
            coordinator=coordinator
        )
    async def initialize(self) -> None: await super().initialize()
    async def shutdown(self) -> None: await super().shutdown()
    async def send_event(self, event_type: EventTypes, data: EventData) -> None: pass
    def validate_request_security(self, request_data: Dict[str, Any]) -> SecurityContext: return SecurityContext(is_secure=True)
    def get_capabilities(self) -> Set[EventTypes]: return set()


class AdapterWithInvalidTypeName(AbstractTransportAdapter):
    TRANSPORT_TYPE_NAME = 123  # Invalid (not a string)

    def __init__(self, coordinator: MockApplicationCoordinator, **config: Any):
        super().__init__(
            transport_id=config.get("transport_id", "invalid_type_default_id"),
            transport_type="actual_invalid_type_name", # Internal type for super()
            coordinator=coordinator
        )
    async def initialize(self) -> None: await super().initialize()
    async def shutdown(self) -> None: await super().shutdown()
    async def send_event(self, event_type: EventTypes, data: EventData) -> None: pass
    def validate_request_security(self, request_data: Dict[str, Any]) -> SecurityContext: return SecurityContext(is_secure=True)
    def get_capabilities(self) -> Set[EventTypes]: return set()


# --- Fixtures ---
@pytest.fixture
def mock_logger_factory():
    logger_mocks: Dict[str, MagicMock] = {}
    def factory(name: str, *args: Any, **kwargs: Any) -> MagicMock:
        if name not in logger_mocks:
            mock = MagicMock()
            # Define standard logging methods
            for level in ["debug", "info", "warning", "error", "critical", "exception", "verbose"]:
                setattr(mock, level, MagicMock(name=f"{name}_{level}_mock"))
            logger_mocks[name] = mock
        return logger_mocks[name]
    factory.logger_mocks = logger_mocks # type: ignore[attr-defined]
    return factory


@pytest_asyncio.fixture
async def registry(mock_logger_factory: Any):
    # Patch get_logger_func where it's used: in transport_adapter_registry and transport_adapter
    with patch('aider_mcp_server.transport_adapter_registry.get_logger_func', mock_logger_factory), \
         patch('aider_mcp_server.transport_adapter.get_logger_func', mock_logger_factory):
        reg = TransportAdapterRegistry(logger_factory=mock_logger_factory)
        yield reg
        # Ensure cleanup, as some tests might not explicitly call shutdown_all
        if reg._adapters or reg._adapter_classes: # Check if not already cleared
            await reg.shutdown_all()


@pytest.fixture
def mock_coordinator() -> MockApplicationCoordinator:
    return MockApplicationCoordinator()


# --- Helper Functions ---
def create_mock_module(name: str, classes: Optional[Dict[str, Type[AbstractTransportAdapter]]] = None, has_path: bool = True) -> ModuleType:
    module = ModuleType(name)
    module.__file__ = f"{name.replace('.', '/')}.py" # Simulate a file path
    if classes:
        for cls_name, cls_obj in classes.items():
            setattr(module, cls_name, cls_obj)
    if has_path:
        # Simulate a package by giving it a __path__
        module.__path__ = [os.path.dirname(module.__file__)] if module.__file__ else ["dummy_path"]
    # If not has_path, it simulates a single module file, __path__ remains unset.
    return module


# --- Test Cases ---
@pytest.mark.asyncio
class TestTransportAdapterRegistry:

    async def test_registry_initialization(self, registry: TransportAdapterRegistry, mock_logger_factory: Any):
        assert not registry._adapter_classes
        assert not registry._adapters
        assert isinstance(registry._lock, asyncio.Lock)
        
        # Check that the registry's own logger was created and used
        registry_logger_name = 'aider_mcp_server.transport_adapter_registry'
        assert registry_logger_name in mock_logger_factory.logger_mocks
        registry_logger = mock_logger_factory.logger_mocks[registry_logger_name]
        registry_logger.info.assert_any_call("TransportAdapterRegistry initialized")

    async def test_discover_adapters_simple_success(self, registry: TransportAdapterRegistry, mock_logger_factory: Any):
        mock_pkg_name = "my_adapters_pkg"
        # Create a mock module containing our MockAdapter
        module_with_adapter = create_mock_module(f"{mock_pkg_name}.adapter_module", {"MyAdapter": MockAdapter})
        # Create a mock package module
        pkg_module = create_mock_module(mock_pkg_name, has_path=True)

        with patch('importlib.import_module') as mock_import_module, \
             patch('pkgutil.iter_modules') as mock_iter_modules:

            # Simulate import_module behavior
            def import_side_effect(name):
                if name == mock_pkg_name: return pkg_module
                if name == f"{mock_pkg_name}.adapter_module": return module_with_adapter
                raise ImportError(f"Test mock: Unknown module {name}")
            mock_import_module.side_effect = import_side_effect
            
            # Simulate iter_modules finding our adapter_module
            mock_iter_modules.return_value = [(None, f"{mock_pkg_name}.adapter_module", False)]

            registry.discover_adapters(mock_pkg_name)
            await asyncio.sleep(0) # Allow async _register_adapter_class to run

            assert MockAdapter.TRANSPORT_TYPE_NAME in registry._adapter_classes
            assert registry._adapter_classes[MockAdapter.TRANSPORT_TYPE_NAME] == MockAdapter
            
            registry_logger = mock_logger_factory.logger_mocks['aider_mcp_server.transport_adapter_registry']
            registry_logger.info.assert_any_call(f"Discovering transport adapters in package: {mock_pkg_name}")
            registry_logger.info.assert_any_call(f"Discovered transport adapter: MyAdapter for type '{MockAdapter.TRANSPORT_TYPE_NAME}'")
            registry_logger.info.assert_any_call("Adapter discovery complete. Found 1 adapter classes.")

    async def test_discover_adapters_from_single_file_module(self, registry: TransportAdapterRegistry, mock_logger_factory: Any):
        mock_pkg_name = "single_file_pkg"
        # Simulate a module that is a single file (no __path__)
        single_file_module = create_mock_module(mock_pkg_name, {"MyAdapterInFile": MockAdapter}, has_path=False)
        
        with patch('importlib.import_module', return_value=single_file_module):
            registry.discover_adapters(mock_pkg_name)
            await asyncio.sleep(0)

            assert MockAdapter.TRANSPORT_TYPE_NAME in registry._adapter_classes
            assert registry._adapter_classes[MockAdapter.TRANSPORT_TYPE_NAME] == MockAdapter
            
            registry_logger = mock_logger_factory.logger_mocks['aider_mcp_server.transport_adapter_registry']
            registry_logger.warning.assert_any_call(f"Package {mock_pkg_name} has no __path__ attribute. Cannot discover modules.")
            registry_logger.info.assert_any_call(f"Discovered transport adapter: MyAdapterInFile for type '{MockAdapter.TRANSPORT_TYPE_NAME}'")

    async def test_discover_adapters_package_import_fails(self, registry: TransportAdapterRegistry, mock_logger_factory: Any):
        mock_pkg_name = "nonexistent_package"
        with patch('importlib.import_module', side_effect=ImportError(f"Cannot import {mock_pkg_name}")):
            registry.discover_adapters(mock_pkg_name)
            await asyncio.sleep(0)

            assert not registry._adapter_classes
            registry_logger = mock_logger_factory.logger_mocks['aider_mcp_server.transport_adapter_registry']
            registry_logger.error.assert_any_call(f"Failed to import package: {mock_pkg_name}")

    async def test_discover_adapters_module_import_fails(self, registry: TransportAdapterRegistry, mock_logger_factory: Any):
        mock_pkg_name = "pkg_with_faulty_module"
        pkg_module = create_mock_module(mock_pkg_name, has_path=True)

        with patch('importlib.import_module') as mock_import_module, \
             patch('pkgutil.iter_modules') as mock_iter_modules:
            
            # Simulate import_module: package imports fine, submodule fails
            mock_import_module.side_effect = lambda name: pkg_module if name == mock_pkg_name else (_ for _ in ()).throw(ImportError("Submodule failed to load"))
            # Simulate iter_modules finding a submodule
            mock_iter_modules.return_value = [(None, f"{mock_pkg_name}.faulty_submodule", False)]

            registry.discover_adapters(mock_pkg_name)
            await asyncio.sleep(0)
            
            assert not registry._adapter_classes
            registry_logger = mock_logger_factory.logger_mocks['aider_mcp_server.transport_adapter_registry']
            registry_logger.error.assert_any_call(
                f"Error discovering adapters in module {mock_pkg_name}.faulty_submodule: Submodule failed to load", exc_info=True
            )

    async def test_discover_adapters_skips_invalid_adapters(self, registry: TransportAdapterRegistry, mock_logger_factory: Any):
        mock_pkg_name = "pkg_with_various_adapters"
        module_content = {
            "GoodAdapter": MockAdapter,
            "NoNameAdapter": AdapterWithoutTypeName,
            "InvalidNameAdapter": AdapterWithInvalidTypeName
        }
        test_module = create_mock_module(f"{mock_pkg_name}.test_mod", module_content)
        pkg_module = create_mock_module(mock_pkg_name, has_path=True)

        with patch('importlib.import_module') as mock_import_module, \
             patch('pkgutil.iter_modules') as mock_iter_modules:
            
            mock_import_module.side_effect = lambda name: pkg_module if name == mock_pkg_name else test_module
            mock_iter_modules.return_value = [(None, f"{mock_pkg_name}.test_mod", False)]

            registry.discover_adapters(mock_pkg_name)
            await asyncio.sleep(0)

            assert MockAdapter.TRANSPORT_TYPE_NAME in registry._adapter_classes
            assert len(registry._adapter_classes) == 1 # Only GoodAdapter

            registry_logger = mock_logger_factory.logger_mocks['aider_mcp_server.transport_adapter_registry']
            registry_logger.warning.assert_any_call(
                f"Adapter class NoNameAdapter in module {mock_pkg_name}.test_mod "
                f"does not define a valid TRANSPORT_TYPE_NAME string attribute. Skipping."
            )
            registry_logger.warning.assert_any_call(
                f"Adapter class InvalidNameAdapter in module {mock_pkg_name}.test_mod "
                f"does not define a valid TRANSPORT_TYPE_NAME string attribute. Skipping."
            )
            registry_logger.info.assert_any_call(f"Discovered transport adapter: GoodAdapter for type '{MockAdapter.TRANSPORT_TYPE_NAME}'")

    async def test_discover_adapters_duplicate_type_name_warning_and_overwrite(self, registry: TransportAdapterRegistry, mock_logger_factory: Any):
        class DuplicateNameAdapter(MockAdapter): # Same TRANSPORT_TYPE_NAME as MockAdapter
            TRANSPORT_TYPE_NAME = "mock_adapter" 

        mock_pkg_name = "pkg_with_duplicates"
        # Module where adapters are defined. Order of definition might matter for inspect.getmembers.
        # We'll control it with a mock for inspect.getmembers.
        module_with_duplicates = create_mock_module(
            f"{mock_pkg_name}.duplicates_mod",
            {"FirstAdapter": MockAdapter, "SecondAdapterWithSameName": DuplicateNameAdapter}
        )
        pkg_module = create_mock_module(mock_pkg_name, has_path=True)

        with patch('importlib.import_module') as mock_import_module, \
             patch('pkgutil.iter_modules') as mock_iter_modules, \
             patch('inspect.getmembers') as mock_getmembers:
            
            mock_import_module.side_effect = lambda name: pkg_module if name == mock_pkg_name else module_with_duplicates
            mock_iter_modules.return_value = [(None, f"{mock_pkg_name}.duplicates_mod", False)]
            
            # Control order from inspect.getmembers: FirstAdapter, then SecondAdapterWithSameName
            mock_getmembers.return_value = [
                ("FirstAdapter", MockAdapter), 
                ("SecondAdapterWithSameName", DuplicateNameAdapter)
            ]

            registry.discover_adapters(mock_pkg_name)
            await asyncio.sleep(0) # Let registration tasks complete

            assert registry._adapter_classes[MockAdapter.TRANSPORT_TYPE_NAME] == DuplicateNameAdapter # Second one overwrites

            registry_logger = mock_logger_factory.logger_mocks['aider_mcp_server.transport_adapter_registry']
            # Check for the warning about overwriting
            registry_logger.warning.assert_any_call(
                f"Duplicate transport type '{MockAdapter.TRANSPORT_TYPE_NAME}' found. "
                f"Class SecondAdapterWithSameName will overwrite FirstAdapter."
            )
            # Check that both were "discovered" before the overwrite logic applied
            registry_logger.info.assert_any_call(f"Discovered transport adapter: FirstAdapter for type '{MockAdapter.TRANSPORT_TYPE_NAME}'")
            registry_logger.info.assert_any_call(f"Discovered transport adapter: SecondAdapterWithSameName for type '{DuplicateNameAdapter.TRANSPORT_TYPE_NAME}'")


    async def test_initialize_adapter_success(self, registry: TransportAdapterRegistry, mock_coordinator: MockApplicationCoordinator, mock_logger_factory: Any):
        # Manually "discover" MockAdapter
        registry._adapter_classes[MockAdapter.TRANSPORT_TYPE_NAME] = MockAdapter
        
        config = {"setting": "A", "transport_id": "adapter_init_success_id"}
        adapter_instance = await registry.initialize_adapter(MockAdapter.TRANSPORT_TYPE_NAME, mock_coordinator, config)

        assert isinstance(adapter_instance, MockAdapter)
        assert adapter_instance.init_called_flag
        assert adapter_instance.get_transport_id() == "adapter_init_success_id"
        assert adapter_instance.config == config
        assert registry.get_adapter("adapter_init_success_id") == adapter_instance

        registry_logger = mock_logger_factory.logger_mocks['aider_mcp_server.transport_adapter_registry']
        registry_logger.info.assert_any_call(f"Initializing transport adapter of type: {MockAdapter.TRANSPORT_TYPE_NAME}")
        registry_logger.info.assert_any_call(
            f"Successfully initialized adapter adapter_init_success_id of type {MockAdapter.TRANSPORT_TYPE_NAME}"
        )

        # Verify AbstractTransportAdapter's initialize logic (registration, subscription)
        mock_coordinator.register_transport.assert_called_once_with("adapter_init_success_id", adapter_instance)
        expected_capabilities = adapter_instance.get_capabilities()
        subscribe_calls = [call("adapter_init_success_id", cap) for cap in expected_capabilities]
        mock_coordinator.subscribe_to_event_type.assert_has_calls(subscribe_calls, any_order=True)
        assert mock_coordinator.subscribe_to_event_type.call_count == len(expected_capabilities)

        # Verify adapter's own logger was used
        adapter_logger_name = f"aider_mcp_server.transport_adapter.MockAdapter.{adapter_instance.get_transport_id()}"
        assert adapter_logger_name in mock_logger_factory.logger_mocks
        adapter_logger = mock_logger_factory.logger_mocks[adapter_logger_name]
        adapter_logger.info.assert_any_call(f"MockAdapter {adapter_instance.get_transport_id()} initialized with config: {config}")

    async def test_initialize_adapter_uses_default_id_if_not_in_config(self, registry: TransportAdapterRegistry, mock_coordinator: MockApplicationCoordinator):
        registry._adapter_classes[MockAdapter.TRANSPORT_TYPE_NAME] = MockAdapter
        adapter_instance = await registry.initialize_adapter(MockAdapter.TRANSPORT_TYPE_NAME, mock_coordinator, {}) # Empty config
        
        assert isinstance(adapter_instance, MockAdapter)
        assert adapter_instance.get_transport_id() == "mock_default_id" # Default from MockAdapter
        assert registry.get_adapter("mock_default_id") == adapter_instance

    async def test_initialize_adapter_unknown_type(self, registry: TransportAdapterRegistry, mock_coordinator: MockApplicationCoordinator, mock_logger_factory: Any):
        adapter_instance = await registry.initialize_adapter("non_existent_type", mock_coordinator, {})
        assert adapter_instance is None
        registry_logger = mock_logger_factory.logger_mocks['aider_mcp_server.transport_adapter_registry']
        registry_logger.error.assert_any_call("No adapter class found for transport type: non_existent_type")

    async def test_initialize_adapter_init_method_fails(self, registry: TransportAdapterRegistry, mock_coordinator: MockApplicationCoordinator, mock_logger_factory: Any):
        registry._adapter_classes[FailingInitAdapter.TRANSPORT_TYPE_NAME] = FailingInitAdapter
        
        adapter_instance = await registry.initialize_adapter(FailingInitAdapter.TRANSPORT_TYPE_NAME, mock_coordinator, {"transport_id": "fail_init_test_id"})
        assert adapter_instance is None
        
        registry_logger = mock_logger_factory.logger_mocks['aider_mcp_server.transport_adapter_registry']
        registry_logger.error.assert_any_call(
            f"Failed to initialize adapter of type {FailingInitAdapter.TRANSPORT_TYPE_NAME}: Simulated initialization failure in FailingInitAdapter",
            exc_info=True
        )
        # Check that the failing adapter's own logger was called before the exception
        adapter_logger_name = f"aider_mcp_server.transport_adapter.FailingInitAdapter.fail_init_test_id"
        assert adapter_logger_name in mock_logger_factory.logger_mocks
        failing_adapter_logger = mock_logger_factory.logger_mocks[adapter_logger_name]
        failing_adapter_logger.error.assert_any_call("Simulating initialization failure for fail_init_test_id")


    async def test_get_adapter_found_and_not_found(self, registry: TransportAdapterRegistry, mock_coordinator: MockApplicationCoordinator):
        registry._adapter_classes[MockAdapter.TRANSPORT_TYPE_NAME] = MockAdapter
        # Initialize an adapter to be found
        config = {"transport_id": "my_test_adapter_id"}
        initialized_adapter = await registry.initialize_adapter(MockAdapter.TRANSPORT_TYPE_NAME, mock_coordinator, config)
        
        assert registry.get_adapter("my_test_adapter_id") == initialized_adapter
        assert registry.get_adapter("non_existent_adapter_id") is None

    async def test_shutdown_all_adapters_success(self, registry: TransportAdapterRegistry, mock_coordinator: MockApplicationCoordinator, mock_logger_factory: Any):
        # Pre-register adapter classes
        registry._adapter_classes[MockAdapter.TRANSPORT_TYPE_NAME] = MockAdapter
        registry._adapter_classes[MockAdapterTwo.TRANSPORT_TYPE_NAME] = MockAdapterTwo

        # Initialize a couple of adapters
        adapter1_config = {"transport_id": "shutdown_adapter_1"}
        adapter1 = await registry.initialize_adapter(MockAdapter.TRANSPORT_TYPE_NAME, mock_coordinator, adapter1_config)
        assert isinstance(adapter1, MockAdapter)

        adapter2_config = {"transport_id": "shutdown_adapter_2"}
        adapter2 = await registry.initialize_adapter(MockAdapterTwo.TRANSPORT_TYPE_NAME, mock_coordinator, adapter2_config)
        assert isinstance(adapter2, MockAdapterTwo)

        assert len(registry._adapters) == 2
        await registry.shutdown_all()

        assert adapter1.shutdown_called_flag
        assert adapter2.shutdown_called_flag
        assert not registry._adapters  # Should be cleared
        assert not registry._adapter_classes # Should be cleared

        # Verify AbstractTransportAdapter's shutdown logic (unregistration)
        mock_coordinator.unregister_transport.assert_any_call("shutdown_adapter_1")
        mock_coordinator.unregister_transport.assert_any_call("shutdown_adapter_2")
        assert mock_coordinator.unregister_transport.call_count == 2
        
        registry_logger = mock_logger_factory.logger_mocks['aider_mcp_server.transport_adapter_registry']
        registry_logger.info.assert_any_call("Shutting down all transport adapters...")
        registry_logger.info.assert_any_call(f"Shutting down adapter {adapter1.get_transport_id()}")
        registry_logger.info.assert_any_call(f"Shutting down adapter {adapter2.get_transport_id()}")
        registry_logger.info.assert_any_call("All transport adapters shut down and registry cleared.")

        # Verify adapter's own logger for shutdown
        adapter1_logger = mock_logger_factory.logger_mocks[f"aider_mcp_server.transport_adapter.MockAdapter.{adapter1.get_transport_id()}"]
        adapter1_logger.info.assert_any_call(f"MockAdapter {adapter1.get_transport_id()} shutdown")

    async def test_shutdown_all_one_adapter_fails_shutdown(self, registry: TransportAdapterRegistry, mock_coordinator: MockApplicationCoordinator, mock_logger_factory: Any):
        registry._adapter_classes[MockAdapter.TRANSPORT_TYPE_NAME] = MockAdapter
        registry._adapter_classes[FailingShutdownAdapter.TRANSPORT_TYPE_NAME] = FailingShutdownAdapter

        good_adapter_config = {"transport_id": "good_one_shutdown"}
        good_adapter = await registry.initialize_adapter(MockAdapter.TRANSPORT_TYPE_NAME, mock_coordinator, good_adapter_config)
        assert isinstance(good_adapter, MockAdapter)

        fail_adapter_config = {"transport_id": "bad_one_shutdown"}
        fail_adapter = await registry.initialize_adapter(FailingShutdownAdapter.TRANSPORT_TYPE_NAME, mock_coordinator, fail_adapter_config)
        assert isinstance(fail_adapter, FailingShutdownAdapter)

        await registry.shutdown_all()

        assert good_adapter.shutdown_called_flag
        assert fail_adapter.shutdown_called_flag # Marked true in its own shutdown before raising
        assert not registry._adapters
        assert not registry._adapter_classes

        # Good adapter's super().shutdown() calls unregister. Failing adapter's shutdown raises before super().
        mock_coordinator.unregister_transport.assert_called_once_with(good_adapter.get_transport_id())
        
        registry_logger = mock_logger_factory.logger_mocks['aider_mcp_server.transport_adapter_registry']
        registry_logger.error.assert_any_call(
            f"Error shutting down adapter {fail_adapter.get_transport_id()}: Simulated shutdown failure in FailingShutdownAdapter",
            exc_info=True
        )
        registry_logger.info.assert_any_call(f"Shutting down adapter {good_adapter.get_transport_id()}")

        # Check failing adapter's logger
        failing_adapter_logger_name = f"aider_mcp_server.transport_adapter.FailingShutdownAdapter.{fail_adapter.get_transport_id()}"
        failing_adapter_logger = mock_logger_factory.logger_mocks[failing_adapter_logger_name]
        failing_adapter_logger.error.assert_any_call(f"Simulating shutdown failure for {fail_adapter.get_transport_id()}")

    async def test_concurrent_initialization_and_shutdown(self, registry: TransportAdapterRegistry, mock_coordinator: MockApplicationCoordinator):
        # "Discover" adapters
        registry._adapter_classes[MockAdapter.TRANSPORT_TYPE_NAME] = MockAdapter
        registry._adapter_classes[MockAdapterTwo.TRANSPORT_TYPE_NAME] = MockAdapterTwo

        num_adapters = 5
        init_tasks = []
        adapter_ids = [f"concurrent_adapter_{i}" for i in range(num_adapters)]

        for i in range(num_adapters):
            adapter_type = MockAdapter.TRANSPORT_TYPE_NAME if i % 2 == 0 else MockAdapterTwo.TRANSPORT_TYPE_NAME
            config = {"transport_id": adapter_ids[i]}
            init_tasks.append(registry.initialize_adapter(adapter_type, mock_coordinator, config))
        
        initialized_adapters: List[Optional[ITransportAdapter]] = await asyncio.gather(*init_tasks)

        assert len(initialized_adapters) == num_adapters
        valid_adapters: List[MockAdapter] = []
        for adapter in initialized_adapters:
            assert adapter is not None
            assert isinstance(adapter, (MockAdapter, MockAdapterTwo)) # Check type
            # Type cast for mypy after assertion
            mock_adapter_instance = typing.cast(MockAdapter, adapter)
            assert mock_adapter_instance.init_called_flag
            assert mock_adapter_instance.get_transport_id() in registry._adapters
            valid_adapters.append(mock_adapter_instance)
        
        assert len(registry._adapters) == num_adapters
        assert mock_coordinator.register_transport.call_count == num_adapters
        # Each adapter subscribes to 2 capabilities (STATUS, HEARTBEAT)
        assert mock_coordinator.subscribe_to_event_type.call_count == num_adapters * 2

        # Now test shutdown_all with these concurrently initialized adapters
        await registry.shutdown_all()

        assert not registry._adapters
        assert not registry._adapter_classes
        assert mock_coordinator.unregister_transport.call_count == num_adapters
        for adapter in valid_adapters:
            assert adapter.shutdown_called_flag
