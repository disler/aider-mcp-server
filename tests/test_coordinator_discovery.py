"""Tests for the coordinator discovery mechanism."""

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from aider_mcp_server.coordinator_discovery import (
    CoordinatorDiscovery,
    CoordinatorInfo,
)


class TestCoordinatorInfo:
    """Test the CoordinatorInfo class."""

    def test_initialization(self):
        """Test initialization of CoordinatorInfo."""
        info = CoordinatorInfo(
            coordinator_id="test_coordinator_1",
            host="localhost",
            port=8000,
            transport_type="test",
            metadata={"test_key": "test_value"},
        )

        # Check basic attributes
        assert info.coordinator_id == "test_coordinator_1"
        assert info.host == "localhost"
        assert info.port == 8000
        assert info.transport_type == "test"
        assert info.metadata == {"test_key": "test_value"}

        # Check that heartbeat is initialized to start_time
        assert info.last_heartbeat == info.start_time

    def test_to_dict(self):
        """Test conversion to dictionary."""
        start_time = time.time()
        info = CoordinatorInfo(
            coordinator_id="test_coordinator_1",
            host="localhost",
            port=8000,
            transport_type="test",
            start_time=start_time,
            metadata={"test_key": "test_value"},
        )

        # Manually update last_heartbeat for verification
        info.last_heartbeat = start_time + 60

        # Convert to dict
        info_dict = info.to_dict()

        # Verify all fields
        assert info_dict["coordinator_id"] == "test_coordinator_1"
        assert info_dict["host"] == "localhost"
        assert info_dict["port"] == 8000
        assert info_dict["transport_type"] == "test"
        assert info_dict["start_time"] == start_time
        assert info_dict["last_heartbeat"] == start_time + 60
        assert info_dict["metadata"] == {"test_key": "test_value"}

    def test_from_dict(self):
        """Test creation from dictionary."""
        info_dict = {
            "coordinator_id": "test_coordinator_1",
            "host": "localhost",
            "port": 8000,
            "transport_type": "test",
            "start_time": 1620000000,
            "last_heartbeat": 1620000060,
            "metadata": {"test_key": "test_value"},
        }

        # Create from dict
        info = CoordinatorInfo.from_dict(info_dict)

        # Verify all fields
        assert info.coordinator_id == "test_coordinator_1"
        assert info.host == "localhost"
        assert info.port == 8000
        assert info.transport_type == "test"
        assert info.start_time == 1620000000
        assert info.last_heartbeat == 1620000060
        assert info.metadata == {"test_key": "test_value"}

    def test_update_heartbeat(self):
        """Test heartbeat update."""
        info = CoordinatorInfo(
            coordinator_id="test_coordinator_1",
            host="localhost",
            port=8000,
        )

        # Record initial heartbeat
        initial_heartbeat = info.last_heartbeat

        # Allow time to pass
        time.sleep(0.01)

        # Update heartbeat
        info.update_heartbeat()

        # Check that heartbeat is updated
        assert info.last_heartbeat > initial_heartbeat

    def test_is_active(self):
        """Test active status checking."""
        info = CoordinatorInfo(
            coordinator_id="test_coordinator_1",
            host="localhost",
            port=8000,
        )

        # Should be active initially
        assert info.is_active(max_age_seconds=30.0)

        # Test with stale heartbeat
        info.last_heartbeat = time.time() - 60  # Set heartbeat to 60 seconds ago
        assert not info.is_active(max_age_seconds=30.0)

        # Test with recent heartbeat
        info.last_heartbeat = time.time() - 15  # Set heartbeat to 15 seconds ago
        assert info.is_active(max_age_seconds=30.0)


@pytest.fixture
def temp_discovery_file():
    """Create a temporary discovery file for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        discovery_file = Path(temp_dir) / "test_coordinator_registry.json"
        yield discovery_file


class TestCoordinatorDiscovery:
    """Test the CoordinatorDiscovery class."""

    @pytest.mark.asyncio
    async def test_initialization_with_default_file(self):
        """Test initialization with default discovery file."""
        # Use a patch to avoid modifying real files
        with patch("pathlib.Path.mkdir"):
            discovery = CoordinatorDiscovery()
            assert discovery.discovery_file.name == "coordinator_registry.json"
            assert "aider_mcp_coordinator" in str(discovery.discovery_file)
            assert discovery.heartbeat_interval == 10.0

    @pytest.mark.asyncio
    async def test_initialization_with_custom_file(self, temp_discovery_file):
        """Test initialization with custom discovery file."""
        discovery = CoordinatorDiscovery(discovery_file=temp_discovery_file)
        assert discovery.discovery_file == temp_discovery_file
        assert discovery.heartbeat_interval == 10.0

    @pytest.mark.asyncio
    async def test_initialization_with_env_var(self, temp_discovery_file):
        """Test initialization with environment variable."""
        with patch.dict(os.environ, {"AIDER_MCP_COORDINATOR_DISCOVERY_FILE": str(temp_discovery_file)}):
            discovery = CoordinatorDiscovery()
            assert discovery.discovery_file == temp_discovery_file

    @pytest.mark.asyncio
    async def test_register_coordinator(self, temp_discovery_file):
        """Test registering a coordinator."""
        discovery = CoordinatorDiscovery(discovery_file=temp_discovery_file)
        
        # Register a coordinator
        coordinator_id = await discovery.register_coordinator(
            host="localhost",
            port=8000,
            transport_type="test",
            metadata={"test_key": "test_value"},
        )
        
        # Verify ID format
        assert coordinator_id.startswith("coordinator_")
        assert len(coordinator_id) > 10
        
        # Verify registry file exists
        assert temp_discovery_file.exists()
        
        # Verify registry content
        registry_data = json.loads(temp_discovery_file.read_text())
        assert len(registry_data) == 1
        assert registry_data[0]["coordinator_id"] == coordinator_id
        assert registry_data[0]["host"] == "localhost"
        assert registry_data[0]["port"] == 8000
        assert registry_data[0]["transport_type"] == "test"
        assert registry_data[0]["metadata"] == {"test_key": "test_value"}
        
        # Verify internal state
        assert discovery._registered_coordinator is not None
        assert discovery._registered_coordinator.coordinator_id == coordinator_id
        
        # Clean up
        await discovery.shutdown()

    @pytest.mark.asyncio
    async def test_discover_coordinators_empty(self, temp_discovery_file):
        """Test discovering coordinators when registry is empty."""
        discovery = CoordinatorDiscovery(discovery_file=temp_discovery_file)
        
        # Discover coordinators (no file yet)
        coordinators = await discovery.discover_coordinators()
        assert len(coordinators) == 0
        
        # Create empty registry file
        temp_discovery_file.write_text("[]")
        
        # Discover coordinators (empty registry)
        coordinators = await discovery.discover_coordinators()
        assert len(coordinators) == 0

    @pytest.mark.asyncio
    async def test_discover_coordinators_with_active(self, temp_discovery_file):
        """Test discovering active coordinators."""
        # Create discovery registry with active coordinator
        registry_data = [{
            "coordinator_id": "test_coordinator_1",
            "host": "localhost",
            "port": 8000,
            "transport_type": "test",
            "start_time": time.time(),
            "last_heartbeat": time.time(),
            "metadata": {"test_key": "test_value"},
        }]
        temp_discovery_file.write_text(json.dumps(registry_data))
        
        # Create discovery instance
        discovery = CoordinatorDiscovery(discovery_file=temp_discovery_file)
        
        # Discover coordinators
        coordinators = await discovery.discover_coordinators()
        assert len(coordinators) == 1
        assert coordinators[0].coordinator_id == "test_coordinator_1"
        assert coordinators[0].is_active()

    @pytest.mark.asyncio
    async def test_discover_coordinators_with_inactive(self, temp_discovery_file):
        """Test discovering coordinators with inactive entries."""
        # Create discovery registry with inactive coordinator
        registry_data = [{
            "coordinator_id": "test_coordinator_1",
            "host": "localhost",
            "port": 8000,
            "transport_type": "test",
            "start_time": time.time() - 100,
            "last_heartbeat": time.time() - 60,  # 60 seconds old (inactive)
            "metadata": {"test_key": "test_value"},
        }]
        temp_discovery_file.write_text(json.dumps(registry_data))
        
        # Create discovery instance
        discovery = CoordinatorDiscovery(discovery_file=temp_discovery_file)
        
        # Discover coordinators (should be filtered out)
        coordinators = await discovery.discover_coordinators(max_age_seconds=30.0)
        assert len(coordinators) == 0

    @pytest.mark.asyncio
    async def test_find_best_coordinator(self, temp_discovery_file):
        """Test finding the best coordinator."""
        # Create discovery registry with multiple coordinators
        registry_data = [
            {
                "coordinator_id": "older_coordinator",
                "host": "localhost",
                "port": 8000,
                "transport_type": "test",
                "start_time": time.time() - 100,
                "last_heartbeat": time.time() - 10,
                "metadata": {"test_key": "test_value"},
            },
            {
                "coordinator_id": "newer_coordinator",
                "host": "localhost",
                "port": 8001,
                "transport_type": "test",
                "start_time": time.time() - 50,
                "last_heartbeat": time.time() - 5,
                "metadata": {"test_key": "test_value"},
            }
        ]
        temp_discovery_file.write_text(json.dumps(registry_data))
        
        # Create discovery instance
        discovery = CoordinatorDiscovery(discovery_file=temp_discovery_file)
        
        # Find best coordinator (should be the newer one)
        coordinator = await discovery.find_best_coordinator()
        assert coordinator is not None
        assert coordinator.coordinator_id == "newer_coordinator"

    @pytest.mark.asyncio
    async def test_heartbeat_loop(self, temp_discovery_file):
        """Test heartbeat loop updates the registry."""
        # Create discovery instance with short heartbeat interval
        discovery = CoordinatorDiscovery(
            discovery_file=temp_discovery_file,
            heartbeat_interval=0.1  # 100ms for faster test
        )
        
        # Register a coordinator to trigger heartbeat
        coordinator_id = await discovery.register_coordinator(
            host="localhost",
            port=8000,
        )
        
        # Wait for at least one heartbeat
        await asyncio.sleep(0.2)  # Should allow at least one heartbeat
        
        # Verify registry has been updated
        registry_data = json.loads(temp_discovery_file.read_text())
        assert len(registry_data) == 1
        assert registry_data[0]["coordinator_id"] == coordinator_id
        
        # Initial heartbeat should match start_time
        initial_heartbeat = discovery._registered_coordinator.start_time
        
        # Current heartbeat should be newer than initial
        current_heartbeat = registry_data[0]["last_heartbeat"]
        assert current_heartbeat > initial_heartbeat
        
        # Clean up
        await discovery.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown(self, temp_discovery_file):
        """Test coordinator shutdown removes entry from registry."""
        discovery = CoordinatorDiscovery(discovery_file=temp_discovery_file)
        
        # Register a coordinator
        coordinator_id = await discovery.register_coordinator(
            host="localhost",
            port=8000,
        )
        
        # Verify registry has entry
        registry_data = json.loads(temp_discovery_file.read_text())
        assert len(registry_data) == 1
        
        # Shutdown the discovery
        await discovery.shutdown()
        
        # Verify registry has no entries
        if temp_discovery_file.exists():  # File might be deleted in cleanup
            registry_data = json.loads(temp_discovery_file.read_text())
            assert len(registry_data) == 0
            
        # Verify heartbeat task is cancelled
        assert discovery._heartbeat_task is None or discovery._heartbeat_task.done()

    @pytest.mark.asyncio
    async def test_async_context_manager(self, temp_discovery_file):
        """Test async context manager functionality."""
        # Use async context manager
        async with CoordinatorDiscovery(discovery_file=temp_discovery_file) as discovery:
            # Register a coordinator
            coordinator_id = await discovery.register_coordinator(
                host="localhost",
                port=8000,
            )
            
            # Verify registry has entry
            registry_data = json.loads(temp_discovery_file.read_text())
            assert len(registry_data) == 1
        
        # After context exit, verify cleanup
        if temp_discovery_file.exists():  # File might be deleted in cleanup
            registry_data = json.loads(temp_discovery_file.read_text())
            assert len(registry_data) == 0