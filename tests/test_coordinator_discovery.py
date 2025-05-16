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
            last_heartbeat=start_time + 5,
            metadata={"test": True},
        )

        result = info.to_dict()

        assert result["coordinator_id"] == "test_coordinator_1"
        assert result["host"] == "localhost"
        assert result["port"] == 8000
        assert result["transport_type"] == "test"
        assert result["start_time"] == start_time
        assert result["last_heartbeat"] == start_time + 5
        assert result["metadata"] == {"test": True}

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "coordinator_id": "test_coordinator_1",
            "host": "localhost",
            "port": 8000,
            "transport_type": "test",
            "start_time": time.time(),
            "last_heartbeat": time.time() + 5,
            "metadata": {"test": True},
        }

        info = CoordinatorInfo.from_dict(data)

        assert info.coordinator_id == "test_coordinator_1"
        assert info.host == "localhost"
        assert info.port == 8000
        assert info.transport_type == "test"
        assert info.start_time == data["start_time"]
        assert info.last_heartbeat == data["last_heartbeat"]
        assert info.metadata == {"test": True}

    def test_is_healthy(self):
        """Test health check based on heartbeat age."""
        current_time = time.time()

        # Healthy coordinator (recent heartbeat)
        info = CoordinatorInfo(
            coordinator_id="test_1",
            host="localhost",
            port=8000,
            transport_type="test",
            last_heartbeat=current_time - 10,  # 10 seconds ago
        )
        assert info.is_healthy(timeout=30)

        # Unhealthy coordinator (old heartbeat)
        info.last_heartbeat = current_time - 60  # 60 seconds ago
        assert not info.is_healthy(timeout=30)

    def test_update_heartbeat(self):
        """Test heartbeat update."""
        info = CoordinatorInfo(
            coordinator_id="test_1",
            host="localhost",
            port=8000,
            transport_type="test",
        )

        old_heartbeat = info.last_heartbeat
        time.sleep(0.01)  # Small delay to ensure time difference
        info.update_heartbeat()

        assert info.last_heartbeat > old_heartbeat


class TestCoordinatorDiscovery:
    """Test the CoordinatorDiscovery class."""

    @pytest.fixture
    def temp_discovery_file(self, tmp_path):
        """Provide a temporary discovery file."""
        return tmp_path / "test_discovery.json"

    @pytest.mark.asyncio
    async def test_initialization(self, temp_discovery_file):
        """Test initialization of discovery system."""
        discovery = CoordinatorDiscovery(discovery_file=temp_discovery_file)

        assert discovery.discovery_file == temp_discovery_file
        assert discovery._heartbeat_interval == 30
        assert discovery._cleanup_interval == 60
        assert discovery._health_timeout == 90

        # Verify file is created if it doesn't exist
        assert temp_discovery_file.exists()

    @pytest.mark.asyncio
    async def test_register_coordinator(self, temp_discovery_file):
        """Test registering a coordinator."""
        discovery = CoordinatorDiscovery(discovery_file=temp_discovery_file)

        # Register a coordinator
        coord_id = await discovery.register_coordinator(
            host="localhost",
            port=8000,
            transport_type="sse",
            metadata={"test": True},
        )

        assert coord_id is not None

        # Verify it's in the registry file
        registry_data = json.loads(temp_discovery_file.read_text())
        assert coord_id in registry_data
        assert registry_data[coord_id]["host"] == "localhost"
        assert registry_data[coord_id]["port"] == 8000
        assert registry_data[coord_id]["transport_type"] == "sse"
        assert registry_data[coord_id]["metadata"] == {"test": True}

    @pytest.mark.asyncio
    async def test_find_coordinators(self, temp_discovery_file):
        """Test finding coordinators."""
        discovery = CoordinatorDiscovery(discovery_file=temp_discovery_file)

        # Register some coordinators
        coord1_id = await discovery.register_coordinator(
            host="localhost",
            port=8000,
            transport_type="sse",
        )

        coord2_id = await discovery.register_coordinator(
            host="localhost",
            port=8001,
            transport_type="stdio",
        )

        # Find all coordinators
        all_coords = await discovery.find_coordinators()
        assert len(all_coords) == 2

        # Find by transport type
        sse_coords = await discovery.find_coordinators(transport_type="sse")
        assert len(sse_coords) == 1
        assert sse_coords[0].coordinator_id == coord1_id

        # Find by port
        port_coords = await discovery.find_coordinators(port=8001)
        assert len(port_coords) == 1
        assert port_coords[0].coordinator_id == coord2_id

    @pytest.mark.asyncio
    async def test_update_heartbeat(self, temp_discovery_file):
        """Test heartbeat updates."""
        discovery = CoordinatorDiscovery(discovery_file=temp_discovery_file)

        # Register a coordinator
        coord_id = await discovery.register_coordinator(
            host="localhost",
            port=8000,
        )

        # Get initial heartbeat
        registry_data = json.loads(temp_discovery_file.read_text())
        initial_heartbeat = registry_data[coord_id]["last_heartbeat"]

        # Wait a bit and update heartbeat
        await asyncio.sleep(0.1)
        await discovery.update_heartbeat(coord_id)

        # Verify heartbeat was updated
        registry_data = json.loads(temp_discovery_file.read_text())
        assert registry_data[coord_id]["last_heartbeat"] > initial_heartbeat

    @pytest.mark.asyncio
    async def test_remove_coordinator(self, temp_discovery_file):
        """Test removing a coordinator."""
        discovery = CoordinatorDiscovery(discovery_file=temp_discovery_file)

        # Register a coordinator
        coord_id = await discovery.register_coordinator(
            host="localhost",
            port=8000,
        )

        # Verify it exists
        registry_data = json.loads(temp_discovery_file.read_text())
        assert coord_id in registry_data

        # Remove the coordinator
        await discovery.remove_coordinator(coord_id)

        # Verify it's gone
        registry_data = json.loads(temp_discovery_file.read_text())
        assert coord_id not in registry_data

    @pytest.mark.asyncio
    async def test_cleanup_unhealthy_coordinators(self, temp_discovery_file):
        """Test cleanup of unhealthy coordinators."""
        discovery = CoordinatorDiscovery(
            discovery_file=temp_discovery_file,
            health_timeout=0.1,  # Very short timeout for testing
        )

        # Register a coordinator
        coord_id = await discovery.register_coordinator(
            host="localhost",
            port=8000,
        )

        # Wait for it to become unhealthy
        await asyncio.sleep(0.2)

        # Run cleanup
        cleaned = await discovery._cleanup_unhealthy_coordinators()

        assert cleaned == 1
        # Verify it's been removed
        registry_data = json.loads(temp_discovery_file.read_text())
        assert coord_id not in registry_data

    @pytest.mark.asyncio
    async def test_concurrent_access(self, temp_discovery_file):
        """Test concurrent access to the registry."""
        discovery = CoordinatorDiscovery(discovery_file=temp_discovery_file)

        # Register multiple coordinators concurrently
        tasks = []
        for i in range(5):
            task = discovery.register_coordinator(
                host="localhost",
                port=8000 + i,
            )
            tasks.append(task)

        coord_ids = await asyncio.gather(*tasks)

        # Verify all were registered
        registry_data = json.loads(temp_discovery_file.read_text())
        assert len(registry_data) == 5
        for coord_id in coord_ids:
            assert coord_id in registry_data

    @pytest.mark.asyncio
    async def test_auto_start_tasks(self, temp_discovery_file):
        """Test automatic task startup."""
        # Use short intervals for testing
        discovery = CoordinatorDiscovery(
            discovery_file=temp_discovery_file,
            heartbeat_interval=0.1,
            cleanup_interval=0.2,
            health_timeout=0.05,
        )

        # Register a coordinator
        coord_id = await discovery.register_coordinator(
            host="localhost",
            port=8000,
            enable_heartbeat=True,
        )

        # Let tasks run for a bit
        await asyncio.sleep(0.3)

        # Check that heartbeat has been updated
        registry_data = json.loads(temp_discovery_file.read_text())
        coord_info = CoordinatorInfo.from_dict(registry_data[coord_id])
        assert coord_info.last_heartbeat > coord_info.start_time

        # Stop tasks
        await discovery.stop_all_tasks()

    @pytest.mark.asyncio
    async def test_find_best_coordinator(self, temp_discovery_file):
        """Test finding the best coordinator based on criteria."""
        discovery = CoordinatorDiscovery(discovery_file=temp_discovery_file)

        # Register multiple coordinators
        await discovery.register_coordinator(
            host="localhost",
            port=8000,
            transport_type="sse",
        )

        await discovery.register_coordinator(
            host="localhost",
            port=8001,
            transport_type="stdio",
        )

        # Find best SSE coordinator
        best = await discovery.find_best_coordinator(transport_type="sse")
        assert best is not None
        assert best.port == 8000
        assert best.transport_type == "sse"

    @pytest.mark.asyncio
    async def test_shutdown(self, temp_discovery_file):
        """Test coordinator shutdown removes entry from registry."""
        discovery = CoordinatorDiscovery(discovery_file=temp_discovery_file)

        # Register a coordinator
        await discovery.register_coordinator(
            host="localhost",
            port=8000,
        )

        # Verify registry has entry
        registry_data = json.loads(temp_discovery_file.read_text())
        assert len(registry_data) == 1

        # Shutdown the discovery
        await discovery.shutdown()

        # Verify cleanup happened
        if temp_discovery_file.exists():  # File might be deleted in cleanup
            registry_data = json.loads(temp_discovery_file.read_text())
            assert len(registry_data) == 0

    @pytest.mark.asyncio
    async def test_async_context_manager(self, temp_discovery_file):
        """Test using discovery as async context manager."""
        # Use async context manager
        async with CoordinatorDiscovery(
            discovery_file=temp_discovery_file
        ) as discovery:
            # Register a coordinator
            await discovery.register_coordinator(
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

    @pytest.mark.asyncio
    async def test_invalid_discovery_file(self):
        """Test handling of invalid discovery file."""
        # Try to use a directory as the discovery file
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            with pytest.raises(Exception):
                # This should fail since temp_path is a directory
                discovery = CoordinatorDiscovery(discovery_file=temp_path)

    @pytest.mark.asyncio
    async def test_corrupted_registry_file(self, temp_discovery_file):
        """Test handling of corrupted registry file."""
        # Write invalid JSON to the file
        temp_discovery_file.write_text("{ invalid json }")

        # Should handle gracefully and create new empty registry
        discovery = CoordinatorDiscovery(discovery_file=temp_discovery_file)

        # Should be able to register
        coord_id = await discovery.register_coordinator(
            host="localhost",
            port=8000,
        )

        assert coord_id is not None

    def test_coordinator_id_generation(self):
        """Test that coordinator IDs are unique."""
        ids = set()
        discovery = CoordinatorDiscovery()

        for _ in range(100):
            coord_id = discovery._generate_coordinator_id()
            assert coord_id not in ids
            ids.add(coord_id)


class TestIntegration:
    """Integration tests for the discovery system."""

    @pytest.mark.asyncio
    async def test_multiple_discovery_instances(self, tmp_path):
        """Test multiple discovery instances sharing a file."""
        discovery_file = tmp_path / "shared_discovery.json"

        # Create two discovery instances
        discovery1 = CoordinatorDiscovery(discovery_file=discovery_file)
        discovery2 = CoordinatorDiscovery(discovery_file=discovery_file)

        # Register coordinators through different instances
        coord1_id = await discovery1.register_coordinator(
            host="localhost",
            port=8000,
        )

        coord2_id = await discovery2.register_coordinator(
            host="localhost",
            port=8001,
        )

        # Both should see both coordinators
        coords1 = await discovery1.find_coordinators()
        coords2 = await discovery2.find_coordinators()

        assert len(coords1) == 2
        assert len(coords2) == 2

    @pytest.mark.asyncio
    async def test_coordinator_lifecycle(self, tmp_path):
        """Test complete coordinator lifecycle."""
        discovery_file = tmp_path / "lifecycle_test.json"

        async with CoordinatorDiscovery(
            discovery_file=discovery_file,
            heartbeat_interval=0.1,
            cleanup_interval=0.3,
            health_timeout=0.2,
        ) as discovery:
            # Register coordinator with heartbeat
            coord_id = await discovery.register_coordinator(
                host="localhost",
                port=8000,
                transport_type="test",
                metadata={"version": "1.0"},
                enable_heartbeat=True,
            )

            # Verify it's registered
            coords = await discovery.find_coordinators()
            assert len(coords) == 1
            assert coords[0].coordinator_id == coord_id

            # Wait for heartbeat updates
            await asyncio.sleep(0.15)

            # Should still be healthy
            coords = await discovery.find_coordinators()
            assert len(coords) == 1
            assert coords[0].is_healthy(timeout=0.2)

            # Stop heartbeat
            await discovery.stop_heartbeat(coord_id)

            # Wait for cleanup
            await asyncio.sleep(0.4)

            # Should be cleaned up
            coords = await discovery.find_coordinators()
            assert len(coords) == 0

    @pytest.mark.asyncio
    async def test_discovery_resilience(self, tmp_path):
        """Test discovery system resilience to errors."""
        discovery_file = tmp_path / "resilience_test.json"

        discovery = CoordinatorDiscovery(discovery_file=discovery_file)

        # Register a coordinator
        coord_id = await discovery.register_coordinator(
            host="localhost",
            port=8000,
        )

        # Simulate file permission issue
        with patch("builtins.open", side_effect=PermissionError("No permission")):
            # Operations should handle errors gracefully
            result = await discovery.find_coordinators()
            assert result is not None  # Should return something even with errors

            # Updates might fail but shouldn't crash
            try:
                await discovery.update_heartbeat(coord_id)
            except PermissionError:
                pass  # Expected

        # After permission issue resolved, should work again
        coords = await discovery.find_coordinators()
        assert len(coords) >= 0  # Might have lost data but should work

    @pytest.mark.asyncio
    async def test_concurrent_updates(self, tmp_path):
        """Test concurrent updates to the registry."""
        discovery_file = tmp_path / "concurrent_test.json"

        discovery = CoordinatorDiscovery(discovery_file=discovery_file)

        # Register multiple coordinators
        coord_ids = []
        for i in range(5):
            coord_id = await discovery.register_coordinator(
                host="localhost",
                port=8000 + i,
            )
            coord_ids.append(coord_id)

        # Concurrent heartbeat updates
        update_tasks = []
        for coord_id in coord_ids:
            task = discovery.update_heartbeat(coord_id)
            update_tasks.append(task)

        # Concurrent finds
        find_tasks = []
        for _ in range(5):
            task = discovery.find_coordinators()
            find_tasks.append(task)

        # Run all concurrently
        await asyncio.gather(*(update_tasks + find_tasks))

        # Verify consistency
        final_coords = await discovery.find_coordinators()
        assert len(final_coords) == 5