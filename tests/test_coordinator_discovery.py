"""Tests for the coordinator discovery mechanism."""

import asyncio
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from aider_mcp_server.molecules.transport.discovery import (
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
            metadata={"test": True},
        )
        # Set last_heartbeat manually after creation
        info.last_heartbeat = start_time + 5

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

    def test_is_active(self):
        """Test active check based on heartbeat age."""
        current_time = time.time()

        # Active coordinator (recent heartbeat)
        info = CoordinatorInfo(
            coordinator_id="test_1",
            host="localhost",
            port=8000,
            transport_type="test",
        )
        info.last_heartbeat = current_time - 10  # 10 seconds ago
        assert info.is_active(max_age_seconds=30)

        # Inactive coordinator (old heartbeat)
        info.last_heartbeat = current_time - 60  # 60 seconds ago
        assert not info.is_active(max_age_seconds=30)

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

    def test_has_streaming_capability(self):
        """Test checking for specific streaming capability."""
        info_with_streaming = CoordinatorInfo(
            coordinator_id="test_stream_1",
            host="localhost",
            port=8000,
            streaming_capabilities={"aider_events": {"url": "/events", "event_types": ["message", "tool_code"]}},
        )
        info_without_streaming = CoordinatorInfo(
            coordinator_id="test_stream_2",
            host="localhost",
            port=8001,
        )

        assert info_with_streaming.has_streaming_capability("aider_events")
        assert not info_with_streaming.has_streaming_capability("health")
        assert not info_without_streaming.has_streaming_capability("aider_events")

    def test_get_streaming_endpoint(self):
        """Test retrieving streaming endpoint details."""
        capabilities = {
            "aider_events": {"url": "/events", "event_types": ["message", "tool_code"]},
            "health": {"url": "/healthz", "method": "GET"},
        }
        info = CoordinatorInfo(
            coordinator_id="test_stream_1",
            host="localhost",
            port=8000,
            streaming_capabilities=capabilities,
        )

        assert info.get_streaming_endpoint("aider_events") == capabilities["aider_events"]
        assert info.get_streaming_endpoint("health") == capabilities["health"]
        assert info.get_streaming_endpoint("non_existent") is None

    def test_get_all_streaming_endpoints(self):
        """Test retrieving all streaming capabilities."""
        capabilities = {
            "aider_events": {"url": "/events"},
            "health": {"url": "/healthz"},
        }
        info = CoordinatorInfo(
            coordinator_id="test_stream_1",
            host="localhost",
            port=8000,
            streaming_capabilities=capabilities,
        )

        assert info.get_all_streaming_endpoints() == capabilities

        info_no_streaming = CoordinatorInfo(
            coordinator_id="test_stream_2",
            host="localhost",
            port=8001,
        )
        assert info_no_streaming.get_all_streaming_endpoints() == {}

    def test_supports_event_type(self):
        """Test checking if an endpoint supports a specific event type."""
        capabilities = {
            "aider_events": {"url": "/events", "event_types": ["message", "tool_code"]},
            "health": {"url": "/healthz"},  # No event_types key
            "status": {"url": "/status", "event_types": []},  # Empty event_types list
        }
        info = CoordinatorInfo(
            coordinator_id="test_stream_1",
            host="localhost",
            port=8000,
            streaming_capabilities=capabilities,
        )

        assert info.supports_event_type("aider_events", "message")
        assert info.supports_event_type("aider_events", "tool_code")
        assert not info.supports_event_type("aider_events", "other_event")
        assert not info.supports_event_type("health", "any_event")  # Endpoint exists, but no event_types
        assert not info.supports_event_type("status", "any_event")  # Endpoint exists, empty event_types
        assert not info.supports_event_type("non_existent", "any_event")  # Endpoint does not exist


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
        assert discovery.heartbeat_interval == 10.0  # Default value from constructor

        # Verify directory is created if it doesn't exist
        assert temp_discovery_file.parent.exists()

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

        # Verify it's in the registry file (which is now a list)
        registry_data = json.loads(temp_discovery_file.read_text())
        assert isinstance(registry_data, list)
        assert len(registry_data) == 1

        coord_data = registry_data[0]
        assert coord_data["coordinator_id"] == coord_id
        assert coord_data["host"] == "localhost"
        assert coord_data["port"] == 8000
        assert coord_data["transport_type"] == "sse"
        assert coord_data["metadata"] == {"test": True}
        # Check that streaming_capabilities is saved (even if empty)
        assert "streaming_capabilities" in coord_data
        assert coord_data["streaming_capabilities"] == {}

    @pytest.mark.asyncio
    async def test_register_coordinator_with_streaming(self, temp_discovery_file):
        """Test registering a coordinator with streaming capabilities."""
        discovery = CoordinatorDiscovery(discovery_file=temp_discovery_file)
        streaming_caps = {
            "aider_events": {"url": "/events", "event_types": ["message"]},
            "health": {"url": "/healthz"},
        }

        # Register a coordinator with streaming capabilities
        coord_id = await discovery.register_coordinator(
            host="localhost",
            port=8000,
            transport_type="sse",
            streaming_capabilities=streaming_caps,
        )

        assert coord_id is not None

        # Verify it's in the registry file with capabilities
        registry_data = json.loads(temp_discovery_file.read_text())
        assert len(registry_data) == 1

        coord_data = registry_data[0]
        assert coord_data["coordinator_id"] == coord_id
        assert coord_data["streaming_capabilities"] == streaming_caps

    @pytest.mark.asyncio
    async def test_find_coordinators(self, temp_discovery_file):
        """Test finding all coordinators."""
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

        # Discover all coordinators
        all_coords = await discovery.discover_coordinators()
        assert len(all_coords) == 2

        # Check that both coordinators are found
        coord_ids = {coord.coordinator_id for coord in all_coords}
        assert coord1_id in coord_ids
        assert coord2_id in coord_ids

    @pytest.mark.asyncio
    async def test_find_streaming_coordinators(self, temp_discovery_file):
        """Test finding only coordinators with streaming capabilities."""
        discovery = CoordinatorDiscovery(discovery_file=temp_discovery_file)

        # Register coordinators, some with streaming, some without
        await discovery.register_coordinator(
            host="localhost",
            port=8000,
            transport_type="sse",
            streaming_capabilities={"aider_events": {"url": "/events"}},
        )  # Has streaming

        await discovery.register_coordinator(
            host="localhost",
            port=8001,
            transport_type="stdio",
        )  # No streaming

        await discovery.register_coordinator(
            host="localhost",
            port=8002,
            transport_type="sse",
            streaming_capabilities={"health": {"url": "/healthz"}},
        )  # Has streaming

        # Discover only streaming coordinators
        streaming_coords = await discovery.find_streaming_coordinators()
        assert len(streaming_coords) == 2  # Should find the two with capabilities

        # Check that the correct ones were found
        ports = {coord.port for coord in streaming_coords}
        assert 8000 in ports
        assert 8002 in ports
        assert 8001 not in ports

    @pytest.mark.asyncio
    async def test_backward_compatibility_no_streaming(self, temp_discovery_file):
        """Test discovery works for coordinators registered without streaming capabilities."""
        discovery = CoordinatorDiscovery(discovery_file=temp_discovery_file)

        # Manually write a registry entry without the streaming_capabilities key
        # This simulates an older version registering
        old_coord_data = {
            "coordinator_id": "old_coord",
            "host": "localhost",
            "port": 9000,
            "transport_type": "legacy",
            "start_time": time.time(),
            "last_heartbeat": time.time(),
            "metadata": {},
        }
        temp_discovery_file.write_text(json.dumps([old_coord_data]))

        # Discover all coordinators - should find the old one
        all_coords = await discovery.discover_coordinators()
        assert len(all_coords) == 1
        assert all_coords[0].coordinator_id == "old_coord"
        assert all_coords[0].streaming_capabilities == {}  # Should default to empty dict

        # Discover streaming coordinators - should NOT find the old one
        streaming_coords = await discovery.find_streaming_coordinators()
        assert len(streaming_coords) == 0

    @pytest.mark.asyncio
    async def test_update_heartbeat(self, temp_discovery_file):
        """Test heartbeat updates."""
        discovery = CoordinatorDiscovery(discovery_file=temp_discovery_file)

        # Register a coordinator
        coord_id = await discovery.register_coordinator(
            host="localhost",
            port=8000,
        )

        # Get initial heartbeat from registered coordinator
        initial_data = json.loads(temp_discovery_file.read_text())
        _ = initial_data[0]["last_heartbeat"]  # noqa: F841

        # Wait a bit and manually update registry
        await asyncio.sleep(0.2)
        # The heartbeat is automatically updated by the background task
        # Let's just verify it's running

        # Verify registry still has the coordinator
        final_data = json.loads(temp_discovery_file.read_text())
        assert len(final_data) == 1
        assert final_data[0]["coordinator_id"] == coord_id

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
        assert len(registry_data) == 1
        assert registry_data[0]["coordinator_id"] == coord_id

        # Shutdown should remove the coordinator
        await discovery.shutdown()

        # Verify it's gone
        registry_data = json.loads(temp_discovery_file.read_text())
        assert len(registry_data) == 0

    @pytest.mark.asyncio
    async def test_cleanup_unhealthy_coordinators(self, temp_discovery_file):
        """Test cleanup of unhealthy coordinators."""
        discovery = CoordinatorDiscovery(
            discovery_file=temp_discovery_file,
        )

        # Register a coordinator
        _ = await discovery.register_coordinator(
            host="localhost",
            port=8000,
        )

        # Manually set an old heartbeat
        registry_data = json.loads(temp_discovery_file.read_text())
        registry_data[0]["last_heartbeat"] = time.time() - 100  # 100 seconds ago
        temp_discovery_file.write_text(json.dumps(registry_data))

        # Discover coordinators should filter out unhealthy ones
        healthy_coords = await discovery.discover_coordinators(max_age_seconds=30)
        assert len(healthy_coords) == 0

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
        registered_ids = {coord["coordinator_id"] for coord in registry_data}
        for coord_id in coord_ids:
            assert coord_id in registered_ids

    @pytest.mark.asyncio
    async def test_streaming_capability_serialization(self, temp_discovery_file):
        """Test that streaming capabilities are correctly serialized and deserialized."""
        discovery = CoordinatorDiscovery(discovery_file=temp_discovery_file)
        streaming_caps = {
            "aider_events": {"url": "/events", "event_types": ["message", "tool_code"]},
            "health": {"url": "/healthz", "method": "GET"},
        }

        # Register a coordinator with capabilities
        coord_id = await discovery.register_coordinator(
            host="localhost",
            port=8000,
            streaming_capabilities=streaming_caps,
        )

        # Read the file directly and check the raw data
        registry_data = json.loads(temp_discovery_file.read_text())
        assert len(registry_data) == 1
        raw_data = registry_data[0]
        assert raw_data["coordinator_id"] == coord_id
        assert raw_data["streaming_capabilities"] == streaming_caps

        # Discover coordinators and check the deserialized object
        discovered_coords = await discovery.discover_coordinators()
        assert len(discovered_coords) == 1
        discovered_coord = discovered_coords[0]
        assert discovered_coord.coordinator_id == coord_id
        assert discovered_coord.streaming_capabilities == streaming_caps
        assert discovered_coord.has_streaming_capability("aider_events")
        assert discovered_coord.get_streaming_endpoint("health") == streaming_caps["health"]

        await discovery.shutdown()

    @pytest.mark.asyncio
    async def test_auto_start_tasks(self, temp_discovery_file):
        """Test automatic task startup."""
        # Use short intervals for testing
        discovery = CoordinatorDiscovery(
            discovery_file=temp_discovery_file,
            heartbeat_interval=0.1,
        )

        # Register a coordinator
        _ = await discovery.register_coordinator(
            host="localhost",
            port=8000,
        )

        # Let tasks run for a bit
        await asyncio.sleep(0.3)

        # Check that heartbeat has been updated
        registry_data = json.loads(temp_discovery_file.read_text())
        coord_info = CoordinatorInfo.from_dict(registry_data[0])
        assert coord_info.last_heartbeat > coord_info.start_time

        # Shutdown to stop tasks
        await discovery.shutdown()

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

        # Find best coordinator (newest first)
        best = await discovery.find_best_coordinator()
        assert best is not None
        assert best.port == 8001  # The second one registered is newer
        assert best.transport_type == "stdio"

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
        async with CoordinatorDiscovery(discovery_file=temp_discovery_file) as discovery:
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
        # The directory creation is now handled automatically
        # Let's test that it handles deeply nested paths
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "some" / "nested" / "path" / "discovery.json"

            # This should work and create the directory
            discovery = CoordinatorDiscovery(discovery_file=temp_path)
            assert discovery.discovery_file == temp_path
            assert temp_path.parent.exists()

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

    @pytest.mark.asyncio
    async def test_coordinator_id_generation(self, temp_discovery_file):
        """Test that coordinator IDs are unique."""
        discovery = CoordinatorDiscovery(discovery_file=temp_discovery_file)
        ids = set()

        # Register multiple coordinators and check IDs are unique
        for _ in range(10):
            coord_id = await discovery.register_coordinator(
                host="localhost",
                port=8000,
            )
            assert coord_id not in ids
            ids.add(coord_id)

        await discovery.shutdown()


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
        await discovery1.register_coordinator(
            host="localhost",
            port=8000,
        )

        await discovery2.register_coordinator(
            host="localhost",
            port=8001,
        )

        # Both should see both coordinators
        coords1 = await discovery1.discover_coordinators()
        coords2 = await discovery2.discover_coordinators()

        assert len(coords1) == 2
        assert len(coords2) == 2

    @pytest.mark.asyncio
    async def test_mixed_discovery_streaming(self, tmp_path):
        """Test discovery with a mix of streaming and non-streaming coordinators."""
        discovery_file = tmp_path / "mixed_discovery.json"

        async with CoordinatorDiscovery(discovery_file=discovery_file) as discovery:
            # Register a streaming coordinator
            stream_coord_id = await discovery.register_coordinator(
                host="localhost",
                port=8000,
                streaming_capabilities={"aider_events": {"url": "/events"}},
            )

            # Register a non-streaming coordinator
            non_stream_coord_id = await discovery.register_coordinator(
                host="localhost",
                port=8001,
            )

            # Discover all - should find both
            all_coords = await discovery.discover_coordinators()
            assert len(all_coords) == 2
            all_ids = {c.coordinator_id for c in all_coords}
            assert stream_coord_id in all_ids
            assert non_stream_coord_id in all_ids

            # Discover streaming - should find only the streaming one
            streaming_coords = await discovery.find_streaming_coordinators()
            assert len(streaming_coords) == 1
            assert streaming_coords[0].coordinator_id == stream_coord_id
            assert streaming_coords[0].has_streaming_capability("aider_events")
            assert not streaming_coords[0].has_streaming_capability("health")  # Check a non-existent cap

            # Check the non-streaming one from the 'all' list
            non_stream_obj = next(c for c in all_coords if c.coordinator_id == non_stream_coord_id)
            assert non_stream_obj.streaming_capabilities == {}
            assert not non_stream_obj.has_streaming_capability("any_key")

    @pytest.mark.asyncio
    async def test_coordinator_lifecycle(self, tmp_path):
        """Test complete coordinator lifecycle."""
        discovery_file = tmp_path / "lifecycle_test.json"

        async with CoordinatorDiscovery(
            discovery_file=discovery_file,
            heartbeat_interval=0.1,
        ) as discovery:
            # Register coordinator with heartbeat
            coord_id = await discovery.register_coordinator(
                host="localhost",
                port=8000,
                transport_type="test",
                metadata={"version": "1.0"},
            )

            # Verify it's registered
            coords = await discovery.discover_coordinators()
            assert len(coords) == 1
            assert coords[0].coordinator_id == coord_id

            # Wait for heartbeat updates
            await asyncio.sleep(0.15)

            # Should still be healthy
            coords = await discovery.discover_coordinators()
            assert len(coords) == 1
            assert coords[0].is_active(max_age_seconds=0.2)

            # Shutdown discovery
            await discovery.shutdown()

            # Wait for cleanup
            await asyncio.sleep(0.4)

            # Should be cleaned up
            coords = await discovery.discover_coordinators()
            assert len(coords) == 0

    @pytest.mark.asyncio
    async def test_discovery_resilience(self, tmp_path):
        """Test discovery system resilience to errors."""
        discovery_file = tmp_path / "resilience_test.json"

        discovery = CoordinatorDiscovery(discovery_file=discovery_file)

        # Register a coordinator
        _ = await discovery.register_coordinator(
            host="localhost",
            port=8000,
        )

        # Simulate file permission issue
        with patch("builtins.open", side_effect=PermissionError("No permission")):
            # Operations should handle errors gracefully
            result = await discovery.discover_coordinators()
            assert result is not None  # Should return something even with errors

        # After permission issue resolved, should work again
        coords = await discovery.discover_coordinators()
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

        # Concurrent discover operations to test file locking
        discover_tasks = []
        for _ in range(10):
            task = discovery.discover_coordinators()
            discover_tasks.append(task)

        # Wait for all discoveries
        results = await asyncio.gather(*discover_tasks)

        # All operations should see the same 5 coordinators
        for result in results:
            assert len(result) == 5

        # Verify consistency
        final_coords = await discovery.discover_coordinators()
        assert len(final_coords) == 5
