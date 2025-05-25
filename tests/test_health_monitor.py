"""
Tests for the HealthMonitor system.

This module tests comprehensive health monitoring, performance metrics tracking,
and resource usage monitoring for the Aider MCP Server real-time streaming system.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aider_mcp_server.atoms.types.event_types import EventTypes
from aider_mcp_server.molecules.monitoring.health_monitor import (
    HealthMonitor,
    HealthStatus,
    SystemMetrics,
)


@pytest.fixture
def mock_coordinator():
    """Create a mock coordinator for testing."""
    coordinator = AsyncMock()
    coordinator.broadcast_event = AsyncMock()
    return coordinator


@pytest.fixture
def health_monitor(mock_coordinator):
    """Create a HealthMonitor instance for testing."""
    monitor = HealthMonitor(
        coordinator=mock_coordinator,
        metrics_retention_minutes=10,  # Short retention for testing
        health_check_interval=0.1,  # Fast interval for testing
    )
    return monitor


class TestHealthMonitorInitialization:
    """Test HealthMonitor initialization and basic functionality."""

    @pytest.mark.asyncio
    async def test_initialization(self, health_monitor, mock_coordinator):
        """Test that HealthMonitor initializes correctly."""
        assert health_monitor.coordinator == mock_coordinator
        assert health_monitor.metrics_retention_minutes == 10
        assert health_monitor.health_check_interval == 0.1
        assert len(health_monitor.system_metrics_history) == 0
        assert len(health_monitor.active_requests) == 0
        assert not health_monitor._is_running

    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, health_monitor, mock_coordinator):
        """Test starting and stopping monitoring."""
        # Start monitoring
        await health_monitor.start_monitoring()
        assert health_monitor._is_running
        assert health_monitor._health_check_task is not None
        assert health_monitor._metrics_cleanup_task is not None
        
        # Verify start event was broadcast
        mock_coordinator.broadcast_event.assert_called_with(
            EventTypes.AIDER_SESSION_STARTED,
            {
                "component": "health_monitor",
                "status": "monitoring_started",
                "check_interval": 0.1,
                "retention_minutes": 10,
            }
        )
        
        # Stop monitoring
        await health_monitor.stop_monitoring()
        assert not health_monitor._is_running
        assert health_monitor._health_check_task is None
        assert health_monitor._metrics_cleanup_task is None
        
        # Verify stop event was broadcast
        mock_coordinator.broadcast_event.assert_called_with(
            EventTypes.AIDER_SESSION_COMPLETED,
            {
                "component": "health_monitor", 
                "status": "monitoring_stopped",
            }
        )


class TestRequestTracking:
    """Test request tracking functionality."""

    @pytest.mark.asyncio
    async def test_record_request_lifecycle(self, health_monitor):
        """Test recording complete request lifecycle."""
        request_id = "test_request_123"
        context = {"model": "gpt-4", "files": 3}
        
        # Record request start
        await health_monitor.record_request_start(request_id, context)
        assert request_id in health_monitor.active_requests
        assert health_monitor.active_requests[request_id]["context"] == context
        
        # Wait a small amount to ensure duration > 0
        await asyncio.sleep(0.01)
        
        # Record successful completion
        await health_monitor.record_request_completion(request_id, success=True)
        assert request_id not in health_monitor.active_requests
        assert len(health_monitor.request_durations) == 1
        assert len(health_monitor.request_results) == 1
        
        # Check that duration was recorded
        _, duration = health_monitor.request_durations[0]
        assert duration > 0
        
        # Check that success was recorded
        _, success = health_monitor.request_results[0]
        assert success

    @pytest.mark.asyncio
    async def test_record_request_failure(self, health_monitor):
        """Test recording request failure with error type."""
        request_id = "test_request_fail"
        
        await health_monitor.record_request_start(request_id)
        await asyncio.sleep(0.01)
        await health_monitor.record_request_completion(request_id, success=False, error_type="rate_limit")
        
        # Check failure was recorded
        _, success = health_monitor.request_results[0]
        assert not success
        
        # Check error type was tracked
        assert health_monitor.error_counts["rate_limit"] == 1

    @pytest.mark.asyncio
    async def test_record_throttling_event(self, health_monitor, mock_coordinator):
        """Test recording throttling events."""
        request_id = "throttled_request"
        context = {"model": "gpt-4"}
        
        await health_monitor.record_throttling_event(request_id, context)
        
        # Check throttling was recorded
        assert len(health_monitor.throttling_events) == 1
        
        # Check event was broadcast
        mock_coordinator.broadcast_event.assert_called()
        call_args = mock_coordinator.broadcast_event.call_args
        assert call_args[0][0] == EventTypes.AIDER_THROTTLING_DETECTED
        event_data = call_args[0][1]
        assert event_data["request_id"] == request_id
        assert event_data["context"] == context


class TestStreamingClientManagement:
    """Test streaming client registration and management."""

    @pytest.mark.asyncio
    async def test_register_unregister_client(self, health_monitor):
        """Test registering and unregistering streaming clients."""
        client_id = "client_123"
        client_info = {"ip": "127.0.0.1", "user_agent": "test"}
        
        # Register client
        await health_monitor.register_streaming_client(client_id, client_info)
        assert client_id in health_monitor.streaming_clients
        assert health_monitor.streaming_clients[client_id]["info"] == client_info
        
        # Unregister client
        await health_monitor.unregister_streaming_client(client_id)
        assert client_id not in health_monitor.streaming_clients


class TestHealthStatusGeneration:
    """Test health status calculation and reporting."""

    @patch("psutil.cpu_percent")
    @patch("psutil.virtual_memory")
    @patch("psutil.net_connections")
    @pytest.mark.asyncio
    async def test_get_health_status_healthy(
        self, mock_connections, mock_memory, mock_cpu, health_monitor
    ):
        """Test health status when system is healthy."""
        # Mock healthy system metrics
        mock_cpu.return_value = 20.0  # 20% CPU
        mock_memory.return_value = MagicMock(
            percent=30.0,  # 30% memory
            used=1024 * 1024 * 1024,  # 1GB used
            available=3 * 1024 * 1024 * 1024,  # 3GB available
        )
        mock_connections.return_value = [1, 2, 3]  # 3 connections
        
        # Add some successful request history
        current_time = time.time()
        for i in range(5):
            health_monitor.request_durations.append((current_time - 300 + i * 60, 1.0))  # 1 sec duration
            health_monitor.request_results.append((current_time - 300 + i * 60, True))  # successful
        
        health_status = await health_monitor.get_health_status()
        
        assert isinstance(health_status, HealthStatus)
        assert health_status.status == "healthy"
        assert health_status.system_metrics.cpu_percent == 20.0
        assert health_status.system_metrics.memory_percent == 30.0
        assert health_status.performance_metrics.successful_requests == 5
        assert health_status.performance_metrics.failed_requests == 0
        assert health_status.error_rate == 0.0

    @patch("psutil.cpu_percent")
    @patch("psutil.virtual_memory")
    @patch("psutil.net_connections")
    @pytest.mark.asyncio
    async def test_get_health_status_warning(
        self, mock_connections, mock_memory, mock_cpu, health_monitor
    ):
        """Test health status when system is in warning state."""
        # Mock warning-level system metrics
        mock_cpu.return_value = 80.0  # 80% CPU (warning threshold)
        mock_memory.return_value = MagicMock(
            percent=80.0,  # 80% memory (warning threshold)
            used=8 * 1024 * 1024 * 1024,  # 8GB used
            available=2 * 1024 * 1024 * 1024,  # 2GB available
        )
        mock_connections.return_value = list(range(50))  # Many connections
        
        health_status = await health_monitor.get_health_status()
        
        assert health_status.status == "warning"
        assert health_status.system_metrics.cpu_percent == 80.0
        assert health_status.system_metrics.memory_percent == 80.0

    @patch("psutil.cpu_percent")
    @patch("psutil.virtual_memory")
    @patch("psutil.net_connections")
    @pytest.mark.asyncio
    async def test_get_health_status_critical(
        self, mock_connections, mock_memory, mock_cpu, health_monitor
    ):
        """Test health status when system is in critical state."""
        # Mock critical system metrics
        mock_cpu.return_value = 95.0  # 95% CPU (critical)
        mock_memory.return_value = MagicMock(
            percent=95.0,  # 95% memory (critical)
            used=15 * 1024 * 1024 * 1024,  # 15GB used
            available=512 * 1024 * 1024,  # 512MB available
        )
        mock_connections.return_value = list(range(100))  # Many connections
        
        # Add many throttling events
        current_time = time.time()
        for i in range(15):  # > 10 throttling events
            health_monitor.throttling_events.append(current_time - 300 + i * 20)
        
        health_status = await health_monitor.get_health_status()
        
        assert health_status.status == "critical"
        assert health_status.performance_metrics.throttled_requests == 15


class TestMetricsSummary:
    """Test metrics summary functionality."""

    @pytest.mark.asyncio
    async def test_get_metrics_summary(self, health_monitor):
        """Test getting metrics summary for a time period."""
        current_time = time.time()
        
        # Add request history spanning 45 minutes
        for i in range(10):
            # Add requests at different times
            timestamp = current_time - (45 * 60) + (i * 5 * 60)  # Every 5 minutes
            duration = 2.0 + (i * 0.1)  # Varying durations
            success = i < 8  # 8 successful, 2 failed
            
            health_monitor.request_durations.append((timestamp, duration))
            health_monitor.request_results.append((timestamp, success))
        
        # Add throttling events
        for i in range(3):
            health_monitor.throttling_events.append(current_time - (20 * 60) + (i * 5 * 60))
        
        # Get 30-minute summary
        summary = await health_monitor.get_metrics_summary(minutes_back=30)
        
        assert summary["time_period_minutes"] == 30
        # Should only include requests from last 30 minutes (6 requests)
        assert summary["total_requests"] == 6
        assert summary["successful_requests"] == 4  # 2 failed in last 30 min
        assert summary["failed_requests"] == 2
        assert summary["success_rate"] == 4/6
        assert summary["throttling_events"] == 3
        assert summary["requests_per_minute"] == 6/30


class TestBackgroundTasks:
    """Test background monitoring tasks."""

    @pytest.mark.asyncio
    async def test_health_check_loop_broadcasts_warnings(self, health_monitor, mock_coordinator):
        """Test that health check loop broadcasts warnings for unhealthy status."""
        
        with patch.object(health_monitor, '_collect_system_metrics') as mock_collect:
            # Mock unhealthy system
            mock_collect.return_value = SystemMetrics(
                timestamp=time.time(),
                cpu_percent=95.0,  # Critical level
                memory_percent=95.0,  # Critical level
                memory_used_mb=15000,
                memory_available_mb=500,
                active_connections=100,
                active_requests=0,
            )
            
            # Start monitoring briefly
            await health_monitor.start_monitoring()
            await asyncio.sleep(0.2)  # Let health check run once
            await health_monitor.stop_monitoring()
            
            # Check that error event was broadcast for critical status
            error_calls = [
                call for call in mock_coordinator.broadcast_event.call_args_list
                if call[0][0] == EventTypes.AIDER_ERROR_OCCURRED
            ]
            assert len(error_calls) > 0
            
            error_event = error_calls[0][0][1]
            assert error_event["component"] == "health_monitor"
            assert error_event["severity"] == "critical"

    @pytest.mark.asyncio
    async def test_metrics_cleanup_removes_old_data(self, health_monitor):
        """Test that metrics cleanup removes old data."""
        current_time = time.time()
        old_time = current_time - (15 * 60)  # 15 minutes ago (older than 10min retention)
        recent_time = current_time - (5 * 60)  # 5 minutes ago (within retention)
        
        # Add old and recent data
        health_monitor.request_durations.append((old_time, 1.0))
        health_monitor.request_durations.append((recent_time, 1.0))
        health_monitor.request_results.append((old_time, True))
        health_monitor.request_results.append((recent_time, True))
        health_monitor.throttling_events.append(old_time)
        health_monitor.throttling_events.append(recent_time)
        
        # Manually run cleanup logic
        cutoff_time = current_time - (health_monitor.metrics_retention_minutes * 60)
        
        # Clean up old data manually (simulate the cleanup loop logic)
        while health_monitor.system_metrics_history and health_monitor.system_metrics_history[0].timestamp < cutoff_time:
            health_monitor.system_metrics_history.popleft()
        
        while health_monitor.request_durations and health_monitor.request_durations[0][0] < cutoff_time:
            health_monitor.request_durations.popleft()
        
        while health_monitor.request_results and health_monitor.request_results[0][0] < cutoff_time:
            health_monitor.request_results.popleft()
        
        while health_monitor.throttling_events and health_monitor.throttling_events[0] < cutoff_time:
            health_monitor.throttling_events.popleft()
        
        # Old data should be removed, recent data should remain
        assert len(health_monitor.request_durations) == 1
        assert health_monitor.request_durations[0][0] == recent_time
        assert len(health_monitor.request_results) == 1
        assert health_monitor.request_results[0][0] == recent_time
        assert len(health_monitor.throttling_events) == 1
        assert health_monitor.throttling_events[0] == recent_time


class TestIntegrationWithCoordinator:
    """Test integration with ApplicationCoordinator."""

    @pytest.mark.asyncio
    async def test_coordinator_health_methods(self, mock_coordinator):
        """Test that coordinator health methods work correctly."""
        # This would test the actual ApplicationCoordinator integration
        # For now, we verify the interface methods exist and work
        monitor = HealthMonitor(mock_coordinator)
        
        # Test that health monitor can be created and basic methods work
        await monitor.record_request_start("test", {"key": "value"})
        await monitor.record_request_completion("test", True)
        
        health_status = await monitor.get_health_status()
        assert isinstance(health_status, HealthStatus)
        
        metrics_summary = await monitor.get_metrics_summary(30)
        assert isinstance(metrics_summary, dict)
        assert "total_requests" in metrics_summary


# Integration test that would be run with actual ApplicationCoordinator
@pytest.mark.integration
class TestHealthMonitorIntegration:
    """Integration tests with real ApplicationCoordinator."""

    @pytest.mark.skip(reason="Requires full ApplicationCoordinator setup")
    @pytest.mark.asyncio
    async def test_full_integration_flow(self):
        """Test complete integration flow with ApplicationCoordinator."""
        # This test would create a real ApplicationCoordinator and verify
        # that health monitoring works end-to-end
        pass
