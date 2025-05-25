"""
Advanced Health Monitor for comprehensive system monitoring.

This module implements health monitoring endpoints, performance metrics tracking,
and resource usage monitoring for the Aider MCP Server real-time streaming system.
"""

import asyncio
import psutil
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from aider_mcp_server.atoms.types.event_types import EventTypes
from aider_mcp_server.interfaces.application_coordinator import IApplicationCoordinator


@dataclass
class SystemMetrics:
    """System resource metrics snapshot."""
    
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    active_connections: int
    active_requests: int


@dataclass
class PerformanceMetrics:
    """Performance metrics for requests and operations."""
    
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_duration: float
    p95_duration: float
    throttled_requests: int
    requests_per_minute: float


@dataclass
class HealthStatus:
    """Overall health status of the system."""
    
    status: str  # "healthy", "warning", "critical"
    timestamp: float
    system_metrics: SystemMetrics
    performance_metrics: PerformanceMetrics
    active_coordinators: int
    streaming_clients: int
    error_rate: float
    details: Dict[str, Any]


class HealthMonitor:
    """Advanced health monitoring with metrics collection and endpoint support."""

    def __init__(
        self,
        coordinator: IApplicationCoordinator,
        metrics_retention_minutes: int = 60,
        health_check_interval: float = 30.0,
    ):
        """
        Initialize the health monitor.

        Args:
            coordinator: ApplicationCoordinator for event broadcasting and system access
            metrics_retention_minutes: How long to retain metrics history
            health_check_interval: Interval between automated health checks
        """
        self.coordinator = coordinator
        self.metrics_retention_minutes = metrics_retention_minutes
        self.health_check_interval = health_check_interval
        
        # Metrics storage with time-based retention
        self.system_metrics_history: deque[SystemMetrics] = deque()
        self.request_durations: deque[Tuple[float, float]] = deque()  # (timestamp, duration)
        self.request_results: deque[Tuple[float, bool]] = deque()  # (timestamp, success)
        self.throttling_events: deque[float] = deque()  # timestamps
        
        # Current state tracking
        self.active_requests: Dict[str, Dict[str, Any]] = {}
        self.streaming_clients: Dict[str, Dict[str, Any]] = {}
        self.error_counts: Dict[str, int] = defaultdict(int)
        
        # Background tasks
        self._health_check_task: Optional["asyncio.Task[None]"] = None
        self._metrics_cleanup_task: Optional["asyncio.Task[None]"] = None
        self._is_running = False

    async def start_monitoring(self) -> None:
        """Start background health monitoring tasks."""
        if self._is_running:
            return
            
        self._is_running = True
        
        # Start health check task
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        # Start metrics cleanup task
        self._metrics_cleanup_task = asyncio.create_task(self._metrics_cleanup_loop())
        
        await self.coordinator.broadcast_event(
            EventTypes.AIDER_SESSION_STARTED,
            {
                "component": "health_monitor",
                "status": "monitoring_started",
                "check_interval": self.health_check_interval,
                "retention_minutes": self.metrics_retention_minutes,
            }
        )

    async def stop_monitoring(self) -> None:
        """Stop background health monitoring tasks."""
        self._is_running = False
        
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None
            
        if self._metrics_cleanup_task:
            self._metrics_cleanup_task.cancel()
            try:
                await self._metrics_cleanup_task
            except asyncio.CancelledError:
                pass
            self._metrics_cleanup_task = None

        await self.coordinator.broadcast_event(
            EventTypes.AIDER_SESSION_COMPLETED,
            {
                "component": "health_monitor", 
                "status": "monitoring_stopped",
            }
        )

    async def get_health_status(self) -> HealthStatus:
        """Get comprehensive current health status."""
        current_time = time.time()
        
        # Collect system metrics
        system_metrics = await self._collect_system_metrics()
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(current_time)
        
        # Determine overall health status
        status = self._determine_health_status(system_metrics, performance_metrics)
        
        # Get coordinator and client counts
        active_coordinators = await self._get_active_coordinator_count()
        streaming_clients_count = len(self.streaming_clients)
        
        # Calculate error rate
        error_rate = self._calculate_error_rate(current_time)
        
        health_status = HealthStatus(
            status=status,
            timestamp=current_time,
            system_metrics=system_metrics,
            performance_metrics=performance_metrics,
            active_coordinators=active_coordinators,
            streaming_clients=streaming_clients_count,
            error_rate=error_rate,
            details={
                "active_requests": len(self.active_requests),
                "recent_throttling_events": len([t for t in self.throttling_events if current_time - t < 300]),  # Last 5 minutes
                "error_breakdown": dict(self.error_counts),
                "monitoring_uptime": current_time if self._is_running else 0,
            }
        )
        
        return health_status

    async def record_request_start(self, request_id: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Record the start of a request for performance tracking."""
        self.active_requests[request_id] = {
            "start_time": time.time(),
            "context": context or {},
        }

    async def record_request_completion(
        self, 
        request_id: str, 
        success: bool, 
        error_type: Optional[str] = None
    ) -> None:
        """Record the completion of a request."""
        current_time = time.time()
        
        if request_id in self.active_requests:
            start_time = self.active_requests[request_id]["start_time"]
            duration = current_time - start_time
            
            # Store metrics
            self.request_durations.append((current_time, duration))
            self.request_results.append((current_time, success))
            
            # Track errors
            if not success and error_type:
                self.error_counts[error_type] += 1
            
            # Remove from active requests
            del self.active_requests[request_id]

    async def record_throttling_event(self, request_id: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Record a throttling event for metrics tracking."""
        current_time = time.time()
        self.throttling_events.append(current_time)
        
        await self.coordinator.broadcast_event(
            EventTypes.AIDER_THROTTLING_DETECTED,
            {
                "request_id": request_id,
                "timestamp": current_time,
                "context": context or {},
                "recent_throttling_count": len([t for t in self.throttling_events if current_time - t < 300]),
            }
        )

    async def register_streaming_client(self, client_id: str, client_info: Dict[str, Any]) -> None:
        """Register a new streaming client for monitoring."""
        self.streaming_clients[client_id] = {
            "connected_at": time.time(),
            "info": client_info,
        }

    async def unregister_streaming_client(self, client_id: str) -> None:
        """Unregister a streaming client."""
        if client_id in self.streaming_clients:
            del self.streaming_clients[client_id]

    async def get_metrics_summary(self, minutes_back: int = 30) -> Dict[str, Any]:
        """Get a summary of metrics for the specified time period."""
        current_time = time.time()
        cutoff_time = current_time - (minutes_back * 60)
        
        # Filter recent data
        recent_durations = [(t, d) for t, d in self.request_durations if t >= cutoff_time]
        recent_results = [(t, s) for t, s in self.request_results if t >= cutoff_time]
        recent_throttling = [t for t in self.throttling_events if t >= cutoff_time]
        recent_system_metrics = [m for m in self.system_metrics_history if m.timestamp >= cutoff_time]
        
        # Calculate summary stats
        total_requests = len(recent_results)
        successful_requests = sum(1 for _, success in recent_results if success)
        failed_requests = total_requests - successful_requests
        
        durations = [d for _, d in recent_durations]
        avg_duration = sum(durations) / len(durations) if durations else 0
        p95_duration = sorted(durations)[int(len(durations) * 0.95)] if durations else 0
        
        requests_per_minute = total_requests / max(minutes_back, 1)
        
        return {
            "time_period_minutes": minutes_back,
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "success_rate": successful_requests / total_requests if total_requests > 0 else 1.0,
            "average_duration_seconds": avg_duration,
            "p95_duration_seconds": p95_duration,
            "throttling_events": len(recent_throttling),
            "requests_per_minute": requests_per_minute,
            "active_requests": len(self.active_requests),
            "streaming_clients": len(self.streaming_clients),
            "system_metrics_count": len(recent_system_metrics),
        }

    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system resource metrics."""
        # Get CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Get memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_mb = memory.used / (1024 * 1024)
        memory_available_mb = memory.available / (1024 * 1024)
        
        # Get connection count (approximate)
        try:
            connections = len(psutil.net_connections())
        except (psutil.AccessDenied, OSError):
            connections = 0
        
        return SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_mb=memory_used_mb,
            memory_available_mb=memory_available_mb,
            active_connections=connections,
            active_requests=len(self.active_requests),
        )

    def _calculate_performance_metrics(self, current_time: float) -> PerformanceMetrics:
        """Calculate performance metrics from collected data."""
        # Use last 10 minutes for performance calculation
        cutoff_time = current_time - 600
        
        recent_durations = [d for t, d in self.request_durations if t >= cutoff_time]
        recent_results = [(t, s) for t, s in self.request_results if t >= cutoff_time]
        recent_throttling = [t for t in self.throttling_events if t >= cutoff_time]
        
        total_requests = len(recent_results)
        successful_requests = sum(1 for _, success in recent_results if success)
        failed_requests = total_requests - successful_requests
        
        avg_duration = sum(recent_durations) / len(recent_durations) if recent_durations else 0
        p95_duration = sorted(recent_durations)[int(len(recent_durations) * 0.95)] if recent_durations else 0
        
        requests_per_minute = total_requests / 10.0  # 10 minute window
        
        return PerformanceMetrics(
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            average_duration=avg_duration,
            p95_duration=p95_duration,
            throttled_requests=len(recent_throttling),
            requests_per_minute=requests_per_minute,
        )

    def _determine_health_status(
        self, 
        system_metrics: SystemMetrics, 
        performance_metrics: PerformanceMetrics
    ) -> str:
        """Determine overall health status based on metrics."""
        # Define thresholds
        if (system_metrics.memory_percent > 90 or 
            system_metrics.cpu_percent > 90 or
            performance_metrics.failed_requests > performance_metrics.successful_requests or
            performance_metrics.throttled_requests > 10):
            return "critical"
        
        if (system_metrics.memory_percent > 75 or 
            system_metrics.cpu_percent > 75 or
            performance_metrics.failed_requests > performance_metrics.successful_requests * 0.1 or
            performance_metrics.throttled_requests > 3):
            return "warning"
        
        return "healthy"

    async def _get_active_coordinator_count(self) -> int:
        """Get the count of active coordinators."""
        # For now, assume 1 coordinator (our current one)
        # This could be expanded to discover multiple coordinators
        return 1

    def _calculate_error_rate(self, current_time: float) -> float:
        """Calculate current error rate over the last 10 minutes."""
        cutoff_time = current_time - 600  # 10 minutes
        recent_results = [(t, s) for t, s in self.request_results if t >= cutoff_time]
        
        if not recent_results:
            return 0.0
        
        failed_requests = sum(1 for _, success in recent_results if not success)
        return failed_requests / len(recent_results)

    async def _health_check_loop(self) -> None:
        """Background task for periodic health checks."""
        while self._is_running:
            try:
                # Collect and store system metrics
                system_metrics = await self._collect_system_metrics()
                self.system_metrics_history.append(system_metrics)
                
                # Get full health status
                health_status = await self.get_health_status()
                
                # Broadcast health update if status is not healthy
                if health_status.status != "healthy":
                    await self.coordinator.broadcast_event(
                        EventTypes.AIDER_ERROR_OCCURRED,
                        {
                            "component": "health_monitor",
                            "severity": health_status.status,
                            "system_metrics": {
                                "cpu_percent": system_metrics.cpu_percent,
                                "memory_percent": system_metrics.memory_percent,
                                "active_requests": len(self.active_requests),
                            },
                            "performance_metrics": {
                                "error_rate": health_status.error_rate,
                                "throttled_requests": health_status.performance_metrics.throttled_requests,
                            },
                        }
                    )
                
                await asyncio.sleep(self.health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log error but continue monitoring
                await self.coordinator.broadcast_event(
                    EventTypes.AIDER_ERROR_OCCURRED,
                    {
                        "component": "health_monitor",
                        "error": "health_check_error",
                        "message": str(e),
                    }
                )
                await asyncio.sleep(self.health_check_interval)

    async def _metrics_cleanup_loop(self) -> None:
        """Background task for cleaning up old metrics data."""
        while self._is_running:
            try:
                current_time = time.time()
                cutoff_time = current_time - (self.metrics_retention_minutes * 60)
                
                # Clean up old data
                while self.system_metrics_history and self.system_metrics_history[0].timestamp < cutoff_time:
                    self.system_metrics_history.popleft()
                
                while self.request_durations and self.request_durations[0][0] < cutoff_time:
                    self.request_durations.popleft()
                
                while self.request_results and self.request_results[0][0] < cutoff_time:
                    self.request_results.popleft()
                
                while self.throttling_events and self.throttling_events[0] < cutoff_time:
                    self.throttling_events.popleft()
                
                # Run cleanup every 5 minutes
                await asyncio.sleep(300)
                
            except asyncio.CancelledError:
                break
            except Exception:
                # Continue on error
                await asyncio.sleep(300)