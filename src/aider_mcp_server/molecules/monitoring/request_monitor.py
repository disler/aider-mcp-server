"""
Request Monitor for tracking request durations and detecting throttling.

This module implements comprehensive request monitoring with automatic throttling detection
for the Aider MCP Server real-time streaming system.
"""

import asyncio
import time
from typing import Any, Dict, Optional
from uuid import uuid4

from aider_mcp_server.atoms.types.event_types import EventTypes
from aider_mcp_server.interfaces.application_coordinator import ApplicationCoordinator


class RequestMonitor:
    """Monitor request durations and detect throttling."""

    def __init__(
        self,
        coordinator: ApplicationCoordinator,
        throttling_threshold: float = 60.0,
        warning_threshold: float = 30.0,
    ):
        """
        Initialize the request monitor.
        
        Args:
            coordinator: ApplicationCoordinator for event broadcasting
            throttling_threshold: Seconds after which to consider request throttled
            warning_threshold: Seconds after which to issue warning
        """
        self.coordinator = coordinator
        self.throttling_threshold = throttling_threshold
        self.warning_threshold = warning_threshold
        self.active_requests: Dict[str, Dict[str, Any]] = {}
        self._monitoring_tasks: Dict[str, asyncio.Task] = {}

    async def track_request(
        self,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Start tracking a request.
        
        Args:
            request_id: Optional request ID (generates one if not provided)
            context: Request context information
            
        Returns:
            The request ID being tracked
        """
        if request_id is None:
            request_id = str(uuid4())

        start_time = time.time()
        self.active_requests[request_id] = {
            "start_time": start_time,
            "context": context or {},
            "warnings_issued": set(),
        }

        # Start monitoring for this request
        self._monitoring_tasks[request_id] = asyncio.create_task(
            self._monitor_request(request_id)
        )

        # Broadcast session start event
        await self.coordinator.broadcast_event(
            EventTypes.AIDER_SESSION_STARTED,
            {
                "request_id": request_id,
                "start_time": start_time,
                "context": context or {},
                "timestamp": start_time,
            },
        )

        return request_id

    async def complete_request(
        self,
        request_id: str,
        success: bool = True,
        result: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Mark a request as completed.
        
        Args:
            request_id: The request ID to complete
            success: Whether the request was successful
            result: Optional result data
        """
        if request_id not in self.active_requests:
            return

        request_info = self.active_requests[request_id]
        end_time = time.time()
        duration = end_time - request_info["start_time"]

        # Cancel monitoring task
        if request_id in self._monitoring_tasks:
            self._monitoring_tasks[request_id].cancel()
            del self._monitoring_tasks[request_id]

        # Broadcast completion event
        await self.coordinator.broadcast_event(
            EventTypes.AIDER_SESSION_COMPLETED,
            {
                "request_id": request_id,
                "start_time": request_info["start_time"],
                "end_time": end_time,
                "duration": duration,
                "success": success,
                "result": result or {},
                "context": request_info["context"],
                "timestamp": end_time,
            },
        )

        # Clean up
        del self.active_requests[request_id]

    async def update_request_progress(
        self,
        request_id: str,
        progress_data: Dict[str, Any],
    ) -> None:
        """
        Update progress for an active request.
        
        Args:
            request_id: The request ID to update
            progress_data: Progress information
        """
        if request_id not in self.active_requests:
            return

        request_info = self.active_requests[request_id]
        current_time = time.time()
        duration = current_time - request_info["start_time"]

        # Broadcast progress event
        await self.coordinator.broadcast_event(
            EventTypes.AIDER_SESSION_PROGRESS,
            {
                "request_id": request_id,
                "duration": duration,
                "progress_data": progress_data,
                "context": request_info["context"],
                "timestamp": current_time,
            },
        )

    async def _monitor_request(self, request_id: str) -> None:
        """
        Monitor a request for throttling and warnings.
        
        Args:
            request_id: The request ID to monitor
        """
        try:
            request_info = self.active_requests[request_id]
            warnings_issued = request_info["warnings_issued"]

            # Wait for warning threshold
            await asyncio.sleep(self.warning_threshold)
            
            if request_id in self.active_requests and "warning" not in warnings_issued:
                await self._issue_warning(request_id)
                warnings_issued.add("warning")

            # Wait for throttling threshold
            remaining_time = self.throttling_threshold - self.warning_threshold
            if remaining_time > 0:
                await asyncio.sleep(remaining_time)

            if request_id in self.active_requests and "throttling" not in warnings_issued:
                await self._detect_throttling(request_id)
                warnings_issued.add("throttling")

                # Continue monitoring for additional throttling events
                await self._continue_throttling_monitoring(request_id)

        except asyncio.CancelledError:
            # Request completed normally
            pass
        except Exception as e:
            # Log error but don't crash monitoring
            await self.coordinator.broadcast_event(
                EventTypes.AIDER_ERROR_OCCURRED,
                {
                    "request_id": request_id,
                    "error_type": "monitoring_error",
                    "error_message": str(e),
                    "timestamp": time.time(),
                },
            )

    async def _issue_warning(self, request_id: str) -> None:
        """Issue a warning for a long-running request."""
        request_info = self.active_requests[request_id]
        current_time = time.time()
        duration = current_time - request_info["start_time"]

        await self.coordinator.broadcast_event(
            EventTypes.AIDER_OPERATION_STATUS,
            {
                "request_id": request_id,
                "status": "long_running_warning",
                "duration": duration,
                "threshold": self.warning_threshold,
                "message": f"Request has been running for {duration:.1f}s (warning threshold: {self.warning_threshold}s)",
                "context": request_info["context"],
                "timestamp": current_time,
            },
        )

    async def _detect_throttling(self, request_id: str) -> None:
        """Detect and broadcast throttling for a request."""
        request_info = self.active_requests[request_id]
        current_time = time.time()
        duration = current_time - request_info["start_time"]

        await self.coordinator.broadcast_event(
            EventTypes.AIDER_THROTTLING_DETECTED,
            {
                "request_id": request_id,
                "duration": duration,
                "threshold": self.throttling_threshold,
                "status": "throttled",
                "severity": "high" if duration > self.throttling_threshold * 2 else "medium",
                "message": f"Request appears throttled - running for {duration:.1f}s",
                "context": request_info["context"],
                "timestamp": current_time,
            },
        )

    async def _continue_throttling_monitoring(self, request_id: str) -> None:
        """Continue monitoring a throttled request for status updates."""
        # Check every 30 seconds for severely throttled requests
        check_interval = 30.0
        
        while request_id in self.active_requests:
            await asyncio.sleep(check_interval)
            
            if request_id not in self.active_requests:
                break
                
            request_info = self.active_requests[request_id]
            current_time = time.time()
            duration = current_time - request_info["start_time"]
            
            # Broadcast periodic updates for severely throttled requests
            await self.coordinator.broadcast_event(
                EventTypes.AIDER_THROTTLING_DETECTED,
                {
                    "request_id": request_id,
                    "duration": duration,
                    "threshold": self.throttling_threshold,
                    "status": "severely_throttled" if duration > self.throttling_threshold * 3 else "throttled",
                    "severity": "critical" if duration > self.throttling_threshold * 5 else "high",
                    "message": f"Request still throttled after {duration:.1f}s",
                    "context": request_info["context"],
                    "timestamp": current_time,
                },
            )

    async def get_active_requests(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all active requests."""
        current_time = time.time()
        active_info = {}
        
        for request_id, request_info in self.active_requests.items():
            duration = current_time - request_info["start_time"]
            active_info[request_id] = {
                "request_id": request_id,
                "start_time": request_info["start_time"],
                "duration": duration,
                "context": request_info["context"],
                "status": self._get_request_status(duration),
                "warnings_issued": list(request_info["warnings_issued"]),
            }
            
        return active_info

    def _get_request_status(self, duration: float) -> str:
        """Get the status of a request based on duration."""
        if duration >= self.throttling_threshold:
            return "throttled"
        elif duration >= self.warning_threshold:
            return "long_running"
        else:
            return "normal"

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitored requests."""
        active_requests = await self.get_active_requests()
        
        total_active = len(active_requests)
        throttled_count = sum(1 for req in active_requests.values() if req["status"] == "throttled")
        long_running_count = sum(1 for req in active_requests.values() if req["status"] == "long_running")
        
        return {
            "total_active_requests": total_active,
            "throttled_requests": throttled_count,
            "long_running_requests": long_running_count,
            "normal_requests": total_active - throttled_count - long_running_count,
            "throttling_threshold": self.throttling_threshold,
            "warning_threshold": self.warning_threshold,
            "timestamp": time.time(),
        }

    async def shutdown(self) -> None:
        """Shutdown the request monitor and clean up resources."""
        # Cancel all monitoring tasks
        for task in self._monitoring_tasks.values():
            task.cancel()
        
        # Wait for tasks to complete cancellation
        if self._monitoring_tasks:
            await asyncio.gather(*self._monitoring_tasks.values(), return_exceptions=True)
        
        # Clear all data
        self.active_requests.clear()
        self._monitoring_tasks.clear()