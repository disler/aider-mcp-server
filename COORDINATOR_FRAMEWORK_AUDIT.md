# Coordinator Framework Architecture Audit

**Audit Date**: December 2024
**Focus**: Multi-transport coordination, error streaming, and throttling monitoring
**Status**: ✅ COMPLETE

## Executive Summary

The current Aider MCP Server implements a **sophisticated coordinator framework** that provides excellent foundations for multi-transport communication, error monitoring, and long-running request management. However, there are **key gaps** in the intended monitoring and error streaming architecture that limit its effectiveness for handling long-running AIDER sessions.

**Current Strengths**: Discovery system, event coordination, rate limiting
**Key Gaps**: Real-time error streaming, throttling detection, cross-transport notifications

## Architectural Vision vs. Current Implementation

### ✅ What Works Well

#### 1. Coordinator Discovery System
**File**: `src/aider_mcp_server/molecules/transport/discovery.py`

**Current Implementation**:
- ✅ **File-based registry** in temp directory (`/tmp/aider_mcp_coordinator/coordinator_registry.json`)
- ✅ **Heartbeat mechanism** (10-second intervals, 30-second timeout)
- ✅ **Multi-transport discovery** (SSE, WebSocket, STDIO)
- ✅ **Automatic cleanup** of inactive coordinators
- ✅ **Environment override** via `AIDER_MCP_COORDINATOR_DISCOVERY_FILE`

**Example Discovery Flow**:
```python
# When STDIO transport starts
discovery = CoordinatorDiscovery()
existing_coordinator = await discovery.find_best_coordinator()

if existing_coordinator:
    # Connect to existing SSE coordinator for monitoring
    sse_url = f"http://{existing_coordinator.host}:{existing_coordinator.port}"
```

#### 2. Event System Architecture
**File**: `src/aider_mcp_server/organisms/coordinators/event_coordinator.py`

**Current Implementation**:
- ✅ **Multi-handler subscription** system
- ✅ **Transport adapter registration** for event broadcasting
- ✅ **Type-safe event handling** with `EventTypes` enum
- ✅ **Cross-transport communication** capabilities

#### 3. Rate Limiting & Error Handling
**File**: `src/aider_mcp_server/atoms/utils/fallback_config.py`

**Current Implementation**:
- ✅ **Provider-specific rate limit detection** (OpenAI, Anthropic, Gemini)
- ✅ **Exponential backoff** with configurable parameters
- ✅ **Fallback model selection** for continued operation
- ✅ **Comprehensive error classification**

## 🔴 Critical Gaps in Current Implementation

### 1. Real-Time Error Streaming Missing

**Problem**: While STDIO runs AIDER tasks, LLM clients have no visibility into:
- Rate limiting events
- Long-running session progress
- Error conditions
- Model fallback switches

**Current Gap**: No active error streaming to SSE endpoints during STDIO operations.

### 2. Throttling Detection Incomplete

**Problem**: The system detects rate limits but doesn't broadcast throttling events in real-time.

**Current Rate Limit Flow**:
```python
# In aider_ai_code.py - rate limit detected but not streamed
except Exception as e:
    should_retry, new_model = await _handle_rate_limit_or_error(...)
    # ❌ No event broadcasting to SSE stream
    # ❌ No coordinator notification
```

**What's Missing**:
- Event broadcasting when rate limits occur
- SSE stream notifications for throttling status
- Progress updates for long-running requests

### 3. Cross-Transport Communication Not Implemented

**Problem**: STDIO and SSE transports operate independently with no coordination.

**Current Architecture Gap**:
```
STDIO Transport    SSE Transport
     |                 |
     |                 |
 [Aider Task]    [Web Client]
     |                 |
     |                 |
  (isolated)       (isolated)
```

**Needed Architecture**:
```
STDIO Transport    SSE Transport
     |                 |
     |                 |
 [Aider Task] ←→ [Coordinator] ←→ [Web Client]
     |                 |              |
Event Broadcasting ----+              |
Rate Limits, Progress, Errors --------+
```

## Detailed Component Analysis

### ApplicationCoordinator Capabilities

**File**: `src/aider_mcp_server/pages/application/coordinator.py`

**✅ Strengths**:
- Singleton pattern for system-wide coordination
- Transport registration and management
- Event broadcasting infrastructure
- Request processing pipeline

**🔴 Gaps**:
- No active monitoring of long-running requests
- No cross-transport event relaying
- No progress tracking for AIDER sessions

### SSE Transport Implementation

**File**: `src/aider_mcp_server/organisms/transports/sse/sse_transport_adapter.py`

**✅ Strengths**:
- FastMCP integration for tool calls
- Event streaming infrastructure via SSE
- Client connection management

**🔴 Gaps**:
- No subscription to STDIO transport events
- No error monitoring from other transports
- No throttling status streaming

### STDIO Transport Implementation

**File**: `src/aider_mcp_server/organisms/transports/stdio/stdio_transport_adapter.py`

**✅ Strengths**:
- Coordinator discovery integration
- MCP standard compliance
- JSON-RPC processing

**🔴 Gaps**:
- No event broadcasting during AIDER execution
- No progress reporting to coordinator
- No error streaming to SSE endpoints

## 🎯 Recommended Implementation Strategy

### Phase 1: Event Streaming Integration

#### 1.1 Enhanced AIDER Tool Error Broadcasting
```python
# In aider_ai_code.py
async def _handle_rate_limit_or_error(...):
    # Current error handling
    should_retry, new_model = handle_error(...)

    # NEW: Broadcast to coordinator
    if coordinator:
        await coordinator.broadcast_event(
            "aider.rate_limit_detected",
            {
                "provider": provider,
                "current_model": current_model,
                "fallback_model": new_model,
                "attempt": attempt,
                "max_retries": max_retries,
                "estimated_delay": delay_seconds
            }
        )
```

#### 1.2 Progress Streaming for Long Sessions
```python
# In aider_ai_code.py
async def _run_aider_session(...):
    # NEW: Progress broadcasting
    await coordinator.broadcast_event(
        "aider.session_started",
        {"files": relative_editable_files, "prompt": ai_coding_prompt[:100]}
    )

    # During execution
    coder_result = coder.run(ai_coding_prompt)

    await coordinator.broadcast_event(
        "aider.session_completed",
        {"success": True, "changes_detected": has_changes}
    )
```

#### 1.3 SSE Endpoint for Real-Time Monitoring
```python
# In SSE transport
@app.route("/events/aider")
async def aider_events_stream(request):
    """Stream AIDER events to web clients"""

    async def event_generator():
        queue = asyncio.Queue()

        # Subscribe to AIDER events
        coordinator.subscribe_to_events([
            "aider.rate_limit_detected",
            "aider.session_started",
            "aider.session_completed",
            "aider.throttling_detected"
        ], queue.put)

        while True:
            event = await queue.get()
            yield f"data: {json.dumps(event)}\n\n"

    return EventSourceResponse(event_generator())
```

### Phase 2: Throttling Detection & Monitoring

#### 2.1 Request Duration Tracking
```python
class RequestMonitor:
    def __init__(self):
        self.active_requests = {}
        self.throttling_threshold = 60.0  # seconds

    async def track_request(self, request_id: str):
        start_time = time.time()
        self.active_requests[request_id] = start_time

        # Monitor for throttling
        asyncio.create_task(self._monitor_throttling(request_id))

    async def _monitor_throttling(self, request_id: str):
        await asyncio.sleep(self.throttling_threshold)

        if request_id in self.active_requests:
            await coordinator.broadcast_event(
                "aider.throttling_detected",
                {
                    "request_id": request_id,
                    "duration": time.time() - self.active_requests[request_id],
                    "status": "long_running"
                }
            )
```

#### 2.2 Health Check Integration
```python
# In coordinator
async def health_check(self):
    """Regular health monitoring"""
    return {
        "active_requests": len(self.request_monitor.active_requests),
        "longest_running": self.request_monitor.get_longest_duration(),
        "rate_limit_status": self.get_rate_limit_status(),
        "transport_status": {
            transport_id: adapter.is_healthy()
            for transport_id, adapter in self.transports.items()
        }
    }
```

### Phase 3: Enhanced Discovery & Coordination

#### 3.1 Transport-Aware Discovery
```python
# Enhanced discovery with transport-specific metadata
coordinator_info = {
    "coordinator_id": "coord_abc123",
    "host": "127.0.0.1",
    "port": 8765,
    "transport_type": "sse",
    "capabilities": [
        "error_streaming",
        "progress_monitoring",
        "throttling_detection"
    ],
    "active_transports": ["sse", "stdio"],
    "health_endpoint": "/health",
    "events_endpoint": "/events/aider"
}
```

#### 3.2 Automatic Cross-Transport Setup
```python
# When STDIO transport starts
async def initialize_stdio_with_monitoring():
    discovery = CoordinatorDiscovery()
    sse_coordinator = await discovery.find_best_coordinator()

    if sse_coordinator:
        # Register for event broadcasting
        await self.register_with_coordinator(sse_coordinator)
        logger.info(f"STDIO transport registered with SSE coordinator {sse_coordinator.coordinator_id}")
    else:
        # Start minimal SSE coordinator for monitoring
        sse_coordinator = await self.start_monitoring_coordinator()
        logger.info("Started new SSE coordinator for monitoring")
```

## Implementation Priorities

### 🔴 High Priority (Critical for User Experience)
1. **Real-time error streaming** - LLM clients need visibility into throttling
2. **Progress broadcasting** - Show long-running session status
3. **Cross-transport event relay** - Connect STDIO operations to SSE streams

### 🟡 Medium Priority (Enhanced Monitoring)
4. **Throttling detection** - Automatic detection of long-running requests
5. **Health monitoring** - System-wide status and performance metrics
6. **Enhanced discovery** - Transport capability negotiation

### 🟢 Low Priority (Nice-to-Have)
7. **Request correlation** - Track requests across transport boundaries
8. **Performance analytics** - Historical data and trends
9. **Advanced fallback** - Intelligent model selection based on performance

## Current vs. Target Architecture

### Current Architecture Limitations
```
┌─────────────────┐    ┌─────────────────┐
│   STDIO MCP     │    │   SSE Server    │
│                 │    │                 │
│ [AIDER Tasks]   │    │ [Web Clients]   │
│       ↓         │    │       ↑         │
│   (isolated)    │    │   (isolated)    │
└─────────────────┘    └─────────────────┘
```

### Target Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   STDIO MCP     │────│   Coordinator   │────│   SSE Server    │
│                 │    │                 │    │                 │
│ [AIDER Tasks]   │    │ [Event Hub]     │    │ [Web Clients]   │
│       ↓         │    │ [Rate Monitor]  │    │       ↑         │
│  Rate Limits────┼────│ [Progress Track]│────│   Live Updates  │
│  Throttling─────┼────│ [Error Stream]  │────│   Error Stream  │
│  Progress───────┼────│ [Discovery]     │────│   Health Status │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Conclusion

The Aider MCP Server has **excellent architectural foundations** for implementing the desired monitoring and coordination system. The discovery mechanism, event system, and rate limiting components are well-designed and production-ready.

**Key Recommendations**:

1. **Immediate**: Implement event broadcasting in AIDER tool execution paths
2. **Short-term**: Add SSE endpoint for real-time error/progress streaming
3. **Medium-term**: Enhance discovery system for automatic cross-transport coordination
4. **Long-term**: Add comprehensive throttling detection and health monitoring

The current framework provides all necessary building blocks - it primarily needs **integration work** to connect the existing components and enable the real-time monitoring capabilities required for long-running AIDER sessions.

---

*This audit provides the roadmap for transforming the current excellent foundation into the fully-realized monitoring and coordination system envisioned for the project.*
