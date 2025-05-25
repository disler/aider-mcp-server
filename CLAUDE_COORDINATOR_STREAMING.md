# Claude Context Reset Instructions - Coordinator Real-Time Streaming

This document contains essential information for implementing real-time error streaming and cross-transport coordination in the Aider MCP Server.

## Project Overview

We are implementing **real-time error streaming and throttling monitoring** to enable LLM clients to monitor STDIO AIDER sessions via SSE streams. This addresses the critical problem where long-running AIDER requests (sometimes hours) provide no visibility into rate limiting, throttling, or progress status.

## Current Primary Task: Real-Time Coordinator Streaming Implementation

🎯 **Current Focus**: Implement cross-transport event broadcasting for real-time monitoring of AIDER sessions.

### Implementation Objectives
- **Event Broadcasting**: Stream rate limits, throttling, and progress from STDIO to SSE
- **Cross-Transport Coordination**: Enable STDIO and SSE transports to share real-time updates
- **Throttling Detection**: Monitor and broadcast long-running request status
- **Client Monitoring**: Provide live updates to web clients about AIDER session health

### Implementation Scope
- Enhanced AIDER tool event broadcasting during execution
- SSE endpoint for real-time error/progress streaming
- Cross-transport event relay system
- Throttling detection and monitoring
- Request correlation across transport boundaries

## 📋 Implementation Tracking

**📊 All implementation tracking is managed in [COORDINATOR_STREAMING_TRACKING.md](./COORDINATOR_STREAMING_TRACKING.md)**

This dedicated tracking document contains:
- **Complete task breakdown** with implementation phases and dependencies
- **Technical implementation details** for each component
- **Integration test plans** for cross-transport communication
- **Quality gates** and validation requirements

**Current Status:** Phase 1 (Event Broadcasting Integration) - Ready to Start

## Implementation Workflow Process

**🎯 SYSTEMATIC REAL-TIME STREAMING IMPLEMENTATION WORKFLOW**

### Current Implementation Phase: Event Broadcasting Integration

1. **Implementation Planning**
   - Follow [COORDINATOR_STREAMING_TRACKING.md](./COORDINATOR_STREAMING_TRACKING.md) phase structure
   - Create dedicated implementation branch: `git checkout -b aider/streaming-phase-X`
   - Focus on specific streaming components per session

2. **Event Broadcasting Development**
   - **Enhance AIDER tool execution** with real-time event broadcasting
   - **Integrate coordinator communication** in rate limit handling
   - **Add progress streaming** for long-running sessions
   - **Test event propagation** across transport boundaries

3. **Cross-Transport Integration**
   - **Implement SSE monitoring endpoints** for real-time updates
   - **Add transport discovery integration** for automatic coordination
   - **Create event relay system** between STDIO and SSE
   - **Validate coordination** with comprehensive integration tests

4. **Quality Validation** (Applied to all implementations)
   - Run `hatch -e dev run pytest` → ALL tests must pass
   - Run `hatch -e dev run ruff check --select=F,E9` → ZERO critical errors
   - Run `hatch -e dev run pre-commit run --all-files` → Clean validation
   - **Integration testing** of cross-transport event flow
   - **Performance testing** of real-time streaming under load

5. **Documentation & Tracking Updates**
   - **Update implementation matrix** with completion status
   - **Document integration patterns** and usage examples
   - **Create monitoring guides** for operational teams
   - **Commit implementation** with comprehensive test coverage

### Future Enhancement Framework

6. **Monitoring & Analytics Setup**
   - **Health check endpoints** for coordinator status
   - **Metrics collection** for streaming performance
   - **Alerting integration** for critical system events
   - **Performance optimization** based on usage patterns

## Git Commit Strategy & Implementation Workflow

### Branch Strategy
- **Main branch**: `feature/atomic-design-reorganization` (current development)
- **Implementation branches**: `aider/streaming-phase-X` (for specific implementation phases)
- **Integration branches**: `aider/integration-X-name` (for cross-component integration)

### Commit Pattern
Each implementation phase gets its own commit following this pattern:
```
feat: implement Streaming Phase X - [Phase Title]

- [Key implementation detail 1]
- [Cross-transport integration status]
- [Event broadcasting capabilities]

✅ Implementation: [Phase completion status]
🔧 Quality: [Quality check status]
📊 Streaming: [Real-time capability status]
🔗 Integration: [Cross-transport coordination status]
📋 Next: [Next implementation phase]

🤖 Generated with [Claude Code](https://claude.ai/code)
Co-Authored-By: Claude <noreply@anthropic.com>
```

### Implementation Phase Workflow
1. **Plan implementation phase** using [COORDINATOR_STREAMING_TRACKING.md](./COORDINATOR_STREAMING_TRACKING.md)
2. **Create implementation branch**: `git checkout -b aider/streaming-phase-X`
3. **Develop streaming components** with comprehensive event integration
4. **Implement cross-transport coordination** with discovery integration
5. **Add integration tests** for event flow validation
6. **Run quality validation**:
   ```bash
   # Run full test suite
   hatch -e dev run pytest

   # Check for critical violations
   hatch -e dev run ruff check --select=F,E9

   # Run pre-commit validation
   hatch -e dev run pre-commit run --all-files

   # Integration test for streaming
   hatch -e dev run pytest tests/integration/test_streaming_coordination.py
   ```
7. **Update implementation matrix** with component status
8. **Create implementation-specific commit** with integration verification
9. **Merge and validate** cross-transport system integration

### Integration Testing Workflow
1. **Unit Tests**: Individual component functionality
2. **Integration Tests**: Cross-transport event flow
3. **End-to-End Tests**: Complete STDIO → SSE streaming workflow
4. **Performance Tests**: Real-time streaming under load
5. **Regression Tests**: Ensure existing functionality preserved

## Implementation Phases Overview

**📋 Complete implementation details in [COORDINATOR_STREAMING_TRACKING.md](./COORDINATOR_STREAMING_TRACKING.md)**

**High-Level Phase Structure:**
- **Phase 1 - Event Broadcasting** (Tasks 1-4): Enhanced AIDER tool event integration
- **Phase 2 - Cross-Transport Coordination** (Tasks 5-8): SSE endpoints and transport relay
- **Phase 3 - Throttling Detection** (Tasks 9-12): Request monitoring and health checks
- **Phase 4 - Integration & Optimization** (Tasks 13-16): Performance and operational readiness

## Testing Strategy

**🚨 CRITICAL: COMPREHENSIVE INTEGRATION TESTING REQUIRED**

For each implementation phase:
1. **MANDATORY Unit tests**: `hatch -e dev run pytest` - ALL TESTS MUST PASS
2. **MANDATORY Integration tests**: Cross-transport event flow validation
3. **MANDATORY Performance tests**: Real-time streaming load testing
4. **MANDATORY Regression tests**: Existing functionality preservation

**⚠️ NEVER PROCEED TO NEXT PHASE WITHOUT:**
- ✅ All tests passing (no failures, no integration errors)
- ✅ Zero F,E9 critical quality violations
- ✅ Cross-transport event flow working correctly
- ✅ Real-time streaming performance validated
- ✅ Clean merge to main branch

## Development Tools

- **Code implementation**: `aider:aider_ai_code` (primary tool)
- **Architecture review**: `sequentialthinking`
- **Integration testing**: Custom test frameworks for cross-transport validation
- **Performance analysis**: Load testing tools for streaming performance
- **Quality**: `hatch -e dev run pre-commit run --all-files`

## Quality Standards Maintained

- ✅ All tests pass: `hatch -e dev run pytest` (comprehensive test coverage)
- ✅ No critical quality issues: Zero F, E9 class violations
- ✅ Integration validated: Cross-transport event flow working
- ✅ Performance verified: Real-time streaming under load
- ✅ Code formatting consistent: `ruff format` applied
- ✅ Pre-commit hooks passing: All quality checks automated

## Directory Structure

- **Event streaming implementation**: `src/aider_mcp_server/molecules/tools/aider_ai_code.py`
- **SSE transport enhancements**: `src/aider_mcp_server/organisms/transports/sse/`
- **Coordinator integration**: `src/aider_mcp_server/pages/application/coordinator.py`
- **Integration tests**: `tests/integration/streaming/`
- **Implementation tracking**: `COORDINATOR_STREAMING_TRACKING.md`

## Context Restart Instructions

When context window requires restart:

1. **Read this document** for current implementation status and workflow
2. **Check COORDINATOR_STREAMING_TRACKING.md** for detailed phase progress
3. **Review latest commits** for implementation status: `git log --oneline -5`
4. **Run integration tests** to verify current state: `hatch -e dev run pytest tests/integration/`
5. **Check quality** status: `hatch -e dev run ruff check --select=F,E9`
6. **Identify next phase** from implementation matrix and dependency chain
7. **Continue with aider-centric workflow** following branch strategy above

## Important Notes

**🚨 REAL-TIME STREAMING IS MISSION-CRITICAL**

- **Maintain pristine quality** after each phase to prevent technical debt
- **ZERO broken integration tests allowed** - ALL cross-transport tests must pass
- **ZERO F,E9 quality violations** - Critical errors must be resolved immediately
- **Use phase-specific commits** with verified integration status in commit messages
- **Focus on robust implementations** over quick fixes - reliability first, always
- **Re-run full integration test suite** after every merge to main branch
- **Update COORDINATOR_STREAMING_TRACKING.md** with detailed progress and integration verification

**⚠️ LESSON LEARNED: Cross-transport integration errors and streaming failures compound rapidly. The systematic quality-first workflow is the ONLY way to maintain a reliable real-time system.**

## Integration Architecture Reference

**Target System Flow:**
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

**Key Integration Points:**
- Event broadcasting from AIDER tool execution
- Cross-transport event relay via Coordinator
- Real-time SSE streaming to web clients
- Automatic discovery and coordination setup