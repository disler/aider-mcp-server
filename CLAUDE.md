# Claude Context Reset Instructions

This document contains essential information to reset Claude's context when working on the SSE Coordinator refactoring project.

## Project Overview

We are refactoring the SSE Coordinator from `../with-sse-mcp` into this branch (`feature/refactor-sse-coordinator`) in a step-by-step manner following the task list in `reference/tasks/tasks.json`. The goal is to implement each feature properly while fixing quality issues and failing tests that exist in the original implementation.

## Current Status (Foundation Commit)

✅ **COMPLETED: Infrastructure & Quality Baseline**
- All test failures resolved (293 tests passing, 7 appropriately skipped)
- All critical quality issues fixed (unused variables, imports, type errors)
- Code formatting and linting issues addressed
- Pristine codebase ready for systematic task-by-task refactoring

📋 **READY FOR: Task 2 - EventSystem Implementation**
- Next task: Implement low-level event broadcasting system
- Dependencies: Task 1 (Base Interfaces) completed
- Expected: Event subscription, filtering, async distribution

## Implementation Progress

### ✅ Task 1: Define Base Interfaces and Types (COMPLETED)
- **Status**: Implemented and tested
- **Files**: `src/aider_mcp_server/interfaces/*`
- **Commit**: 24c5505 feat: implement Task 1 - Define Base Interfaces and Types

### 🎯 Task 2: Implement EventSystem (NEXT)
- **Status**: Ready to implement
- **Dependencies**: Task 1 ✅
- **Details**: Create low-level event broadcasting with subscription/filtering
- **Files to create**: `src/aider_mcp_server/event_system.py` (enhanced)
- **Tests to create**: Comprehensive event system tests

## Workflow Process

1. **Identify next task** from `reference/tasks/tasks.json` dependency chain
2. **Use aider for implementation** with comprehensive context
3. **Implement with proper testing** using sequentialthinking for review
4. **Run quality checks** after each task completion
5. **Create task-specific commit** following established pattern
6. **Update this document** with progress

## Git Commit Strategy

Each task gets its own commit following this pattern:
```
feat: implement Task X - [Task Title]

- [Key implementation detail 1]
- [Key implementation detail 2]
- [Test coverage details]

✅ Tests: [Test status]
🔧 Quality: [Quality check status]  
🔗 Dependencies: [Completed dependencies]
📋 Next: [Next task]

🤖 Generated with [Claude Code](https://claude.ai/code)
Co-Authored-By: Claude <noreply@anthropic.com>
```

## Task Sequence (Dependency Order)

1. ✅ **Task 1: Define Base Interfaces and Types** - Create foundational interfaces
2. 🎯 **Task 2: Implement EventSystem** - Low-level event broadcasting system  
3. **Task 3: Implement TransportAdapterRegistry** - Registry for transport lifecycle
4. **Task 4: Implement EventCoordinator** - Event distribution to transports
5. **Task 5: Implement RequestProcessor** - Request handling and routing
6. **Task 6: Implement SSE Transport Adapter** - Server-Sent Events transport
7. **Task 7: Implement Stdio Transport Adapter** - Standard I/O transport
8. **Task 8: Implement HandlerRegistry** - Registry for request handlers
9. **Task 9: Implement ApplicationCoordinator** - Central coordinator (singleton)
10. **Task 10: Implement Initialization Sequence** - Proper initialization avoiding circular dependencies
11. **Task 11: Implement Error Handling System** - Comprehensive error handling
12. **Task 12: Implement Logging System** - Structured logging system
13. **Task 13: Implement Configuration System** - Application configuration
14. **Task 14: Implement Transport Discovery Mechanism** - Transport discovery
15. **Task 15: Implement Integration Tests** - End-to-end testing

## Testing Strategy

For each task:
1. **Unit tests during development**: `hatch -e dev run pytest`
2. **Specific test runs**: `hatch -e dev run pytest tests/test_specific_file.py`
3. **Quality assessment**: `hatch -e dev run pre-commit run --all-files`
4. **Document progress** in this file

## Development Tools

- **Code implementation**: `aider:aider_ai_code` (primary tool)
- **Idea review**: `sequentialthinking`
- **Alternative solutions**: `just-prompt` to consult models like Gemini
- **Testing**: `hatch -e dev run pytest`
- **Quality**: `hatch -e dev run pre-commit run --all-files`

## Quality Standards Maintained

- ✅ All tests pass: `hatch -e dev run pytest` (293 passed, 7 skipped)
- ✅ No critical quality issues: Zero F, E9 class violations
- ✅ Type safety verified: `mypy` clean on key files
- ✅ Code formatting consistent: `ruff format` applied
- ⚠️ Style warnings acceptable: 128 E501 line-too-long (non-blocking)

## Directory Structure

- **Key interfaces**: `src/aider_mcp_server/interfaces/`
- **Transport adapters**: `src/aider_mcp_server/`
- **Tests**: `tests/`
- **Task reference**: `reference/tasks/tasks.json`

## Context Restart Instructions

When context window requires restart:

1. **Read this document** for current status
2. **Check git log** for latest progress: `git log --oneline -5`
3. **Run tests** to verify current state: `hatch -e dev run pytest --tb=no -q`
4. **Check quality** status: `hatch -e dev run ruff check --select=F,E9`
5. **Identify next task** from dependency chain above
6. **Continue with aider-centric workflow**

## Important Notes

- **Maintain pristine quality** after each phase to prevent technical debt
- **Use task-specific commits** for clear progression tracking
- **Focus on solid implementations** over quick fixes
- **Test thoroughly** before moving to next task
- **Update this document** when significant progress is made