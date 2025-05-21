# Claude Context Reset Instructions

This document contains essential information to reset Claude's context when working on the SSE Coordinator refactoring project.

## Project Overview

We are refactoring the SSE Coordinator from `../with-sse-mcp` into this branch (`feature/refactor-sse-coordinator`) in a step-by-step manner following the task list in `../with-sse-mcp/tasks/tasks.json`. The goal is to implement each feature properly while fixing quality issues and failing tests that exist in the original implementation.

## Workflow Process

1. Identify the next task to implement from `../with-sse-mcp/tasks/tasks.json`
2. Use `aider:aider_ai_code` for code creation/modification
3. Implement the task with proper testing using sequentialthinking for thorough review
4. Use just-prompt to consult other models like Gemini for solution comparison
5. After successful implementation, mark the task as complete in our tracking system
6. When merging back to development, we'll rebase the original `with-sse-mcp` branch to remove the implemented commits

## Implementation Order

We are following the dependency chain from the tasks.json file:

1. **Task 1: Define Base Interfaces and Types** - Create foundational interfaces for TransportAdapter, EventHandler, RequestHandler, and data classes
2. **Task 2: Implement EventSystem** - Low-level event broadcasting system
3. **Task 3: Implement TransportAdapterRegistry** - Registry for managing transport adapters lifecycle
4. **Task 4: Implement EventCoordinator** - Event distribution to transport adapters
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

## Implementation Tracking

For each task, record the following in `IMPLEMENTATION_TRACKING.md`:
- Task ID and name
- Implementation status
- Corresponding files
- Tests created or fixed
- Commit hash that completes the task
- Notes on issues resolved

## Testing Strategy

For each feature:
1. Run unit tests during development: `hatch -e dev run pytest`
2. Run specific test: `hatch -e dev run pytest tests/test_specific_file.py`
3. Run quality assessment: `hatch -e dev run pre-commit run --all-files`
4. Document test fixes in `IMPLEMENTATION_TRACKING.md`

## Development Tools

- Code implementation: `aider:aider_ai_code`
- Idea review: `sequentialthinking`
- Alternative solutions: `just-prompt` to consult models like Gemini

## Directory Structure

- Key interfaces are in `src/aider_mcp_server/interfaces/`
- Transport adapters are in `src/aider_mcp_server/`
- Tests are in `tests/`

## Quality Standards

- Ensure all tests pass: `hatch -e dev run pytest`
- Verify all quality checks pass: `hatch -e dev run pre-commit run --all-files`
- Maintain consistent implementation style throughout the codebase
- Focus on solid, correct implementations over quick fixes