# Claude Context Reset Instructions

This document contains essential information to reset Claude's context when working on the SSE Coordinator refactoring project.

## Project Overview

We are refactoring the SSE Coordinator from `../with-sse-mcp` into this branch (`feature/refactor-sse-coordinator`) in a step-by-step manner following the task list in `reference/tasks/tasks.json`. The goal is to implement each feature properly while fixing quality issues and failing tests that exist in the original implementation.

## Current Status

✅ **Foundation Complete**: Infrastructure & Quality Baseline established
- All test failures resolved (432 tests passing, 18 appropriately skipped)
- All critical quality issues fixed (unused variables, imports, type errors)
- Code formatting and linting issues addressed
- Pristine codebase ready for systematic task-by-task refactoring

## 📋 Implementation Progress

**📊 All task implementation tracking is managed in [IMPLEMENTATION_TRACKING.md](./IMPLEMENTATION_TRACKING.md)**

This dedicated tracking document contains:
- **Complete task status matrix** with implementation files and test coverage
- **Detailed progress notes** for each completed task with commit hashes
- **Current next task identification** based on dependency chains
- **Implementation issues and resolutions** for each phase

**Quick Status:** 7/15 core tasks + 2/5 modernization tasks completed → Next: Task 7.3 (Update SSE Transport) or Task 8 (HandlerRegistry)

## Workflow Process

**🎯 METICULOUS QUALITY-FIRST WORKFLOW - NO EXCEPTIONS**

1. **Identify next task** from `reference/tasks/tasks.json` dependency chain
2. **Create dedicated aider branch**: `git checkout -b aider/task-X-name`
3. **Implement with aider** using comprehensive context and proper patterns
4. **🔴 MANDATORY QUALITY GATE**: Before any commit:
   - Run `hatch -e dev run pytest` → ALL tests must pass
   - Run `hatch -e dev run ruff check --select=F,E9` → ZERO critical errors
   - Run `hatch -e dev run pre-commit run --all-files` → Clean validation
5. **Create task-specific commit** with verified quality status in message
6. **Merge to feature branch**: Switch to `feature/refactor-sse-coordinator` and merge
7. **🔴 MANDATORY RE-VERIFICATION**: After merge, re-run all quality checks
8. **Update PR** with milestone progress and quality verification
9. **Keep aider branch** until project completion for reference

**⛔ ZERO TOLERANCE POLICY:**
- No commits with failing tests
- No commits with F,E9 quality violations
- No proceeding to next tasks without pristine quality baseline

## Git Commit Strategy & PR Workflow

### Branch Strategy
- **Feature branch**: `feature/refactor-sse-coordinator` (tracks overall progress)
- **Work branches**: `aider/task-X-name` (for individual task implementation)
- **PR tracking**: https://github.com/MementoRC/aider-mcp-server/pull/18

### Commit Pattern
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

### Milestone Workflow
1. **Implement** task in `aider/task-X` branch
2. **Verify** quality and tests pass
3. **Merge** to `feature/refactor-sse-coordinator`
4. **CI Verification (CRITICAL)**
   ```bash
   # Push to remote to trigger CI
   git push origin feature/refactor-sse-coordinator

   # Check PR status
   gh pr list --head feature/refactor-sse-coordinator

   # Monitor CI checks (takes ~90s for tests to complete)
   gh pr checks <PR_NUMBER>

   # Wait for all checks to pass before proceeding to next task
   # Expected checks:
   # - lint_and_types (~30s)
   # - test (Python 3.9-3.12) (~90s each)
   # - CodeQL analysis (~90s)
   # - build-and-deploy (~10s)

   # If any checks fail, fix issues and repeat from step 4
   ```
5. **Update PR Summary** (IMPORTANT: Edit PR description, do NOT post comments)
   ```bash
   # Update the PR description/summary with milestone progress
   gh pr edit <PR_NUMBER> --body "$(cat <<'EOF'
   [Updated PR description with current progress and milestones]
   EOF
   )"
   ```
6. **Monitor CI** through PR interface
7. **Keep aider branch** for reference until project completion

## PR Management Best Practices

**🚨 IMPORTANT: UPDATE PR DESCRIPTION, NOT COMMENTS**

- **DO**: Use `gh pr edit <PR_NUMBER> --body` to update the PR summary/description with milestones
- **DON'T**: Post individual comments for each task completion (clutters PR discussion)
- **Reason**: The PR description serves as the authoritative progress summary, while comments are for discussions

**PR Description Structure:**
```markdown
# SSE Coordinator Refactoring + MCP SDK v1.9.1 Modernization

## Progress Summary
- **9/15 core tasks completed** + 2/5 modernization tasks ✅
- **Current Phase:** [Phase description]
- **Quality Status:** 480 tests passing, zero F,E9 violations

## Recent Milestones
- ✅ Task 10: InitializationSequence - Complete lifecycle management
- ✅ Task 9: ApplicationCoordinator - Central singleton coordination
- [... other completed tasks]

## Next Steps
- [Next planned tasks and dependencies]

## Quality Verification
- All CI checks passing
- Pristine quality baseline maintained
```

## Task Sequence Overview

**📋 Complete task details, dependency chains, and current status: [IMPLEMENTATION_TRACKING.md](./IMPLEMENTATION_TRACKING.md)**

**High-Level Phase Structure:**
- **Phase 1 - Foundation** (Tasks 1-3): ✅ **Completed** - Interfaces, EventSystem, TransportRegistry
- **Phase 2 - Core Coordination** (Tasks 4-5): 🎯 **Next** - EventCoordinator, RequestProcessor
- **Phase 3 - Transport Layer** (Tasks 6-7): Transport adapters (SSE, Stdio)
- **Phase 4 - Application Layer** (Tasks 8-9): HandlerRegistry, ApplicationCoordinator
- **Phase 5 - Integration** (Tasks 10-15): Initialization, Error handling, Configuration, Discovery, Tests

## Testing Strategy

**🚨 CRITICAL: NO SUBSTITUTE FOR QUALITY/TESTS AFTER EACH TASK**

For each task completion:
1. **MANDATORY Unit tests**: `hatch -e dev run pytest` - ALL TESTS MUST PASS
2. **MANDATORY Quality checks**: `hatch -e dev run ruff check --select=F,E9` - ZERO critical errors
3. **MANDATORY Pre-commit**: `hatch -e dev run pre-commit run --all-files` - Clean validation
4. **MANDATORY Documentation**: Update IMPLEMENTATION_TRACKING.md with detailed progress

**⚠️ NEVER PROCEED TO NEXT TASK WITHOUT:**
- ✅ All tests passing (no failures, no collection errors)
- ✅ Zero F,E9 critical quality violations
- ✅ Proper commit with quality verification in message
- ✅ Clean merge to feature branch

## Development Tools

- **Code implementation**: `aider:aider_ai_code` (primary tool)
- **Idea review**: `sequentialthinking`
- **Alternative solutions**: `just-prompt` to consult models like Gemini
- **Testing**: `hatch -e dev run pytest`
- **Quality**: `hatch -e dev run pre-commit run --all-files`

## Quality Standards Maintained

- ✅ All tests pass: `hatch -e dev run pytest` (432 passed, 18 skipped)
- ✅ No critical quality issues: Zero F, E9 class violations
- ✅ Type safety verified: `mypy` clean on all files
- ✅ Code formatting consistent: `ruff format` applied
- ✅ Pre-commit hooks passing: All quality checks automated
- ⚠️ Style warnings acceptable: E501 line-too-long (non-blocking)

## Directory Structure

- **Key interfaces**: `src/aider_mcp_server/interfaces/`
- **Transport adapters**: `src/aider_mcp_server/`
- **Tests**: `tests/`
- **Task reference**: `reference/tasks/tasks.json`

## Context Restart Instructions

When context window requires restart:

1. **Read this document** for current status and workflow
2. **Check IMPLEMENTATION_TRACKING.md** for detailed progress
3. **Review PR status**: https://github.com/MementoRC/aider-mcp-server/pull/18
4. **Check git log** for latest progress: `git log --oneline -5`
5. **Run tests** to verify current state: `hatch -e dev run pytest --tb=no -q`
6. **Check quality** status: `hatch -e dev run ruff check --select=F,E9`
7. **Identify next task** from dependency chain and IMPLEMENTATION_TRACKING.md
8. **Continue with aider-centric workflow** following branch strategy above

## Important Notes

**🚨 QUALITY IS NON-NEGOTIABLE**

- **Maintain pristine quality** after each phase to prevent technical debt
- **ZERO broken tests allowed** - ALL tests must pass before proceeding
- **ZERO F,E9 quality violations** - Critical errors must be resolved immediately
- **Use task-specific commits** with verified quality status in commit messages
- **Focus on solid implementations** over quick fixes - quality first, always
- **Re-run full test suite** after every merge to feature branch
- **Update IMPLEMENTATION_TRACKING.md** with detailed progress and quality verification

**⚠️ LESSON LEARNED: Import errors, test collection failures, and quality violations compound rapidly. The meticulous quality-first workflow is the ONLY way to maintain a healthy codebase.**
