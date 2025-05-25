# Claude Context Reset Instructions

This document contains essential information for working on AIDER MCP Server compatibility reviews and ongoing development.

## Project Overview

This is an Aider MCP Server that provides comprehensive AI coding task offloading capabilities following atomic design principles. The server implements full AIDER v0.83.1 API compatibility through Model Context Protocol (MCP) interfaces.

## Current Primary Task: AIDER Compliance Review

🎯 **Current Focus**: Comprehensive review of MCP server compatibility with AIDER v0.83.1 functionality and API evolution tracking.

### Review Objectives
- **API Compatibility**: Verify all AIDER v0.83.1 features are properly supported
- **Interface Compliance**: Ensure MCP tools match AIDER's programmatic interfaces  
- **Feature Parity**: Confirm architectural patterns align with AIDER's capabilities
- **Evolution Tracking**: Establish framework for future AIDER version compatibility

### Review Scope
- Core AIDER integration points (Coder class, Model configuration, Git operations)
- MCP tool implementations vs AIDER command-line interfaces
- Architecture patterns and extensibility hooks
- Missing features and enhancement opportunities

## 📋 AIDER Compliance Tracking

**📊 All compliance review tracking is managed in [AIDER_COMPLIANCE_TRACKING.md](./AIDER_COMPLIANCE_TRACKING.md)**

This dedicated tracking document contains:
- **Complete compliance matrix** with feature status across AIDER v0.83.1
- **Phase-by-phase review plan** with specific tasks and deliverables
- **Gap analysis** identifying missing features and enhancement opportunities
- **Action items** prioritized by impact and implementation complexity

**Current Status:** Phase 1 ✅ COMPLETE (100% Core API Compatibility) | Phase 2 📋 READY

## Review Workflow Process

**🎯 SYSTEMATIC AIDER COMPLIANCE REVIEW WORKFLOW**

### Current Review Phase: Core API Compatibility

1. **Review Planning**
   - Follow [AIDER_COMPLIANCE_TRACKING.md](./AIDER_COMPLIANCE_TRACKING.md) phase structure
   - Create dedicated review branch: `git checkout -b aider/compliance-review-phase-X`
   - Focus on specific compliance areas per session

2. **API Compatibility Analysis**
   - **Compare AIDER v0.83.1 source** with MCP server implementations
   - **Test parameter compatibility** using automated compatibility checks
   - **Validate interface compliance** through integration testing
   - **Document findings** in compliance matrix

3. **Gap Identification & Prioritization**
   - **Identify missing features** not exposed through MCP tools
   - **Assess implementation complexity** and impact
   - **Prioritize enhancements** based on user value and feasibility
   - **Plan implementation roadmap** for identified gaps

4. **Quality Validation** (Applied to any implementations)
   - Run `hatch -e dev run pytest` → ALL tests must pass
   - Run `hatch -e dev run ruff check --select=F,E9` → ZERO critical errors  
   - Run `hatch -e dev run pre-commit run --all-files` → Clean validation
   - **Compatibility regression testing** against existing AIDER integrations

5. **Documentation & Tracking Updates**
   - **Update compliance matrix** with findings and status
   - **Document enhancement recommendations** with implementation notes
   - **Create action items** with priority assignments
   - **Commit review findings** with comprehensive documentation

### Future Version Compatibility Setup

6. **Version Monitoring Framework**
   - **Set up GitHub watch** for AIDER repository releases
   - **Create automated compatibility checks** for new versions
   - **Establish regular review schedule** (quarterly or on major releases)
   - **Document upgrade process** for systematic compatibility maintenance

## Git Commit Strategy & Review Workflow

### Branch Strategy
- **Main branch**: `feature/atomic-design-reorganization` (current development)
- **Review branches**: `aider/compliance-review-phase-X` (for specific review phases)
- **Enhancement branches**: `aider/enhancement-X-name` (for implementing identified improvements)

### Commit Pattern
Each review phase or enhancement gets its own commit following this pattern:
```
review: AIDER v0.83.1 compliance Phase X - [Phase Title]

- [Key findings 1]
- [Gap analysis results]
- [Compliance matrix updates]

✅ Review: [Phase completion status]
🔧 Quality: [Quality check status if changes made]
📊 Compliance: [Overall compliance score]
📋 Next: [Next review phase]

🤖 Generated with [Claude Code](https://claude.ai/code)
Co-Authored-By: Claude <noreply@anthropic.com>
```

### Review Phase Workflow
1. **Plan review phase** using [AIDER_COMPLIANCE_TRACKING.md](./AIDER_COMPLIANCE_TRACKING.md)
2. **Create review branch**: `git checkout -b aider/compliance-review-phase-X`
3. **Conduct systematic review** of specific compliance areas
4. **Document findings** in compliance matrix and action items
5. **Commit review results** with comprehensive documentation
6. **Merge to main branch** and update tracking documents
7. **Proceed to next phase** based on priority and dependencies

### Enhancement Implementation Workflow
1. **Select enhancement** from prioritized action items
2. **Create enhancement branch**: `git checkout -b aider/enhancement-X-name`
3. **Implement enhancement** using aider-centric approach
4. **Run quality validation**:
   ```bash
   # Run full test suite
   hatch -e dev run pytest

   # Check for critical violations
   hatch -e dev run ruff check --select=F,E9

   # Run pre-commit validation
   hatch -e dev run pre-commit run --all-files
   ```
5. **Update compliance matrix** with enhancement status
6. **Create enhancement-specific commit** with quality verification
7. **Merge and validate** overall system integration

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
