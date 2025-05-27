# Development Environment Guide

This project uses **dual environment management** with both `uv` and `hatch` for different purposes.

## Environment Overview

- **UV (.venv)**: Used for MCP server execution and runtime dependencies
- **Hatch**: Used for development tasks, testing, and quality checks

## Quick Start

### 1. Initial Setup
```bash
# Install dependencies and sync environments
hatch -e dev run sync-all
```

### 2. Daily Development Workflow
```bash
# Run tests
hatch -e dev run test

# Format and lint
hatch -e dev run format
hatch -e dev run lint

# All quality checks
hatch -e dev run check-all
```

### 3. When Dependencies Change
```bash
# After modifying pyproject.toml dependencies
uv lock --upgrade           # Update lock file
uv sync --refresh --dev     # Refresh .venv
hatch env prune             # Clean hatch environments

# Or use the convenience script
hatch -e dev run sync-all
```

## Environment Management Commands

### UV Commands (for MCP server)
```bash
uv sync                     # Sync .venv with lock file
uv sync --refresh           # Force refresh .venv
uv sync --dev              # Include dev dependencies
uv lock --upgrade          # Update dependencies
uv run mcp-aider-server    # Run MCP server
```

### Hatch Commands (for development)
```bash
hatch -e dev run test       # Run tests
hatch -e dev run lint       # Run linting
hatch -e dev run format     # Format code
hatch env show              # Show environments
hatch env prune             # Clean environments
```

## Troubleshooting

### "Module not found" or Import Errors
```bash
# Refresh both environments
uv sync --refresh --dev
hatch env prune
hatch -e dev run sync-all
```

### MCP Server Issues
```bash
# Test MCP server directly
uv run mcp-aider-server --help

# Check .venv dependencies
.venv/bin/python -c "import aider; print('OK')"
```

### Environment Conflicts
```bash
# Remove problematic .venv and recreate
rm -rf .venv
uv sync --dev
```

## Quality Gates

Before committing:
```bash
hatch -e dev run check-all  # Runs format + lint + test
```

## Architecture Notes

- **UV manages runtime**: The `.venv` created by UV is used when the MCP server runs
- **Hatch manages development**: Testing, linting, formatting use hatch environments
- **Lock file authority**: `uv.lock` is the source of truth for dependency versions
- **Dual sync required**: Both environments need refresh when dependencies change