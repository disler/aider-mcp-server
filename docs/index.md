# aider-mcp-server Documentation

Welcome to the official documentation for **aider-mcp-server**, an enhanced pytest failure analysis and fix suggestion tool.

## Installation

Install from PyPI:
```bash
pip install aider-mcp-server
```

Or install from source:
```bash
git clone https://github.com/MementoRC/aider-mcp-server.git
cd aider-mcp-server
pip install -e .[dev]
```

## Documentation Structure

- **README**: Core usage, CLI flags, and API examples ([README.md](../README.md)).
- **Configuration**: See the `mkdocs.yml` and docs/ for this site’s configuration.
- **Source Code**: Explore modules under `src/pytest_analyzer/` for internals.

## Building Locally

```bash
pip install mkdocs
mkdocs serve
```