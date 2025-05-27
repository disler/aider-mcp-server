#!/bin/bash
# Environment synchronization script

set -e

echo "🔄 Synchronizing all environments..."

# Update uv lock file
echo "📦 Updating uv lock file..."
uv lock --upgrade

# Refresh uv virtual environment
echo "🔄 Refreshing uv .venv environment..."
uv sync --refresh --dev

# Update hatch environments
echo "🏠 Updating hatch environments..."
hatch env prune
hatch -e dev run pip install --upgrade pip

echo "✅ All environments synchronized!"
echo ""
echo "Next steps:"
echo "- Test with: hatch -e dev run pytest"
echo "- Verify MCP server: uv run mcp-aider-server --help"
