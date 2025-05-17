#!/bin/bash
# Test CLI verbose flag

echo "=== Test without verbose flag ==="
python -m aider_mcp_server --help | grep -A 2 verbose

echo -e "\n=== Test verbose flag functionality ==="
echo "Testing that the verbose flag would enable DEBUG logging..."
echo "To fully test, you would need a git repository and run:"
echo "python -m aider_mcp_server --verbose --current-working-dir /path/to/git/repo"
