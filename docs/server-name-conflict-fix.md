# Fix for MCP Server Name Conflict Issue

## Problem Statement

When using the `setup-aider-mcp` command with the `--also-just-prompt` flag, users were encountering an error when trying to add both servers to Claude Code:

```
user@host:~$ claude mcp add-json aider-mcp-server -s local '{"type": "stdio", "command": "uv", "args": ["--directory", "/path/to/aider-mcp-server", "run", "aider-mcp-server", "--editor-model", "gemini/gemini-2.5-pro-exp-03-25", "--current-working-dir", "/path/to/your/project"]}'
A server with the name aider-mcp-server already exists.
```

This occurred because servers can exist in multiple scopes (project, local, user), and the `claude mcp list` command doesn't show project-specific servers. When a server exists in the project scope but we try to add it to the local scope, we get the "already exists" error.

## Solution

The fix involves generating a bash script that:
1. Removes servers from ALL possible scopes (project, local, user) before adding
2. Uses error handling (`2>/dev/null || true`) to ignore errors when a server doesn't exist in a particular scope
3. Then adds the server to the local scope
4. Provides proper error handling and user feedback

## Implementation Details

### 1. Updated `generate_claude_command` Function

The function now generates a complete bash script instead of individual commands:

```python
def generate_claude_command(aider_dir: str, model: str, target_dir: str, has_mcp_file: bool = False, just_prompt_config: Optional[Dict] = None) -> str:
    """Generate a bash script to properly add MCP servers to Claude Code."""
    # ... generates a bash script that checks and removes existing servers
```

### 2. Bash Script Features

The generated script includes:
- Bash shebang and error handling (`set -e`)
- Server existence checks using `grep`
- Automatic removal of existing servers
- Proper JSON escaping for bash
- Success/failure messages

### 3. Script Generation Example

For a configuration with both servers, the script will generate:

```bash
#!/bin/bash
# Script to add MCP servers to Claude Code
# This script will automatically remove existing servers before adding new ones
# from all possible scopes (project, local, user) before adding the new configuration

# Configure aider-mcp-server
echo "Removing any existing aider-mcp-server server..."
# Remove from project scope
claude mcp remove -s project aider-mcp-server 2>/dev/null || true
# Remove from local scope
claude mcp remove -s local aider-mcp-server 2>/dev/null || true
# Remove from user/global scope
claude mcp remove -s user aider-mcp-server 2>/dev/null || true

echo "Adding aider-mcp-server server..."
claude mcp add-json aider-mcp-server -s local '{"type": "stdio", ...}'
if [ $? -eq 0 ]; then
    echo "aider-mcp-server server added successfully!"
else
    echo "Failed to add aider-mcp-server server. Please check the error message above."
    exit 1
fi

# Similar section for just-prompt server...
```

### 4. User Experience Improvements

- The script is automatically saved to `add_mcp_servers.sh` in the target directory
- The script is made executable (`chmod +x`)
- Clear instructions are provided on how to run the script
- The script provides user-friendly feedback during execution

## Testing

The fix has been thoroughly tested with:
1. Unit tests for the `generate_claude_command` function
2. Tests for proper JSON escaping
3. Tests for handling multiple servers
4. Tests for various server name formats

All tests pass, ensuring the fix handles edge cases properly.

## Usage

After running the setup command:

```bash
setup-aider-mcp --also-just-prompt
```

Users will receive:
1. A displayed version of the script they can copy and run
2. A saved script file they can execute directly: `./add_mcp_servers.sh`

This approach ensures that servers can be successfully added to Claude Code even if they already exist, resolving the original issue.