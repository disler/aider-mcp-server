# Using Environment Variables with setup-aider-mcp

The `setup-aider-mcp` command now supports using environment variables to specify the location of both the aider-mcp-server and just-prompt installations, making it easier to use from any directory.

## AIDER_MCP_SERVER_PATH

Use `AIDER_MCP_SERVER_PATH` to specify the location of your aider-mcp-server installation.

### Setting the Environment Variable

You can set the `AIDER_MCP_SERVER_PATH` environment variable in several ways:

### 1. In your shell profile (permanent):

Add this line to your `.bashrc`, `.zshrc`, or similar:
```bash
export AIDER_MCP_SERVER_PATH="/path/to/aider-mcp-server"
```

### 2. In a .env file (project-specific):

Create a `.env` file in your project directory:
```
AIDER_MCP_SERVER_PATH=/path/to/aider-mcp-server
```

The script now automatically reads this file, so you can simply run:
```bash
setup-aider-mcp setup
```

No need to source the file first - the script will automatically check for these variables in your .env file.

### 3. For a single command (temporary):

```bash
AIDER_MCP_SERVER_PATH=/path/to/aider-mcp-server setup-aider-mcp setup
```

## Priority Order

The script looks for the aider-mcp-server installation in this order:
1. Command-line argument (`--aider-dir`)
2. Environment variable (`AIDER_MCP_SERVER_PATH`)
3. Relative to the script location
4. Current working directory (searching up to 3 levels)

## Example

Once you have set the environment variable, you can run `setup-aider-mcp` from any directory:

```bash
cd /path/to/your/project
setup-aider-mcp setup  # Will find aider-mcp-server using the env var
```

This eliminates the need to specify `--aider-dir` every time.

## JUST_PROMPT_PATH

Use `JUST_PROMPT_PATH` to specify the location of your just-prompt installation when using the `--also-just-prompt` option.

### Setting the Environment Variable

You can set the `JUST_PROMPT_PATH` environment variable in the same ways as described above:

### 1. In your shell profile (permanent):

Add this line to your `.bashrc`, `.zshrc`, or similar:
```bash
export JUST_PROMPT_PATH="/path/to/just-prompt"
```

### 2. In a .env file (project-specific):

Create a `.env` file in your project directory:
```
JUST_PROMPT_PATH=/path/to/just-prompt
```

The script will automatically read this file, with no need to manually source it.

### 3. For a single command (temporary):

```bash
JUST_PROMPT_PATH=/path/to/just-prompt setup-aider-mcp setup --also-just-prompt
```

## Example Using Both Environment Variables

You can use both environment variables together for a completely streamlined setup:

```bash
# Set both environment variables
export AIDER_MCP_SERVER_PATH=/path/to/aider-mcp-server
export JUST_PROMPT_PATH=/path/to/just-prompt

# Run the setup command with minimal options
cd /path/to/your/project
setup-aider-mcp setup --also-just-prompt
```

Or for a one-line command:

```bash
AIDER_MCP_SERVER_PATH=/path/to/aider-mcp-server JUST_PROMPT_PATH=/path/to/just-prompt setup-aider-mcp setup --also-just-prompt
```