# setup-aider-mcp Utility

The `setup-aider-mcp` utility is our contribution to the aider-mcp-server project. It simplifies the setup and management of aider-mcp configurations.

## Features

- **Easy Setup**: Automatically configures aider-mcp for your project
- **Model Management**: Change models on the fly without manual config editing
- **Model Listing**: See available models based on your API keys
- **Just-Prompt Integration**: Optional integration with the just-prompt MCP plugin
- **Subcommand Support**: Organized commands for different tasks

## Commands

### setup (default)
```bash
setup-aider-mcp setup
# or simply:
setup-aider-mcp
```

Sets up aider-mcp configuration for your project. Options:
- `--model`: Specify model directly 
- `--also-just-prompt`: Include just-prompt configuration
- `--current-dir`: Target project directory
- `--aider-dir`: Path to aider-mcp-server installation

### change-model
```bash
setup-aider-mcp change-model
# or with model:
setup-aider-mcp change-model --model "openai/gpt-4o"
```

Changes the model in existing configuration.

### list-models
```bash
setup-aider-mcp list-models
```

Lists available models based on configured API keys.

## Key Files

- `src/aider_mcp_server/setup_aider_mcp.py`: Main setup script with subcommand support
- `src/aider_mcp_server/scripts/setup_aider_mcp_cli.py`: CLI entry point
- `docs/setup-aider-mcp-commands.md`: Detailed documentation

## Testing

All functionality is tested:
- `test_setup_aider_mcp.py`: Original setup tests
- `test_just_prompt_integration.py`: Just-prompt integration tests
- `test_setup_subcommands.py`: Subcommand functionality tests

## Usage Example

```bash
# Initial setup
cd /path/to/your/project
git init  # Required by aider
setup-aider-mcp --also-just-prompt

# Later, change model
setup-aider-mcp change-model --model "gemini/gemini-2.5-pro"

# Check available models
setup-aider-mcp list-models
```