# setup-aider-mcp Commands

The `setup-aider-mcp` utility now supports multiple subcommands for enhanced functionality:

## Commands

### setup

The default command that sets up the aider-mcp configuration.

```bash
# Basic setup
setup-aider-mcp setup

# With model selection
setup-aider-mcp setup --model "gemini/gemini-2.5-pro-exp-03-25"

# With just-prompt integration
setup-aider-mcp setup --also-just-prompt

# Full options
setup-aider-mcp setup \
  --current-dir /path/to/project \
  --aider-dir /path/to/aider-mcp-server \
  --model "openai/gpt-4o" \
  --also-just-prompt \
  --just-prompt-dir /path/to/just-prompt \
  --just-prompt-models "o:gpt-4o,a:claude-3-5-haiku"
```

### change-model

Change the model in an existing configuration.

```bash
# Interactive model selection
setup-aider-mcp change-model

# Direct model change
setup-aider-mcp change-model --model "openai/gpt-4o"

# Specify project directory
setup-aider-mcp change-model --current-dir /path/to/project --model "gemini/gemini-2.5-pro-exp-03-25"
```

### list-models

List available models based on configured API keys.

```bash
# List available models
setup-aider-mcp list-models

# Specify aider installation path
setup-aider-mcp list-models --aider-dir /path/to/aider-mcp-server
```

## Default Behavior

When run without any command, `setup-aider-mcp` defaults to the `setup` command:

```bash
# These are equivalent
setup-aider-mcp
setup-aider-mcp setup
```

## Examples

### Setting up a New Project

```bash
cd /path/to/your/project
git init  # Ensure it's a git repository
setup-aider-mcp setup --also-just-prompt
```

### Changing Models in Existing Project

```bash
cd /path/to/your/project
setup-aider-mcp change-model --model "openai/gpt-4o"
```

### Checking Available Models

```bash
setup-aider-mcp list-models
```

## Model Configuration

The tool supports the following models by default:

- **Gemini**: `gemini/gemini-2.5-pro-exp-03-25`
- **OpenAI**: `openai/gpt-4o`
- **Anthropic**: `anthropic/claude-3-opus-20240229`
- **OpenRouter**: `openrouter/openrouter/quasar-alpha`
- **Fireworks**: `fireworks_ai/accounts/fireworks/models/llama4-maverick-instruct-basic`

To use a model, ensure the corresponding API key is set:

- `GEMINI_API_KEY` for Gemini
- `OPENAI_API_KEY` for OpenAI
- `ANTHROPIC_API_KEY` for Anthropic
- `OPENROUTER_API_KEY` for OpenRouter
- `FIREWORKS_API_KEY` for Fireworks

## Configuration File

The tool creates or modifies a `.mcp.json` file in your project directory. After making changes, you'll need to run the generated bash script to apply the configuration to Claude Code.

## Tips

1. Always run the tool from within a git repository - Aider requires git to function properly.
2. Use `list-models` to see which models are available based on your configured API keys.
3. The `change-model` command is useful for quickly switching between models without recreating the entire configuration.
4. When using `--also-just-prompt`, the tool will configure both aider-mcp and just-prompt in a single `.mcp.json` file.