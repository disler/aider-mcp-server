# ✅ COMPLETED: Integrate just-prompt Setup with setup-aider-mcp

## Overview
Add functionality to the `setup-aider-mcp` command to also set up the just-prompt MCP plugin, since they work well together. This will allow users to install both tools with a single command.

## Implementation Details

### 1. Add Command Line Option
Add a new option to `setup-aider-mcp`:
```bash
setup-aider-mcp --also-just-prompt
```

### 2. Integration Steps
When the `--also-just-prompt` flag is provided, the command should:

1. **Check for just-prompt installation**
   - Look for just-prompt in the following locations:
     - Default locations and nearby directories
     - User-specified path via additional option: `--just-prompt-dir PATH`
   - Verify the installation by checking for `src/just_prompt` directory

2. **Read environment configuration**
   - Check for `.env` file in the just-prompt directory
   - Read API keys for available providers (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
   - Warn if no API keys are configured

3. **Model selection**
   - If interactive mode: prompt user to select from available providers
   - If non-interactive: use all available providers with default models
   - Allow override with `--just-prompt-models` option

4. **Create MCP configuration**
   - Add just-prompt configuration to the existing `.mcp.json` file
   - The config should look like:
   ```json
   {
     "mcpServers": {
       "aider": { ... existing aider config ... },
       "just-prompt": {
         "type": "stdio",
         "command": "uv",
         "args": ["--directory", "/path/to/just-prompt", "run", "just-prompt", "--default-models", "selected-models"],
         "env": {}
       }
     }
   }
   ```

5. **Install with Claude Code**
   - Run `claude mcp install` with the updated configuration

### 3. Code to Reference
The implementation should reference the existing just-prompt setup script:
- just-prompt source code in the `src/just_prompt/setup_just_prompt.py` file

Key functions to adapt:
- `find_just_prompt_server()` - for locating the installation
- `read_env_file()` - for reading API keys
- `get_available_providers()` - for determining which providers have keys
- `select_models()` - for model selection logic
- `create_mcp_config()` - for configuration generation

### 4. Default Model Mapping
Use the same default models as just-prompt:
```python
DEFAULT_MODELS = {
    "openai": "o:gpt-4o-mini",
    "anthropic": "a:claude-3-5-haiku",
    "gemini": "g:gemini-2.0-flash",
    "groq": "q:llama-3.1-70b-versatile",
    "deepseek": "d:deepseek-coder",
    "ollama": "l:llama3.1",
}
```

### 5. Error Handling
- Gracefully handle missing just-prompt installation
- Warn about missing API keys
- Don't fail the aider setup if just-prompt setup fails

### 6. Help Documentation
Update the help text for `setup-aider-mcp` to include:
```
--also-just-prompt      Also set up the just-prompt MCP plugin
--just-prompt-dir PATH  Path to just-prompt installation
--just-prompt-models    Comma-separated list of models (e.g., 'o:gpt-4o,a:claude-3-5-haiku')
```

## Testing
1. Test installation with and without the flag
2. Test with missing just-prompt installation
3. Test with missing API keys
4. Test model selection in interactive and non-interactive modes
5. Test configuration merge with existing .mcp.json

## Notes
- This integration makes sense because aider and just-prompt complement each other well
- The setup should be optional to avoid forcing users who only want aider
- Consider adding a reciprocal option in just-prompt to install aider

## ✅ Completion Notes

This task has been successfully completed. All requested features have been implemented:

### Implemented Features
1. ✅ Command-line options: `--also-just-prompt`, `--just-prompt-dir`, `--just-prompt-models`
2. ✅ Functions to find and configure just-prompt:
   - `find_just_prompt_server()`
   - `read_just_prompt_env_file()`
   - `get_just_prompt_providers()`
   - `select_just_prompt_models()`
   - `create_just_prompt_config()`
3. ✅ Merged configuration support in `.mcp.json`
4. ✅ Default location detection logic
5. ✅ Interactive and non-interactive model selection
6. ✅ Error handling for missing installation or API keys
7. ✅ Help documentation updated
8. ✅ Comprehensive test coverage added:
   - `test_just_prompt_integration.py` - Unit tests for all functions
   - `test_just_prompt_integration_cli.py` - CLI argument testing
   - `test_generate_claude_command.py` - Script generation tests

### Usage Examples

```bash
# Basic usage - will prompt for model selection
setup-aider-mcp --also-just-prompt

# Specify custom just-prompt location
setup-aider-mcp --also-just-prompt --just-prompt-dir /path/to/just-prompt

# Pre-select models (non-interactive)
setup-aider-mcp --also-just-prompt --just-prompt-models "o:gpt-4o,a:claude-3-5-haiku"

# All options combined
setup-aider-mcp --also-just-prompt --just-prompt-dir /custom/path --just-prompt-models "o:gpt-4o,g:gemini-2.0-flash"
```

All tests are passing and the implementation fully meets the requirements specified in this TODO.