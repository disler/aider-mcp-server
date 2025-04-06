# Aider MCP Server - Experimental
> Model context protocol server for offloading AI coding work to Aider, enhancing development efficiency and flexibility.

[![smithery badge](https://smithery.ai/badge/@disler/aider-mcp-server)](https://smithery.ai/server/@disler/aider-mcp-server)

## Overview

This server allows Claude Code to offload AI coding tasks to Aider, the best open source AI coding assistant. By delegating certain coding tasks to Aider, we can reduce costs, gain control over our coding model and operate Claude Code in a more orchestrative way to review and revise code.

## Setup

1. Install dependencies:

```bash
uv sync
```

2. Create your environment file:

```bash
cp .env.sample .env
```

3. Configure your API keys in the `.env` file:

```
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

## Test run the server

```bash
uv run aider-mcp-server
```

## Testing

To run all tests:

```bash
uv run pytest
```

To run specific tests:

```bash
# Test listing models
uv run pytest src/aider_mcp_server/tests/atoms/tools/test_aider_list_models.py

# Test AI coding
uv run pytest src/aider_mcp_server/tests/atoms/tools/test_aider_ai_code.py
```

Note: The AI coding tests require a valid API key for the Gemini model. Make sure to set it in your `.env` file before running the tests.

## Add this MCP server to Claude Code

### Add with `gemini-2.5-pro-exp-03-25`

```bash
claude mcp add aider-mcp-server -s local \
  -- \
  uv --directory . \
  run aider-mcp-server \
  --editor-model "gemini/gemini-2.5-pro-exp-03-25" \
  --current-working-dir "."
```

### Add with `gemini-2.5-pro-preview-03-25`

```bash
claude mcp add aider-mcp-server -s local \
  -- \
  uv --directory . \
  run aider-mcp-server \
  --editor-model "gemini/gemini-2.5-pro-preview-03-25" \
  --current-working-dir "."
```

### Add with `quasar-alpha`

```bash
claude mcp add aider-mcp-server -s local \
  -- \
  uv --directory . \
  run aider-mcp-server \
  --editor-model "openrouter/openrouter/quasar-alpha" \
  --current-working-dir "."
```

### Add with `llama4-maverick-instruct-basic`

```bash
claude mcp add aider-mcp-server -s local \
  -- \
  uv --directory . \
  run aider-mcp-server \
  --editor-model "fireworks_ai/accounts/fireworks/models/llama4-maverick-instruct-basic" \
  --current-working-dir "."
```

## Usage

This MCP server provides the following functionalities:

1. **Offload AI coding tasks to Aider**:
   - Takes a prompt and file paths
   - Uses Aider to implement the requested changes
   - Returns success or failure

2. **List available models**:
   - Provides a list of models matching a substring
   - Useful for discovering supported models

## Available Tools

This MCP server exposes the following tools:

### 1. `aider_ai_code`

This tool allows you to run Aider to perform AI coding tasks based on a provided prompt and specified files.

**Parameters:**

- `ai_coding_prompt` (string, required): The natural language instruction for the AI coding task.
- `relative_editable_files` (list of strings, required): A list of file paths (relative to the `current_working_dir`) that Aider is allowed to modify. If a file doesn't exist, it will be created.
- `relative_readonly_files` (list of strings, optional): A list of file paths (relative to the `current_working_dir`) that Aider can read for context but cannot modify. Defaults to an empty list `[]`.
- `model` (string, optional): The primary AI model Aider should use for generating code. Defaults to `"gemini/gemini-2.5-pro-exp-03-25"`. You can use the `list_models` tool to find other available models.
- `editor_model` (string, optional): The AI model Aider should use for editing/refining code, particularly when using architect mode. If not provided, the primary `model` might be used depending on Aider's internal logic. Defaults to `None`.
- `use_architect` (boolean, optional): Whether to use Aider's architect mode, which often involves a planning step before coding. Defaults to `True`.
- `use_git` (boolean, optional): Whether Aider should interact with the git repository (e.g., for context or diffs). Defaults to `True`.

**Example Usage (within an MCP request):**

```json
{
  "name": "aider_ai_code",
  "parameters": {
    "ai_coding_prompt": "Refactor the calculate_sum function in calculator.py to handle potential TypeError exceptions.",
    "relative_editable_files": ["src/calculator.py"],
    "relative_readonly_files": ["docs/requirements.txt"],
    "model": "openai/gpt-4o"
  }
}
```

**Returns:**

- A simple string indicating the outcome: `"success"` or `"failure"`.

### 2. `list_models`

This tool lists available AI models supported by Aider that match a given substring.

**Parameters:**

- `substring` (string, required): The substring to search for within the names of available models.

**Example Usage (within an MCP request):**

```json
{
  "name": "list_models",
  "parameters": {
    "substring": "gemini"
  }
}
```

**Returns:**

- A list of model name strings that match the provided substring. Example: `["gemini/gemini-1.5-flash", "gemini/gemini-1.5-pro", "gemini/gemini-pro"]`

## Architecture

The server is structured as follows:

- **Server layer**: Handles MCP protocol communication
- **Atoms layer**: Individual, pure functional components
  - **Tools**: Specific capabilities (AI coding, listing models)
  - **Utils**: Constants and helper functions
  - **Data Types**: Type definitions using Pydantic

All components are thoroughly tested for reliability.

## Codebase Structure

The project is organized into the following main directories and files:

```
.
├── ai_docs                   # Documentation related to AI models and examples
│   ├── just-prompt-example-mcp-server.xml
│   └── programmable-aider-documentation.md
├── pyproject.toml            # Project metadata and dependencies
├── README.md                 # This file
├── specs                     # Specification documents
│   └── init-aider-mcp-exp.md
├── src                       # Source code directory
│   └── aider_mcp_server      # Main package for the server
│       ├── __init__.py       # Package initializer
│       ├── __main__.py       # Main entry point for the server executable
│       ├── atoms             # Core, reusable components (pure functions)
│       │   ├── __init__.py
│       │   ├── data_types.py # Pydantic models for data structures
│       │   ├── logging.py    # Custom logging setup
│       │   ├── tools         # Individual tool implementations
│       │   │   ├── __init__.py
│       │   │   ├── aider_ai_code.py # Logic for the aider_ai_code tool
│       │   │   └── aider_list_models.py # Logic for the list_models tool
│       │   └── utils.py      # Utility functions and constants (like default models)
│       ├── server.py         # MCP server logic, tool registration, request handling
│       └── tests             # Unit and integration tests
│           ├── __init__.py
│           └── atoms         # Tests for the atoms layer
│               ├── __init__.py
│               ├── test_logging.py # Tests for logging
│               └── tools     # Tests for the tools
│                   ├── __init__.py
│                   ├── test_aider_ai_code.py # Tests for AI coding tool
│                   └── test_aider_list_models.py # Tests for model listing tool
```

- **`src/aider_mcp_server`**: Contains the main application code.
  - **`atoms`**: Holds the fundamental building blocks. These are designed to be pure functions or simple classes with minimal dependencies.
    - **`tools`**: Each file here implements the core logic for a specific MCP tool (`aider_ai_code`, `list_models`).
    - **`utils.py`**: Contains shared constants like default model names.
    - **`data_types.py`**: Defines Pydantic models for request/response structures, ensuring data validation.
    - **`logging.py`**: Sets up a consistent logging format for console and file output.
  - **`server.py`**: Orchestrates the MCP server. It initializes the server, registers the tools defined in the `atoms/tools` directory, handles incoming requests, routes them to the appropriate tool logic, and sends back responses according to the MCP protocol.
  - **`__main__.py`**: Provides the command-line interface entry point (`aider-mcp-server`), parsing arguments like `--editor-model` and starting the server defined in `server.py`.
  - **`tests`**: Contains tests mirroring the structure of the `src` directory, ensuring that each component (especially atoms) works as expected.

