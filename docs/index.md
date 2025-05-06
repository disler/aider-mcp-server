# Aider MCP Server

## Overview

Aider MCP Server is a specialized application that integrates [Aider](https://github.com/paul-gauthier/aider), an AI-powered coding assistant, with the Model Context Protocol (MCP). This integration allows AI coding capabilities to be accessed through multiple transport mechanisms, including Standard I/O (stdio) and Server-Sent Events (SSE), making AI-assisted coding more accessible across different platforms and interfaces.

The server offloads AI coding tasks to Aider, providing a standardized way to request code modifications, receive real-time progress updates, and get results through various communication channels.

## Key Features

- **Multiple Transport Mechanisms**: Support for both stdio (command-line) and SSE (web-based) communications
- **Flexible Architecture**: Modular design with transport adapters and central coordination
- **Real-time Progress Updates**: Event-based system for tracking operation progress
- **Robust Error Handling**: Rate limit detection with model fallback capabilities
- **Diff Caching**: Optimizes repeated operations by caching file differences
- **Security Controls**: Built-in security validation for requests

## Architecture

The architecture of Aider MCP Server is built around several key components:

### Transport Layer

- **AbstractTransportAdapter**: Base abstract class that defines the interface for all transport adapters
- **StdioTransportAdapter**: Implements communication via standard input/output streams
- **SSETransportAdapter**: Implements Server-Sent Events for web-based communications

### Coordination

- **ApplicationCoordinator**: Central singleton that:
  - Manages transport registrations
  - Routes requests to appropriate handlers
  - Broadcasts events back to subscribed transports
  - Handles request lifecycle management

### Handlers

- **process_aider_ai_code_request**: Processes AI coding tasks using Aider
- **process_list_models_request**: Lists available AI models

### Event System

The system uses an event-based architecture with event types including:
- `STATUS`: General status updates
- `PROGRESS`: Operation progress information
- `TOOL_RESULT`: Final results from tool operations
- `HEARTBEAT`: Regular connectivity checks

## Installation

### From PyPI

```bash
pip install aider-mcp-server
```

### From Source

```bash
git clone https://github.com/yourusername/aider-mcp-server.git
cd aider-mcp-server
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
pre-commit install
```

## Configuration

Aider MCP Server can be configured using environment variables:

```
# API Keys for different AI services
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GOOGLE_API_KEY=your_google_api_key
GEMINI_API_KEY=your_gemini_api_key

# Default model to use
AIDER_MODEL=gpt-4o
```

You can create a `.env` file in your project directory with these variables, or set them in your environment before running the server.

## Running the Server

The server can be run in three different modes:

### Standard I/O Mode (Default)

Ideal for CLI applications, plugins, or extensions that can communicate via stdio.

```bash
mcp-aider-stdio --current-working-dir /path/to/your/repository
```

### SSE Mode

For web applications that need to receive real-time updates.

```bash
mcp-aider-sse --current-working-dir /path/to/your/repository --host localhost --port 5005
```

### Multi-Transport Mode

Combines both stdio and SSE transport mechanisms.

```bash
mcp-aider-multi --current-working-dir /path/to/your/repository --host localhost --port 5005
```

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--server-mode` | Server communication mode (`stdio`, `sse`, or `multi`) | `stdio` |
| `--editor-model` | Primary AI model for Aider | `gpt-4o` (configurable) |
| `--current-working-dir` | Path to the working directory (must be a valid git repository) | Required |
| `--host` | Host address for SSE/Multi server | `localhost` |
| `--port` | Port number for SSE/Multi server | `5005` |

## Available MCP Tools

### aider_ai_code

This tool processes AI coding tasks based on a prompt and set of files.

#### Parameters

- `ai_coding_prompt` (string): The prompt for the AI to execute
- `relative_editable_files` (array): List of relative paths to files that can be edited
- `relative_readonly_files` (array, optional): List of relative paths to files that can be read but not edited
- `model` (string, optional): The model to use (defaults to the server's configured model)

#### Response

```json
{
  "success": true,
  "diff": "...git diff or file content...",
  "is_cached_diff": false,
  "rate_limit_info": {
    "encountered": false,
    "retries": 0,
    "fallback_model": null
  }
}
```

### list_models

Lists available AI models that match a given substring.

#### Parameters

- `substring` (string, optional): Substring to match against available models

#### Response

```json
{
  "success": true,
  "models": ["gpt-4", "gpt-4o", "gpt-3.5-turbo", "claude-3-opus", "claude-3-sonnet", "gemini-pro", ...]
}
```

## API Endpoints (SSE Mode)

When running in SSE mode, the server exposes these HTTP endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/sse` | GET | Establishes an SSE connection for receiving events |
| `/message` | POST | Submits tool requests (aider_ai_code, list_models) |

### SSE Events

When connected to the `/sse` endpoint, clients can receive these event types:

- `connected`: Initial connection established
- `STATUS`: Status updates about operations
- `PROGRESS`: Progress updates during operations
- `TOOL_RESULT`: Final results from tool operations
- `HEARTBEAT`: Regular connectivity checks

## Integration Examples

### Python Client Example (SSE)

```python
import asyncio
import json
import httpx
from sse_starlette.sse import EventSourceResponse

async def main():
    # Connect to SSE endpoint
    async with httpx.AsyncClient(timeout=None) as client:
        sse_task = asyncio.create_task(listen_for_events(client))
        
        # Send a coding request
        response = await client.post(
            "http://localhost:5005/message",
            json={
                "request_id": "req-1",
                "name": "aider_ai_code",
                "parameters": {
                    "ai_coding_prompt": "Add documentation to the function",
                    "relative_editable_files": ["src/example.py"],
                    "relative_readonly_files": [],
                    "model": "gpt-4o"
                }
            }
        )
        
        print(f"Request response: {response.json()}")
        await sse_task

async def listen_for_events(client):
    async with client.stream("GET", "http://localhost:5005/sse") as response:
        async for line in response.aiter_lines():
            if line.startswith("data:"):
                data = json.loads(line[5:].strip())
                print(f"Received event: {data}")
                
                # If it's a tool result, we're done
                if data.get("type") == "TOOL_RESULT":
                    break

if __name__ == "__main__":
    asyncio.run(main())
```

### MCP Client Example

```python
import asyncio
import mcp

async def main():
    client = mcp.Client()
    await client.connect("localhost", 5005)

    result = await client.call(
        "aider_ai_code",
        {
            "ai_coding_prompt": "Add a test for the hello function",
            "relative_editable_files": ["main.py", "test.py"],
            "relative_readonly_files": [],
            "model": "gpt-4o"
        }
    )

    print(result)

asyncio.run(main())
```

## Security Considerations

- The server validates all requests and requires a valid git repository as the working directory
- For production environments, consider adding authentication mechanisms when using SSE mode
- API keys for AI services are handled securely through environment variables

## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure your AI service API keys are correctly set in your environment.
2. **Git Repository Errors**: The working directory must be a valid git repository.
3. **Permission Issues**: Ensure the server has read/write access to the specified files.
4. **Rate Limiting**: If you encounter rate limit errors, the server will attempt to use fallback models, but you may need to wait or use a different service.

### Logs

Logs are stored in `~/.aider-mcp-server/logs/` by default. Check these logs for detailed error information.

## Development

### Project Structure

```
src/aider_mcp_server/
├── __init__.py           # Package initialization
├── __main__.py           # Entry point
├── app.py                # Application setup
├── atoms/                # Core utilities and data structures
│   ├── atoms_utils.py    # Utility functions
│   ├── data_types.py     # Data type definitions
│   ├── diff_cache.py     # Cache for file diffs
│   ├── event_types.py    # Event type definitions
│   ├── logging.py        # Logging configuration
│   └── tools/            # Tool implementations
├── handlers.py           # Request handlers
├── multi_transport_server.py  # Combined transport server
├── progress_reporter.py  # Progress reporting
├── security.py           # Security validation
├── server.py             # Base server implementation
├── sse_server.py         # SSE server implementation
├── sse_transport_adapter.py  # SSE transport adapter
├── stdio_transport_adapter.py # Stdio transport adapter
├── transport_adapter.py  # Abstract transport interface
└── transport_coordinator.py  # Request coordination
```

### Running Tests

```bash
pytest
```

## License

MIT

## Additional Resources

- [Aider Documentation](https://aider.chat/docs/)
- [MCP Protocol](https://github.com/Microsoft/mcp)
- [SSE Specification](https://html.spec.whatwg.org/multipage/server-sent-events.html)