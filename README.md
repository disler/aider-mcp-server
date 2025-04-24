# Aider MCP Server

This project provides an MCP (Modular Code Processing) server that offloads AI coding tasks to [Aider](https://github.com/paul-gauthier/aider).

## Installation

```bash
pip install aider-mcp-server
```

Or install from source:

```bash
git clone https://github.com/yourusername/aider-mcp-server.git
cd aider-mcp-server
pip install -e .
```

## Usage

### Starting the server

```bash
mcp-aider-server
```

By default, the server runs on `localhost:5005`.

### Configuration

The server can be configured using environment variables:

- `AIDER_MODEL`: The default AI model to use (defaults to "gpt-4o")
- Add your API keys for the AI services used by Aider

You can create a `.env` file in the project directory with these variables.

### Making requests

The server exposes the following MCP methods:

- `aider.process`: Process files with Aider based on a prompt

Example client:

```python
import asyncio
import mcp

async def main():
    client = mcp.Client()
    await client.connect("localhost", 5005)

    result = await client.call(
        "aider.process",
        {
            "files": {
                "main.py": "def hello():\n    print('Hello, world!')\n",
                "test.py": "import unittest\n\nclass TestMain(unittest.TestCase):\n    pass\n"
            },
            "prompt": "Add a test for the hello function",
            "model": "gpt-4o"
        }
    )

    print(result)

asyncio.run(main())
```

## Development

### Setup

```bash
pip install -e ".[dev]"
pre-commit install
```

### Running tests

```bash
pytest
```

## License

MIT
