import argparse
import asyncio
from aider_mcp_server.server import serve
from aider_mcp_server.atoms.utils import DEFAULT_EDITOR_MODEL

def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Aider MCP Server - Offload AI coding tasks to Aider")
    
    # Add arguments
    parser.add_argument(
        "--editor-model", 
        type=str, 
        default=DEFAULT_EDITOR_MODEL,
        help=f"Editor model to use (default: {DEFAULT_EDITOR_MODEL})"
    )
    
    args = parser.parse_args()
    
    # Run the server asynchronously
    asyncio.run(serve(
        editor_model=args.editor_model
    ))

if __name__ == "__main__":
    main()