import json
import os
import shutil
import subprocess
import sys  # Import sys for stderr
from pathlib import Path  # Import Path
from typing import Any, Dict, List, Optional, Tuple, Union  # Add Optional

# Use absolute imports from the package root
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from ...atoms.logging.logger import get_logger
from ...atoms.security.context import ANONYMOUS_SECURITY_CONTEXT
from ...atoms.utils.atoms_utils import DEFAULT_EDITOR_MODEL
from ...organisms.processors.handlers import (
    process_aider_ai_code_request,
    process_list_models_request,
)

# Configure logging
logger = get_logger(__name__)

# Define MCP tools
AIDER_AI_CODE_TOOL = Tool(
    name="aider_ai_code",
    description="Run Aider to perform AI coding tasks based on the provided prompt and files",
    inputSchema={
        "type": "object",
        "properties": {
            "ai_coding_prompt": {
                "type": "string",
                "description": "The prompt for the AI to execute",
            },
            "relative_editable_files": {
                "type": "array",
                "description": "LIST of relative paths to files that can be edited",
                "items": {"type": "string"},
            },
            "relative_readonly_files": {
                "type": "array",
                "description": "LIST of relative paths to files that can be read but not edited, add files that are not editable but useful for context",
                "items": {"type": "string"},
            },
            "model": {
                "type": "string",
                "description": "The primary AI model Aider should use for generating code, leave blank unless model is specified in the request",
            },
            "architect_mode": {
                "type": "boolean",
                "description": "Enable two-phase code generation with an architect model planning first, then an editor model implementing",
                "default": False,
            },
            "editor_model": {
                "type": "string",
                "description": "The secondary AI model to use for code implementation when architect_mode is enabled",
            },
            "auto_accept_architect": {
                "type": "boolean",
                "description": "Automatically accept architect suggestions without confirmation",
                "default": True,
            },
            "include_raw_diff": {
                "type": "boolean",
                "description": "Whether to include raw diff in the response (not recommended)",
                "default": False,
            },
        },
        "required": ["ai_coding_prompt", "relative_editable_files"],
    },
)

LIST_MODELS_TOOL = Tool(
    name="list_models",
    description="List available models that match the provided substring",
    inputSchema={
        "type": "object",
        "properties": {
            "substring": {
                "type": "string",
                "description": "Substring to match against available models",
            }
        },
    },
)


def is_git_repository(directory: Union[str, Path]) -> Tuple[bool, Optional[str]]:
    """
    Check if the specified directory is a git repository.

    Args:
        directory (Union[str, Path]): The directory path to check.

    Returns:
        Tuple[bool, Optional[str]]: A tuple containing a boolean indicating if it's a git repo,
                                    and an error message string if it's not or an error occurred.
    """
    try:
        git_executable = shutil.which("git")
        if not git_executable:
            return False, "Git executable not found"

        # Make sure the directory exists and convert to Path if it's a string
        directory_path = Path(directory) if isinstance(directory, str) else directory
        if not directory_path.is_dir():
            return False, f"Directory does not exist: {directory_path}"

        # Validate directory is a legitimate path before passing to subprocess
        abs_path = directory_path.resolve()

        # Using a list of arguments instead of shell=True prevents command injection
        result = subprocess.run(  # noqa: S603
            [git_executable, "-C", str(abs_path), "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode == 0 and result.stdout.strip() == "true":
            return True, None
        else:
            return False, result.stderr.strip() or "Directory is not a git repository"

    except subprocess.SubprocessError as e:
        return False, f"Error checking git repository: {str(e)}"
    except Exception as e:
        return False, f"Unexpected error checking git repository: {str(e)}"


async def handle_request(
    request: Dict[str, Any],
    current_working_dir: str,
    editor_model: str,
) -> Dict[str, Any]:
    """
    Handle incoming MCP requests according to the MCP protocol (for stdio server).

    Args:
        request (Dict[str, Any]): The request JSON.
        current_working_dir (str): The current working directory. Must be a valid git repository.
        editor_model (str): The editor model to use.

    Returns:
        Dict[str, Any]: The response JSON.
    """
    try:
        # Validate current_working_dir is provided and is a git repository
        if not current_working_dir:
            error_msg = "Error: current_working_dir is required. Please provide a valid git repository path."
            logger.error(error_msg)
            return {"success": False, "error": error_msg}

        # MCP protocol requires 'name' and 'parameters' fields
        if "name" not in request:
            logger.error("Error: Received request missing 'name' field.")
            return {"success": False, "error": "Missing 'name' field in request"}

        request_type = request.get("name")
        params = request.get("parameters", {})

        logger.info(f"Received request: Type='{request_type}', CWD='{current_working_dir}'")

        # Validate that the current_working_dir is a git repository before changing to it
        is_git_repo, error_message = is_git_repository(current_working_dir)
        if not is_git_repo:
            error_msg = (
                f"Error: The specified directory '{current_working_dir}' is not a valid git repository: {error_message}"
            )
            logger.error(error_msg)
            return {"success": False, "error": error_msg}

        # Set working directory for the request
        try:
            os.chdir(current_working_dir)
            logger.info(f"Changed working directory to: {current_working_dir}")
        except Exception as e:
            error_msg = f"Failed to change working directory to {current_working_dir}: {e}"
            logger.critical(error_msg)
            return {"success": False, "error": error_msg}

        # Route to the appropriate handler based on request type
        if request_type == "aider_ai_code":
            result = await process_aider_ai_code_request(
                request_id="stdio",  # Using fixed ID for stdio mode
                transport_id="stdio",  # Using fixed ID for stdio mode
                params=params,  # Using the correct parameter name 'params' instead of 'parameters'
                security_context=ANONYMOUS_SECURITY_CONTEXT,
                editor_model=editor_model,
                current_working_dir=current_working_dir,
            )
            return result
        elif request_type == "list_models":
            result = await process_list_models_request(
                request_id="stdio",  # Using fixed ID for stdio mode
                transport_id="stdio",  # Using fixed ID for stdio mode
                params=params,  # Using the correct parameter name 'params' instead of 'parameters'
                security_context=ANONYMOUS_SECURITY_CONTEXT,
            )
            return result
        else:
            # Unknown request type
            logger.warning(f"Warning: Unknown request type received: {request_type}")
            return {"success": False, "error": f"Unknown request type: {request_type}"}

    except Exception as e:
        # Handle any errors
        logger.exception(f"Critical Error: Unhandled exception during request processing: {str(e)}")
        return {"success": False, "error": f"Internal server error: {str(e)}"}


async def serve(
    editor_model: str = DEFAULT_EDITOR_MODEL,
    current_working_dir: Optional[str] = None,  # Make CWD optional here, but validate
) -> None:
    """
    Start the MCP server over stdio.

    Args:
        editor_model (str, optional): The editor model to use. Defaults to DEFAULT_EDITOR_MODEL.
        current_working_dir (str, optional): The current working directory. Must be provided and be a valid git repository.

    Raises:
        ValueError: If current_working_dir is not provided or is not a git repository.
    """
    logger.info("Starting Aider MCP Server (stdio mode)")
    logger.info(f"Editor Model: {editor_model}")

    # ... (unchanged code) ...

    # Create the MCP server instance for stdio
    server: Server[List[TextContent]] = Server("aider-mcp-server")

    @server.list_tools()
    async def list_tools() -> List[Tool]:
        return [AIDER_AI_CODE_TOOL, LIST_MODELS_TOOL]

    @server.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        logger.info(f"Received Tool Call (stdio): Name='{name}'")
        # logger.debug(f"Arguments: {arguments}") # Log arguments only at debug level

        try:
            # Use handle_request which calls the appropriate handler
            if current_working_dir is None:
                raise ValueError("current_working_dir must be provided")
            result_dict = await handle_request(
                {"name": name, "parameters": arguments},
                current_working_dir,  # Use the validated current_working_dir
                editor_model,
            )
            # Ensure result is always a dict
            result_dict = (
                result_dict
                if isinstance(result_dict, dict)
                else {
                    "success": False,
                    "error": f"Internal server error: handle_request did not return a dict for tool '{name}'. Got: {type(result_dict)}",
                }
            )

            return [TextContent(type="text", text=json.dumps(result_dict))]
        except Exception as e:
            logger.exception(f"Error: Exception during stdio tool call '{name}': {e}")
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "success": False,
                            "error": f"Error processing tool {name}: {str(e)}",
                        }
                    ),
                )
            ]

    # Initialize and run the server
    try:
        options = server.create_initialization_options()
        logger.info("Initializing stdio server connection...")
        async with stdio_server() as (read_stream, write_stream):
            logger.info("Server running. Waiting for requests...")
            await server.run(read_stream, write_stream, options, raise_exceptions=True)
    except Exception as e:
        logger.exception(f"Critical Error: Server stopped due to unhandled exception: {e}")
        # Exit with error code if server crashes
        sys.exit(1)  # Ensure exit on critical error
    finally:
        logger.info("Aider MCP Server (stdio mode) shutting down.")
