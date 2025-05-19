import json
import os
import os.path
from typing import List, Dict, Any, Union

from aider.coders import Coder
from aider.io import InputOutput
from aider.models import Model

from aider_mcp_server.atoms.logging import get_logger
from aider_mcp_server.atoms.tools.aider_common import _format_response

# Configure logging for this module
logger = get_logger(__name__)


def ask_aider(
    ai_coding_prompt: str,
    relative_readonly_files: List[str] = [],
    model: str = "gemini/gemini-2.5-pro-exp-03-25",
    working_dir: str = None,
) -> str:
    """
    Run Aider in ask mode to get information based on the provided prompt and reference files.

    Args:
        ai_coding_prompt (str): The prompt for the AI to execute.
        relative_readonly_files (List[str], optional): List of files that can be read but not edited. Defaults to [].
        model (str, optional): The model to use. Defaults to "gemini/gemini-2.5-pro-exp-03-25".
        working_dir (str, required): The working directory where git repository is located and files are stored.

    Returns:
        str: JSON formatted response with Aider's answer
    """
    logger.info("Starting ask_aider process in ask mode.")
    logger.info(f"Prompt: '{ai_coding_prompt}'")

    # Working directory must be provided
    if not working_dir:
        error_msg = "Error: working_dir is required for ask_aider"
        logger.error(error_msg)
        return json.dumps({"success": False, "response": error_msg})

    logger.info(f"Working directory: {working_dir}")
    logger.info(f"Readonly files: {relative_readonly_files}")
    logger.info(f"Model: {model}")

    try:
        # Configure the model
        logger.info("Configuring AI model...")
        ai_model = Model(model)
        logger.info(f"Configured model: {model}")
        logger.info("AI model configured.")

        # Create the coder instance in ask mode
        logger.info("Creating Aider coder instance for ask mode...")
        # Use working directory for chat history file if provided
        history_dir = working_dir
        abs_readonly_files = [
            os.path.join(working_dir, file) for file in relative_readonly_files
        ]
        chat_history_file = os.path.join(history_dir, ".aider.chat.history.md")
        logger.info(f"Using chat history file: {chat_history_file}")

        # Create standard InputOutput instance
        io = InputOutput(
            yes=True,
            chat_history_file=chat_history_file,
        )
        
        # Use the explicit "ask" edit_format to create an AskCoder instance
        coder = Coder.create(
            main_model=ai_model,
            io=io,
            edit_format="ask",  # This specifies the AskCoder class
            fnames=[],  # No editable files in ask mode
            read_only_fnames=abs_readonly_files,
            auto_commits=False,
            suggest_shell_commands=False,
            detect_urls=False,
            use_git=True
        )
        logger.info("Aider coder instance created successfully in ask mode.")

        # Run the ask session
        logger.info("Starting Aider ask session...")
        result = coder.run(ai_coding_prompt)
        logger.info("Aider ask session finished.")

        # In ask mode, we just return the response directly
        response = {
            "success": True,
            "response": result
        }

    except Exception as e:
        logger.exception(f"Critical Error in ask_aider: {str(e)}")
        response = {
            "success": False,
            "response": f"Unhandled Error during Aider execution: {str(e)}",
        }

    formatted_response = json.dumps(response, indent=4)
    logger.info(f"ask_aider process completed. Success: {response.get('success')}")
    logger.info(f"Formatted response length: {len(formatted_response)}")
    return formatted_response