import json
import os
import os.path
from typing import List

from aider.coders import Coder
from aider.io import InputOutput
from aider.models import Model

from aider_mcp_server.atoms.logging import get_logger
from aider_mcp_server.atoms.tools.aider_common import _process_coder_results, _format_response

# Configure logging for this module
logger = get_logger(__name__)


def code_with_aider(
    ai_coding_prompt: str,
    relative_editable_files: List[str],
    relative_readonly_files: List[str] = [],
    model: str = "gemini/gemini-2.5-pro-exp-03-25",
    working_dir: str = None,
) -> str:
    """
    Run Aider to perform AI coding tasks based on the provided prompt and files.

    Args:
        ai_coding_prompt (str): The prompt for the AI to execute.
        relative_editable_files (List[str]): List of files that can be edited.
        relative_readonly_files (List[str], optional): List of files that can be read but not edited. Defaults to [].
        model (str, optional): The model to use. Defaults to "gemini/gemini-2.5-pro-exp-03-25".
        working_dir (str, required): The working directory where git repository is located and files are stored.

    Returns:
        Dict[str, Any]: {'success': True/False, 'diff': str with git diff output}
    """
    logger.info("Starting code_with_aider process.")
    logger.info(f"Prompt: '{ai_coding_prompt}'")

    # Working directory must be provided
    if not working_dir:
        error_msg = "Error: working_dir is required for code_with_aider"
        logger.error(error_msg)
        return json.dumps({"success": False, "diff": error_msg})

    logger.info(f"Working directory: {working_dir}")
    logger.info(f"Editable files: {relative_editable_files}")
    logger.info(f"Readonly files: {relative_readonly_files}")
    logger.info(f"Model: {model}")

    try:
        # Configure the model
        logger.info("Configuring AI model...")  # Point 1: Before init
        ai_model = Model(model)
        logger.info(f"Configured model: {model}")
        logger.info("AI model configured.")  # Point 2: After init

        # Create the coder instance
        logger.info("Creating Aider coder instance...")
        # Use working directory for chat history file if provided
        history_dir = working_dir
        abs_editable_files = [
            os.path.join(working_dir, file) for file in relative_editable_files
        ]
        abs_readonly_files = [
            os.path.join(working_dir, file) for file in relative_readonly_files
        ]
        chat_history_file = os.path.join(history_dir, ".aider.chat.history.md")
        logger.info(f"Using chat history file: {chat_history_file}")

        coder = Coder.create(
            main_model=ai_model,
            io=InputOutput(
                yes=True,
                chat_history_file=chat_history_file,
            ),
            fnames=abs_editable_files,
            read_only_fnames=abs_readonly_files,
            auto_commits=False,  # We'll handle commits separately
            suggest_shell_commands=False,
            detect_urls=False,
            use_git=True,  # Always use git
        )
        logger.info("Aider coder instance created successfully.")

        # Run the coding session
        logger.info("Starting Aider coding session...")  # Point 3: Before run
        result = coder.run(ai_coding_prompt)
        logger.info(f"Aider coding session result: {result}")
        logger.info("Aider coding session finished.")  # Point 4: After run

        # Process the results after the coder has run
        logger.info("Processing coder results...")  # Point 5: Processing results
        try:
            response = _process_coder_results(relative_editable_files, working_dir)
            logger.info("Coder results processed.")
        except Exception as e:
            logger.exception(
                f"Error processing coder results: {str(e)}"
            )  # Point 6: Error
            response = {
                "success": False,
                "diff": f"Error processing files after execution: {str(e)}",
            }

    except Exception as e:
        logger.exception(
            f"Critical Error in code_with_aider: {str(e)}"
        )  # Point 6: Error
        response = {
            "success": False,
            "diff": f"Unhandled Error during Aider execution: {str(e)}",
        }

    formatted_response = _format_response(response)
    logger.info(
        f"code_with_aider process completed. Success: {response.get('success')}"
    )
    logger.info(
        f"Formatted response: {formatted_response}"
    )  # Log complete response for debugging
    return formatted_response