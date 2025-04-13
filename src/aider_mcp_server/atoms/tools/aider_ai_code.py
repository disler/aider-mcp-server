import json
from typing import List, Optional, Dict, Any, Union
import os
import os.path
import subprocess
from aider.models import Model
from aider.coders import Coder
from aider.io import InputOutput
from aider_mcp_server.atoms.logging import get_logger

# Try to import dotenv for environment variable loading
try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False

# Configure logging for this module
logger = get_logger(__name__)

def load_env_files(working_dir=None):
    """Load environment variables from .env files in relevant directories."""
    if not HAS_DOTENV:
        logger.warning("python-dotenv not installed. Cannot load .env files.")
        return

    # List of potential locations for .env files in order of precedence
    env_locations = []

    # Add working_dir if provided
    if working_dir:
        env_locations.append(working_dir)

    # Add current directory
    env_locations.append(os.getcwd())

    # Add parent directory of current directory
    env_locations.append(os.path.dirname(os.getcwd()))

    # Add user's home directory
    env_locations.append(os.path.expanduser("~"))

    # Load .env from each location if it exists
    for location in env_locations:
        env_path = os.path.join(location, ".env")
        if os.path.isfile(env_path):
            logger.info(f"Loading environment variables from {env_path}")
            load_dotenv(env_path)
            # Don't break - load all .env files to allow for overrides

def check_api_keys(working_dir=None):
    """Check if necessary API keys are set in the environment and log their status."""
    # First load any .env files
    load_env_files(working_dir)

    keys_to_check = {
        "OPENAI_API_KEY": "OpenAI",
        "GOOGLE_API_KEY": "Google/Gemini",
        "GEMINI_API_KEY": "Google/Gemini (alternative)",
        "ANTHROPIC_API_KEY": "Anthropic/Claude",
        "AZURE_OPENAI_API_KEY": "Azure OpenAI",
        "VERTEX_AI_API_KEY": "Vertex AI"
    }
    
    logger.info("Checking API keys in environment...")
    for key, provider in keys_to_check.items():
        if os.environ.get(key):
            logger.info(f"✓ {provider} API key found ({key})")
        else:
            logger.warning(f"✗ {provider} API key missing ({key})")
    
    # Special handling for Gemini/Google - check if we need to copy between variables
    if os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        logger.info("Setting GOOGLE_API_KEY from GEMINI_API_KEY for compatibility")
        os.environ["GOOGLE_API_KEY"] = os.environ.get("GEMINI_API_KEY")

# Type alias for response dictionary
ResponseDict = Dict[str, Union[bool, str]]


def _get_changes_diff_or_content(
    relative_editable_files: List[str], working_dir: str = None
) -> str:
    """
    Get the git diff for the specified files, or their content if git fails.

    Args:
        relative_editable_files: List of files to check for changes
        working_dir: The working directory where the git repo is located
    """
    diff = ""
    # Log current directory for debugging - safely handle directory issues
    try:
        current_dir = os.getcwd()
        logger.info(f"Current directory during diff: {current_dir}")
    except FileNotFoundError:
        logger.warning("Current working directory is invalid or has been deleted.")
        current_dir = "Unknown (deleted or inaccessible)"

    if working_dir:
        logger.info(f"Using working directory: {working_dir}")

    # Always attempt to use git
    # Normalize paths - for git command we need paths relative to working_dir
    normalized_paths = []
    for file_path in relative_editable_files:
        if working_dir and os.path.isabs(file_path):
            # If it's an absolute path and working_dir is provided, make it relative to working_dir
            try:
                rel_path = os.path.relpath(file_path, working_dir)
                normalized_paths.append(rel_path)
                logger.info(f"Normalized path: {file_path} -> {rel_path}")
            except ValueError:
                # If paths are on different drives (Windows), just use the basename
                normalized_paths.append(os.path.basename(file_path))
                logger.info(f"Using basename: {file_path} -> {os.path.basename(file_path)}")
        else:
            normalized_paths.append(file_path)
            logger.info(f"Using as-is: {file_path}")

    files_arg = " ".join(normalized_paths)
    logger.info(f"Attempting to get git diff for: {files_arg}")

    try:
        # Use git -C to specify the repository directory
        if working_dir:
            diff_cmd = f"git -C {working_dir} diff -- {files_arg}"
        else:
            diff_cmd = f"git diff -- {files_arg}"

        logger.info(f"Running git command: {diff_cmd}")
        diff = subprocess.check_output(
            diff_cmd, shell=True, text=True, stderr=subprocess.PIPE
        )
        logger.info("Successfully obtained git diff.")
    except subprocess.CalledProcessError as e:
        logger.warning(
            f"Git diff command failed with exit code {e.returncode}. Error: {e.stderr.strip()}"
        )
        logger.warning("Falling back to reading file contents.")
        diff = "File contents after editing (git not used):\n\n"
        for file_path in relative_editable_files:
            # Get the correct full path without duplicating the working dir
            full_path = file_path
            if not os.path.isabs(file_path) and working_dir:
                full_path = os.path.join(working_dir, file_path)

            logger.info(f"Reading content from: {full_path}")

            if os.path.exists(full_path):
                try:
                    with open(full_path, "r") as f:
                        content = f.read()
                        diff += f"--- {file_path} ---\n{content}\n\n"
                        logger.info(f"Read content for {file_path}")
                except Exception as read_e:
                    logger.error(
                        f"Failed reading file {full_path} for content fallback: {read_e}"
                    )
                    diff += f"--- {file_path} --- (Error reading file)\n\n"
            else:
                logger.warning(f"File {full_path} not found during content fallback.")
                diff += f"--- {file_path} --- (File not found)\n\n"
    except Exception as e:
        logger.error(f"Unexpected error getting git diff: {str(e)}")
        diff = f"Error getting git diff: {str(e)}\n\n"  # Provide error in diff string as fallback
    return diff


def _check_for_meaningful_changes(
    relative_editable_files: List[str], working_dir: str = None
) -> bool:
    """
    Check if the edited files contain meaningful content.

    Args:
        relative_editable_files: List of files to check
        working_dir: The working directory where files are located
    """
    for file_path in relative_editable_files:
        # Get the correct full path without duplicating the working dir
        full_path = file_path
        if not os.path.isabs(file_path) and working_dir:
            full_path = os.path.join(working_dir, file_path)

        logger.info(f"Checking for meaningful content in: {full_path}")

        if os.path.exists(full_path):
            try:
                with open(full_path, "r") as f:
                    content = f.read()
                    # Check if the file has more than just whitespace or a single comment line,
                    # or contains common code keywords. This is a heuristic.
                    stripped_content = content.strip()
                    # Improve detection - check for function/class definitions more thoroughly
                    if stripped_content and (
                        len(stripped_content.split("\n")) > 1
                        or any(
                            kw in content
                            for kw in [
                                "def ",
                                "class ",
                                "import ",
                                "from ",
                                "async def ",
                                "return ",
                            ]
                        )
                    ):
                        logger.info(f"Meaningful content found in: {file_path}")
                        return True
                    logger.info(f"No meaningful content found in {file_path}, content: '{stripped_content}'")
            except Exception as e:
                logger.error(
                    f"Failed reading file {full_path} during meaningful change check: {e}"
                )
                # If we can't read it, we can't confirm meaningful change from this file
                continue
        else:
            logger.info(
                f"File not found or empty, skipping meaningful check: {full_path}"
            )

    logger.info("No meaningful changes detected in any editable files.")
    return False


def _process_coder_results(
    relative_editable_files: List[str], working_dir: str = None
) -> ResponseDict:
    """
    Process the results after Aider has run, checking for meaningful changes
    and retrieving the diff or content.

    Args:
        relative_editable_files: List of files that were edited
        working_dir: The working directory where the git repo is located

    Returns:
        Dictionary with success status and diff output
    """
    diff_output = _get_changes_diff_or_content(relative_editable_files, working_dir)
    logger.info("Checking for meaningful changes in edited files...")
    has_meaningful_content = _check_for_meaningful_changes(
        relative_editable_files, working_dir
    )

    if has_meaningful_content:
        logger.info("Meaningful changes found. Processing successful.")
        return {"success": True, "diff": diff_output}
    else:
        logger.warning(
            "No meaningful changes detected. Processing marked as unsuccessful."
        )
        # Even if no meaningful content, provide the diff/content if available
        return {
            "success": False,
            "diff": diff_output
            or "No meaningful changes detected and no diff/content available.",
        }


def _format_response(response: ResponseDict) -> str:
    """
    Format the response dictionary as a JSON string.

    Args:
        response: Dictionary containing success status and diff output

    Returns:
        JSON string representation of the response
    """
    return json.dumps(response, indent=4)


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
    
    # Check API keys at the beginning - now passing working_dir
    check_api_keys(working_dir)

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
        logger.info(f"Attempting to initialize model: {model}")
        try:
            # Check environment variables
            api_key_env = None
            if "openai" in model.lower():
                api_key_env = os.environ.get("OPENAI_API_KEY")
                logger.info(f"OpenAI API key present: {bool(api_key_env)}")
            elif "gemini" in model.lower() or "google" in model.lower():
                # Check both possible environment variable names for Gemini
                api_key_env = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
                logger.info(f"Google/Gemini API key present: {bool(api_key_env)}")
                # If using GEMINI_API_KEY, set GOOGLE_API_KEY for compatibility with aider
                if not os.environ.get("GOOGLE_API_KEY") and os.environ.get("GEMINI_API_KEY"):
                    os.environ["GOOGLE_API_KEY"] = os.environ.get("GEMINI_API_KEY")
                    logger.info("Set GOOGLE_API_KEY from GEMINI_API_KEY for compatibility")
            elif "anthropic" in model.lower():
                api_key_env = os.environ.get("ANTHROPIC_API_KEY")
                logger.info(f"Anthropic API key present: {bool(api_key_env)}")
            
            if not api_key_env:
                logger.warning(f"No API key found for model type: {model}")
            
            ai_model = Model(model)
            logger.info(f"Successfully configured model: {model}")
        except Exception as model_error:
            logger.exception(f"Error initializing model {model}: {str(model_error)}")
            raise
        logger.info("AI model configured.")  # Point 2: After init

        # Create the coder instance
        logger.info("Creating Aider coder instance...")
        # Use working directory for chat history file if provided
        history_dir = working_dir
        # Handle both absolute and relative paths correctly for editable files
        abs_editable_files = []
        for file in relative_editable_files:
            if os.path.isabs(file):
                abs_editable_files.append(file)
            else:
                abs_editable_files.append(os.path.join(working_dir, file))

        # Same for readonly files
        abs_readonly_files = []
        for file in relative_readonly_files:
            if os.path.isabs(file):
                abs_readonly_files.append(file)
            else:
                abs_readonly_files.append(os.path.join(working_dir, file))

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
        try:
            result = coder.run(ai_coding_prompt)
            logger.info(f"Aider coding session result: {result}")
        except Exception as run_error:
            logger.exception(f"Error during Aider coding session: {str(run_error)}")
            # Check if it's an authentication error
            error_str = str(run_error).lower()
            if any(term in error_str for term in ["auth", "api key", "credential", "unauthorized", "permission"]):
                logger.critical("Authentication error detected. Please check your API keys.")
                return json.dumps({
                    "success": False,
                    "diff": f"Authentication error: {str(run_error)}. Please check your API keys and permissions."
                })
            raise
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
