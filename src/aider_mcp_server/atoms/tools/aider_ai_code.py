import json
import os
import os.path
import pathlib
import subprocess
import time
from typing import Dict, List, Union

from aider.coders import Coder
from aider.io import InputOutput
from aider.models import Model

from aider_mcp_server.atoms.logging import get_logger
from aider_mcp_server.atoms.utils.fallback_config import (
    detect_rate_limit_error,
    get_fallback_model,
)

# Try to import dotenv for environment variable loading
try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False

# Configure logging for this module
logger = get_logger(__name__)

# Load fallback configuration
try:
    # Get the directory where this module is located
    current_dir = pathlib.Path(__file__).parent.absolute()
    # Navigate up from tools/aider_ai_code.py to the repository root (4 levels up)
    repo_root = current_dir.parents[3]  # atoms/tools -> atoms -> aider_mcp_server -> src -> repo_root
    mcp_json_path = os.path.join(repo_root, ".rate-limit-fallback.json")
    
    # Try alternative locations if the default location doesn't work
    if not os.path.exists(mcp_json_path):
        # Try the parent directory of the repo root
        mcp_json_path = os.path.join(repo_root.parent, ".rate-limit-fallback.json")
    
    if not os.path.exists(mcp_json_path):
        # Try the current working directory
        mcp_json_path = os.path.join(os.getcwd(), ".rate-limit-fallback.json")
    
    logger.info(f"Loading fallback config from: {mcp_json_path}")
    with open(mcp_json_path, "r") as f:
        fallback_config = json.load(f)["fallback_config"]
    logger.info("Successfully loaded fallback configuration")
except Exception as e:
    logger.warning(f"Error loading fallback config: {e}")
    # Use default fallback configuration
    fallback_config = {
        "openai": {
            "rate_limit_errors": ["rate_limit_exceeded", "insufficient_quota"],
            "backoff_factor": 2,
            "initial_delay": 1,
            "max_retries": 5,
            "fallback_models": ["gpt-3.5-turbo", "gpt-3.5-turbo-16k"]
        },
        "anthropic": {
            "rate_limit_errors": ["rate_limit_exceeded"],
            "backoff_factor": 2,
            "initial_delay": 1,
            "max_retries": 5,
            "fallback_models": ["claude-2", "claude-instant-1"]
        },
        "gemini": {
            "rate_limit_errors": ["quota_exceeded", "rate_limit_exceeded"],
            "backoff_factor": 2,
            "initial_delay": 1,
            "max_retries": 5,
            "fallback_models": ["gemini-pro", "gemini-1.0-pro"]
        }
    }

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

def _normalize_file_paths(relative_editable_files: List[str], working_dir: str = None) -> List[str]:
    """
    Normalize file paths to be relative to working_dir for git commands.

    Args:
        relative_editable_files: List of files to normalize
        working_dir: The working directory to make paths relative to

    Returns:
        List of normalized file paths
    """
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

    return normalized_paths

def _get_git_diff(normalized_paths: List[str], working_dir: str = None) -> str:
    """
    Execute git diff command and return the output.

    Args:
        normalized_paths: List of normalized file paths
        working_dir: The directory where git commands should be executed

    Returns:
        Git diff output as string or raises exception if command fails
    """
    git_cmd = ["git"]
    if working_dir:
        git_cmd.extend(["-C", working_dir])

    git_cmd.append("diff")
    git_cmd.append("--")
    git_cmd.extend(normalized_paths)  # Add all file paths as separate arguments

    logger.info(f"Running git command: {git_cmd}")
    result = subprocess.run(  # noqa: S603 - Using list of args instead of shell=True is safe
        git_cmd,
        capture_output=True,
        text=True,
        check=False  # Don't raise exception on non-zero exit
    )

    if result.returncode == 0:
        logger.info("Successfully obtained git diff.")
        return result.stdout
    else:
        raise subprocess.CalledProcessError(
            result.returncode, git_cmd, result.stdout, result.stderr
        )

def _read_file_contents(relative_editable_files: List[str], working_dir: str = None) -> str:
    """
    Read contents of files as a fallback when git diff fails.

    Args:
        relative_editable_files: List of files to read
        working_dir: The working directory where files are located

    Returns:
        String containing file contents with headers
    """
    content = "File contents after editing (git not used):\n\n"

    for file_path in relative_editable_files:
        # Get the correct full path without duplicating the working dir
        full_path = file_path
        if not os.path.isabs(file_path) and working_dir:
            full_path = os.path.join(working_dir, file_path)

        logger.info(f"Reading content from: {full_path}")

        if os.path.exists(full_path):
            try:
                with open(full_path, "r") as f:
                    file_content = f.read()
                    content += f"--- {file_path} ---\n{file_content}\n\n"
                    logger.info(f"Read content for {file_path}")
            except Exception as read_e:
                logger.error(
                    f"Failed reading file {full_path} for content fallback: {read_e}"
                )
                content += f"--- {file_path} --- (Error reading file)\n\n"
        else:
            logger.warning(f"File {full_path} not found during content fallback.")
            content += f"--- {file_path} --- (File not found)\n\n"

    return content

def get_changes_diff_or_content(
    relative_editable_files: List[str], working_dir: str = None
) -> str:
    """
    Get the git diff for the specified files, or their content if git fails.

    Args:
        relative_editable_files: List of files to check for changes
        working_dir: The working directory where the git repo is located

    Returns:
        String containing git diff or file contents
    """
    # Log current directory for debugging
    try:
        current_dir = os.getcwd()
        logger.info(f"Current directory during diff: {current_dir}")
    except FileNotFoundError:
        logger.warning("Current working directory is invalid or has been deleted.")

    if working_dir:
        logger.info(f"Using working directory: {working_dir}")

    try:
        # Step 1: Normalize file paths for git command
        normalized_paths = _normalize_file_paths(relative_editable_files, working_dir)

        # Step 2: Try to get git diff
        return _get_git_diff(normalized_paths, working_dir)

    except subprocess.CalledProcessError as e:
        logger.warning(
            f"Git diff command failed with exit code {e.returncode}. Error: {e.stderr.strip()}"
        )
        logger.warning("Falling back to reading file contents.")

        # Step 3: Fall back to reading file contents
        return _read_file_contents(relative_editable_files, working_dir)

    except Exception as e:
        logger.error(f"Unexpected error getting git diff: {str(e)}")
        return f"Error getting git diff: {str(e)}\n\n"

# Keep original function name but call the refactored implementation
def _get_changes_diff_or_content(
    relative_editable_files: List[str], working_dir: str = None
) -> str:
    """
    Get the git diff for the specified files, or their content if git fails.

    Args:
        relative_editable_files: List of files to check for changes
        working_dir: The working directory where the git repo is located
    """
    return get_changes_diff_or_content(relative_editable_files, working_dir)

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

def _configure_model(model: str) -> Model:
    """
    Configure and initialize the AI model.

    Args:
        model: Model identifier string

    Returns:
        Configured Model instance
    """
    logger.info(f"Attempting to initialize model: {model}")

    # Check environment variables
    api_key_env = None
    if "openai" in model.lower():
        api_key_env = os.environ.get("OPENAI_API_KEY")
        logger.info(f"OpenAI API key present: {bool(api_key_env)}")
    elif "gemini" in model.lower() or "google" in model.lower():
        api_key_env = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        logger.info(f"Google/Gemini API key present: {bool(api_key_env)}")
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
    return ai_model

def _setup_aider_coder(
    ai_model: Model,
    working_dir: str,
    abs_editable_files: List[str],
    abs_readonly_files: List[str]
) -> Coder:
    """
    Create and configure the Aider coder instance.

    Args:
        ai_model: The configured Model instance
        working_dir: Directory where files are located
        abs_editable_files: List of files that can be edited (absolute paths)
        abs_readonly_files: List of files that are read-only (absolute paths)

    Returns:
        Configured Coder instance
    """
    logger.info("Creating Aider coder instance...")
    history_dir = working_dir
    chat_history_file = os.path.join(history_dir, ".aider.chat.history.md")
    logger.info(f"Using chat history file: {chat_history_file}")

    coder = Coder.create(
        main_model=ai_model,
        io=InputOutput(yes=True, chat_history_file=chat_history_file),
        fnames=abs_editable_files,
        read_only_fnames=abs_readonly_files,
        auto_commits=False,
        suggest_shell_commands=False,
        detect_urls=False,
        use_git=True,
    )
    logger.info("Aider coder instance created successfully.")
    return coder

def _run_aider_session(
    coder: Coder,
    ai_coding_prompt: str,
    relative_editable_files: List[str],
    working_dir: str
) -> ResponseDict:
    """
    Run the Aider coding session and process the results.

    Args:
        coder: The configured Coder instance
        ai_coding_prompt: The prompt to send to the AI
        relative_editable_files: List of files that can be edited (relative paths)
        working_dir: Directory where files are located

    Returns:
        Dictionary with success status and diff output
    """
    logger.info("Starting Aider coding session...")
    result = coder.run(ai_coding_prompt)
    logger.info(f"Aider coding session result: {result}")

    # Process the results after the coder has run
    logger.info("Processing coder results...")
    response = _process_coder_results(relative_editable_files, working_dir)
    logger.info("Coder results processed.")
    return response

def code_with_aider(
    ai_coding_prompt: str,
    relative_editable_files: List[str],
    relative_readonly_files: List[str] = None,
    model: str = "gemini/gemini-2.5-flash-preview-04-17",
    working_dir: str = None,
) -> str:
    """
    Run Aider to perform AI coding tasks based on the provided prompt and files.

    Args:
        ai_coding_prompt (str): The prompt for the AI to execute.
        relative_editable_files (List[str]): List of files that can be edited.
        relative_readonly_files (List[str], optional): List of files that can be read but not edited. Defaults to [].
        model (str, optional): The model to use. Defaults to "gemini/gemini-2.5-flash-preview-04-17".
        working_dir: The working directory where git repository is located and files are stored.

    Returns:
        str: JSON string containing 'success', 'diff', and additional rate limit information.
    """
    if relative_readonly_files is None:
        relative_readonly_files = []
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
    logger.info(f"Initial model: {model}")

    response = {
        "success": False,
        "diff": "",
        "rate_limit_info": {
            "encountered": False,
            "retries": 0,
            "fallback_model": None
        }
    }

    # Convert relative paths to absolute paths
    abs_editable_files = [os.path.join(working_dir, f) if not os.path.isabs(f) else f for f in relative_editable_files]
    abs_readonly_files = [os.path.join(working_dir, f) if not os.path.isabs(f) else f for f in relative_readonly_files]

    # Determine the provider and get retry configuration
    provider = "openai" if "openai" in model.lower() else "anthropic" if "anthropic" in model.lower() else "gemini"
    max_retries = fallback_config[provider]["max_retries"]
    initial_delay = fallback_config[provider]["initial_delay"]
    backoff_factor = fallback_config[provider]["backoff_factor"]

    try:
        for attempt in range(max_retries + 1):
            try:
                # Configure the model and create the coder
                ai_model = _configure_model(model)
                coder = _setup_aider_coder(ai_model, working_dir, abs_editable_files, abs_readonly_files)
                
                # Run the session and get results
                response = _run_aider_session(coder, ai_coding_prompt, relative_editable_files, working_dir)
                
                # If successful, exit the retry loop
                break

            except Exception as e:
                logger.warning(f"Error during Aider execution (Attempt {attempt + 1}): {str(e)}")
                
                if detect_rate_limit_error(e, provider):
                    logger.info(f"Rate limit detected for {provider}. Attempting fallback...")
                    response["rate_limit_info"]["encountered"] = True
                    response["rate_limit_info"]["retries"] += 1

                    if attempt < max_retries:
                        delay = initial_delay * (backoff_factor ** attempt)
                        logger.info(f"Retrying after {delay} seconds...")
                        time.sleep(delay)
                        model = get_fallback_model(model, provider)
                        response["rate_limit_info"]["fallback_model"] = model
                        logger.info(f"Falling back to model: {model}")
                    else:
                        logger.error("Max retries reached. Unable to complete the request.")
                        raise
                else:
                    # If it's not a rate limit error, re-raise the exception
                    raise

    except Exception as e:
        logger.exception(f"Critical Error in code_with_aider: {str(e)}")
        response.update({
            "success": False,
            "diff": f"Unhandled Error during Aider execution: {str(e)}",
        })

    formatted_response = json.dumps(response, indent=4)
    logger.info(f"code_with_aider process completed. Success: {response['success']}")
    return formatted_response
