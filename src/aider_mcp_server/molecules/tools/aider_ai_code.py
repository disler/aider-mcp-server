import asyncio
import json
import os
import os.path
import pathlib
import subprocess

# External imports - no stubs available
from typing import Any, Dict, List, Optional, TypedDict, Union

from aider.coders import Coder
from aider.io import InputOutput
from aider.models import Model

from aider_mcp_server.atoms.logging.logger import get_logger

# Internal imports
from aider_mcp_server.atoms.utils.diff_cache import DiffCache
from aider_mcp_server.atoms.utils.fallback_config import (
    detect_rate_limit_error,
    get_fallback_model,
)
from aider_mcp_server.molecules.tools.aider_compatibility import (
    filter_supported_params,
    get_aider_version,
    get_supported_coder_create_params,
    get_supported_coder_params,
)
from aider_mcp_server.molecules.tools.changes_summarizer import (
    get_file_status_summary,
    summarize_changes,
)


# Create a subclass of InputOutput that overrides tool_error to do nothing
class SilentInputOutput(InputOutput):  # type: ignore[misc]
    """A subclass of InputOutput that overrides tool_error to do nothing."""

    def tool_error(self, message: str = "", strip: bool = True) -> None:
        """Override to do nothing with error messages."""
        pass


class ResponseDict(TypedDict, total=False):
    """Type for Aider response dictionary."""

    success: bool
    changes_summary: Dict[str, Any]  # Holds the summarized changes
    file_status: Dict[str, Any]  # Holds file status information (retained for backward compatibility)
    rate_limit_info: Optional[
        Dict[str, Union[bool, int, str, None]]
    ]  # Optional - only included when rate limits are encountered
    is_cached_diff: bool  # Added for backward compatibility
    diff: Optional[str]  # Raw diff output, optional
    api_key_status: Optional[Dict[str, Any]]  # Information about API key status
    warnings: Optional[List[str]]  # List of warnings to display to the user


# Try to import dotenv for environment variable loading
try:
    from dotenv import load_dotenv

    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False


def _check_individual_api_keys(keys_to_check: Dict[str, str], result: Dict[str, Any]) -> None:
    """Helper to check individual API keys and update result."""
    logger.info("Checking API keys in environment...")
    for key, provider in keys_to_check.items():
        if os.environ.get(key):
            logger.info(f"✓ {provider} API key found ({key})")
            result["found"].append(key)
            result["any_keys_found"] = True
        else:
            logger.warning(f"✗ {provider} API key missing ({key})")
            result["missing"].append(key)


def _handle_gemini_api_key_alias(result: Dict[str, Any]) -> None:
    """Helper to handle GEMINI_API_KEY and GOOGLE_API_KEY aliasing."""
    if os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if gemini_key is not None:  # Explicit check for None
            logger.info("Setting GOOGLE_API_KEY from GEMINI_API_KEY for compatibility")
            os.environ["GOOGLE_API_KEY"] = gemini_key
            missing_list = result["missing"]
            if isinstance(missing_list, list) and "GOOGLE_API_KEY" in missing_list:
                missing_list.remove("GOOGLE_API_KEY")
                found_list = result["found"]
                if isinstance(found_list, list):
                    found_list.append("GOOGLE_API_KEY")


def _determine_available_providers(provider_keys: Dict[str, List[str]], result: Dict[str, Any]) -> None:
    """Helper to determine available providers based on found keys."""
    for provider, keys in provider_keys.items():
        found_list = result["found"]
        available_providers = result["available_providers"]
        missing_providers = result["missing_providers"]

        if isinstance(found_list, list) and any(key in found_list for key in keys):
            if isinstance(available_providers, list):
                available_providers.append(provider)
        elif isinstance(missing_providers, list):
            missing_providers.append(provider)


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

    logger.debug(f"Loading fallback config from: {mcp_json_path}")
    with open(mcp_json_path, "r") as f:
        fallback_config = json.load(f)["fallback_config"]
    logger.debug("Successfully loaded fallback configuration")
except Exception as e:
    logger.warning(f"Error loading fallback config: {e}")
    # Use default fallback configuration
    fallback_config = {
        "openai": {
            "rate_limit_errors": ["rate_limit_exceeded", "insufficient_quota"],
            "backoff_factor": 2,
            "initial_delay": 1,
            "max_retries": 5,
            "fallback_models": ["gpt-3.5-turbo", "gpt-3.5-turbo-16k"],
        },
        "anthropic": {
            "rate_limit_errors": ["rate_limit_exceeded"],
            "backoff_factor": 2,
            "initial_delay": 1,
            "max_retries": 5,
            "fallback_models": ["claude-2", "claude-instant-1"],
        },
        "gemini": {
            "rate_limit_errors": ["quota_exceeded", "rate_limit_exceeded"],
            "backoff_factor": 2,
            "initial_delay": 1,
            "max_retries": 5,
            "fallback_models": ["gemini-pro", "gemini-1.0-pro"],
        },
    }


# Initialize to None, will be set by init_diff_cache
diff_cache: Optional[DiffCache] = None


async def init_diff_cache() -> None:
    """Initializes the module-level DiffCache."""
    global diff_cache
    if diff_cache is not None:
        # Already initialized
        logger.warning("DiffCache already initialized.")
        return

    logger.info("Initializing DiffCache...")
    # Create the instance
    new_cache = DiffCache()
    # Await the start method
    await new_cache.start()
    # Assign to the module-level variable
    diff_cache = new_cache
    logger.info("DiffCache initialized.")


async def shutdown_diff_cache() -> None:
    """Shuts down the module-level DiffCache."""
    global diff_cache
    if diff_cache is None:
        logger.warning("DiffCache not initialized, nothing to shut down.")
        return

    logger.info("Shutting down DiffCache...")
    await diff_cache.shutdown()
    diff_cache = None  # Reset the global variable
    logger.info("DiffCache shut down.")


def load_env_files(working_dir: Optional[str] = None) -> None:
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


def check_api_keys(working_dir: Optional[str] = None) -> Dict[str, Any]:
    """Check if necessary API keys are set in the environment and return status.

    Args:
        working_dir: Optional working directory to search for .env files

    Returns:
        Dict with API key status info including:
        - missing: List of missing API key env vars
        - found: List of found API key env vars
        - available_providers: List of providers with valid keys
        - missing_providers: List of providers with missing keys
        - any_keys_found: Boolean indicating if any keys were found
    """
    # First load any .env files
    load_env_files(working_dir)

    keys_to_check = {
        "OPENAI_API_KEY": "OpenAI",
        "GOOGLE_API_KEY": "Google/Gemini",
        "GEMINI_API_KEY": "Google/Gemini (alternative)",
        "ANTHROPIC_API_KEY": "Anthropic/Claude",
        "AZURE_OPENAI_API_KEY": "Azure OpenAI",
        "VERTEX_AI_API_KEY": "Vertex AI",
    }

    provider_keys = {
        "gemini": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
        "openai": ["OPENAI_API_KEY", "AZURE_OPENAI_API_KEY"],
        "anthropic": ["ANTHROPIC_API_KEY"],
        "vertexai": ["VERTEX_AI_API_KEY"],
    }

    result: Dict[str, Any] = {
        "missing": [],
        "found": [],
        "available_providers": [],
        "missing_providers": [],
        "any_keys_found": False,
    }

    _check_individual_api_keys(keys_to_check, result)
    _handle_gemini_api_key_alias(result)
    _determine_available_providers(provider_keys, result)

    return result


def _normalize_file_paths(relative_editable_files: List[str], working_dir: Optional[str] = None) -> List[str]:
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


def _get_git_diff(normalized_paths: List[str], working_dir: Optional[str] = None) -> str:
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
        check=False,  # Don't raise exception on non-zero exit
    )

    if result.returncode == 0:
        logger.info("Successfully obtained git diff.")
        return result.stdout
    else:
        raise subprocess.CalledProcessError(result.returncode, git_cmd, result.stdout, result.stderr)


def _read_file_contents(relative_editable_files: List[str], working_dir: Optional[str] = None) -> str:
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
                logger.error(f"Failed reading file {full_path} for content fallback: {read_e}")
                content += f"--- {file_path} --- (Error reading file)\n\n"
        else:
            logger.warning(f"File {full_path} not found during content fallback.")
            content += f"--- {file_path} --- (File not found)\n\n"

    return content


def get_changes_diff_or_content(relative_editable_files: List[str], working_dir: Optional[str] = None) -> str:
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
        logger.warning(f"Git diff command failed with exit code {e.returncode}. Error: {e.stderr.strip()}")
        logger.warning("Falling back to reading file contents.")

        # Step 3: Fall back to reading file contents
        return _read_file_contents(relative_editable_files, working_dir)

    except Exception as e:
        logger.error(f"Unexpected error getting git diff: {str(e)}")
        return f"Error getting git diff: {str(e)}\n\n"


# Keep original function name but call the refactored implementation
def _get_changes_diff_or_content(relative_editable_files: List[str], working_dir: Optional[str] = None) -> str:
    """
    Get the git diff for the specified files, or their content if git fails.

    Args:
        relative_editable_files: List of files to check for changes
        working_dir: The working directory where the git repo is located
    """
    return get_changes_diff_or_content(relative_editable_files, working_dir)


def _normalize_model_name(model: str) -> str:
    """
    Normalize the model name to a consistent format.

    For example:
    - gemini-pro -> gemini/gemini-pro
    - gpt-4 -> openai/gpt-4

    Args:
        model: The model name to normalize

    Returns:
        Normalized model name
    """
    # If the model already has a provider prefix, return it as is
    if "/" in model:
        return model

    # Based on model name, add appropriate provider prefix
    if model.startswith("gemini-") or model.startswith("gemini:"):
        return f"gemini/{model.replace(':', '-')}"
    elif model.startswith("gpt-") or model.startswith("openai:"):
        return f"openai/{model.replace(':', '-')}"
    elif model.startswith("claude-") or model.startswith("anthropic:"):
        return f"anthropic/{model.replace(':', '-')}"
    else:
        # Default to gemini if provider can't be determined
        return f"gemini/{model}"


def _determine_provider(model: str) -> str:
    """
    Extract the provider from the model name.

    Args:
        model: The model name in normalized format (provider/model)

    Returns:
        Provider name
    """
    if "/" in model:
        provider, _ = model.split("/", 1)
        return provider

    # Fallback for unnormalized model names
    if model.startswith("gemini-"):
        return "gemini"
    elif model.startswith("gpt-"):
        return "openai"
    elif model.startswith("claude-"):
        return "anthropic"
    else:
        # Default to gemini if provider can't be determined
        return "gemini"


def _configure_model(model: str, editor_model: Optional[str] = None, architect_mode: bool = False) -> Model:
    """
    Configure the Aider model based on the model name.

    Args:
        model: The model name in normalized format (provider/model)
        editor_model: Optional secondary model for implementation when architect_mode is enabled
        architect_mode: Whether to use the two-phase architecture mode

    Returns:
        Aider Model instance
    """
    logger.info(f"Configuring model: {model}, architect_mode={architect_mode}")

    # For testing purposes (when we know the model will fail), use a simple model name
    if model == "non_existent_model_123456789":
        logger.info(f"Using deliberately non-existent model for testing: {model}")
        return Model(model)

    # Use the actual requested model instead of hardcoding
    aider_model_name = model
    logger.info(f"Using requested model: {aider_model_name}")

    # Configure model based on architect mode setting
    if architect_mode:
        if editor_model:
            logger.info(f"Using editor model: {editor_model}")
            return Model(aider_model_name, editor_model=editor_model)
        else:
            # Use the same model for both architect and editor roles
            logger.info(f"Using same model for architect and editor: {aider_model_name}")
            return Model(aider_model_name, editor_model=aider_model_name)
    else:
        # Standard (non-architect) configuration
        return Model(aider_model_name)


def _convert_to_absolute_paths(relative_paths: List[str], working_dir: Optional[str]) -> List[str]:
    """
    Convert relative file paths to absolute paths.

    Args:
        relative_paths: List of file paths (possibly relative)
        working_dir: Working directory to resolve relative paths against

    Returns:
        List of absolute file paths
    """
    if not working_dir:
        # If no working dir, assume paths are already absolute
        return relative_paths

    # Convert each path to absolute if it's not already
    absolute_paths = []
    for path in relative_paths:
        if os.path.isabs(path):
            absolute_paths.append(path)
        else:
            absolute_paths.append(os.path.abspath(os.path.join(working_dir, path)))

    return absolute_paths


def _setup_aider_coder(
    model: Model,
    working_dir: str,
    abs_editable_files: List[str],
    abs_readonly_files: List[str],
    architect_mode: bool = False,
    auto_accept_architect: bool = True,
) -> Coder:
    """
    Set up an Aider Coder instance with the given configuration.

    Args:
        model: The configured Aider model
        working_dir: Working directory for git operations
        abs_editable_files: List of absolute paths to files that can be edited
        abs_readonly_files: List of absolute paths to files that can be read but not edited
        architect_mode: Whether to use two-phase architecture mode
        auto_accept_architect: Whether to automatically accept architect suggestions

    Returns:
        Configured Aider Coder instance
    """
    logger.info("Setting up Aider coder...")

    # Log aider version for debugging
    aider_version = get_aider_version()
    logger.info(f"Using aider version: {aider_version}")

    # Set chat history file path in the working directory if possible
    chat_history_file = None
    if working_dir:
        try:
            chat_history_dir = os.path.join(working_dir, ".aider")
            os.makedirs(chat_history_dir, exist_ok=True)
            chat_history_file = os.path.join(chat_history_dir, "chat_history.md")
        except Exception as e:
            logger.warning(f"Could not create chat history directory: {e}")

    # Create no-op functions to replace output methods
    def noop_output(*args: Any, **kwargs: Any) -> None:
        pass

    # Create an IO instance for the Coder that won't require interactive prompting
    # Add verbose=False to suppress progress output
    io = SilentInputOutput(
        pretty=False,  # Disable fancy output
        yes=True,  # Always say yes to prompts
        fancy_input=False,  # Disable fancy input to avoid prompt_toolkit usage
        chat_history_file=chat_history_file,  # Set chat history file if available
    )

    io.yes_to_all = True  # Automatically say yes to all prompts
    io.dry_run = False  # Ensure we're not in dry-run mode

    # Create no-op functions for output methods to suppress output
    def noop(*args: Any, **kwargs: Any) -> None:
        pass

    # Redirect output to no-op functions
    io.output = noop
    io.tool_output = noop

    # Set quiet mode to True to suppress unnecessary output
    io.quiet = True
    # For the GitRepo, we need to import the class from aider (if available)
    try:
        from aider.repo import GitRepo  # No stubs available for aider.repo

        # Create a GitRepo instance
        try:
            # Check if a .git folder exists to determine if this is a git repo
            git_dir = os.path.join(working_dir, ".git")
            is_git_repo = os.path.isdir(git_dir)

            if is_git_repo:
                logger.info(f"Found git repository at {working_dir}")
                git_repo = GitRepo(
                    io=io,
                    fnames=abs_editable_files,
                    git_dname=working_dir,
                    models=model.commit_message_models(),
                )
                logger.info(f"Successfully initialized GitRepo with root: {git_repo.root}")
            else:
                logger.warning(f"No .git directory found at {working_dir}, will set repo=None")
                git_repo = None
        except Exception as e:
            logger.warning(f"Could not initialize GitRepo: {e}, will set repo=None")
            git_repo = None
    except ImportError:
        logger.warning("Could not import GitRepo from aider.repo, will set repo=None")
        git_repo = None

    # Parameters for Coder.create method (different from __init__)
    create_params = {
        "main_model": model,
        "io": io,
        "edit_format": "architect" if architect_mode else None,
    }

    # Parameters that go directly to Coder.__init__ via kwargs
    init_params = {
        "fnames": abs_editable_files,
        "read_only_fnames": abs_readonly_files,
        "repo": git_repo,
        "show_diffs": True,  # Show diffs to help debugging
        "auto_commits": False,
        "dirty_commits": False,
        "use_git": True if git_repo else False,
        "stream": False,
        "suggest_shell_commands": False,
        "detect_urls": False,
        "verbose": True,  # Enable verbose mode for more debugging info
        "auto_accept_architect": auto_accept_architect if architect_mode else True,
    }

    logger.info(f"Setting up Aider Coder with params: create_params={create_params}, init_params={init_params}")

    # Get supported parameters for create method
    supported_create_params = get_supported_coder_create_params()
    logger.info(f"Supported Coder.create parameters: {supported_create_params}")

    # Get supported parameters for init method
    supported_init_params = get_supported_coder_params()
    logger.info(f"Supported Coder.__init__ parameters: {supported_init_params}")

    # Filter parameters based on what's actually supported
    filtered_create = filter_supported_params(create_params, supported_create_params)
    filtered_init = filter_supported_params(init_params, supported_init_params)

    # Combine create params with init params as kwargs
    final_params = filtered_create.copy()
    final_params.update(filtered_init)

    logger.info(f"Creating Coder with parameters: {list(final_params.keys())}")

    # Create the Coder instance using parameters compatible with the installed version
    coder = Coder.create(**final_params)

    return coder


def _check_for_meaningful_changes(relative_editable_files: List[str], working_dir: Optional[str] = None) -> bool:
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
                    # Check if the file has any non-whitespace content
                    stripped_content = content.strip()

                    # Check if it's just a comment line
                    is_just_comment = stripped_content.startswith("#") and len(stripped_content.split("\n")) == 1

                    # Consider any non-empty file as meaningful, except for files with just comments
                    if stripped_content and not is_just_comment:
                        logger.info(f"Content found in file, considering meaningful: {file_path}")
                        return True
                    elif is_just_comment:
                        logger.info(f"File only contains comments, not considered meaningful: {file_path}")
                    else:
                        # If completely empty, continue to next file
                        logger.info(f"File is empty, no content found: {file_path}")
            except Exception as e:
                logger.error(f"Failed reading file {full_path} during meaningful change check: {e}")
                # If we can't read it, we can't confirm meaningful change from this file
                continue
        else:
            logger.info(f"File not found or empty, skipping meaningful check: {full_path}")

    logger.info("No meaningful content detected in any editable files.")
    return False


async def _handle_diff_cache_processing(
    cache_key: str,
    raw_diff_output: str,
    use_diff_cache: bool,
    clear_cached_for_unchanged: bool,
) -> tuple[str, bool]:
    """Handles diff cache logic and returns final diff content and cache status."""
    global diff_cache
    is_cached_diff = False
    final_diff_content = raw_diff_output or "No git-tracked changes detected."

    if use_diff_cache and diff_cache is not None:
        logger.info(f"Attempting to use diff cache for key: {cache_key}")
        try:
            changes_from_cache = await diff_cache.compare_and_cache(
                cache_key,
                {"diff": raw_diff_output},  # Wrap the diff string in a dict as expected by cache
                clear_cached_for_unchanged,
            )
            is_cached_diff = True
            logger.info("Diff cache operation successful.")

            if diff_cache is not None:  # Log cache stats
                stats = diff_cache.get_stats()
                logger.info(
                    f"Diff cache stats: Hits={stats.get('hits')}, Misses={stats.get('misses')}, Total={stats.get('total_accesses')}, Size={stats.get('current_size')} bytes, Max Size={stats.get('max_size')} bytes, Hit Rate={stats.get('hit_rate', 0.0):.2f}"
                )

            if not changes_from_cache:  # Empty dict or None means no changes detected by cache
                logger.info("Cache comparison detected no changes.")
                final_diff_content = "No git-tracked changes detected by cache comparison."
            else:  # Changes were detected by cache
                logger.info("Cache comparison detected changes.")
                final_diff_content = changes_from_cache.get("diff", "Error retrieving changes from cache.")
                if not final_diff_content:  # Should not happen if changes_from_cache is not empty
                    logger.warning(
                        "Cache comparison returned empty diff string despite changes_from_cache not being empty."
                    )
                    final_diff_content = "No git-tracked changes detected by cache comparison."
        except Exception as e:
            logger.error(f"Error using diff cache for key {cache_key}: {e}")
            # Fallback to using the raw diff_output if cache fails
            final_diff_content = raw_diff_output or "No git-tracked changes detected."
            logger.warning("Falling back to raw diff output due to cache error.")
    else:
        logger.info("Diff cache is disabled or not initialized.")
        # Use the raw diff_output if cache is disabled
        final_diff_content = raw_diff_output or "No git-tracked changes detected."
    return final_diff_content, is_cached_diff


def _update_summary_from_file_status(
    changes_summary: Dict[str, Any], file_status: Dict[str, Any], success: bool
) -> None:
    """Updates changes_summary with information from file_status if needed."""
    if (
        success  # Only update if overall operation is considered a success
        and file_status.get("has_changes")
        and (not changes_summary.get("summary") or "No git-tracked changes detected" in changes_summary["summary"])
    ):
        changes_summary["summary"] = (
            "No git-tracked changes detected, but " + file_status.get("status_summary", "").lower()
        )

        # Ensure 'files' in file_status is a list before iterating
        fs_files = file_status.get("files")
        if isinstance(fs_files, list) and not changes_summary.get("files"):
            changes_summary["files"] = [
                {"name": f["name"], "operation": f["operation"], "source": "filesystem"} for f in fs_files
            ]

        if "stats" not in changes_summary:
            changes_summary["stats"] = {}

        current_stats = changes_summary["stats"]  # Known to be a dict now

        fs_files_created = file_status.get("files_created", 0)
        if fs_files_created > 0:
            current_stats["files_created"] = fs_files_created

        fs_files_modified = file_status.get("files_modified", 0)
        if fs_files_modified > 0:
            current_stats["files_modified"] = fs_files_modified

        if current_stats:  # Check if stats dict is not empty
            total_changes = sum(
                v
                for k, v in current_stats.items()
                if k in ["files_created", "files_modified", "files_deleted"] and isinstance(v, int) and v > 0
            )
            if total_changes > 0:
                current_stats["total_files_changed"] = total_changes
                current_stats["source"] = "filesystem"


def _finalize_response_diff_field(
    response: ResponseDict, success: bool, changes_summary: Dict[str, Any], file_status: Dict[str, Any]
) -> None:
    """Sets the 'diff' field in the response for backward compatibility."""
    # Ensure diff field is present, primarily for backward compatibility with tests.
    # It should reflect the most relevant summary of changes or errors.
    if success:
        # If successful, the diff field should contain the summary of changes.
        # Prioritize git changes summary, then filesystem changes summary.
        if changes_summary.get("summary") and "No git-tracked changes detected" not in changes_summary["summary"]:
            response["diff"] = changes_summary["summary"]
        elif file_status.get("has_changes") and file_status.get("status_summary"):
            response["diff"] = file_status["status_summary"]
        else:  # Fallback if somehow success is true but no summary is available
            response["diff"] = changes_summary.get("summary", "Changes detected, summary unavailable.")
    else:
        # If not successful, diff field should contain an error message or indication of no changes.
        # If a specific error message is in changes_summary, use that.
        if changes_summary.get("summary") and "Error:" in changes_summary["summary"]:
            response["diff"] = changes_summary["summary"]
        elif "diff" in response and response["diff"]:  # If a diff field with an error was already set
            pass  # Keep the existing error message in diff
        else:  # Default for unsuccessful with no specific error in summary
            response["diff"] = changes_summary.get("summary", "No changes detected or operation failed.")


async def _process_coder_results(
    relative_editable_files: List[str],
    working_dir: Optional[str] = None,
    use_diff_cache: bool = True,
    clear_cached_for_unchanged: bool = True,
) -> ResponseDict:
    """
    Process the results after Aider has run, checking for meaningful changes
    and retrieving the diff or content, potentially using the diff cache.

    Args:
        relative_editable_files: List of files that were edited
        working_dir: The working directory where the git repo is located
        use_diff_cache: Whether to use the diff cache.
        clear_cached_for_unchanged: If using cache, whether to clear the entry
                                     if no changes are detected by the cache.

    Returns:
        Dictionary with success status and diff output
    """
    global diff_cache
    logger.info("Processing coder results...")

    # Initialize diff_cache if it's None and we're using it
    if use_diff_cache and diff_cache is None:
        logger.info("Initializing diff_cache in _process_coder_results")
        await init_diff_cache()

    raw_diff_output = get_changes_diff_or_content(relative_editable_files, working_dir)
    logger.info(f"Raw diff output obtained (length: {len(raw_diff_output)}).")

    has_meaningful_content = _check_for_meaningful_changes(relative_editable_files, working_dir)
    logger.info(f"Meaningful content detected: {has_meaningful_content}")

    cache_key = f"{working_dir}:{':'.join(sorted(relative_editable_files))}"
    final_diff_content, is_cached_diff = await _handle_diff_cache_processing(
        cache_key, raw_diff_output, use_diff_cache, clear_cached_for_unchanged
    )

    changes_summary = summarize_changes(final_diff_content)
    logger.info(f"Generated changes summary: {changes_summary['summary']}")

    file_status = get_file_status_summary(relative_editable_files, working_dir)
    logger.info(f"File status check: {file_status['status_summary']}")

    success = has_meaningful_content or file_status.get("has_changes", False)

    if has_meaningful_content and "git diff" in raw_diff_output:  # Check raw_diff_output for git diff presence
        if "stats" not in changes_summary:
            changes_summary["stats"] = {}
        changes_summary["stats"]["source"] = "git"

    response: ResponseDict = {
        "success": success,
        "changes_summary": changes_summary,
        "file_status": file_status,
        "is_cached_diff": is_cached_diff,
        # diff field will be set by _finalize_response_diff_field
    }

    _update_summary_from_file_status(changes_summary, file_status, success)
    _finalize_response_diff_field(response, success, changes_summary, file_status)

    if success:
        logger.info("Meaningful changes found or file status shows changes. Processing successful.")
    else:
        logger.warning("No changes detected in git or filesystem. Processing marked as unsuccessful.")

    logger.info("Coder results processed.")
    return response


async def _run_aider_session(
    coder: Coder,
    ai_coding_prompt: str,
    relative_editable_files: List[str],
    working_dir: str,
    use_diff_cache: bool,
    clear_cached_for_unchanged: bool,
) -> ResponseDict:
    """
    Run the Aider coding session and process the results.

    Args:
        coder: The configured Coder instance
        ai_coding_prompt: The prompt to send to the AI
        relative_editable_files: List of files that can be edited (relative paths)
        working_dir: Directory where files are located
        use_diff_cache: Whether to use the diff cache.
        clear_cached_for_unchanged: If using cache, whether to clear the entry
                                     if no changes are detected by the cache.

    Returns:
        Dictionary with success status and diff output
    """
    logger.info("Starting Aider coding session...")

    _log_file_states_before_after_run("before", relative_editable_files, working_dir)

    # Run the coder and capture output
    # The result of coder.run is often None or not directly used for the final JSON output's content.
    # The primary effect is file modification, which is then assessed.
    _ = _capture_output_and_run_coder(coder, ai_coding_prompt)  # Result of run is logged by helper

    _log_file_states_before_after_run("after", relative_editable_files, working_dir)

    # Process the results after the coder has run
    response: ResponseDict = await _process_coder_results(
        relative_editable_files,
        working_dir,
        use_diff_cache,
        clear_cached_for_unchanged,
    )
    return response


def _capture_output_and_run_coder(coder: Coder, ai_coding_prompt: str) -> Optional[str]:
    """Captures stdout/stderr and runs coder.run. Returns the result of coder.run."""
    import sys
    from io import StringIO

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    stdout_capture = StringIO()
    stderr_capture = StringIO()

    run_result = None
    try:
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture
        # Assuming coder.run might return something, like a status or summary string
        run_result = coder.run(ai_coding_prompt)
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    captured_stdout = stdout_capture.getvalue()
    captured_stderr = stderr_capture.getvalue()

    if captured_stdout:
        logger.warning(f"Captured stdout from Aider: {captured_stdout[:200]}...")
    if captured_stderr:
        logger.warning(f"Captured stderr from Aider: {captured_stderr[:200]}...")

    logger.info(f"coder.run completed, result: {run_result}")
    return str(run_result) if run_result is not None else None


def _log_file_states_before_after_run(stage: str, relative_editable_files: List[str], working_dir: str) -> None:
    """Logs the state of editable files (exists, size, partial content)."""
    logger.info(f"Logging file states {stage.upper()} coder run...")
    for file_path in relative_editable_files:
        # Ensure working_dir is used correctly to form full_path
        full_path = os.path.join(working_dir, os.path.normpath(file_path))

        if os.path.exists(full_path):
            try:
                size = os.path.getsize(full_path)
                logger.info(f"{stage.upper()} RUN: File {full_path} exists, size: {size} bytes")
                if size > 0:
                    with open(full_path, "r", errors="ignore") as f:  # Add errors='ignore' for robustness
                        content = f.read(100)
                        logger.info(
                            f"{stage.upper()} RUN: File '{file_path}' (partial content): {content[:50].encode('unicode_escape').decode()}..."
                        )
                elif stage.lower() == "after":  # Only warn if empty *after* run
                    logger.warning(f"AFTER RUN: File {full_path} exists but is EMPTY!")
            except Exception as e:
                logger.error(f"{stage.upper()} RUN: Error accessing file {full_path}: {e}")
        else:
            logger.info(f"{stage.upper()} RUN: File {full_path} does not exist.")
            if stage.lower() == "after":  # Only warn if not existing *after* run
                logger.warning(f"AFTER RUN: File {full_path} still does not exist after coder.run!")


async def _execute_with_retry(
    ai_coding_prompt: str,
    relative_editable_files: List[str],
    abs_editable_files: List[str],
    abs_readonly_files: List[str],
    working_dir: str,
    model: str,
    provider: str,
    use_diff_cache: bool,
    clear_cached_for_unchanged: bool,
    architect_mode: bool = False,
    editor_model: Optional[str] = None,
    auto_accept_architect: bool = True,
) -> ResponseDict:
    """
    Execute Aider with retry logic for rate limit handling.

    Args:
        ai_coding_prompt: The prompt for the AI
        relative_editable_files: List of editable files (relative paths)
        abs_editable_files: List of editable files (absolute paths)
        abs_readonly_files: List of read-only files (absolute paths)
        working_dir: Working directory
        model: Model identifier
        provider: Provider name
        use_diff_cache: Whether to use the diff cache.
        clear_cached_for_unchanged: If using cache, whether to clear the entry
                                     if no changes are detected by the cache.
        architect_mode: Whether to use two-phase architecture mode
        editor_model: Optional secondary model for implementation when architect_mode is enabled
        auto_accept_architect: Whether to automatically accept architect suggestions

    Returns:
        Response dictionary
    """
    empty_summary = summarize_changes("")
    empty_status = {"has_changes": False, "status_summary": "No changes detected."}
    response: ResponseDict = {
        "success": False,
        "changes_summary": empty_summary,
        "file_status": empty_status,
        "is_cached_diff": False,
        "rate_limit_info": {"encountered": False, "retries": 0, "fallback_model": None},
    }

    max_retries = fallback_config.get(provider, {}).get("max_retries", 3)  # Default retries
    initial_delay = fallback_config.get(provider, {}).get("initial_delay", 1)
    backoff_factor = fallback_config.get(provider, {}).get("backoff_factor", 2)
    current_model = model

    for attempt in range(max_retries + 1):
        try:
            ai_model = _configure_model(current_model, editor_model, architect_mode)
            coder = _setup_aider_coder(
                ai_model, working_dir, abs_editable_files, abs_readonly_files, architect_mode, auto_accept_architect
            )
            session_response = await _run_aider_session(
                coder,
                ai_coding_prompt,
                relative_editable_files,
                working_dir,
                use_diff_cache,
                clear_cached_for_unchanged,
            )
            response.update(session_response)
            # If _run_aider_session was successful, it sets response["success"] = True
            if response.get("success"):
                break  # Successful execution, exit retry loop

            # If not successful but no exception (e.g. no changes made), still break if it's the first attempt
            # or if it's not a rate limit error (which would be caught by except block)
            if attempt == 0 and not response.get("success"):  # No changes on first try
                logger.info("Aider session completed without errors but no changes were made.")
                break

        except Exception as e:
            should_retry, new_model_or_error = await _handle_rate_limit_or_error(
                e, provider, attempt, max_retries, initial_delay, backoff_factor, current_model, response
            )
            if should_retry:
                current_model = new_model_or_error  # This is the new model name
                # response['rate_limit_info'] is updated by the helper
            else:  # Not a retriable error or max retries reached for rate limit
                # Error details are set in response by the helper before raising
                raise  # Re-raise the exception that _handle_rate_limit_or_error decided not to retry

    return response


async def _handle_rate_limit_or_error(
    e: Exception,
    provider: str,
    attempt: int,
    max_retries: int,
    initial_delay: float,
    backoff_factor: float,
    current_model: str,
    response: ResponseDict,
) -> tuple[bool, str]:
    """
    Handles rate limits and other errors during Aider execution.
    Returns: (should_retry, new_model_if_retry_else_error_message)
    Updates response with error details.
    """
    logger.warning(f"Error during Aider execution (Attempt {attempt + 1}/{max_retries + 1}): {str(e)}")

    # Ensure rate_limit_info exists and is a dict
    rli = response.get("rate_limit_info")
    if not isinstance(rli, dict):
        rli = {"encountered": False, "retries": 0, "fallback_model": None}
        response["rate_limit_info"] = rli

    # Ensure 'retries' key exists with an int value
    if not isinstance(rli.get("retries"), int):
        rli["retries"] = 0

    if detect_rate_limit_error(e, provider):
        logger.info(f"Rate limit detected for {provider}. Attempting fallback...")
        rli["encountered"] = True
        rli["retries"] = rli.get("retries", 0) + 1  # type: ignore

        if attempt < max_retries:
            delay = initial_delay * (backoff_factor**attempt)
            logger.info(f"Retrying after {delay:.2f} seconds...")
            await asyncio.sleep(delay)
            new_model = get_fallback_model(current_model, provider)
            rli["fallback_model"] = new_model
            logger.info(f"Falling back to model: {new_model}")
            return True, new_model
        else:
            error_msg = f"Max retries ({max_retries}) reached for rate limit. Unable to complete request."
            logger.error(f"{error_msg} Last error: {str(e)}")
            response["success"] = False
            response["diff"] = response.get("diff") or f"Error: {error_msg}"
            cs = response.get("changes_summary")
            if not isinstance(cs, dict) or not cs.get("summary"):
                cs = summarize_changes("")
                cs["summary"] = f"Error: {error_msg}"
                response["changes_summary"] = cs
            raise Exception(f"{error_msg} Last error: {str(e)}") from e
    else:
        # Non-rate-limit error
        error_msg = f"Unhandled error during Aider execution: {str(e)}"
        logger.error(error_msg, exc_info=True)  # Log with traceback
        response["success"] = False
        response["diff"] = response.get("diff") or f"Error: {error_msg}"
        cs = response.get("changes_summary")
        if not isinstance(cs, dict) or not cs.get("summary"):
            cs = summarize_changes("")
            cs["summary"] = f"Error: {error_msg}"
            response["changes_summary"] = cs
        # For non-rate-limit errors, we don't retry through this helper.
        # The original exception is re-raised to be handled by the caller.
        raise  # Re-raise the original exception


async def code_with_aider(
    ai_coding_prompt: str,
    relative_editable_files: List[str],
    relative_readonly_files: Optional[List[str]] = None,
    model: str = "gemini/gemini-2.5-flash-preview-04-17",
    working_dir: Optional[str] = None,
    use_diff_cache: bool = True,
    clear_cached_for_unchanged: bool = True,
    architect_mode: bool = False,
    editor_model: Optional[str] = None,
    auto_accept_architect: bool = True,
    include_raw_diff: bool = False,
) -> str:
    """
    Run Aider to perform AI coding tasks based on the provided prompt and files.

    Args:
        ai_coding_prompt (str): The prompt for the AI to execute.
        relative_editable_files (List[str]): List of files that can be edited.
        relative_readonly_files (List[str], optional): List of files that can be read but not edited. Defaults to [].
        model (str, optional): The model to use. Defaults to "gemini/gemini-2.5-flash-preview-04-17".
        working_dir: The working directory where git repository is located and files are stored.
        use_diff_cache: Whether to use the diff cache. Defaults to True.
        clear_cached_for_unchanged: If using cache, whether to clear the entry
                                     if no changes are detected by the cache. Defaults to True.
        architect_mode (bool, optional): Enable two-phase code generation with an architect model
                                       planning first, then an editor model implementing. Defaults to False.
        editor_model (str, optional): The secondary AI model to use for code implementation when
                                    architect_mode is enabled. Defaults to None.
        auto_accept_architect (bool, optional): Automatically accept architect suggestions without
                                              confirmation. Defaults to True.
        include_raw_diff (bool, optional): Whether to include raw diff in the response (not recommended).
                                        Defaults to False.

    Returns:
        str: JSON string containing 'success', 'changes_summary', 'file_status', and other relevant information.
    """
    if relative_readonly_files is None:
        relative_readonly_files = []

    await _initial_setup_and_logging(
        ai_coding_prompt,
        relative_editable_files,
        relative_readonly_files,
        model,
        working_dir,
        use_diff_cache,
        clear_cached_for_unchanged,
        architect_mode,
        editor_model,
        auto_accept_architect,
    )

    original_model = model
    normalized_model_name = _normalize_model_name(model)
    provider = _determine_provider(normalized_model_name)

    # This working_dir check is critical and should lead to an early exit if failed.
    if not working_dir:
        error_msg = "Error: working_dir is required for code_with_aider"
        logger.error(error_msg)
        # Ensure a valid JSON response for this critical error
        return json.dumps(
            {
                "success": False,
                "changes_summary": {"summary": error_msg},
                "error": error_msg,  # Explicit error field
                "api_key_status": check_api_keys(None),  # Basic API key status
            }
        )

    key_status, _ = _handle_api_key_checks_and_warnings(working_dir, provider)  # Initial check
    if not key_status["any_keys_found"]:
        error_msg = "Error: No API keys found for any provider. Please set at least one API key."
        logger.error(error_msg)
        return json.dumps(
            {
                "success": False,
                "error": error_msg,
                "api_key_status": key_status,
                "warnings": [error_msg],
                "changes_summary": {"summary": error_msg},
            }
        )

    abs_editable_files = _convert_to_absolute_paths(relative_editable_files, working_dir)
    abs_readonly_files = _convert_to_absolute_paths(relative_readonly_files, working_dir)

    # Initialize response structure for error cases before try-except
    response: ResponseDict = {
        "success": False,
        "changes_summary": {"summary": "Operation did not complete successfully."},
        "file_status": {"has_changes": False, "status_summary": "No changes detected."},
        "is_cached_diff": False,
        "rate_limit_info": {"encountered": False, "retries": 0, "fallback_model": None},
    }

    # Capture stdout/stderr
    import sys
    from io import StringIO

    old_stdout, old_stderr = sys.stdout, sys.stderr
    stdout_capture, stderr_capture = StringIO(), StringIO()
    sys.stdout, sys.stderr = stdout_capture, stderr_capture

    actual_model_used = normalized_model_name  # Will be updated by _execute_with_retry if fallback occurs

    try:
        session_response = await _execute_with_retry(
            ai_coding_prompt,
            relative_editable_files,
            abs_editable_files,
            abs_readonly_files,
            working_dir,
            normalized_model_name,
            provider,
            use_diff_cache,
            clear_cached_for_unchanged,
            architect_mode,
            editor_model,
            auto_accept_architect,
        )
        response.update(session_response)
        # Update actual_model_used if fallback occurred
        if response.get("rate_limit_info", {}).get("fallback_model"):  # type: ignore
            actual_model_used = response["rate_limit_info"]["fallback_model"]  # type: ignore

    except TypeError as te:
        if "'bool' object is not callable" in str(te):  # Specific known issue
            logger.exception(f"Caught bool not callable error: {str(te)}")
            error_msg = "Error: Aider's internal tool_error method issue (bool not callable). Functionality unaffected."
            response["changes_summary"]["summary"] = error_msg
            response["diff"] = error_msg  # For backward compatibility
            # This is not a failure of the coding task itself, so success might still be true if changes were made.
            # However, to be safe, mark as unsuccessful if this specific error occurs.
            response["success"] = False
        else:  # Other TypeErrors
            logger.exception(f"Unhandled TypeError in code_with_aider: {str(te)}")
            response["changes_summary"]["summary"] = f"Unhandled TypeError: {str(te)}"
            response["diff"] = f"Unhandled TypeError: {str(te)}"
            response["success"] = False
    except Exception as e:
        logger.exception(f"Critical Error in code_with_aider: {str(e)}")
        response["changes_summary"]["summary"] = f"Unhandled Error: {str(e)}"
        response["diff"] = f"Unhandled Error: {str(e)}"  # For backward compatibility
        response["success"] = False
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        captured_stdout, captured_stderr = stdout_capture.getvalue(), stderr_capture.getvalue()
        if captured_stdout:
            logger.warning(f"Captured stdout: {captured_stdout[:500]}...")
        if captured_stderr:
            logger.warning(f"Captured stderr: {captured_stderr[:500]}...")

    _finalize_aider_response(response, key_status, original_model, actual_model_used, provider, include_raw_diff)

    formatted_response = json.dumps(response, indent=4)
    logger.info(f"code_with_aider process completed. Success: {response.get('success', False)}")
    return formatted_response


async def _initial_setup_and_logging(
    ai_coding_prompt: str,
    relative_editable_files: List[str],
    relative_readonly_files: List[str],
    model: str,  # This is original_model
    working_dir: Optional[str],
    use_diff_cache: bool,
    clear_cached_for_unchanged: bool,
    architect_mode: bool,
    editor_model: Optional[str],
    auto_accept_architect: bool,
) -> None:
    """Performs initial setup and logging for code_with_aider."""
    global diff_cache  # Ensure diff_cache is accessible
    logger.info("--- Starting code_with_aider ---")
    logger.info(f"Prompt: '{ai_coding_prompt[:100]}...'")  # Log truncated prompt

    if use_diff_cache and diff_cache is None:
        logger.info("Initializing DiffCache for code_with_aider...")
        await init_diff_cache()  # Ensure this is awaited

    if not working_dir:
        logger.error("CRITICAL: working_dir is None in _initial_setup_and_logging. This should not happen.")
        # This state indicates a programming error, as working_dir should be validated before this call.
        return

    logger.info(f"Working directory: {working_dir}")
    logger.info(f"Editable files ({len(relative_editable_files)}): {relative_editable_files[:3]}...")
    logger.info(f"Readonly files ({len(relative_readonly_files)}): {relative_readonly_files[:3]}...")
    logger.info(f"Initial model: {model}")
    logger.info(f"Use diff cache: {use_diff_cache}, Clear cached for unchanged: {clear_cached_for_unchanged}")
    logger.info(f"Architect mode: {architect_mode}")
    if architect_mode:
        logger.info(f"Editor model: {editor_model if editor_model else 'Same as main model'}")
        logger.info(f"Auto accept architect: {auto_accept_architect}")


def _handle_api_key_checks_and_warnings(  # This function is mostly for initial check
    working_dir: Optional[str],
    provider_requested: str,  # No response needed here
) -> tuple[Dict[str, Any], bool]:
    """Checks API keys. Returns key_status and if requested provider has keys."""
    key_status = check_api_keys(working_dir)  # Loads .env files

    # Log general API key status
    if not key_status["any_keys_found"]:
        logger.error("CRITICAL: No API keys found for ANY provider.")
    else:
        logger.info(f"Available providers with keys: {key_status['available_providers']}")
        if key_status["missing_providers"]:
            logger.warning(f"Providers missing keys: {key_status['missing_providers']}")

    # Check for the specifically requested provider
    provider_has_keys = provider_requested in key_status["available_providers"]
    if not provider_has_keys:
        logger.warning(
            f"API key for the initially requested provider '{provider_requested}' is missing or invalid."
            " Fallback mechanisms will be attempted if other provider keys are available."
        )
    return key_status, provider_has_keys


def _update_api_key_status_in_response(
    response: ResponseDict,
    key_status: Dict[str, Any],
    requested_provider: str,
    actual_model_used: str,
    original_model_requested: str,
) -> None:
    """Updates the API key status information in the response."""
    actual_provider_used = _determine_provider(actual_model_used)
    response["api_key_status"] = {
        "available_providers": key_status.get("available_providers", []),
        "missing_providers": key_status.get("missing_providers", []),
        "requested_provider": requested_provider,
        "used_provider": actual_provider_used,
        "original_model_requested": original_model_requested,
        "actual_model_used": actual_model_used,
    }


def _add_provider_warning_to_response(
    response: ResponseDict,
    key_status: Dict[str, Any],
    requested_provider: str,
    actual_provider_used: str,
    actual_model_used: str,
) -> None:
    """Adds a warning if the requested provider's key was missing."""
    if requested_provider not in key_status.get("available_providers", []):
        warning_msg = (
            f"Warning: API key for the initially requested provider '{requested_provider}' was missing. "
            f"The system attempted to use provider '{actual_provider_used}' with model '{actual_model_used}'."
        )
        if "warnings" not in response:
            response["warnings"] = []
        # Ensure warnings is a list
        if not isinstance(response.get("warnings"), list):
            response["warnings"] = []

        if warning_msg not in response["warnings"]:  # type: ignore
            response["warnings"].append(warning_msg)  # type: ignore


def _handle_diff_field_in_response(response: ResponseDict, include_raw_diff: bool) -> None:
    """Ensures 'diff' field is present or removed based on include_raw_diff and success."""
    if "diff" not in response or not response["diff"]:
        summary_text = response.get("changes_summary", {}).get("summary", "No changes or error information available.")
        response["diff"] = summary_text

    if not include_raw_diff and response.get("success", False):
        if "diff" in response:
            # Only delete if it's not an error message that might be in 'diff'
            if not (response["diff"] and "Error:" in response["diff"]):
                del response["diff"]


def _cleanup_file_status_in_response(response: ResponseDict) -> None:
    """Cleans up the file_status field in the response."""
    fs = response.get("file_status")
    if isinstance(fs, dict):
        for key in ["files_created", "files_modified"]:
            if fs.get(key) == 0:
                del fs[key]
        if (
            not fs.get("has_changes")
            and not fs.get("files_created")
            and not fs.get("files_modified")
            and not fs.get("files")
        ):
            response["file_status"] = {
                "has_changes": False,
                "status_summary": fs.get("status_summary", "No filesystem changes detected."),
            }


def _cleanup_changes_summary_in_response(response: ResponseDict) -> None:
    """Cleans up the changes_summary field in the response."""
    cs = response.get("changes_summary")
    if isinstance(cs, dict):
        cs_stats = cs.get("stats")
        if isinstance(cs_stats, dict):
            keys_to_del = [k for k, v in cs_stats.items() if v == 0]
            for k in keys_to_del:
                del cs_stats[k]
            if not cs_stats:
                del cs["stats"]
        if isinstance(cs.get("files"), list) and not cs["files"]:
            del cs["files"]


def _finalize_aider_response(
    response: ResponseDict,
    key_status: Dict[str, Any],  # Result from initial check_api_keys
    original_model_requested: str,
    actual_model_used: str,
    requested_provider: str,
    include_raw_diff: bool,
) -> None:
    """Finalizes the response dictionary with API key status, warnings, and cleanup."""
    _update_api_key_status_in_response(
        response, key_status, requested_provider, actual_model_used, original_model_requested
    )

    actual_provider_used = _determine_provider(actual_model_used)  # Recalculate for warning
    _add_provider_warning_to_response(response, key_status, requested_provider, actual_provider_used, actual_model_used)

    _handle_diff_field_in_response(response, include_raw_diff)
    _cleanup_file_status_in_response(response)
    _cleanup_changes_summary_in_response(response)
