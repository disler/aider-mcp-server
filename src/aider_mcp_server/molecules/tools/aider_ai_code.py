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

# Internal imports
from aider_mcp_server.atoms.utils.diff_cache import DiffCache
from aider_mcp_server.atoms.logging.logger import get_logger
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
from aider_mcp_server.atoms.utils.fallback_config import (
    detect_rate_limit_error,
    get_fallback_model,
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

    # Define provider to key mappings
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

    logger.info("Checking API keys in environment...")
    for key, provider in keys_to_check.items():
        if os.environ.get(key):
            logger.info(f"✓ {provider} API key found ({key})")
            result["found"].append(key)
            result["any_keys_found"] = True
        else:
            logger.warning(f"✗ {provider} API key missing ({key})")
            result["missing"].append(key)

    # Special handling for Gemini/Google - check if we need to copy between variables
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

    # Determine available providers
    for provider, keys in provider_keys.items():
        found_list = result["found"]
        available_providers = result["available_providers"]
        missing_providers = result["missing_providers"]

        if isinstance(found_list, list) and any(key in found_list for key in keys):
            if isinstance(available_providers, list):
                available_providers.append(provider)
        elif isinstance(missing_providers, list):
            missing_providers.append(provider)

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

    # Get the raw diff output from git or file contents
    diff_output = get_changes_diff_or_content(relative_editable_files, working_dir)
    logger.info(f"Raw diff output obtained (length: {len(diff_output)}).")

    # Check for meaningful content in the edited files
    logger.info("Checking for meaningful changes in edited files...")
    has_meaningful_content = _check_for_meaningful_changes(relative_editable_files, working_dir)
    logger.info(f"Meaningful content detected: {has_meaningful_content}")

    cache_key = f"{working_dir}:{':'.join(sorted(relative_editable_files))}"  # Use sorted files for consistent key
    changes_from_cache: Optional[Dict[str, Any]] = None
    is_cached_diff = False
    final_diff_content = diff_output or "No git-tracked changes detected."  # Default fallback

    if use_diff_cache and diff_cache is not None:
        logger.info(f"Attempting to use diff cache for key: {cache_key}")
        try:
            # Pass the raw diff_output as the 'new_diff' data to compare_and_cache
            # The cache will compare this against the old cached diff and return the changes.
            changes_from_cache = await diff_cache.compare_and_cache(
                cache_key,
                {"diff": diff_output},  # Wrap the diff string in a dict as expected by cache
                clear_cached_for_unchanged,
            )
            is_cached_diff = True
            logger.info("Diff cache operation successful.")

            # Log cache statistics
            if diff_cache is not None:
                stats = diff_cache.get_stats()  # This is a synchronous method, not async
                logger.info(
                    f"Diff cache stats: Hits={stats.get('hits')}, Misses={stats.get('misses')}, Total={stats.get('total_accesses')}, Size={stats.get('current_size')} bytes, Max Size={stats.get('max_size')} bytes, Hit Rate={stats.get('hit_rate'):.2f}"
                )

            # Safety check for cache result type
            changes_is_none = changes_from_cache is None
            # In reality, the cache implementation never returns None, so mypy sees this as unreachable.
            # We're keeping this code for safety, but we need to silence the unreachable warning.

            # Determine the final diff content based on cache result
            if changes_is_none:
                # This branch is for defensive programming only
                logger.warning("Cache returned None - bypassing type check for safety")
                final_diff_content = "Error retrieving changes from cache."
            elif not changes_from_cache:  # Empty dict means no changes detected by cache
                logger.info("Cache comparison detected no changes.")
                final_diff_content = "No git-tracked changes detected by cache comparison."
            else:  # Changes were detected by cache
                logger.info("Cache comparison detected changes.")
                # The changes_from_cache dict contains the diff of the diffs.
                # We want to return the actual diff content that represents the changes.
                # The 'diff' key in the changes_from_cache dict holds the diff string.
                final_diff_content = changes_from_cache.get("diff", "Error retrieving changes from cache.")
                if not final_diff_content:
                    logger.warning(
                        "Cache comparison returned empty diff string despite changes_from_cache not being empty."
                    )
                    final_diff_content = "No git-tracked changes detected by cache comparison."

        except Exception as e:
            logger.error(f"Error using diff cache for key {cache_key}: {e}")
            is_cached_diff = False
            # Fallback to using the raw diff_output if cache fails
            final_diff_content = diff_output or "No git-tracked changes detected."
            logger.warning("Falling back to raw diff output due to cache error.")
    else:
        logger.info("Diff cache is disabled.")
        # Use the raw diff_output if cache is disabled
        final_diff_content = diff_output or "No git-tracked changes detected."

    # Generate a summary of the changes
    changes_summary = summarize_changes(final_diff_content)
    logger.info(f"Generated changes summary: {changes_summary['summary']}")

    # Check file status as a fallback for detecting changes
    file_status = get_file_status_summary(relative_editable_files, working_dir)
    logger.info(f"File status check: {file_status['status_summary']}")

    # Determine success based on meaningful content or file status
    success = has_meaningful_content or file_status["has_changes"]

    # For git-tracked changes, add a source field
    if has_meaningful_content and "git diff" in diff_output:
        if "stats" not in changes_summary:
            changes_summary["stats"] = {}
        changes_summary["stats"]["source"] = "git"

    # Create base response with only essential information
    response: ResponseDict = {
        "success": success,
        "changes_summary": changes_summary,
        "file_status": file_status,  # Include for backward compatibility with tests
        "is_cached_diff": is_cached_diff,  # Include for backward compatibility with tests
        "diff": final_diff_content
        or changes_summary.get("summary", "No changes detected."),  # Always include diff for backward compatibility
    }

    # Use file_status to update changes_summary if it's empty or indicates no changes
    if (
        success
        and file_status["has_changes"]
        and (not changes_summary["summary"] or "No git-tracked changes detected" in changes_summary["summary"])
    ):
        # Update changes_summary with information from file_status, but clearly indicate source
        changes_summary["summary"] = "No git-tracked changes detected, but " + file_status["status_summary"].lower()
        # If files were created/modified, make sure they appear in the changes_summary files list
        if "files" in file_status and not changes_summary.get("files"):
            changes_summary["files"] = [
                {"name": f["name"], "operation": f["operation"], "source": "filesystem"} for f in file_status["files"]
            ]
        # Update stats from file_status counts
        if "files_created" in file_status and file_status["files_created"] > 0:
            if "stats" not in changes_summary:
                changes_summary["stats"] = {}
            changes_summary["stats"]["files_created"] = file_status["files_created"]
        if "files_modified" in file_status and file_status["files_modified"] > 0:
            if "stats" not in changes_summary:
                changes_summary["stats"] = {}
            changes_summary["stats"]["files_modified"] = file_status["files_modified"]
        if "stats" in changes_summary and changes_summary["stats"]:
            total_changes = sum(
                v
                for k, v in changes_summary["stats"].items()
                if k in ["files_created", "files_modified", "files_deleted"] and v > 0
            )
            if total_changes > 0:
                changes_summary["stats"]["total_files_changed"] = total_changes
                # Add source information for clarity
                changes_summary["stats"]["source"] = "filesystem"

    # For backward compatibility, include a human-readable summary in the diff field,
    # but only if there are meaningful changes to report
    if success:
        # Add clarity about the source of changes
        if "No git-tracked changes detected" in changes_summary["summary"] and file_status["has_changes"]:
            response["diff"] = changes_summary["summary"]
        elif file_status["has_changes"]:
            response["diff"] = changes_summary["summary"]
        else:
            response["diff"] = changes_summary["summary"]

    if success:
        if has_meaningful_content:
            logger.info("Meaningful changes found. Processing successful.")
        else:
            logger.info("No meaningful content detected, but file status shows changes. Processing successful.")
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
    try:
        # Capture any stdout that might interfere with JSON response
        import sys
        from io import StringIO

        # Redirect stdout and stderr temporarily
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        stdout_capture = StringIO()
        stderr_capture = StringIO()
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture

        # Check files before running coder to see if they exist/have content
        for file_path in relative_editable_files:
            full_path = os.path.join(working_dir, file_path) if working_dir else file_path
            if os.path.exists(full_path):
                size = os.path.getsize(full_path)
                logger.info(f"BEFORE RUN: File {full_path} exists, size: {size} bytes")
                if size > 0:
                    with open(full_path, "r") as f:
                        content = f.read(100)  # Just read the first 100 chars to log
                        logger.info(f"BEFORE RUN: File contains: {content}...")
            else:
                logger.info(f"BEFORE RUN: File {full_path} does not exist yet")

        # Run the actual coder function
        logger.info(f"Running coder.run with prompt: {ai_coding_prompt}")
        result = coder.run(ai_coding_prompt)
        logger.info(f"coder.run completed, result: {result}")

        # Check files after running coder to see if they exist/have content
        for file_path in relative_editable_files:
            full_path = os.path.join(working_dir, file_path) if working_dir else file_path
            if os.path.exists(full_path):
                size = os.path.getsize(full_path)
                logger.info(f"AFTER RUN: File {full_path} exists, size: {size} bytes")
                if size > 0:
                    with open(full_path, "r") as f:
                        content = f.read(100)  # Just read the first 100 chars to log
                        logger.info(f"AFTER RUN: File contains: {content}...")
                else:
                    logger.warning(f"AFTER RUN: File {full_path} exists but is EMPTY!")
            else:
                logger.warning(f"AFTER RUN: File {full_path} still does not exist after coder.run!")

        # Capture any output that was written
        captured_stdout = stdout_capture.getvalue()
        captured_stderr = stderr_capture.getvalue()

        # Restore stdout and stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr

        if captured_stdout:
            logger.warning(f"Captured stdout from Aider: {captured_stdout[:200]}...")
        if captured_stderr:
            logger.warning(f"Captured stderr from Aider: {captured_stderr[:200]}...")

        logger.info(f"Aider coding session result: {result}")
    except Exception as e:
        # Make sure to restore stdout and stderr even if there's an error
        if "old_stdout" in locals():
            sys.stdout = old_stdout
        if "old_stderr" in locals():
            sys.stderr = old_stderr
        raise e

    # Process the results after the coder has run
    response: ResponseDict = await _process_coder_results(  # Await the async function
        relative_editable_files,
        working_dir,
        use_diff_cache,
        clear_cached_for_unchanged,
    )
    return response


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
    # Create empty summary objects
    empty_summary = summarize_changes("")
    empty_status = {
        "has_changes": False,
        "status_summary": "No changes detected.",
    }

    response: ResponseDict = {
        "success": False,
        "changes_summary": empty_summary,
        "file_status": empty_status,  # For backward compatibility
        "is_cached_diff": False,  # For backward compatibility
    }

    max_retries = fallback_config[provider]["max_retries"]
    initial_delay = fallback_config[provider]["initial_delay"]
    backoff_factor = fallback_config[provider]["backoff_factor"]

    current_model = model  # Use a variable for the model that might change during retries

    for attempt in range(max_retries + 1):
        try:
            # Configure the model and create the coder
            ai_model = _configure_model(current_model, editor_model, architect_mode)  # Use current_model
            coder = _setup_aider_coder(
                ai_model,
                working_dir,
                abs_editable_files,
                abs_readonly_files,
                architect_mode,
                auto_accept_architect,
            )

            # Run the session and get results
            session_response = await _run_aider_session(  # Await the async function
                coder,
                ai_coding_prompt,
                relative_editable_files,
                working_dir,
                use_diff_cache,
                clear_cached_for_unchanged,
            )
            response.update(session_response)

            # Success, exit the retry loop
            break

        except Exception as e:
            logger.warning(f"Error during Aider execution (Attempt {attempt + 1}): {str(e)}")

            if detect_rate_limit_error(e, provider):
                logger.info(f"Rate limit detected for {provider}. Attempting fallback...")
                # Add or update rate_limit_info - only when actually encountered
                if "rate_limit_info" not in response:
                    response["rate_limit_info"] = {"encountered": True, "retries": 1, "fallback_model": None}
                else:
                    rate_limit_info = response.get("rate_limit_info")
                    if isinstance(rate_limit_info, dict):
                        rate_limit_info["encountered"] = True
                        if "retries" in rate_limit_info and isinstance(rate_limit_info["retries"], int):
                            rate_limit_info["retries"] += 1

                if attempt < max_retries:
                    delay = initial_delay * (backoff_factor**attempt)
                    logger.info(f"Retrying after {delay} seconds...")
                    await asyncio.sleep(delay)  # Use asyncio.sleep in async function
                    current_model = get_fallback_model(current_model, provider)  # Update current_model
                    if "rate_limit_info" not in response:
                        response["rate_limit_info"] = {
                            "encountered": True,
                            "retries": 1,
                            "fallback_model": current_model,
                        }
                    else:
                        rate_limit_info = response.get("rate_limit_info")
                        if isinstance(rate_limit_info, dict):
                            rate_limit_info["fallback_model"] = current_model
                    logger.info(f"Falling back to model: {current_model}")
                else:
                    logger.error("Max retries reached. Unable to complete the request.")
                    # Update response with final error state before re-raising
                    error_msg = f"Max retries reached due to rate limit or other error: {str(e)}"
                    response["success"] = False
                    if "diff" not in response:  # Only add diff if it doesn't exist
                        response["diff"] = error_msg

                    # Add error summary
                    error_summary = summarize_changes("")
                    error_summary["summary"] = "Error: " + error_msg
                    response["changes_summary"] = error_summary
                    raise
            else:
                # If it's not a rate limit error, update response with error and re-raise
                error_msg = f"Error during Aider execution: {str(e)}"
                response["success"] = False
                if "diff" not in response:  # Only add diff if it doesn't exist
                    response["diff"] = error_msg

                # Add error summary
                error_summary = summarize_changes("")
                error_summary["summary"] = "Error: " + error_msg
                response["changes_summary"] = error_summary
                raise

    return response


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
    global diff_cache
    if relative_readonly_files is None:
        relative_readonly_files = []
    logger.info("Starting code_with_aider process.")
    logger.info(f"Prompt: '{ai_coding_prompt}'")

    # Initialize diff_cache if it's None and we're using it
    if use_diff_cache and diff_cache is None:
        logger.info("Initializing diff_cache for code_with_aider")
        await init_diff_cache()

    # Check API keys at the beginning and store the result
    key_status = check_api_keys(working_dir)

    # Store original model for reference
    original_model = model

    # Validate working directory
    if not working_dir:
        error_msg = "Error: working_dir is required for code_with_aider"
        logger.error(error_msg)
        return json.dumps({"success": False, "changes_summary": {"summary": error_msg}})

    logger.info(f"Working directory: {working_dir}")
    logger.info(f"Editable files: {relative_editable_files}")
    logger.info(f"Readonly files: {relative_readonly_files}")
    logger.info(f"Initial model: {model}")
    logger.info(f"Use diff cache: {use_diff_cache}")
    logger.info(f"Clear cached for unchanged: {clear_cached_for_unchanged}")
    logger.info(f"Architect mode: {architect_mode}")
    if architect_mode:
        logger.info(f"Editor model: {editor_model if editor_model else 'Same as main model'}")
        logger.info(f"Auto accept architect: {auto_accept_architect}")

    # Normalize model name
    model = _normalize_model_name(model)
    logger.info(f"Normalized model name: {model}")

    # Determine provider
    provider = _determine_provider(model)
    logger.info(f"Using provider: {provider}")

    # Check if we have any API keys at all
    if not key_status["any_keys_found"]:
        # No API keys found at all - this is a critical error
        error_msg = "Error: No API keys found for any provider. Please set at least one API key."
        logger.error(error_msg)
        no_keys_response = {
            "success": False,
            "error": error_msg,
            "api_key_status": key_status,
            "warnings": [error_msg],
            "changes_summary": {"summary": error_msg},
        }
        return json.dumps(no_keys_response)

    # Check if the requested provider has missing keys
    provider_has_keys = provider in key_status["available_providers"]

    # Warn about missing keys for the requested provider
    if not provider_has_keys and key_status["available_providers"]:
        available_provider = key_status["available_providers"][0]
        warning_msg = f"Warning: No API keys found for {provider}. Using {available_provider} instead."
        logger.warning(warning_msg)

        # We will continue with the execution, but note that a different provider might be used
        # This will be handled by the fallback mechanism in _execute_with_retry

    # Convert to absolute paths
    abs_editable_files = _convert_to_absolute_paths(relative_editable_files, working_dir)
    abs_readonly_files = _convert_to_absolute_paths(relative_readonly_files, working_dir)

    # Capture stdout/stderr at the earliest point to catch any output
    import sys
    from io import StringIO

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    stdout_capture = StringIO()
    stderr_capture = StringIO()

    # Redirect stdout and stderr
    sys.stdout = stdout_capture
    sys.stderr = stderr_capture

    try:
        # Execute with retry logic
        response: ResponseDict = await _execute_with_retry(
            ai_coding_prompt,
            relative_editable_files,
            abs_editable_files,
            abs_readonly_files,
            working_dir,
            model,
            provider,
            use_diff_cache,
            clear_cached_for_unchanged,
            architect_mode,
            editor_model,
            auto_accept_architect,
        )
    except TypeError as te:
        if "'bool' object is not callable" in str(te):
            # Handle the specific bool not callable error
            logger.exception(f"Caught bool not callable error: {str(te)}")
            error_msg = "Error during Aider execution: Unable to call tool_error method due to being set to a boolean. This is a known issue with no impact on functionality."
            empty_summary = summarize_changes("")
            empty_summary["summary"] = "Error: " + error_msg
            empty_status = {
                "has_changes": False,
                "status_summary": "No changes detected.",
            }

            type_error_response: ResponseDict = {
                "success": False,
                "changes_summary": empty_summary,
                "file_status": empty_status,
                "rate_limit_info": {
                    "encountered": False,
                    "retries": 0,
                    "fallback_model": None,
                },
                "is_cached_diff": False,
                "diff": "Error: " + error_msg,
            }
            response = type_error_response
        else:
            # Re-raise other TypeError exceptions
            raise
    except Exception as e:
        logger.exception(f"Critical Error in code_with_aider: {str(e)}")
        # Create error response since the previous one failed
        error_msg = f"Unhandled Error during Aider execution: {str(e)}"
        empty_summary = summarize_changes("")
        empty_summary["summary"] = "Error: " + error_msg
        empty_status = {
            "has_changes": False,
            "status_summary": "No changes detected.",
        }

        # Use a properly typed definition
        response = {
            "success": False,
            "changes_summary": empty_summary,
            "file_status": empty_status,
            "rate_limit_info": {
                "encountered": False,
                "retries": 0,
                "fallback_model": None,
            },
            "is_cached_diff": False,
            "diff": "Error: " + error_msg,
        }
    finally:
        # Always restore stdout and stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr

        # Log any captured output
        captured_stdout = stdout_capture.getvalue()
        captured_stderr = stderr_capture.getvalue()

        if captured_stdout:
            logger.warning(f"Captured stdout: {captured_stdout}")
        if captured_stderr:
            logger.warning(f"Captured stderr: {captured_stderr}")

    # Add API key status information to the response
    response["api_key_status"] = {
        "available_providers": key_status["available_providers"],
        "missing_providers": key_status["missing_providers"],
        "used_provider": provider,
        "original_model": original_model,
        "actual_model": model,  # This may be different if fallback was used in _execute_with_retry
    }

    # Add warnings if the provider is missing keys
    available_providers = key_status["available_providers"]
    if isinstance(available_providers, list) and provider not in available_providers:
        warning_msg = f"Warning: No API keys found for provider {provider}. Some functionality may be limited."
        if "warnings" not in response:
            response["warnings"] = []
        warnings_list = response.get("warnings")
        if isinstance(warnings_list, list):
            warnings_list.append(warning_msg)

    # Ensure diff field is present for tests (even though we may remove it later)
    if "diff" not in response and "changes_summary" in response and response["changes_summary"].get("summary"):
        response["diff"] = response["changes_summary"]["summary"]

    # If not requested to include the raw diff, remove it from successful responses
    # to keep the response more concise (we already have better info in changes_summary).
    # But keep it for errors as it may contain important debugging info.
    if not include_raw_diff and response.get("success", False):
        if "diff" in response:
            del response["diff"]

    # Clean up the response by removing default/zero values
    if "file_status" in response:
        file_status = response["file_status"]
        # Remove zero counts
        for key in ["files_created", "files_modified"]:
            if key in file_status and (file_status[key] == 0 or not file_status[key]):
                del file_status[key]
        # If there are no changes in file_status, simplify it to just the essential info
        if not file_status.get("has_changes", False) or (
            "files" in file_status
            and not file_status["files"]
            and "files_created" not in file_status
            and "files_modified" not in file_status
        ):
            file_status = {
                "has_changes": file_status.get("has_changes", False),
                "status_summary": file_status.get("status_summary", "No changes detected."),
            }
            response["file_status"] = file_status

    if "changes_summary" in response and "stats" in response["changes_summary"]:
        stats = response["changes_summary"]["stats"]
        # Remove zero counts
        keys_to_check = [
            "files_created",
            "files_modified",
            "files_deleted",
            "lines_added",
            "lines_removed",
            "total_files_changed",
        ]
        for key in keys_to_check:
            if key in stats and (stats[key] == 0 or not stats[key]):
                del stats[key]
        # If stats is empty after removing zeros, remove it
        if not stats:
            del response["changes_summary"]["stats"]
        # If files list is empty, remove it to keep response clean
        if "files" in response["changes_summary"] and not response["changes_summary"]["files"]:
            del response["changes_summary"]["files"]

    # Convert the response to a proper JSON string
    formatted_response = json.dumps(response, indent=4)
    logger.info(f"code_with_aider process completed. Success: {response['success']}")
    return formatted_response
