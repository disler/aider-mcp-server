import asyncio
import json
import os
import os.path
import pathlib
import time

# External imports
from typing import TYPE_CHECKING, Any, Dict, List, Optional, TypedDict

# Add TYPE_CHECKING import for coordinator
if TYPE_CHECKING:
    from aider_mcp_server.interfaces.application_coordinator import IApplicationCoordinator

from aider.coders import AskCoder
from aider.io import InputOutput
from aider.models import Model

from aider_mcp_server.atoms.logging.logger import get_logger
from aider_mcp_server.atoms.types.event_types import EventTypes
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

# Create a subclass of InputOutput that overrides tool_error to do nothing
class SilentInputOutput(InputOutput):  # type: ignore[misc]
    """A subclass of InputOutput that overrides tool_error to do nothing."""

    def tool_error(self, message: str = "", strip: bool = True) -> None:
        """Override to do nothing with error messages."""
        pass


class AskResponseDict(TypedDict, total=False):
    """Type for Aider Ask Mode response dictionary."""

    success: bool
    response: str  # The explanation/answer from Aider
    rate_limit_info: Optional[
        Dict[str, Any]
    ]  # Optional - only included when rate limits are encountered
    api_key_status: Optional[Dict[str, Any]]  # Information about API key status
    warnings: Optional[List[str]]  # List of warnings to display to the user


# Configure logging for this module
logger = get_logger(__name__)

# Try to import dotenv for environment variable loading
try:
    from dotenv import load_dotenv

    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False

# Load fallback configuration (reuse from aider_ai_code)
try:
    # Get the directory where this module is located
    current_dir = pathlib.Path(__file__).parent.absolute()
    # Navigate up from tools/aider_ai_ask.py to the repository root (4 levels up)
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
    """Check if necessary API keys are set in the environment and return status."""
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

    # Check individual API keys
    logger.info("Checking API keys in environment...")
    for key, provider in keys_to_check.items():
        if os.environ.get(key):
            logger.info(f"✓ {provider} API key found ({key})")
            result["found"].append(key)
            result["any_keys_found"] = True
        else:
            logger.warning(f"✗ {provider} API key missing ({key})")
            result["missing"].append(key)

    # Handle GEMINI_API_KEY alias
    if os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if gemini_key is not None:
            logger.info("Setting GOOGLE_API_KEY from GEMINI_API_KEY for compatibility")
            os.environ["GOOGLE_API_KEY"] = gemini_key
            if "GOOGLE_API_KEY" in result["missing"]:
                result["missing"].remove("GOOGLE_API_KEY")
                result["found"].append("GOOGLE_API_KEY")

    # Determine available providers
    for provider, keys in provider_keys.items():
        if any(key in result["found"] for key in keys):
            result["available_providers"].append(provider)
        else:
            result["missing_providers"].append(provider)

    return result


def _normalize_model_name(model: str) -> str:
    """Normalize the model name to a consistent format."""
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
    """Extract the provider from the model name."""
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


def _configure_model(model: str, architect_mode: bool = False) -> Model:
    """Configure the Aider model based on the model name."""
    logger.info(f"Configuring model for Ask Mode: {model}, architect_mode={architect_mode}")

    # For testing purposes (when we know the model will fail), use a simple model name
    if model == "non_existent_model_123456789":
        logger.info(f"Using deliberately non-existent model for testing: {model}")
        return Model(model)

    # Use the actual requested model
    aider_model_name = model
    logger.info(f"Using requested model: {aider_model_name}")

    # For Ask Mode, we don't need architect mode typically, but support it if requested
    if architect_mode:
        logger.info(f"Using model with architect mode: {aider_model_name}")
        return Model(aider_model_name, editor_model=aider_model_name)
    else:
        # Standard configuration for Ask Mode
        return Model(aider_model_name)


def _convert_to_absolute_paths(relative_paths: List[str], working_dir: Optional[str]) -> List[str]:
    """Convert relative file paths to absolute paths."""
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


def _setup_aider_ask_coder(
    model: Model,
    working_dir: str,
    abs_readonly_files: List[str],
    architect_mode: bool = False,
) -> AskCoder:
    """Set up an Aider AskCoder instance for question-answering."""
    logger.info("Setting up Aider AskCoder...")

    # Log aider version for debugging
    aider_version = get_aider_version()
    logger.info(f"Using aider version: {aider_version}")

    # Set chat history file path in the working directory if possible
    chat_history_file = None
    if working_dir:
        try:
            chat_history_dir = os.path.join(working_dir, ".aider")
            os.makedirs(chat_history_dir, exist_ok=True)
            chat_history_file = os.path.join(chat_history_dir, "ask_chat_history.md")
        except Exception as e:
            logger.warning(f"Could not create chat history directory: {e}")

    # Create an IO instance for the AskCoder
    io = SilentInputOutput(
        pretty=False,  # Disable fancy output
        yes=True,  # Always say yes to prompts
        fancy_input=False,  # Disable fancy input to avoid prompt_toolkit usage
        chat_history_file=chat_history_file,  # Set chat history file if available
    )

    io.yes_to_all = True  # Automatically say yes to all prompts
    io.dry_run = False  # Ensure we're not in dry-run mode
    io.quiet = True  # Set quiet mode to suppress output

    # Create no-op functions for output methods to suppress output
    def noop(*args: Any, **kwargs: Any) -> None:
        pass

    # Redirect output to no-op functions
    io.output = noop
    io.tool_output = noop

    # For AskCoder, we don't need a GitRepo since we're not modifying files
    # Parameters for Coder.create method (different from __init__)
    create_params = {
        "main_model": model,
        "io": io,
        "edit_format": "ask",  # This is the key - tells Aider to use AskCoder
    }

    # Parameters that go directly to Coder.__init__ via kwargs
    init_params = {
        "fnames": [],  # No editable files for Ask Mode
        "read_only_fnames": abs_readonly_files,
        "repo": None,  # No git repo needed for Ask Mode
        "show_diffs": False,  # No diffs in Ask Mode
        "auto_commits": False,
        "dirty_commits": False,
        "use_git": False,  # No git operations in Ask Mode
        "stream": False,
        "suggest_shell_commands": False,
        "detect_urls": False,
        "verbose": True,  # Enable verbose mode for more debugging info
    }

    logger.info(f"Setting up AskCoder with params: create_params={create_params}, init_params={init_params}")

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

    logger.info(f"Creating AskCoder with parameters: {list(final_params.keys())}")

    # Create the AskCoder instance using parameters compatible with the installed version
    coder = AskCoder.create(**final_params)

    return coder


async def _run_ask_session(
    coder: AskCoder,
    ai_coding_prompt: str,
) -> str:
    """Run the Aider Ask session and return the response."""
    logger.info("Starting Aider Ask session...")

    # Capture stdout/stderr during coder run
    import sys
    from io import StringIO

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    stdout_capture = StringIO()
    stderr_capture = StringIO()

    try:
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture
        # Run the ask session - this should return the explanation
        result = coder.run(ai_coding_prompt)
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    captured_stdout = stdout_capture.getvalue()
    captured_stderr = stderr_capture.getvalue()

    if captured_stdout:
        logger.debug(f"Captured stdout from Aider Ask: {captured_stdout[:200]}...")
    if captured_stderr:
        logger.debug(f"Captured stderr from Aider Ask: {captured_stderr[:200]}...")

    logger.info(f"Ask session completed, result type: {type(result)}")
    
    # The result should be the explanation text
    if result is None:
        return "Ask mode completed but no response was returned."
    
    return str(result)


async def _execute_ask_with_retry(
    ai_coding_prompt: str,
    abs_readonly_files: List[str],
    working_dir: str,
    model: str,
    provider: str,
    architect_mode: bool = False,
    coordinator: Optional["IApplicationCoordinator"] = None,
) -> AskResponseDict:
    """Execute Aider Ask Mode with retry logic for rate limit handling."""
    response: AskResponseDict = {
        "success": False,
        "response": "No response generated.",
        "rate_limit_info": {"encountered": False, "retries": 0, "fallback_model": None},
    }

    max_retries = fallback_config.get(provider, {}).get("max_retries", 3)
    initial_delay = fallback_config.get(provider, {}).get("initial_delay", 1)
    backoff_factor = fallback_config.get(provider, {}).get("backoff_factor", 2)
    current_model = model

    for attempt in range(max_retries + 1):
        try:
            ai_model = _configure_model(current_model, architect_mode)
            coder = _setup_aider_ask_coder(ai_model, working_dir, abs_readonly_files, architect_mode)
            ask_response = await _run_ask_session(coder, ai_coding_prompt)
            
            response["success"] = True
            response["response"] = ask_response
            break  # Successful execution, exit retry loop

        except Exception as e:
            should_retry, new_model_or_error = await _handle_ask_rate_limit_or_error(
                e, provider, attempt, max_retries, initial_delay, backoff_factor, current_model, response, coordinator
            )
            if should_retry:
                current_model = new_model_or_error  # This is the new model name
            else:
                raise  # Re-raise the exception

    return response


async def _handle_ask_rate_limit_or_error(
    e: Exception,
    provider: str,
    attempt: int,
    max_retries: int,
    initial_delay: float,
    backoff_factor: float,
    current_model: str,
    response: AskResponseDict,
    coordinator: Optional["IApplicationCoordinator"] = None,
) -> tuple[bool, str]:
    """Handle rate limits and other errors during Aider Ask execution."""
    logger.warning(f"Error during Aider Ask execution (Attempt {attempt + 1}/{max_retries + 1}): {str(e)}")

    # Ensure rate_limit_info exists
    rli = response.get("rate_limit_info")
    if not isinstance(rli, dict):
        rli = {"encountered": False, "retries": 0, "fallback_model": None}
        response["rate_limit_info"] = rli

    if not isinstance(rli.get("retries"), int):
        rli["retries"] = 0

    if detect_rate_limit_error(e, provider):
        logger.info(f"Rate limit detected for {provider}. Attempting fallback...")
        rli["encountered"] = True
        rli["retries"] = rli.get("retries", 0) + 1

        # Broadcast rate limit event
        if coordinator:
            try:
                delay = initial_delay * (backoff_factor**attempt)
                new_model = get_fallback_model(current_model, provider)
                await coordinator.broadcast_event(
                    EventTypes.AIDER_RATE_LIMIT_DETECTED,
                    {
                        "provider": provider,
                        "current_model": current_model,
                        "fallback_model": new_model,
                        "attempt": attempt + 1,
                        "max_retries": max_retries,
                        "estimated_delay": delay,
                        "error_message": str(e),
                        "timestamp": time.time(),
                        "will_retry": attempt < max_retries,
                        "mode": "ask",
                    },
                )
                logger.info("Broadcasted aider.rate_limit_detected event for Ask Mode")
            except Exception as broadcast_error:
                logger.warning(f"Failed to broadcast rate_limit_detected event: {broadcast_error}")

        if attempt < max_retries:
            delay = initial_delay * (backoff_factor**attempt)
            logger.info(f"Retrying Ask Mode after {delay:.2f} seconds...")
            await asyncio.sleep(delay)
            new_model = get_fallback_model(current_model, provider)
            rli["fallback_model"] = new_model
            logger.info(f"Falling back to model: {new_model}")
            return True, new_model
        else:
            error_msg = f"Max retries ({max_retries}) reached for rate limit in Ask Mode. Unable to complete request."
            logger.error(f"{error_msg} Last error: {str(e)}")
            response["success"] = False
            response["response"] = f"Error: {error_msg}"
            raise Exception(f"{error_msg} Last error: {str(e)}") from e
    else:
        # Non-rate-limit error
        error_msg = f"Unhandled error during Aider Ask execution: {str(e)}"
        logger.error(error_msg, exc_info=True)
        response["success"] = False
        response["response"] = f"Error: {error_msg}"
        raise  # Re-raise the original exception


async def ask_with_aider(
    ai_coding_prompt: str,
    relative_readonly_files: Optional[List[str]] = None,
    model: str = "gemini/gemini-2.5-flash-preview-04-17",
    working_dir: Optional[str] = None,
    architect_mode: bool = False,
    coordinator: Optional["IApplicationCoordinator"] = None,
) -> str:
    """
    Run Aider in Ask Mode to get explanations about code without making any changes.

    Args:
        ai_coding_prompt (str): The question or explanation request for Aider.
        relative_readonly_files (List[str], optional): List of files that can be read for context.
                                                     Defaults to [].
        model (str, optional): The model to use. Defaults to "gemini/gemini-2.5-flash-preview-04-17".
        working_dir: The working directory where files are located.
        architect_mode (bool, optional): Enable architect mode if needed. Defaults to False.
        coordinator (IApplicationCoordinator, optional): Coordinator instance for event broadcasting.

    Returns:
        str: JSON string containing 'success' and 'response' with the explanation.
    """
    if relative_readonly_files is None:
        relative_readonly_files = []

    logger.info("--- Starting ask_with_aider ---")
    logger.info(f"Question: '{ai_coding_prompt[:100]}...'")
    logger.info(f"Working directory: {working_dir}")
    logger.info(f"Readonly files ({len(relative_readonly_files)}): {relative_readonly_files[:3]}...")
    logger.info(f"Model: {model}")
    logger.info(f"Architect mode: {architect_mode}")

    # Validate working directory
    if not working_dir:
        error_msg = "Error: working_dir is required for ask_with_aider"
        logger.error(error_msg)
        return json.dumps(
            {
                "success": False,
                "response": error_msg,
                "api_key_status": check_api_keys(None),
            }
        )

    original_model = model
    normalized_model_name = _normalize_model_name(model)
    provider = _determine_provider(normalized_model_name)

    # Check API keys
    key_status = check_api_keys(working_dir)
    if not key_status["any_keys_found"]:
        error_msg = "Error: No API keys found for any provider. Please set at least one API key."
        logger.error(error_msg)
        return json.dumps(
            {
                "success": False,
                "response": error_msg,
                "api_key_status": key_status,
                "warnings": [error_msg],
            }
        )

    # Convert to absolute paths
    abs_readonly_files = _convert_to_absolute_paths(relative_readonly_files, working_dir)

    try:
        # Broadcast session start event
        if coordinator:
            try:
                await coordinator.broadcast_event(
                    EventTypes.AIDER_SESSION_STARTED,
                    {
                        "prompt": ai_coding_prompt[:100] + "..." if len(ai_coding_prompt) > 100 else ai_coding_prompt,
                        "readonly_files": relative_readonly_files,
                        "model": original_model,
                        "working_dir": working_dir,
                        "architect_mode": architect_mode,
                        "mode": "ask",
                        "timestamp": time.time(),
                    },
                )
                logger.info("Broadcasted aider.session_started event for Ask Mode")
            except Exception as e:
                logger.warning(f"Failed to broadcast session_started event: {e}")

        # Execute Ask Mode
        response = await _execute_ask_with_retry(
            ai_coding_prompt,
            abs_readonly_files,
            working_dir,
            normalized_model_name,
            provider,
            architect_mode,
            coordinator,
        )

        # Determine actual model used
        rate_limit_info = response.get("rate_limit_info")
        fallback_model = None
        if isinstance(rate_limit_info, dict):
            fallback_model = rate_limit_info.get("fallback_model")
        actual_model_used = str(fallback_model) if fallback_model else normalized_model_name

        # Broadcast session completion event
        if coordinator:
            try:
                # Prepare values for the event, handling possible None for rate_limit_info
                rate_limit_info_dict = response.get("rate_limit_info")
                rate_limit_encountered_for_event = False
                fallback_used_for_event = False
                if rate_limit_info_dict is not None:
                    rate_limit_encountered_for_event = bool(rate_limit_info_dict.get("encountered", False))
                    fallback_used_for_event = bool(rate_limit_info_dict.get("fallback_model"))

                await coordinator.broadcast_event(
                    EventTypes.AIDER_SESSION_COMPLETED,
                    {
                        "success": response.get("success", False),
                        "model_used": actual_model_used,
                        "original_model": original_model,
                        "rate_limit_encountered": rate_limit_encountered_for_event,
                        "fallback_used": fallback_used_for_event,
                        "mode": "ask",
                        "timestamp": time.time(),
                    },
                )
                logger.info("Broadcasted aider.session_completed event for Ask Mode")
            except Exception as e:
                logger.warning(f"Failed to broadcast session_completed event: {e}")

        # Add API key status to response
        response["api_key_status"] = {
            "available_providers": key_status.get("available_providers", []),
            "missing_providers": key_status.get("missing_providers", []),
            "requested_provider": provider,
            "used_provider": _determine_provider(actual_model_used),
            "original_model_requested": original_model,
            "actual_model_used": actual_model_used,
        }

        # Add warning if requested provider's key was missing
        actual_provider_used = _determine_provider(actual_model_used)
        if provider not in key_status.get("available_providers", []):
            warning_msg = (
                f"Warning: API key for the initially requested provider '{provider}' was missing. "
                f"The system used provider '{actual_provider_used}' with model '{actual_model_used}'."
            )
            if "warnings" not in response:
                response["warnings"] = []
            if warning_msg not in response["warnings"]:  # type: ignore
                response["warnings"].append(warning_msg)  # type: ignore

        formatted_response = json.dumps(response, indent=4)
        logger.info(f"ask_with_aider process completed. Success: {response.get('success', False)}")
        return formatted_response

    except Exception as e:
        logger.error(f"Critical Error in ask_with_aider: {str(e)}", exc_info=True)
        error_response = {
            "success": False,
            "response": f"Unhandled Error during Ask Mode execution: {str(e)}",
            "api_key_status": key_status,
        }
        return json.dumps(error_response, indent=4)
