import json
import os
import time
from typing import Dict, List, Tuple, TypedDict


class ProviderConfig(TypedDict):
    rate_limit_errors: List[str]
    backoff_factor: float
    initial_delay: float
    max_retries: int
    fallback_models: List[str]


# Default fallback configuration - this will be overridden by .rate-limit-fallback.json if available
DEFAULT_FALLBACK_CONFIG: Dict[str, ProviderConfig] = {
    "openai": {
        "rate_limit_errors": ["rate_limit_exceeded", "insufficient_quota"],
        "backoff_factor": 2.0,
        "initial_delay": 1.0,
        "max_retries": 5,
        "fallback_models": ["gpt-3.5-turbo", "gpt-3.5-turbo-16k"],
    },
    "anthropic": {
        "rate_limit_errors": ["rate_limit_exceeded"],
        "backoff_factor": 2.0,
        "initial_delay": 1.0,
        "max_retries": 5,
        "fallback_models": ["claude-3-sonnet-20240229", "claude-instant-1"],
    },
    "gemini": {
        "rate_limit_errors": ["quota_exceeded", "resource_exhausted"],
        "backoff_factor": 2.0,
        "initial_delay": 1.0,
        "max_retries": 5,
        "fallback_models": ["gemini-pro", "gemini-1.0-pro"],
    },
}

# Try to load config from .rate-limit-fallback.json
try:
    # First try to find .rate-limit-fallback.json in the current directory
    mcp_json_paths = [
        ".rate-limit-fallback.json",
        os.path.join(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                )
            ),
            ".rate-limit-fallback.json",
        ),
    ]

    mcp_json_path = None
    for path in mcp_json_paths:
        if os.path.exists(path):
            mcp_json_path = path
            break

    if mcp_json_path:
        with open(mcp_json_path, "r") as f:
            config_data = json.load(f)
            if "fallback_config" in config_data:
                DEFAULT_FALLBACK_CONFIG.update(config_data["fallback_config"])
except Exception as e:
    print(f"Error loading fallback config from .rate-limit-fallback.json: {e}")


def detect_rate_limit_error(error: Exception, provider: str) -> bool:
    """
    Detect if the given error is a rate limit error for the specified provider.

    Args:
        error: The exception to check
        provider: The provider (openai, anthropic, gemini)

    Returns:
        True if it's a rate limit error, False otherwise
    """
    error_message = str(error).lower()

    # Standardize provider name
    if provider.lower() == "google":
        provider = "gemini"

    # Get the rate limit error patterns for this provider
    if provider not in DEFAULT_FALLBACK_CONFIG:
        return False

    error_patterns = DEFAULT_FALLBACK_CONFIG[provider]["rate_limit_errors"]
    return any(err.lower() in error_message for err in error_patterns)


def get_retry_delay(attempt: int, provider: str) -> float:
    """
    Calculate the retry delay using exponential backoff.

    Args:
        attempt: The current retry attempt (0-indexed)
        provider: The provider (openai, anthropic, gemini)

    Returns:
        The delay in seconds before the next retry
    """
    if provider not in DEFAULT_FALLBACK_CONFIG:
        # Use default values if provider not found
        return float(2.0 * (2**attempt))

    config = DEFAULT_FALLBACK_CONFIG[provider]
    backoff_factor = config["backoff_factor"]
    initial_delay = config["initial_delay"]
    delay = initial_delay * (backoff_factor**attempt)
    return min(delay, 60.0)  # Cap the delay at 60 seconds


def handle_rate_limit(provider: str, attempt: int) -> Tuple[bool, float]:
    """
    Handle rate limit error and return whether to retry and the delay.

    Args:
        provider: The provider (openai, anthropic, gemini)
        attempt: The current retry attempt (0-indexed)

    Returns:
        Tuple of (should_retry, delay_seconds)
    """
    if provider not in DEFAULT_FALLBACK_CONFIG:
        return False, 0.0

    config = DEFAULT_FALLBACK_CONFIG[provider]
    max_retries = config["max_retries"]
    if attempt < max_retries:
        delay = get_retry_delay(attempt, provider)
        return True, delay
    return False, 0.0


def apply_rate_limit_delay(delay: float) -> None:
    """
    Apply the rate limit delay by sleeping for the specified duration.

    Args:
        delay: The delay in seconds
    """
    time.sleep(delay)


def get_fallback_model(current_model: str, provider: str) -> str:
    """
    Get the next fallback model for the specified provider.

    Args:
        current_model: The current model that hit a rate limit
        provider: The provider (openai, anthropic, gemini)

    Returns:
        The next fallback model to try, or the current model if no fallbacks are available
    """
    if provider not in DEFAULT_FALLBACK_CONFIG:
        return current_model

    # Get the fallback models for this provider
    fallback_models = DEFAULT_FALLBACK_CONFIG[provider].get("fallback_models", [])

    # If there are no fallback models, return the current model
    if not fallback_models:
        return current_model

    # If the current model is in the list, return the next one
    try:
        idx = fallback_models.index(current_model)
        if idx + 1 < len(fallback_models):
            return fallback_models[idx + 1]
        else:
            return fallback_models[0]
    except (ValueError, IndexError):
        # If the current model is not in the list, return the first fallback model
        return fallback_models[0] if fallback_models else current_model
