import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from ...atoms.logging.logger import get_logger

# Use absolute imports from the package root
from ...atoms.security.context import SecurityContext
from ...atoms.types.mcp_types import (
    OperationResult,
    RequestParameters,
)
from ...molecules.tools.aider_ai_code import code_with_aider
from ...molecules.tools.aider_list_models import list_models

if TYPE_CHECKING:
    from ...organisms.coordinators.transport_coordinator import ApplicationCoordinator

# Configure logging for this module
logger = get_logger(__name__)


def _validate_ai_coding_prompt(
    request_id: str, params: RequestParameters
) -> Tuple[Optional[str], Optional[OperationResult]]:
    """Validate the AI coding prompt parameter."""
    ai_coding_prompt = params.get("ai_coding_prompt")
    if not ai_coding_prompt or not isinstance(ai_coding_prompt, str):
        logger.error(f"Request {request_id}: Missing or invalid 'ai_coding_prompt'.")
        return None, {
            "success": False,
            "error": "Missing or invalid 'ai_coding_prompt' parameter.",
        }
    return ai_coding_prompt, None


def _validate_editable_files(
    request_id: str, params: RequestParameters
) -> Tuple[Optional[List[str]], Optional[OperationResult]]:
    """Validate the editable files parameter."""
    relative_editable_files_raw = params.get("relative_editable_files")
    if not relative_editable_files_raw or not isinstance(relative_editable_files_raw, list):
        logger.error(f"Request {request_id}: Missing or invalid 'relative_editable_files'. Expected list.")
        return None, {
            "success": False,
            "error": "Missing or invalid 'relative_editable_files' parameter. Expected list.",
        }

    # Ensure all items in the list are strings
    if not all(isinstance(f, str) for f in relative_editable_files_raw):
        logger.error(f"Request {request_id}: 'relative_editable_files' must contain only strings.")
        return None, {
            "success": False,
            "error": "'relative_editable_files' must contain only strings.",
        }

    return relative_editable_files_raw, None


def _process_readonly_files(request_id: str, params: RequestParameters) -> List[str]:
    """Process and validate the readonly files parameter."""
    relative_readonly_files_raw = params.get("relative_readonly_files", [])

    if not isinstance(relative_readonly_files_raw, list):
        logger.warning(
            f"Request {request_id}: Invalid 'relative_readonly_files' format (expected list, got {type(relative_readonly_files_raw)}). Using empty list."
        )
        return []

    if not all(isinstance(f, str) for f in relative_readonly_files_raw):
        logger.warning(
            f"Request {request_id}: 'relative_readonly_files' contained non-string elements. Filtering them out."
        )
        return [f for f in relative_readonly_files_raw if isinstance(f, str)]

    return relative_readonly_files_raw


def _determine_model(request_id: str, params: RequestParameters, editor_model: str) -> str:
    """Determine which model to use based on request parameters and defaults."""
    request_model = params.get("model")

    if request_model and not isinstance(request_model, str):
        logger.warning(
            f"Request {request_id}: Invalid 'model' parameter type (expected string, got {type(request_model)}). Ignoring."
        )
        request_model = None

    return request_model if request_model else editor_model


async def _execute_aider_code(
    request_id: str,
    ai_coding_prompt: str,
    relative_editable_files: List[str],
    relative_readonly_files: List[str],
    model_to_use: str,
    current_working_dir: str,
    params: RequestParameters,
    coordinator: Optional["ApplicationCoordinator"] = None,
) -> OperationResult:
    """Execute the Aider code generation and process the results."""
    try:
        # Extract additional optional parameters
        architect_mode = params.get("architect_mode", False)
        editor_model = params.get("editor_model", None)
        auto_accept_architect = params.get("auto_accept_architect", True)
        include_raw_diff = params.get("include_raw_diff", False)

        # Log the architect mode configuration
        if architect_mode:
            logger.info(f"Request {request_id}: Using architect mode with editor model: {editor_model or model_to_use}")

        # Call the underlying tool function which returns a JSON string
        # The actual implementation always returns a string
        result_json_str = await code_with_aider(
            ai_coding_prompt=ai_coding_prompt,
            relative_editable_files=relative_editable_files,
            relative_readonly_files=relative_readonly_files,
            model=model_to_use,
            working_dir=current_working_dir,
            architect_mode=architect_mode,
            editor_model=editor_model,
            auto_accept_architect=auto_accept_architect,
            include_raw_diff=include_raw_diff,
            coordinator=coordinator,
        )

        result = _parse_aider_result(request_id, result_json_str)

        # If there are warnings in the response related to API keys, log them
        if "warnings" in result:
            for warning in result["warnings"]:
                if "API key" in warning or "api key" in warning.lower():
                    logger.warning(f"Request {request_id}: API key warning: {warning}")

            # Add a user-friendly message if there are API key warnings
            if any("API key" in warning or "api key" in warning.lower() for warning in result["warnings"]):
                result["user_message"] = "⚠️ API key issue detected. Some features may not work as expected."

        # If there's API key information, log it at debug level
        if "api_key_status" in result:
            logger.debug(f"Request {request_id}: API key status: {result['api_key_status']}")

        return result

    except Exception as e:
        logger.exception(f"Request {request_id}: Unhandled exception during code_with_aider execution: {e}")
        return {
            "success": False,
            "error": "An unexpected error occurred during AI coding",
            "details": str(e),
        }


def _parse_aider_result(request_id: str, result_json_str: str) -> OperationResult:
    """Parse the JSON result from the Aider code generation."""
    try:
        result_dict = json.loads(result_json_str)

        # Ensure the result is a dictionary
        if not isinstance(result_dict, dict):
            raise json.JSONDecodeError("Result is not a JSON object", result_json_str, 0)

        # Ensure 'success' field exists in the final dictionary
        if "success" not in result_dict:
            logger.warning(
                f"Request {request_id}: 'success' field missing in code_with_aider result. Assuming failure."
            )
            result_dict["success"] = False

        logger.info(f"Request {request_id}: AI Coding completed - Success: {result_dict.get('success')}")
        return result_dict

    except json.JSONDecodeError as e:
        logger.error(f"Request {request_id}: Failed to parse JSON response from code_with_aider: {e}")
        logger.error(f"Request {request_id}: Received raw response: {result_json_str}")
        return {
            "success": False,
            "error": "Failed to process AI coding result",
            "details": str(e),
        }


async def process_aider_ai_code_request(
    request_id: str,
    transport_id: str,
    params: RequestParameters,
    security_context: SecurityContext,
    use_diff_cache: bool = True,
    clear_cached_for_unchanged: bool = True,
    editor_model: str = "",  # Passed by coordinator/server based on config
    current_working_dir: str = "",  # Passed by coordinator/server based on config
    coordinator: Optional["ApplicationCoordinator"] = None,  # For event broadcasting
) -> Dict[str, Any]:
    """
    Process an aider_ai_code request.

    Args:
        request_id (str): The unique ID for this request.
        transport_id (str): The ID of the transport that initiated the request.
        params (Dict[str, Any]): The request parameters.
        security_context (SecurityContext): Security context for the request.
        editor_model (str): The default editor model configured for the server instance.
        current_working_dir (str): The validated working directory for the server instance.

    Returns:
        Dict[str, Any]: The response data including 'success' field.
    """
    logger.debug(
        f"Handler 'process_aider_ai_code_request' invoked for request_id: {request_id}, transport_id: {transport_id}"
    )
    logger.debug(f"Security Context: {security_context}")  # Log context at debug

    # Validate AI coding prompt
    ai_coding_prompt, error = _validate_ai_coding_prompt(request_id, params)
    if error:
        return error

    # Validate editable files
    relative_editable_files, error = _validate_editable_files(request_id, params)
    if error:
        return error

    # Process readonly files
    relative_readonly_files = _process_readonly_files(request_id, params)

    # Determine which model to use
    model_to_use = _determine_model(request_id, params, editor_model)

    # Log request details
    logger.debug(
        f"Request {request_id}: AI Coding Request: Prompt='{ai_coding_prompt[:50] if ai_coding_prompt else ''}...'"
    )
    logger.debug(f"Request {request_id}: Editable files: {relative_editable_files}")
    logger.debug(f"Request {request_id}: Readonly files: {relative_readonly_files}")
    logger.debug(f"Request {request_id}: Model to use: {model_to_use} (Editor default: {editor_model})")
    logger.debug(f"Request {request_id}: Working directory: {current_working_dir}")

    # Execute the Aider code generation
    if ai_coding_prompt is None or relative_editable_files is None:
        return {
            "success": False,
            "error": "Missing required parameters",
            "details": "AI coding prompt or editable files are missing",
        }

    return await _execute_aider_code(
        request_id,
        ai_coding_prompt,
        relative_editable_files,
        relative_readonly_files,
        model_to_use,
        current_working_dir,
        params,
        coordinator,
    )


async def process_list_models_request(
    request_id: str,
    transport_id: str,
    params: RequestParameters,
    security_context: SecurityContext,
    use_diff_cache: bool = True,
    clear_cached_for_unchanged: bool = True,
) -> OperationResult:
    """
    Process a list_models request.

    Args:
        request_id (str): The unique ID for this request.
        transport_id (str): The ID of the transport that initiated the request.
        params (Dict[str, Any]): The request parameters.
        security_context (SecurityContext): Security context for the request.

    Returns:
        Dict[str, Any]: The response data containing the list of models and 'success' field.
    """
    logger.debug(
        f"Handler 'process_list_models_request' invoked for request_id: {request_id}, transport_id: {transport_id}"
    )
    logger.debug(f"Security Context: {security_context}")

    # Substring is optional, default to empty string
    substring = params.get("substring", "")
    if not isinstance(substring, str):
        logger.warning(
            f"Request {request_id}: Invalid 'substring' parameter type (expected string, got {type(substring)}). Using empty string."
        )
        substring = ""

    logger.debug(f"Request {request_id}: List Models Request: Substring='{substring}'")

    try:
        models = list_models(substring)
        logger.debug(f"Request {request_id}: Found {len(models)} models matching '{substring}'")
        return {"models": models, "success": True}
    except Exception as e:
        logger.exception(f"Request {request_id}: Error during list_models execution: {e}")
        return {
            "models": [],
            "success": False,
            "error": "Failed to list models",
            "details": str(e),
        }
