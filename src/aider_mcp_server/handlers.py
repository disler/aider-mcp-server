import json
from typing import Any, Dict

from aider_mcp_server.atoms.logging import get_logger
from aider_mcp_server.atoms.tools.aider_ai_code import code_with_aider
from aider_mcp_server.atoms.tools.aider_list_models import list_models

# Use absolute imports from the package root
from aider_mcp_server.security import SecurityContext

# Configure logging for this module
logger = get_logger(__name__)


async def process_aider_ai_code_request(
    request_id: str,
    transport_id: str,
    params: Dict[str, Any],
    security_context: SecurityContext,
    editor_model: str, # Passed by coordinator/server based on config
    current_working_dir: str, # Passed by coordinator/server based on config
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
    logger.info(f"Handler 'process_aider_ai_code_request' invoked for request_id: {request_id}, transport_id: {transport_id}")
    logger.debug(f"Security Context: {security_context}") # Log context at debug

    # --- Parameter Extraction and Validation ---
    ai_coding_prompt = params.get("ai_coding_prompt")
    if not ai_coding_prompt or not isinstance(ai_coding_prompt, str):
        logger.error(f"Request {request_id}: Missing or invalid 'ai_coding_prompt'.")
        return {"success": False, "error": "Missing or invalid 'ai_coding_prompt' parameter."}

    relative_editable_files_raw = params.get("relative_editable_files")
    if not relative_editable_files_raw or not isinstance(relative_editable_files_raw, list):
        logger.error(f"Request {request_id}: Missing or invalid 'relative_editable_files'. Expected list.")
        return {"success": False, "error": "Missing or invalid 'relative_editable_files' parameter. Expected list."}
    # Ensure all items in the list are strings
    if not all(isinstance(f, str) for f in relative_editable_files_raw):
        logger.error(f"Request {request_id}: 'relative_editable_files' must contain only strings.")
        return {"success": False, "error": "'relative_editable_files' must contain only strings."}
    relative_editable_files = relative_editable_files_raw

    # Readonly files are optional, default to empty list
    relative_readonly_files_raw = params.get("relative_readonly_files", [])
    if not isinstance(relative_readonly_files_raw, list):
        logger.warning(f"Request {request_id}: Invalid 'relative_readonly_files' format (expected list, got {type(relative_readonly_files_raw)}). Using empty list.")
        relative_readonly_files = []
    elif not all(isinstance(f, str) for f in relative_readonly_files_raw):
         logger.warning(f"Request {request_id}: 'relative_readonly_files' contained non-string elements. Filtering them out.")
         relative_readonly_files = [f for f in relative_readonly_files_raw if isinstance(f, str)]
    else:
        relative_readonly_files = relative_readonly_files_raw

    # Model is optional, use editor_model if not provided
    request_model = params.get("model")
    if request_model and not isinstance(request_model, str):
        logger.warning(f"Request {request_id}: Invalid 'model' parameter type (expected string, got {type(request_model)}). Ignoring.")
        request_model = None

    model_to_use = request_model if request_model else editor_model

    # --- Logging ---
    logger.info(f"Request {request_id}: AI Coding Request: Prompt='{ai_coding_prompt[:50]}...'")
    logger.info(f"Request {request_id}: Editable files: {relative_editable_files}")
    logger.info(f"Request {request_id}: Readonly files: {relative_readonly_files}")
    logger.info(f"Request {request_id}: Model to use: {model_to_use} (Editor default: {editor_model})")
    logger.info(f"Request {request_id}: Working directory: {current_working_dir}")

    # --- Execute Tool ---
    try:
        # Call the underlying tool function - properly await the async function
        result_json_str = await code_with_aider(
            ai_coding_prompt=ai_coding_prompt,
            relative_editable_files=relative_editable_files,
            relative_readonly_files=relative_readonly_files,
            model=model_to_use,
            working_dir=current_working_dir,
        )

        # Parse the JSON string result from the tool
        try:
            # Ensure result_json_str is actually a string
            if not isinstance(result_json_str, str):
                logger.error(f"Request {request_id}: Expected string from code_with_aider, got {type(result_json_str)}")
                return {"success": False, "error": "Unexpected response type from code_with_aider", "details": f"Expected string, got {type(result_json_str)}"}
                
            result_dict = json.loads(result_json_str)
            # Ensure the result is a dictionary
            if not isinstance(result_dict, dict):
                 raise json.JSONDecodeError("Result is not a JSON object", result_json_str, 0)
        except json.JSONDecodeError as e:
            logger.error(f"Request {request_id}: Failed to parse JSON response from code_with_aider: {e}")
            logger.error(f"Request {request_id}: Received raw response: {result_json_str}")
            return {"success": False, "error": "Failed to process AI coding result", "details": str(e)}

        # Ensure 'success' field exists in the final dictionary
        if "success" not in result_dict:
            logger.warning(f"Request {request_id}: 'success' field missing in code_with_aider result. Assuming failure.")
            result_dict["success"] = False

        logger.info(f"Request {request_id}: AI Coding Request Completed. Success: {result_dict.get('success')}")
        return result_dict

    except Exception as e:
        logger.exception(f"Request {request_id}: Unhandled exception during code_with_aider execution: {e}")
        return {"success": False, "error": "An unexpected error occurred during AI coding", "details": str(e)}


async def process_list_models_request(
    request_id: str,
    transport_id: str,
    params: Dict[str, Any],
    security_context: SecurityContext,
) -> Dict[str, Any]:
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
    logger.info(f"Handler 'process_list_models_request' invoked for request_id: {request_id}, transport_id: {transport_id}")
    logger.debug(f"Security Context: {security_context}")

    # Substring is optional, default to empty string
    substring = params.get("substring", "")
    if not isinstance(substring, str):
        logger.warning(f"Request {request_id}: Invalid 'substring' parameter type (expected string, got {type(substring)}). Using empty string.")
        substring = ""

    logger.info(f"Request {request_id}: List Models Request: Substring='{substring}'")

    try:
        models = list_models(substring)
        logger.info(f"Request {request_id}: Found {len(models)} models matching '{substring}'")
        return {"models": models, "success": True}
    except Exception as e:
        logger.exception(f"Request {request_id}: Error during list_models execution: {e}")
        return {"models": [], "success": False, "error": "Failed to list models", "details": str(e)}
