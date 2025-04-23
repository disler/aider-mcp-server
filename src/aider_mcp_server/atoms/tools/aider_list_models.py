from typing import List

from aider.models import fuzzy_match_models


def list_models(substring: str) -> List[str]:
    """
    List available models that match the provided substring.
    
    Args:
        substring (str): Substring to match against available models.
    
    Returns:
        List[str]: List of model names matching the substring.
    """
    return fuzzy_match_models(substring)