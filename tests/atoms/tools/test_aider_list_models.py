import pytest
from aider_mcp_server.atoms.tools.aider_list_models import list_models

def test_list_models_openai():
    """Test that list_models returns GPT-4o model when searching for openai."""
    models = list_models("openai")
    assert any("gpt-4o" in model for model in models), "Expected to find GPT-4o model in the list"
    
def test_list_models_gemini():
    """Test that list_models returns Gemini models when searching for gemini."""
    models = list_models("gemini")
    assert any("gemini" in model.lower() for model in models), "Expected to find Gemini models in the list"
    
def test_list_models_empty():
    """Test that list_models with an empty string returns all models."""
    models = list_models("")
    assert len(models) > 0, "Expected to get at least some models with empty string"
    
def test_list_models_nonexistent():
    """Test that list_models with a nonexistent model returns an empty list."""
    models = list_models("this_model_does_not_exist_12345")
    assert len(models) == 0, "Expected to get no models with a nonexistent model name"