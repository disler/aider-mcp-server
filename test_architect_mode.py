#!/usr/bin/env python3
"""Test Architecture Mode Implementation"""

import sys
sys.path.insert(0, 'src')

# Test the architecture mode parameter handling
def test_architecture_mode():
    print("=== TESTING ARCHITECTURE MODE ===")
    
    # Test 1: Check edit_format parameter handling
    print("\n1. Testing edit_format parameter:")
    
    # Simulate architect mode True
    architect_mode = True
    edit_format = "architect" if architect_mode else None
    print(f"  architect_mode=True -> edit_format='{edit_format}'")
    
    # Simulate architect mode False  
    architect_mode = False
    edit_format = "architect" if architect_mode else None
    print(f"  architect_mode=False -> edit_format='{edit_format}'")
    
    # Test 2: Check auto_accept_architect parameter
    print("\n2. Testing auto_accept_architect parameter:")
    
    architect_mode = True
    auto_accept_architect = True
    result = auto_accept_architect if architect_mode else True
    print(f"  architect_mode=True, auto_accept_architect=True -> {result}")
    
    architect_mode = False
    auto_accept_architect = True  
    result = auto_accept_architect if architect_mode else True
    print(f"  architect_mode=False, auto_accept_architect=True -> {result}")
    
    # Test 3: Check Model configuration for architect mode
    print("\n3. Testing Model configuration:")
    from aider.models import Model
    
    # Test with architect mode enabled
    model_name = "gemini/gemini-2.5-pro-preview-05-06"
    editor_model = "gemini/gemini-2.5-flash-preview-04-17"
    
    print(f"  Primary model: {model_name}")
    print(f"  Editor model: {editor_model}")
    
    # Test Model creation with editor model
    try:
        model_with_editor = Model(model_name, editor_model=editor_model)
        print(f"  ✅ Model with editor created successfully")
    except Exception as e:
        print(f"  ❌ Model with editor failed: {e}")
        
    # Test Model creation without editor model
    try:
        model_without_editor = Model(model_name)
        print(f"  ✅ Model without editor created successfully")
    except Exception as e:
        print(f"  ❌ Model without editor failed: {e}")

if __name__ == "__main__":
    test_architecture_mode()