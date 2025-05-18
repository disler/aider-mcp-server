#!/usr/bin/env python3
"""
Script to inspect actual parameters supported by aider's Coder class.
This helps us understand what parameters we can use.
"""

import inspect

from packaging import version

try:
    import aider
    from aider.coders import Coder
except ImportError:
    print("aider library not installed")
    exit(1)

print(f"aider version: {aider.__version__}")
print(f"aider path: {aider.__file__}")

# Check Coder.__init__ parameters
print("\nCoder.__init__ parameters:")
coder_init_sig = inspect.signature(Coder.__init__)
for param_name, param in coder_init_sig.parameters.items():
    if param_name == "self":
        continue
    print(
        f"  {param_name}: {param.annotation if param.annotation != inspect.Parameter.empty else 'Any'} = {param.default if param.default != inspect.Parameter.empty else 'REQUIRED'}"
    )

# Check Coder.create parameters
print("\nCoder.create parameters:")
coder_create_sig = inspect.signature(Coder.create)
for param_name, param in coder_create_sig.parameters.items():
    if param_name == "cls":
        continue
    print(
        f"  {param_name}: {param.annotation if param.annotation != inspect.Parameter.empty else 'Any'} = {param.default if param.default != inspect.Parameter.empty else 'REQUIRED'}"
    )

# Check actual source of create method to see what it accepts
try:
    create_source = inspect.getsource(Coder.create)
    print("\nCoder.create method source (first 50 lines):")
    lines = create_source.split("\n")[:50]
    for i, line in enumerate(lines):
        print(f"{i + 1:3d}: {line}")
except Exception as e:
    print(f"Could not get source: {e}")

# Check for any version-specific attributes
print("\nVersion-specific attributes:")
aider_version = version.parse(aider.__version__)
print(f"  Version object: {aider_version}")
print(f"  Major: {aider_version.major}")
print(f"  Minor: {aider_version.minor}")
print(f"  Micro: {aider_version.micro}")

# Let's also check what the actual implementation does with kwargs
print("\nChecking Coder base classes:")
for base in Coder.__mro__:
    print(f"  {base}")
    if hasattr(base, "__init__"):
        try:
            sig = inspect.signature(base.__init__)
            if "**kwargs" in str(sig) or "kwargs" in sig.parameters:
                print("    Has **kwargs support")
        except Exception:  # noqa: S110
            pass
