#!/usr/bin/env python3
"""
Validate the multi-client project structure and dependencies.

This script verifies that all required directories and configuration
constants are properly set up for the multi-client HTTP server architecture.
"""

import sys
from pathlib import Path


def validate_project_structure():
    """Validate that all required directories exist."""
    required_dirs = [
        "src/aider_mcp_server/managers",
        "src/aider_mcp_server/utils/multi_client",
        "src/aider_mcp_server/organisms/multi_client",
        "tests/managers",
        "tests/utils/multi_client",
        "tests/organisms/multi_client",
    ]

    project_root = Path(__file__).parent.parent
    missing_dirs = []

    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if not full_path.exists():
            missing_dirs.append(dir_path)
        else:
            print(f"✅ {dir_path}")

    if missing_dirs:
        print(f"\n❌ Missing directories: {missing_dirs}")
        return False

    return True


def validate_configuration():
    """Validate that configuration constants are properly defined."""
    try:
        # Add project root to path
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root / "src"))

        from aider_mcp_server.atoms.utils.config_constants import (
            CLIENT_SESSION_TIMEOUT,
            DEFAULT_MANAGER_PORT,
            MAX_CONCURRENT_CLIENTS,
            MULTI_CLIENT_PORT_RANGE_END,
            MULTI_CLIENT_PORT_RANGE_START,
            WORKSPACE_BASE_DIR,
        )

        print(f"✅ Multi-client port range: {MULTI_CLIENT_PORT_RANGE_START}-{MULTI_CLIENT_PORT_RANGE_END}")
        print(f"✅ Max concurrent clients: {MAX_CONCURRENT_CLIENTS}")
        print(f"✅ Session timeout: {CLIENT_SESSION_TIMEOUT}s")
        print(f"✅ Workspace base dir: {WORKSPACE_BASE_DIR}")
        print(f"✅ Manager port: {DEFAULT_MANAGER_PORT}")

        return True

    except ImportError as e:
        print(f"❌ Configuration import failed: {e}")
        return False


def main():
    """Run all validation checks."""
    print("🔍 Validating Multi-Client HTTP Server Project Structure\n")

    print("📁 Checking directory structure...")
    structure_ok = validate_project_structure()

    print("\n⚙️ Checking configuration constants...")
    config_ok = validate_configuration()

    if structure_ok and config_ok:
        print("\n✅ Project structure validation PASSED")
        print("🚀 Ready for multi-client implementation!")
        return 0
    else:
        print("\n❌ Project structure validation FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
