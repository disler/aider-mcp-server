"""
Security permissions enumeration for the Aider MCP Server.

Defines the set of possible permissions within the system.
"""

import enum
from typing import Optional


class Permissions(enum.Enum):
    """Defines the set of possible permissions within the system."""

    EXECUTE_AIDER = "execute_aider"  # Permission to execute aider commands
    VIEW_CONFIG = "view_config"      # Permission to view configuration
    LIST_MODELS = "list_models"      # Permission to list available models
    READ = "read"                    # Basic read permission
    WRITE = "write"                  # Basic write permission

    @classmethod
    def from_string(cls, value: str) -> Optional["Permissions"]:
        """Convert a string value to a Permissions enum member."""
        for member in cls:
            if member.value == value:
                return member
        return None