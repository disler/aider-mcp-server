"""
Fallback logger factory utilities to eliminate duplication across modules.

This module provides a standardized fallback logger pattern that was
previously duplicated across multiple files in the codebase.
"""

import logging
import typing
from typing import Any

from aider_mcp_server.atoms.types.mcp_types import LoggerFactory, LoggerProtocol


def get_logger_with_fallback(module_name: str) -> LoggerProtocol:
    """
    Get a logger with fallback capability for the specified module.

    This function attempts to use the custom logger from atoms.logging.logger,
    but falls back to a standard logger implementation if not available.

    Args:
        module_name: Name of the module requesting the logger

    Returns:
        LoggerProtocol: A logger instance compatible with the expected interface
    """
    try:
        from aider_mcp_server.atoms.logging.logger import get_logger as custom_get_logger

        return typing.cast(LoggerProtocol, custom_get_logger(module_name))
    except ImportError:
        return _create_fallback_logger(module_name)


def _create_fallback_logger(name: str, *args: Any, **kwargs: Any) -> LoggerProtocol:
    """
    Create a fallback logger implementation.

    This provides the same logger pattern that was duplicated across multiple
    files in the codebase.
    """
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        if logger.level == logging.NOTSET:
            logger.setLevel(logging.INFO)

    class CustomLogger(LoggerProtocol):
        def debug(self, message: str, **kwargs: Any) -> None:
            logger.debug(message, **kwargs)

        def info(self, message: str, **kwargs: Any) -> None:
            logger.info(message, **kwargs)

        def warning(self, message: str, **kwargs: Any) -> None:
            logger.warning(message, **kwargs)

        def error(self, message: str, **kwargs: Any) -> None:
            logger.error(message, **kwargs)

        def critical(self, message: str, **kwargs: Any) -> None:
            logger.critical(message, **kwargs)

        def exception(self, message: str, **kwargs: Any) -> None:
            logger.exception(message, **kwargs)

        def verbose(self, message: str, **kwargs: Any) -> None:
            logger.debug(message, **kwargs)

    return CustomLogger()


def get_fallback_logger_factory() -> LoggerFactory:
    """
    Get a logger factory function for modules that need the factory pattern.

    Returns:
        LoggerFactory: A factory function that creates loggers with fallback
    """
    try:
        from aider_mcp_server.atoms.logging.logger import get_logger as custom_get_logger

        return typing.cast(LoggerFactory, custom_get_logger)
    except ImportError:
        return _create_fallback_logger
