import pytest
import logging
from pathlib import Path

from aider_mcp_server.atoms.logging import Logger, get_logger


def test_logger_creation_and_file_output(tmp_path):
    """Test Logger instance creation using get_logger and log file existence with fixed name."""
    log_dir = tmp_path / "logs"
    logger_name = "test_logger_creation"
    expected_log_file = log_dir / "aider_mcp_server.log" # Fixed log file name

    # --- Test get_logger ---
    logger = get_logger(
        name=logger_name,
        log_dir=log_dir,
        level=logging.INFO,
    )
    assert logger is not None, "Logger instance from get_logger should be created"
    assert logger.name == logger_name

    # Log a message to ensure file handling is triggered
    logger.info("Initial log message.")

    # Verify log directory and file exist
    assert log_dir.exists(), f"Log directory should be created by get_logger at {log_dir}"
    assert log_dir.is_dir(), f"Log path created by get_logger should be a directory"
    assert expected_log_file.exists(), f"Log file should be created by get_logger at {expected_log_file}"
    assert expected_log_file.is_file(), f"Log path created by get_logger should point to a file"


def test_log_levels_and_output(tmp_path):
    """Test logging at different levels to the fixed log file using get_logger."""
    log_dir = tmp_path / "logs"
    logger_name = "test_logger_levels"
    expected_log_file = log_dir / "aider_mcp_server.log" # Fixed log file name

    # Instantiate our custom logger with DEBUG level using get_logger
    logger = get_logger(
        name=logger_name,
        log_dir=log_dir,
        level=logging.DEBUG,
    )

    # Log messages at different levels
    messages = {
        logging.DEBUG: "This is a debug message.",
        logging.INFO: "This is an info message.",
        logging.WARNING: "This is a warning message.",
        logging.ERROR: "This is an error message.",
        logging.CRITICAL: "This is a critical message.",
    }

    logger.debug(messages[logging.DEBUG])
    logger.info(messages[logging.INFO])
    logger.warning(messages[logging.WARNING])
    logger.error(messages[logging.ERROR])
    logger.critical(messages[logging.CRITICAL])

    # Verify file output
    assert expected_log_file.exists(), "Log file should exist for level testing"

    file_content = expected_log_file.read_text()

    # Verify file output contains messages and level indicators
    for level, msg in messages.items():
        level_name = logging.getLevelName(level)
        assert msg in file_content, f"Message '{msg}' not found in file content"
        assert level_name in file_content, f"Level '{level_name}' not found in file content"
        assert logger_name in file_content, f"Logger name '{logger_name}' not found in file content"


def test_log_level_filtering(tmp_path):
    """Test that messages below the set log level are filtered using get_logger."""
    log_dir = tmp_path / "logs"
    logger_name = "test_logger_filtering"
    expected_log_file = log_dir / "aider_mcp_server.log" # Fixed log file name

    # Instantiate the logger with WARNING level using get_logger
    logger = get_logger(
        name=logger_name,
        log_dir=log_dir,
        level=logging.WARNING,
    )

    # Log messages at different levels
    debug_msg = "This debug message should NOT appear."
    info_msg = "This info message should NOT appear."
    warning_msg = "This warning message SHOULD appear."
    error_msg = "This error message SHOULD appear."
    critical_msg = "This critical message SHOULD appear." # Add critical for completeness

    logger.debug(debug_msg)
    logger.info(info_msg)
    logger.warning(warning_msg)
    logger.error(error_msg)
    logger.critical(critical_msg)

    # Verify file output filtering
    assert expected_log_file.exists(), "Log file should exist for filtering testing"

    file_content = expected_log_file.read_text()

    assert debug_msg not in file_content, "Debug message should be filtered from file"
    assert info_msg not in file_content, "Info message should be filtered from file"
    assert warning_msg in file_content, "Warning message should appear in file"
    assert error_msg in file_content, "Error message should appear in file"
    assert critical_msg in file_content, "Critical message should appear in file"
    assert logging.getLevelName(logging.DEBUG) not in file_content, "DEBUG level indicator should be filtered from file"
    assert logging.getLevelName(logging.INFO) not in file_content, "INFO level indicator should be filtered from file"
    assert logging.getLevelName(logging.WARNING) in file_content, "WARNING level indicator should appear in file"
    assert logging.getLevelName(logging.ERROR) in file_content, "ERROR level indicator should appear in file"
    assert logging.getLevelName(logging.CRITICAL) in file_content, "CRITICAL level indicator should appear in file"
    assert logger_name in file_content, f"Logger name '{logger_name}' should appear in file content"


def test_log_appending(tmp_path):
    """Test that log messages are appended to the existing log file."""
    log_dir = tmp_path / "logs"
    logger_name_1 = "test_logger_append_1"
    logger_name_2 = "test_logger_append_2"
    expected_log_file = log_dir / "aider_mcp_server.log" # Fixed log file name

    # First logger instance and message
    logger1 = get_logger(
        name=logger_name_1,
        log_dir=log_dir,
        level=logging.INFO,
    )
    message1 = "First message to append."
    logger1.info(message1)

    # Ensure some time passes or context switches if needed, though file handler should manage appending
    # Second logger instance (or could reuse logger1) and message
    logger2 = get_logger(
        name=logger_name_2, # Can use a different name or the same
        log_dir=log_dir,
        level=logging.INFO,
    )
    message2 = "Second message to append."
    logger2.info(message2)

    # Verify both messages are in the file
    assert expected_log_file.exists(), "Log file should exist for appending test"
    file_content = expected_log_file.read_text()

    assert message1 in file_content, "First message not found in appended log file"
    assert logger_name_1 in file_content, "First logger name not found in appended log file"
    assert message2 in file_content, "Second message not found in appended log file"
    assert logger_name_2 in file_content, "Second logger name not found in appended log file"
