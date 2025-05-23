import asyncio
import unittest
from unittest.mock import MagicMock, patch

from aider_mcp_server.atoms.logging import Logger as ProjectLogger
from aider_mcp_server.error_handling import (
    AiderMCPError,
    ConfigurationError,
    ErrorHandler,
    EventError,
    HandlerError,
    InitializationError,
    TransportError,
)

# This global mock instance will be assigned to ErrorHandler._logger
# allowing us to control its behavior across all tests for ErrorHandler static methods.
mock_error_handler_class_logger = MagicMock(spec=ProjectLogger)


class TestErrorHandling(unittest.IsolatedAsyncioTestCase):
    original_error_handler_logger = None

    @classmethod
    def setUpClass(cls):
        # Replace ErrorHandler._logger with our mock for the duration of these tests.
        # This is crucial because ErrorHandler._logger is a class attribute initialized
        # when the error_handling.py module is loaded.
        cls.original_error_handler_logger = ErrorHandler._logger
        ErrorHandler._logger = mock_error_handler_class_logger

    @classmethod
    def tearDownClass(cls):
        # Restore the original ErrorHandler._logger.
        if cls.original_error_handler_logger is not None:
            ErrorHandler._logger = cls.original_error_handler_logger

    def setUp(self):
        # Reset the mock for ErrorHandler's internal logger before each test.
        mock_error_handler_class_logger.reset_mock()
        # Create a fresh mock for tests that pass a custom logger instance.
        self.mock_custom_logger = MagicMock(spec=ProjectLogger)

    def tearDown(self):
        # Stop any patchers that might have been started within test methods.
        patch.stopall()

    def test_custom_exception_hierarchy(self):
        self.assertIsInstance(TransportError("test"), AiderMCPError)
        self.assertIsInstance(HandlerError("test"), AiderMCPError)
        self.assertIsInstance(EventError("test"), AiderMCPError)
        self.assertIsInstance(InitializationError("test"), AiderMCPError)
        self.assertIsInstance(ConfigurationError("test"), AiderMCPError)
        self.assertIsInstance(AiderMCPError("test"), Exception)

        exceptions_to_test = [
            AiderMCPError,
            TransportError,
            HandlerError,
            EventError,
            InitializationError,
            ConfigurationError,
        ]
        for exc_type in exceptions_to_test:
            with self.assertRaises(exc_type):
                raise exc_type("Specific message for " + exc_type.__name__)

    def _raise_and_format_exception(self, exc_to_raise: Exception):
        try:
            raise exc_to_raise
        except Exception as e:
            return ErrorHandler.format_exception(e)

    def test_format_exception_standard_exception(self):
        exception = ValueError("A standard error occurred")
        formatted_error = self._raise_and_format_exception(exception)

        self.assertEqual(formatted_error["type"], "error")
        self.assertEqual(formatted_error["error"]["type"], "ValueError")
        self.assertEqual(formatted_error["error"]["message"], "A standard error occurred")
        self.assertIn("Traceback (most recent call last):", formatted_error["error"]["traceback"])
        self.assertIn("raise exc_to_raise", formatted_error["error"]["traceback"])
        self.assertIn("ValueError: A standard error occurred", formatted_error["error"]["traceback"])

    def test_format_exception_custom_exception(self):
        exception = TransportError("A transport layer error")
        formatted_error = self._raise_and_format_exception(exception)

        self.assertEqual(formatted_error["type"], "error")
        self.assertEqual(formatted_error["error"]["type"], "TransportError")
        self.assertEqual(formatted_error["error"]["message"], "A transport layer error")
        self.assertIn("TransportError: A transport layer error", formatted_error["error"]["traceback"])

    def test_format_exception_unicode_message(self):
        unicode_message = "こんにちは世界"  # Hello world in Japanese
        exception = HandlerError(unicode_message)
        formatted_error = self._raise_and_format_exception(exception)

        self.assertEqual(formatted_error["error"]["message"], unicode_message)
        self.assertIn(f"HandlerError: {unicode_message}", formatted_error["error"]["traceback"])

    def test_format_exception_no_message(self):
        exception = AiderMCPError()  # Exception with no message
        formatted_error = self._raise_and_format_exception(exception)

        # str(AiderMCPError()) might be empty or the class name depending on __str__
        # For a generic Exception(), str(e) is often empty.
        # Let's assume it results in an empty string if no args passed.
        self.assertEqual(formatted_error["error"]["message"], "")
        self.assertIn("AiderMCPError", formatted_error["error"]["traceback"])

    def test_format_exception_none_message(self):
        exception = EventError(None)  # Exception with None as message
        formatted_error = self._raise_and_format_exception(exception)

        self.assertEqual(formatted_error["error"]["message"], "None")  # str(None) is "None"
        self.assertIn("EventError: None", formatted_error["error"]["traceback"])

    def test_log_exception_default_logger(self):
        exception = InitializationError("Init failed")
        ErrorHandler.log_exception(exception)

        mock_error_handler_class_logger.error.assert_called_once_with("Error: Init failed", exc_info=True)

    def test_log_exception_with_context(self):
        exception = ConfigurationError("Config parse error")
        context = "loading settings.json"
        ErrorHandler.log_exception(exception, context=context)

        mock_error_handler_class_logger.error.assert_called_once_with(
            f"Error in {context}: Config parse error", exc_info=True
        )

    def test_log_exception_custom_logger(self):
        exception = AiderMCPError("Generic MCP failure")
        ErrorHandler.log_exception(exception, logger_instance=self.mock_custom_logger)

        self.mock_custom_logger.error.assert_called_once_with("Error: Generic MCP failure", exc_info=True)
        mock_error_handler_class_logger.error.assert_not_called()  # Ensure default logger wasn't used

    def test_log_exception_no_context(self):  # Same as default logger test, explicit for clarity
        exception = TransportError("Connection lost")
        ErrorHandler.log_exception(exception, context=None)  # Explicitly pass None context

        mock_error_handler_class_logger.error.assert_called_once_with("Error: Connection lost", exc_info=True)

    @patch("aider_mcp_server.error_handling.ErrorHandler.format_exception")
    @patch("aider_mcp_server.error_handling.ErrorHandler.log_exception")
    def test_handle_exception(self, mock_log_exception, mock_format_exception):
        exception = HandlerError("Request handling failed")
        context = "processing user request"
        mock_formatted_response = {"type": "error", "error": {"type": "HandlerError", "message": "..."}}
        mock_format_exception.return_value = mock_formatted_response

        result = ErrorHandler.handle_exception(exception, context=context, logger_instance=self.mock_custom_logger)

        mock_log_exception.assert_called_once_with(exception, context, self.mock_custom_logger)
        mock_format_exception.assert_called_once_with(exception)
        self.assertEqual(result, mock_formatted_response)

    @patch("asyncio.get_event_loop")
    def test_install_global_exception_handler_and_handler_logic(self, mock_get_event_loop):
        # Scenario 1: Install with default loop and ErrorHandler._logger
        mock_loop_default = MagicMock(spec=asyncio.AbstractEventLoop)
        mock_get_event_loop.return_value = mock_loop_default

        ErrorHandler.install_global_exception_handler()

        mock_get_event_loop.assert_called_once()
        mock_loop_default.set_exception_handler.assert_called_once()
        handler_func_default_logger = mock_loop_default.set_exception_handler.call_args[0][0]
        self.assertTrue(callable(handler_func_default_logger))
        mock_error_handler_class_logger.info.assert_called_with(
            f"Global asyncio exception handler installed on loop {id(mock_loop_default)}."
        )
        mock_error_handler_class_logger.reset_mock()  # Reset after install log

        # Test the handler captured from Scenario 1 (uses ErrorHandler._logger)
        test_exception = ValueError("Async crash")
        mock_future = MagicMock()
        context_with_exc_and_future = {
            "message": "Unhandled error in task",
            "exception": test_exception,
            "future": mock_future,
        }
        handler_func_default_logger(mock_loop_default, context_with_exc_and_future)
        mock_error_handler_class_logger.error.assert_called_once_with(
            f"Unhandled exception in asyncio event loop (Future: {mock_future}): ValueError: Async crash",
            exc_info=test_exception,
        )
        mock_error_handler_class_logger.reset_mock()

        context_with_msg_only = {"message": "Some async issue, no exception object"}
        handler_func_default_logger(mock_loop_default, context_with_msg_only)
        mock_error_handler_class_logger.error.assert_called_once_with(
            "Unhandled exception in asyncio event loop: Some async issue, no exception object"
        )
        mock_error_handler_class_logger.reset_mock()

        context_with_exc_no_future = {
            "message": "Unhandled error in task",
            "exception": test_exception,  # No future
        }
        handler_func_default_logger(mock_loop_default, context_with_exc_no_future)
        mock_error_handler_class_logger.error.assert_called_once_with(
            "Unhandled exception in asyncio event loop: ValueError: Async crash",  # No future in log
            exc_info=test_exception,
        )
        mock_error_handler_class_logger.reset_mock()

        # Scenario 2: Install with custom loop and custom logger
        mock_loop_custom = MagicMock(spec=asyncio.AbstractEventLoop)

        ErrorHandler.install_global_exception_handler(loop=mock_loop_custom, logger_instance=self.mock_custom_logger)
        mock_loop_custom.set_exception_handler.assert_called_once()
        handler_func_custom_logger = mock_loop_custom.set_exception_handler.call_args[0][0]
        self.assertTrue(callable(handler_func_custom_logger))
        self.mock_custom_logger.info.assert_called_with(
            f"Global asyncio exception handler installed on loop {id(mock_loop_custom)}."
        )
        self.mock_custom_logger.reset_mock()  # Reset after install log

        # Test the handler captured from Scenario 2 (uses self.mock_custom_logger)
        handler_func_custom_logger(mock_loop_custom, context_with_exc_and_future)
        self.mock_custom_logger.error.assert_called_once_with(
            f"Unhandled exception in asyncio event loop (Future: {mock_future}): ValueError: Async crash",
            exc_info=test_exception,
        )

    def test_error_handler_internal_logger_is_mocked(self):
        # This test confirms that ErrorHandler._logger is indeed our global mock instance.
        self.assertIs(ErrorHandler._logger, mock_error_handler_class_logger)


if __name__ == "__main__":
    unittest.main()
