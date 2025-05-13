def _setup_mock_reporter_context_for_reporter(
    coordinator, request_id_val, operation_name_val
):
    """Set up mock reporter context for reporter handler test."""
    mock_reporter_context = coordinator.get_progress_reporter.return_value

    async def reporter_update_side_effect(message, status="in_progress", details=None):
        await coordinator.update_request(request_id_val, status, message, details)

    mock_reporter_context.update.side_effect = reporter_update_side_effect

    async def reporter_aenter_side_effect():
        await coordinator.update_request(
            request_id_val,
            "starting",
            f"Operation '{operation_name_val}' started.",
            None,
        )
        return mock_reporter_context

    mock_reporter_context.__aenter__.side_effect = reporter_aenter_side_effect

    async def reporter_aexit_side_effect(exc_type, exc_val, exc_tb):
        if exc_type is None:
            await coordinator.update_request(
                request_id_val, "completed", "Operation completed successfully.", None
            )
        else:
            error_msg = f"Operation '{operation_name_val}' failed: {exc_val}"
            await coordinator.update_request(
                request_id_val,
                "error",
                error_msg,
                {"exception_type": str(exc_type.__name__)},
            )

    mock_reporter_context.__aexit__.side_effect = reporter_aexit_side_effect

    return mock_reporter_context
