import enum


class EventTypes(enum.Enum):
    """
    Defines the types of events that can be sent between the server and clients.
    These are used by transport adapters (like SSE, WebSocket) and the coordinator.
    """

    STATUS = "status"  # General status updates (e.g., request starting, completed, failed)
    PROGRESS = "progress"  # Detailed progress updates during an operation
    TOOL_RESULT = "tool_result"  # Final result from a tool execution (success or failure)
    HEARTBEAT = "heartbeat"  # Periodic message to keep the connection alive

    # AIDER-specific event types for real-time streaming
    AIDER_SESSION_STARTED = "aider.session_started"  # AIDER session begins
    AIDER_SESSION_PROGRESS = "aider.session_progress"  # AIDER progress updates
    AIDER_SESSION_COMPLETED = "aider.session_completed"  # AIDER session completes
    AIDER_RATE_LIMIT_DETECTED = "aider.rate_limit_detected"  # Rate limit encountered
    AIDER_THROTTLING_DETECTED = "aider.throttling_detected"  # Long-running request detected
    AIDER_ERROR_OCCURRED = "aider.error_occurred"  # AIDER error events

    # Add other event types here as needed
    # For example:
    # LOG = "log"               # Server-side log messages
    # NOTIFICATION = "notification" # General notifications
