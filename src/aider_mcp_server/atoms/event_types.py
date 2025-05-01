import enum

class EventTypes(enum.Enum):
    """
    Defines the types of events that can be sent between the server and clients.
    These are used by transport adapters (like SSE, WebSocket) and the coordinator.
    """
    STATUS = "status"           # General status updates (e.g., request starting, completed, failed)
    PROGRESS = "progress"       # Detailed progress updates during an operation
    TOOL_RESULT = "tool_result" # Final result from a tool execution (success or failure)
    HEARTBEAT = "heartbeat"     # Periodic message to keep the connection alive

    # Add other event types here as needed
    # For example:
    # LOG = "log"               # Server-side log messages
    # NOTIFICATION = "notification" # General notifications
