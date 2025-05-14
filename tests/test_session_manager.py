import asyncio
from unittest.mock import AsyncMock, MagicMock
from datetime import timedelta  # Added import for timedelta

import pytest

from aider_mcp_server.session_manager import Session, SessionManager
from aider_mcp_server.security import Permissions

@pytest.fixture
def session_manager():
    return SessionManager(start_cleanup=False)

@pytest.mark.asyncio
async def test_get_or_create_session(session_manager):
    # Test getting or creating a session
    transport_id = "test_transport_id"
    session = await session_manager.get_or_create_session(transport_id)

    assert session.transport_id == transport_id
    assert session.creation_time is not None
    assert session.last_accessed_time is not None
    assert session.user_info is None
    assert session.permissions == set()
    assert session.custom_data == {}

@pytest.mark.asyncio
async def test_update_session(session_manager):
    # Test updating a session
    transport_id = "test_transport_id"
    data = {"key": "value"}
    session = await session_manager.get_or_create_session(transport_id)
    updated_session = await session_manager.update_session(transport_id, data)

    assert updated_session is not None
    assert updated_session.custom_data == data

@pytest.mark.asyncio
async def test_remove_session(session_manager):
    # Test removing a session
    transport_id = "test_transport_id"
    await session_manager.get_or_create_session(transport_id)
    await session_manager.remove_session(transport_id)

    sessions = await session_manager.get_all_sessions()
    assert transport_id not in sessions

@pytest.mark.asyncio
async def test_check_permission(session_manager):
    # Test checking permissions
    transport_id = "test_transport_id"
    permissions = {Permissions.EXECUTE_AIDER, Permissions.VIEW_CONFIG}
    await session_manager.get_or_create_session(transport_id)
    session = await session_manager.set_permissions(transport_id, permissions)

    has_execute_aider_permission = await session_manager.check_permission(transport_id, Permissions.EXECUTE_AIDER)
    has_view_config_permission = await session_manager.check_permission(transport_id, Permissions.VIEW_CONFIG)
    has_list_models_permission = await session_manager.check_permission(transport_id, Permissions.LIST_MODELS)

    assert has_execute_aider_permission
    assert has_view_config_permission
    assert not has_list_models_permission

@pytest.mark.asyncio
async def test_cleanup_expired_sessions(session_manager):
    # Test cleaning up expired sessions
    transport_id = "test_transport_id"
    session = await session_manager.get_or_create_session(transport_id)
    session.last_accessed_time = session.creation_time - timedelta(seconds=session_manager.session_timeout + 1)

    # Manually call the cleanup method instead of waiting for the task
    await session_manager.cleanup_expired_sessions(run_once=True)
    
    # Check that the session was removed
    sessions = await session_manager.get_all_sessions()
    assert transport_id not in sessions

@pytest.mark.asyncio
async def test_get_all_sessions(session_manager):
    # Test getting all sessions
    transport_id1 = "test_transport_id1"
    transport_id2 = "test_transport_id2"
    await session_manager.get_or_create_session(transport_id1)
    await session_manager.get_or_create_session(transport_id2)

    sessions = await session_manager.get_all_sessions()

    assert transport_id1 in sessions
    assert transport_id2 in sessions

@pytest.mark.asyncio
async def test_set_permissions(session_manager):
    # Test setting permissions
    transport_id = "test_transport_id"
    permissions = {Permissions.EXECUTE_AIDER, Permissions.VIEW_CONFIG}
    await session_manager.get_or_create_session(transport_id)
    session = await session_manager.set_permissions(transport_id, permissions)

    assert session is not None
    assert session.permissions == permissions
