"""Unit tests for SessionManager class."""

import pytest
import time
import threading
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.agents.core.session_manager import SessionManager, SessionInfo


class TestSessionInfoModel:
    """Tests for SessionInfo Pydantic model."""

    def test_session_info_creation(self):
        """Test SessionInfo model creation."""
        now = datetime.now()
        session = SessionInfo(
            session_id="test-123",
            created_at=now,
            last_activity=now,
            expires_at=now + timedelta(minutes=30),
        )

        assert session.session_id == "test-123"
        assert session.created_at == now
        assert session.query_count == 0
        assert session.total_tokens_used == 0

    def test_session_info_defaults(self):
        """Test SessionInfo default values."""
        now = datetime.now()
        session = SessionInfo(
            session_id="test",
            created_at=now,
            last_activity=now,
            expires_at=now,
        )

        assert session.query_count == 0
        assert session.total_tokens_used == 0
        assert session.total_processing_time_ms == 0
        assert session.documents_processed == []
        assert session.files_processed == []
        assert session.files_in_context == []


class TestSessionManagerInit:
    """Tests for SessionManager initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        manager = SessionManager()
        assert manager.timeout_minutes == 30
        assert manager.max_cache_size == 10

    def test_init_with_custom_values(self):
        """Test initialization with custom values."""
        manager = SessionManager(timeout_minutes=15, max_cache_size=5)
        assert manager.timeout_minutes == 15
        assert manager.max_cache_size == 5

    def test_init_empty_sessions(self):
        """Test sessions dict is initialized empty."""
        manager = SessionManager()
        assert len(manager.sessions) == 0

    def test_init_empty_cache(self):
        """Test response cache is initialized empty."""
        manager = SessionManager()
        assert len(manager.response_cache) == 0

    def test_repr(self):
        """Test string representation."""
        manager = SessionManager()
        repr_str = repr(manager)
        assert "timeout_minutes=30" in repr_str
        assert "active_sessions=" in repr_str


class TestGetOrCreateSession:
    """Tests for get_or_create_session method."""

    def test_create_new_session_no_id(self):
        """Test creating new session without ID."""
        manager = SessionManager()
        session = manager.get_or_create_session()

        assert session.session_id is not None
        assert len(session.session_id) == 36  # UUID format
        assert session.query_count == 0

    def test_create_new_session_with_id(self):
        """Test creating new session with provided ID."""
        manager = SessionManager()
        session = manager.get_or_create_session(session_id="custom-id")

        assert session.session_id == "custom-id"

    def test_get_existing_session(self):
        """Test retrieving existing session."""
        manager = SessionManager()
        session1 = manager.get_or_create_session(session_id="session-1")
        session1_time = session1.created_at

        time.sleep(0.1)
        session2 = manager.get_or_create_session(session_id="session-1")

        assert session2.session_id == "session-1"
        assert session2.created_at == session1_time
        assert session2.last_activity > session1_time

    def test_expired_session_creates_new(self):
        """Test that expired session creates new one."""
        manager = SessionManager(timeout_minutes=0)  # Immediate expiry
        session1 = manager.get_or_create_session(session_id="session-1")
        created_at1 = session1.created_at

        time.sleep(0.1)
        session2 = manager.get_or_create_session(session_id="session-1")

        # New session created with same ID
        assert session2.created_at > created_at1

    def test_session_timeout_extended_on_access(self):
        """Test that session timeout is extended on access."""
        manager = SessionManager(timeout_minutes=1)
        session = manager.get_or_create_session(session_id="session-1")
        original_expiry = session.expires_at

        time.sleep(0.1)
        manager.get_or_create_session(session_id="session-1")
        session = manager.get_session("session-1")

        assert session.expires_at > original_expiry

    def test_create_multiple_sessions(self):
        """Test creating multiple independent sessions."""
        manager = SessionManager()
        session1 = manager.get_or_create_session(session_id="session-1")
        session2 = manager.get_or_create_session(session_id="session-2")

        assert session1.session_id != session2.session_id
        assert manager.active_count == 2


class TestGetSession:
    """Tests for get_session method."""

    def test_get_existing_session(self):
        """Test getting existing session."""
        manager = SessionManager()
        manager.get_or_create_session(session_id="session-1")

        session = manager.get_session("session-1")
        assert session is not None
        assert session.session_id == "session-1"

    def test_get_nonexistent_session(self):
        """Test getting non-existent session returns None."""
        manager = SessionManager()
        session = manager.get_session("unknown")
        assert session is None

    def test_get_expired_session_returns_none(self):
        """Test that expired session returns None."""
        manager = SessionManager(timeout_minutes=0)
        manager.get_or_create_session(session_id="session-1")

        time.sleep(0.1)
        session = manager.get_session("session-1")
        assert session is None


class TestUpdateSession:
    """Tests for update_session method."""

    def test_update_existing_session(self):
        """Test updating existing session fields."""
        manager = SessionManager()
        manager.get_or_create_session(session_id="session-1")

        result = manager.update_session(
            "session-1",
            query_count=5,
            total_tokens_used=1000
        )

        assert result is True
        session = manager.get_session("session-1")
        assert session.query_count == 5
        assert session.total_tokens_used == 1000

    def test_update_nonexistent_session(self):
        """Test updating non-existent session returns False."""
        manager = SessionManager()
        result = manager.update_session("unknown", query_count=5)
        assert result is False

    def test_update_updates_last_activity(self):
        """Test that update updates last_activity timestamp."""
        manager = SessionManager()
        session = manager.get_or_create_session(session_id="session-1")
        original_activity = session.last_activity

        time.sleep(0.1)
        manager.update_session("session-1", query_count=1)

        updated = manager.get_session("session-1")
        assert updated.last_activity > original_activity

    def test_update_unknown_field_ignored(self):
        """Test that unknown fields are ignored."""
        manager = SessionManager()
        manager.get_or_create_session(session_id="session-1")

        # Should not raise
        manager.update_session("session-1", unknown_field="value")
        session = manager.get_session("session-1")
        assert not hasattr(session, "unknown_field") or session.unknown_field is None

    def test_update_list_fields(self):
        """Test updating list fields."""
        manager = SessionManager()
        manager.get_or_create_session(session_id="session-1")

        manager.update_session(
            "session-1",
            documents_processed=["doc1.pdf", "doc2.pdf"],
            files_in_context=["file1.xlsx"]
        )

        session = manager.get_session("session-1")
        assert session.documents_processed == ["doc1.pdf", "doc2.pdf"]
        assert session.files_in_context == ["file1.xlsx"]


class TestCleanupExpiredSessions:
    """Tests for cleanup_expired_sessions method."""

    def test_cleanup_removes_expired(self):
        """Test that expired sessions are removed."""
        manager = SessionManager(timeout_minutes=0)
        manager.get_or_create_session(session_id="session-1")
        manager.get_or_create_session(session_id="session-2")

        time.sleep(0.1)
        cleaned = manager.cleanup_expired_sessions()

        assert cleaned == 2
        assert manager.get_session("session-1") is None
        assert manager.get_session("session-2") is None

    def test_cleanup_preserves_active(self):
        """Test that active sessions are preserved."""
        manager = SessionManager(timeout_minutes=60)
        manager.get_or_create_session(session_id="session-1")

        cleaned = manager.cleanup_expired_sessions()

        assert cleaned == 0
        assert manager.get_session("session-1") is not None

    def test_cleanup_removes_response_cache(self):
        """Test that cleanup also removes response cache."""
        manager = SessionManager(timeout_minutes=0)
        manager.get_or_create_session(session_id="session-1")
        manager.cache_response("session-1", "hash1", {"data": "value"})

        time.sleep(0.1)
        manager.cleanup_expired_sessions()

        assert manager.get_cached_response("session-1", "hash1") is None

    def test_cleanup_partial(self):
        """Test cleanup only removes expired sessions."""
        manager = SessionManager(timeout_minutes=0)
        manager.get_or_create_session(session_id="session-expired")

        time.sleep(0.1)

        # Create a new session with longer timeout
        manager.timeout_minutes = 60
        manager.get_or_create_session(session_id="session-active")

        # Reset timeout for cleanup
        manager.timeout_minutes = 0
        cleaned = manager.cleanup_expired_sessions()

        # Only the expired session should be cleaned
        assert cleaned == 1
        assert manager.get_session("session-expired") is None


class TestResponseCaching:
    """Tests for response caching methods."""

    def test_cache_and_retrieve(self):
        """Test caching and retrieving response."""
        manager = SessionManager()
        manager.get_or_create_session(session_id="session-1")

        response = {"answer": "test"}
        manager.cache_response("session-1", "query-hash", response)

        cached = manager.get_cached_response("session-1", "query-hash")
        assert cached == response

    def test_get_uncached_returns_none(self):
        """Test that uncached query returns None."""
        manager = SessionManager()
        cached = manager.get_cached_response("session-1", "unknown-hash")
        assert cached is None

    def test_cache_eviction_at_max(self):
        """Test LRU eviction when cache is full."""
        manager = SessionManager(max_cache_size=2)
        manager.get_or_create_session(session_id="session-1")

        manager.cache_response("session-1", "hash1", {"data": 1})
        manager.cache_response("session-1", "hash2", {"data": 2})
        manager.cache_response("session-1", "hash3", {"data": 3})

        # First one should be evicted
        assert manager.get_cached_response("session-1", "hash1") is None
        assert manager.get_cached_response("session-1", "hash2") is not None
        assert manager.get_cached_response("session-1", "hash3") is not None

    def test_cache_update_existing_key(self):
        """Test updating existing cache key."""
        manager = SessionManager()
        manager.get_or_create_session(session_id="session-1")

        manager.cache_response("session-1", "hash1", {"v": 1})
        manager.cache_response("session-1", "hash1", {"v": 2})

        cached = manager.get_cached_response("session-1", "hash1")
        assert cached["v"] == 2

    def test_cache_different_sessions_independent(self):
        """Test that cache is independent per session."""
        manager = SessionManager()
        manager.get_or_create_session(session_id="session-1")
        manager.get_or_create_session(session_id="session-2")

        manager.cache_response("session-1", "hash1", {"session": 1})
        manager.cache_response("session-2", "hash1", {"session": 2})

        assert manager.get_cached_response("session-1", "hash1")["session"] == 1
        assert manager.get_cached_response("session-2", "hash1")["session"] == 2

    def test_cache_creates_session_entry(self):
        """Test that caching creates session cache entry if not exists."""
        manager = SessionManager()

        manager.cache_response("new-session", "hash1", {"data": "test"})

        assert "new-session" in manager.response_cache
        assert manager.get_cached_response("new-session", "hash1") is not None


class TestDeleteSession:
    """Tests for delete_session method."""

    def test_delete_existing_session(self):
        """Test deleting existing session."""
        manager = SessionManager()
        manager.get_or_create_session(session_id="session-1")

        result = manager.delete_session("session-1")

        assert result is True
        assert manager.get_session("session-1") is None

    def test_delete_nonexistent_session(self):
        """Test deleting non-existent session returns False."""
        manager = SessionManager()
        result = manager.delete_session("unknown")
        assert result is False

    def test_delete_clears_cache(self):
        """Test that delete also clears response cache."""
        manager = SessionManager()
        manager.get_or_create_session(session_id="session-1")
        manager.cache_response("session-1", "hash1", {"data": "value"})

        manager.delete_session("session-1")

        assert manager.get_cached_response("session-1", "hash1") is None

    def test_delete_preserves_other_sessions(self):
        """Test delete only affects specified session."""
        manager = SessionManager()
        manager.get_or_create_session(session_id="session-1")
        manager.get_or_create_session(session_id="session-2")

        manager.delete_session("session-1")

        assert manager.get_session("session-1") is None
        assert manager.get_session("session-2") is not None


class TestActiveCount:
    """Tests for active_count property."""

    def test_active_count_empty(self):
        """Test active count on empty manager."""
        manager = SessionManager()
        assert manager.active_count == 0

    def test_active_count_with_sessions(self):
        """Test active count with sessions."""
        manager = SessionManager(timeout_minutes=60)
        manager.get_or_create_session(session_id="session-1")
        manager.get_or_create_session(session_id="session-2")

        assert manager.active_count == 2

    def test_active_count_excludes_expired(self):
        """Test that active count excludes expired sessions."""
        manager = SessionManager(timeout_minutes=0)
        manager.get_or_create_session(session_id="session-1")

        time.sleep(0.1)
        assert manager.active_count == 0

    def test_active_count_after_delete(self):
        """Test active count after deleting session."""
        manager = SessionManager(timeout_minutes=60)
        manager.get_or_create_session(session_id="session-1")
        manager.get_or_create_session(session_id="session-2")

        manager.delete_session("session-1")

        assert manager.active_count == 1


class TestThreadSafety:
    """Tests for thread-safety."""

    def test_concurrent_session_creation(self):
        """Test concurrent session creation."""
        manager = SessionManager()

        def create_session(i):
            return manager.get_or_create_session(session_id=f"session-{i}")

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(create_session, i) for i in range(100)]
            sessions = [f.result() for f in as_completed(futures)]

        assert len(sessions) == 100
        assert manager.active_count == 100

    def test_concurrent_same_session_access(self):
        """Test concurrent access to same session."""
        manager = SessionManager()
        manager.get_or_create_session(session_id="shared-session")
        errors = []

        def access_session():
            try:
                for _ in range(50):
                    manager.get_or_create_session(session_id="shared-session")
                    manager.update_session("shared-session", query_count=1)
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(access_session) for _ in range(10)]
            for f in as_completed(futures):
                f.result()

        assert len(errors) == 0

    def test_concurrent_cache_access(self):
        """Test concurrent cache access."""
        manager = SessionManager()
        manager.get_or_create_session(session_id="session-1")

        def cache_and_get(i):
            manager.cache_response("session-1", f"hash-{i}", {"i": i})
            return manager.get_cached_response("session-1", f"hash-{i}")

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(cache_and_get, i) for i in range(50)]
            results = [f.result() for f in as_completed(futures)]

        # All should succeed (no race conditions)
        assert all(r is not None for r in results)

    def test_concurrent_cleanup_and_access(self):
        """Test cleanup running concurrently with access."""
        manager = SessionManager(timeout_minutes=0)
        errors = []

        def create_sessions():
            try:
                for i in range(50):
                    manager.get_or_create_session(session_id=f"session-{i}")
                    time.sleep(0.01)
            except Exception as e:
                errors.append(e)

        def run_cleanup():
            try:
                for _ in range(10):
                    manager.cleanup_expired_sessions()
                    time.sleep(0.05)
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=2) as executor:
            executor.submit(create_sessions)
            executor.submit(run_cleanup)

        assert len(errors) == 0

    def test_concurrent_delete_and_access(self):
        """Test delete running concurrently with access."""
        manager = SessionManager()
        errors = []

        def access_sessions():
            try:
                for i in range(50):
                    manager.get_or_create_session(session_id=f"session-{i % 10}")
            except Exception as e:
                errors.append(e)

        def delete_sessions():
            try:
                for i in range(10):
                    manager.delete_session(f"session-{i}")
                    time.sleep(0.01)
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=2) as executor:
            executor.submit(access_sessions)
            executor.submit(delete_sessions)

        assert len(errors) == 0
