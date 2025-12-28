"""Thread-safe singleton pattern implementation.

This module provides a reusable base class for implementing the singleton
pattern with double-checked locking for thread safety.
"""

import threading
from abc import ABC
from typing import ClassVar, Optional, TypeVar

T = TypeVar('T', bound='ThreadSafeSingleton')


class ThreadSafeSingleton(ABC):
    """Abstract base class for thread-safe singletons.

    Implements double-checked locking pattern to ensure only one instance
    is created even under concurrent access.

    Usage:
        class MySingleton(ThreadSafeSingleton):
            def _initialize(self):
                # One-time initialization logic
                self.some_resource = create_resource()

            def do_something(self):
                return self.some_resource.process()

        # Get instance (creates on first call)
        instance = MySingleton.get_instance()

    Note:
        Subclasses should implement `_initialize()` for one-time setup.
        Do NOT override `__new__` or `__init__` in subclasses.
    """

    _instance: ClassVar[Optional['ThreadSafeSingleton']] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()
    _initialized: bool = False

    def __new__(cls: type[T]) -> T:
        """Create singleton instance with double-checked locking."""
        if cls._instance is None:
            with cls._lock:
                # Double-check after acquiring lock
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance  # type: ignore

    def __init__(self) -> None:
        """Initialize singleton (runs only once)."""
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._initialize()
                    self._initialized = True

    def _initialize(self) -> None:
        """Override in subclasses for one-time initialization.

        This method is called exactly once when the singleton is first
        created. Use this instead of __init__ for initialization logic.
        """
        pass

    @classmethod
    def get_instance(cls: type[T]) -> T:
        """Get the singleton instance.

        This is the preferred way to access the singleton.

        Returns:
            The singleton instance.
        """
        return cls()

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (primarily for testing).

        Warning:
            This should only be used in tests. Using this in production
            code may lead to resource leaks or inconsistent state.
        """
        with cls._lock:
            if cls._instance is not None:
                # Call cleanup if available
                if hasattr(cls._instance, '_cleanup'):
                    try:
                        cls._instance._cleanup()
                    except Exception:
                        pass
                cls._instance = None

    def _cleanup(self) -> None:
        """Override in subclasses for cleanup logic.

        Called when reset_instance() is invoked. Use this to release
        resources, close connections, etc.
        """
        pass
