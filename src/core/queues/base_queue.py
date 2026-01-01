"""Background queue base class with dedicated event loop.

Provides common infrastructure for async queues running in background threads.
"""

import asyncio
import atexit
import logging
import queue
import threading
from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar

from src.core.patterns import ThreadSafeSingleton

logger = logging.getLogger(__name__)

# Event type parameter
T = TypeVar('T')


class BackgroundQueue(Generic[T], ThreadSafeSingleton, ABC):
    """
    Abstract base class for background queue processing.

    Provides common infrastructure for:
    - Thread-safe singleton pattern
    - Background thread with dedicated event loop
    - Queue management (enqueue, process, shutdown)
    - Database connection cleanup

    Subclasses must implement:
    - `_get_queue_name()`: Return queue name for logging
    - `_process_event(event)`: Process a single event

    Example:
        class MyQueue(BackgroundQueue[MyEvent]):
            def _get_queue_name(self) -> str:
                return "my-queue"

            async def _process_event(self, event: MyEvent) -> None:
                # Process the event
                await some_service.handle(event)

        # Usage
        queue = MyQueue.get_instance()
        queue.enqueue(MyEvent(...))
    """

    def _initialize(self) -> None:
        """Initialize queue resources."""
        self._queue: queue.Queue[Optional[T]] = queue.Queue(maxsize=1000)
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._shutdown_event = threading.Event()
        self._started = False

        # Register cleanup on interpreter shutdown
        atexit.register(self.shutdown)

    @abstractmethod
    def _get_queue_name(self) -> str:
        """Return the queue name for logging and thread naming.

        Returns:
            Queue name string (e.g., 'audit-queue', 'usage-queue')
        """
        pass

    @abstractmethod
    async def _process_event(self, event: T) -> None:
        """Process a single event from the queue.

        Args:
            event: The event to process

        Raises:
            Any exception - will be caught and logged
        """
        pass

    def start(self) -> None:
        """Start the background processing thread."""
        if self._started:
            return

        self._shutdown_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name=self._get_queue_name(),
            daemon=True,
        )
        self._thread.start()
        self._started = True
        logger.info(f"{self._get_queue_name()} started")

    def _run_loop(self) -> None:
        """Background thread with persistent event loop."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_until_complete(self._process_events())
        except Exception as e:
            logger.error(f"{self._get_queue_name()} loop error: {e}")
        finally:
            # Proper cleanup of database connections
            try:
                self._loop.run_until_complete(self._cleanup_db())
            except Exception as e:
                logger.warning(f"Error during {self._get_queue_name()} DB cleanup: {e}")
            self._loop.close()
            self._loop = None
            logger.info(f"{self._get_queue_name()} loop stopped")

    async def _process_events(self) -> None:
        """Process events from queue until shutdown."""
        # Initialize database connection for this loop once
        await self._init_db_for_loop()

        while not self._shutdown_event.is_set():
            try:
                # Non-blocking get with timeout
                event = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self._queue.get(timeout=1.0)
                )

                if event is None:
                    # Poison pill - shutdown signal
                    break

                try:
                    await self._process_event(event)
                except Exception as e:
                    error_str = str(e).lower()
                    # Check if connection error - try to reinitialize
                    if "connection" in error_str and ("closed" in error_str or "lost" in error_str):
                        logger.warning(
                            f"{self._get_queue_name()} connection lost, reinitializing..."
                        )
                        await self._reinit_db_connection()
                        # Retry once after reinit
                        try:
                            await self._process_event(event)
                        except Exception as retry_e:
                            logger.warning(
                                f"Failed to process {self._get_queue_name()} event after retry: {retry_e}"
                            )
                    else:
                        logger.warning(
                            f"Failed to process {self._get_queue_name()} event: {e}"
                        )

            except queue.Empty:
                continue
            except Exception as e:
                if not self._shutdown_event.is_set():
                    logger.error(f"Error in {self._get_queue_name()} loop: {e}")

    async def _init_db_for_loop(self) -> None:
        """Initialize database session factory for this event loop."""
        try:
            from src.db.connection import db
            await db.get_engine_async()
            logger.debug(f"{self._get_queue_name()} database initialized for loop")
        except Exception as e:
            logger.warning(f"Failed to initialize {self._get_queue_name()} database: {e}")

    async def _reinit_db_connection(self) -> None:
        """Reinitialize database connection after failure."""
        try:
            from src.db.connection import db
            # Close current session factory for this loop
            await db.close()
            # Reinitialize
            await db.get_engine_async()
            logger.info(f"{self._get_queue_name()} database connection reinitialized")
        except Exception as e:
            logger.warning(f"Failed to reinitialize {self._get_queue_name()} database: {e}")

    async def _cleanup_db(self) -> None:
        """Cleanup database connections for this event loop."""
        try:
            from src.db.connection import db
            await db.close()
            logger.debug(f"{self._get_queue_name()} database connections closed")
        except Exception as e:
            logger.warning(f"Error closing {self._get_queue_name()} DB connections: {e}")

    def enqueue(self, event: T) -> None:
        """Add event to queue (non-blocking).

        Args:
            event: The event to enqueue
        """
        if not self._started:
            self.start()

        try:
            self._queue.put_nowait(event)
        except queue.Full:
            logger.warning(f"{self._get_queue_name()} full, dropping event")

    def shutdown(self, wait: bool = True, timeout: float = 5.0) -> None:
        """Shutdown the queue gracefully.

        Args:
            wait: If True, wait for thread to finish
            timeout: Maximum seconds to wait for shutdown
        """
        if not self._started:
            return

        logger.info(f"Shutting down {self._get_queue_name()}...")
        self._shutdown_event.set()

        # Send poison pill
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

        if wait and self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

        self._started = False
        logger.info(f"{self._get_queue_name()} shutdown complete")

    @property
    def queue_size(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()

    @property
    def is_running(self) -> bool:
        """Check if queue is running."""
        return self._started and self._thread is not None and self._thread.is_alive()
