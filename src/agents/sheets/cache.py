"""File cache manager for SheetsAgent.

Provides efficient LRU caching for DataFrame file data.
"""

import logging
from collections import OrderedDict
from typing import Dict, Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class FileCache:
    """Manages cached file data to avoid redundant reads using efficient OrderedDict LRU.

    Provides DataFrame caching with configurable max size and LRU eviction.
    Thread-safe for read operations (copy on read).
    """

    def __init__(self, max_size: int = 50):
        """Initialize the file cache.

        Args:
            max_size: Maximum number of files to cache (default: 50)
        """
        self.cache: OrderedDict[str, pd.DataFrame] = OrderedDict()
        self.max_size = max_size
        self._hits = 0
        self._misses = 0

    def get(self, file_path: str) -> Optional[pd.DataFrame]:
        """Get cached DataFrame if available, moving to end (most recently used).

        Args:
            file_path: Path to the file (local or GCS)

        Returns:
            Copy of cached DataFrame if found, None otherwise
        """
        if file_path in self.cache:
            # Move to end to mark as recently used
            self.cache.move_to_end(file_path)
            self._hits += 1
            logger.debug(f"Cache hit for {file_path}")
            return self.cache[file_path].copy()
        self._misses += 1
        return None

    def put(self, file_path: str, df: pd.DataFrame):
        """Cache DataFrame with LRU eviction using OrderedDict.

        Args:
            file_path: Path to the file (local or GCS)
            df: DataFrame to cache
        """
        if file_path in self.cache:
            # Update existing entry and move to end
            self.cache.move_to_end(file_path)
            self.cache[file_path] = df.copy()
        else:
            # Evict oldest if at capacity
            if len(self.cache) >= self.max_size:
                oldest_file, _ = self.cache.popitem(last=False)
                logger.debug(f"Evicted {oldest_file} from cache")

            self.cache[file_path] = df.copy()
        logger.debug(f"Cached {file_path} (shape: {df.shape})")

    def clear(self):
        """Clear all cached data and reset statistics."""
        self.cache.clear()
        self._hits = 0
        self._misses = 0
        logger.debug("File cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with size, max_size, hits, misses, hit_rate_percent
        """
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_percent": round(hit_rate, 2)
        }

    def __len__(self) -> int:
        """Return the number of cached files."""
        return len(self.cache)

    def __contains__(self, file_path: str) -> bool:
        """Check if a file is in the cache."""
        return file_path in self.cache
