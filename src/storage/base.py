"""Abstract base class for storage backends."""

from abc import ABC, abstractmethod
from typing import Optional, List


class StorageBackend(ABC):
    """Abstract storage backend interface."""

    @abstractmethod
    async def save(self, content: str, filename: str, directory: str = "") -> str:
        """
        Save content to storage.

        Args:
            content: Content to save
            filename: Name of the file
            directory: Subdirectory within storage (optional)

        Returns:
            Full URI/path of saved file
        """
        pass

    @abstractmethod
    async def read(self, path: str, use_prefix: bool = True) -> Optional[str]:
        """
        Read content from storage.

        Args:
            path: Full path or URI to the file
            use_prefix: If True, prepend configured prefix to relative paths.
                       If False, use path directly without prefix.

        Returns:
            File content as string, or None if not found
        """
        pass

    @abstractmethod
    async def exists(self, path: str, use_prefix: bool = True) -> bool:
        """
        Check if file exists in storage.

        Args:
            path: Full path or URI to check
            use_prefix: If True, prepend configured prefix to relative paths.
                       If False, use path directly without prefix.

        Returns:
            True if file exists
        """
        pass

    @abstractmethod
    async def list_files(
        self, directory: str, extension: Optional[str] = None
    ) -> List[str]:
        """
        List files in directory.

        Args:
            directory: Directory to list
            extension: Filter by file extension (e.g., ".md")

        Returns:
            List of file URIs/paths
        """
        pass

    @abstractmethod
    async def delete(self, path: str) -> bool:
        """
        Delete file from storage.

        Args:
            path: Full path or URI to delete

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    def get_uri(self, path: str) -> str:
        """
        Get full URI for a relative path.

        Args:
            path: Relative path within storage

        Returns:
            Full URI (e.g., gs://bucket/prefix/path)
        """
        pass
