"""Environment variable utilities.

This module provides utilities for parsing environment variables
with proper type conversion and default value handling.
"""

import os
from typing import Optional


def parse_bool_env(key: str, default: bool = True) -> bool:
    """Parse a boolean value from an environment variable.

    Args:
        key: The environment variable name.
        default: Default value if the environment variable is not set.

    Returns:
        True if the value is 'true' (case-insensitive), False otherwise.
        Returns the default if the environment variable is not set.

    Examples:
        >>> os.environ["MY_VAR"] = "true"
        >>> parse_bool_env("MY_VAR")
        True
        >>> os.environ["MY_VAR"] = "false"
        >>> parse_bool_env("MY_VAR")
        False
        >>> parse_bool_env("UNSET_VAR", default=False)
        False
    """
    value = os.getenv(key)
    if value is None:
        return default
    return value.lower() == "true"


def parse_int_env(key: str, default: int) -> int:
    """Parse an integer value from an environment variable.

    Args:
        key: The environment variable name.
        default: Default value if the environment variable is not set or invalid.

    Returns:
        The integer value from the environment variable, or the default.
    """
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def parse_float_env(key: str, default: float) -> float:
    """Parse a float value from an environment variable.

    Args:
        key: The environment variable name.
        default: Default value if the environment variable is not set or invalid.

    Returns:
        The float value from the environment variable, or the default.
    """
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def parse_str_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """Parse a string value from an environment variable.

    Args:
        key: The environment variable name.
        default: Default value if the environment variable is not set.

    Returns:
        The string value from the environment variable, or the default.
    """
    return os.getenv(key, default)
