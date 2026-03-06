"""Module for memory usage utilities."""

import getpass
import logging

import psutil


def readable_memory_usage(memory_usage: int) -> str:
    """
    Convert memory usage in bytes to a human-readable string.

    Parameters
    ----------
    memory_usage : int
        Memory usage in bytes.

    Returns
    -------
    str
        Human-readable memory usage string.
    """
    if memory_usage < 1024:
        return f"{memory_usage} B"
    elif memory_usage < 1024**2:
        return f"{memory_usage // 1024} KB"
    elif memory_usage < 1024**3:
        return f"{memory_usage // (1024**2)} MB"
    elif memory_usage < 1024**4:
        return f"{memory_usage // (1024**3)} GB"
    else:
        return f"{memory_usage // (1024**4)} TB"


def get_memory_usage_by_user(user_name: str | None = None) -> int:
    """
    Get the total memory usage of all processes owned by a specific user.

    Parameters
    ----------
    user_name : str, optional
        The username to check memory usage for. If None, uses the current user.

    Returns
    -------
    int
        Total memory usage in bytes for the specified user.
    """
    if user_name is None:
        user_name = getpass.getuser()
    total_memory = 0
    try:
        for proc in psutil.process_iter(["username", "memory_info"]):
            if proc.info["username"] == user_name:
                total_memory += proc.info["memory_info"].rss  # rss is the resident set size (actual memory usage)
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        # Handle processes that might terminate or be inaccessible
        logging.debug("Could not access process info for PID %d", proc.pid)
    return total_memory


def check_over_memory(memory_threshold_in_gb: int | float, user_name: str | None = None) -> bool:
    """
    Check if the memory usage of a specific user exceeds a given threshold in gigabytes.

    Parameters
    ----------
    memory_threshold_in_gb : int | float
        Memory threshold in gigabytes.
    user_name : str, optional
        The username to check memory usage for. If None, uses the current user.

    Returns
    -------
    bool
        True if memory usage exceeds the threshold, False otherwise.
    """
    return get_memory_usage_by_user(user_name=user_name) > memory_threshold_in_gb * 1024 * 1024 * 1024
