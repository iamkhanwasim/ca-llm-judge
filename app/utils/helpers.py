"""
Shared utility functions for the LLM Judge application.
"""
import logging

logger = logging.getLogger(__name__)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.

    Args:
        numerator: The numerator
        denominator: The denominator
        default: Default value to return if denominator is zero

    Returns:
        Result of division or default value
    """
    if denominator == 0:
        return default
    return numerator / denominator


def truncate_string(text: str, max_length: int = 100) -> str:
    """
    Truncate a string to a maximum length.

    Args:
        text: The string to truncate
        max_length: Maximum length

    Returns:
        Truncated string with ellipsis if needed
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format a float as a percentage string.

    Args:
        value: Float value between 0 and 1
        decimals: Number of decimal places

    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"
