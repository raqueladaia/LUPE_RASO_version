"""
Filename Utilities for LUPE Analysis Tool

This module provides utility functions for handling and manipulating filenames,
particularly for DeepLabCut CSV files.

Usage:
    from src.utils.filename_utils import extract_partial_filename

    partial_name = extract_partial_filename('mouse01DLC_resnet50.csv')
    # Returns: 'mouse01'
"""

from pathlib import Path


def extract_partial_filename(csv_path: str) -> str:
    """
    Extract the partial filename before 'DLC' from a CSV file path.

    This function extracts the portion of the base filename that comes before
    the 'DLC' marker. This is used to create organized output folders.

    Args:
        csv_path (str): Full path to the CSV file

    Returns:
        str: Partial filename before 'DLC' (e.g., 'mouse01' from 'mouse01DLC_tracking.csv')

    Example:
        >>> extract_partial_filename('/path/to/mouse01DLC_resnet50.csv')
        'mouse01'
        >>> extract_partial_filename('/path/to/Subject_123_DLC_analysis.csv')
        'Subject_123_'
    """
    # Get the base filename without extension
    base_name = Path(csv_path).stem

    # Find 'DLC' in the filename (case-insensitive)
    dlc_index = base_name.upper().find('DLC')

    if dlc_index != -1:
        # Return everything before 'DLC'
        return base_name[:dlc_index]
    else:
        # If 'DLC' is not found, return the full base name
        return base_name
