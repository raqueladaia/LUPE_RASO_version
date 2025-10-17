"""
AMPS Summary Reader Utility

This module provides functions to read LUPE analysis summary CSV files
and extract parameters needed for LUPE-AMPS analysis.

The summary CSVs are generated during LUPE analysis and contain metadata
about the recording (framerate, duration, frame count, etc.).

Usage:
    from src.utils.amps_summary_reader import read_parameters_from_summary

    # Get parameters for AMPS analysis from summary CSV
    params = read_parameters_from_summary('outputs/mouse01/mouse01_behaviors.csv')

    print(f"Recording Length: {params['recording_length_min']} minutes")
    print(f"Original Framerate: {params['original_fps']} fps")
"""

import pandas as pd
from pathlib import Path
from typing import Dict


def read_parameters_from_summary(behavior_csv_path: str) -> Dict[str, float]:
    """
    Read recording parameters from the corresponding LUPE summary CSV file.

    This function takes a behavior CSV file path (e.g., 'mouse01_behaviors.csv')
    and reads the corresponding summary CSV file (e.g., 'mouse01_summary.csv')
    to extract the recording parameters needed for LUPE-AMPS analysis.

    Args:
        behavior_csv_path (str): Path to the behavior CSV file
                                 (e.g., 'outputs/mouse01/mouse01_behaviors.csv')

    Returns:
        dict: Dictionary containing:
            - 'recording_length_min' (float): Recording duration in minutes
            - 'original_fps' (float): Original framerate in frames per second
            - 'total_frames' (int): Total number of frames in the recording

    Raises:
        FileNotFoundError: If the summary CSV file does not exist
        ValueError: If the summary CSV cannot be parsed or required fields are missing

    Example:
        >>> params = read_parameters_from_summary('outputs/mouse01/mouse01_behaviors.csv')
        >>> print(params)
        {'recording_length_min': 30.0, 'original_fps': 60.0, 'total_frames': 108000}
    """
    behavior_path = Path(behavior_csv_path)

    # Construct summary CSV path by replacing _behaviors.csv with _summary.csv
    # Handle both cases: mouse01_behaviors.csv -> mouse01_summary.csv
    # and mouse01DLC_behaviors.csv -> mouse01DLC_summary.csv
    behavior_name = behavior_path.stem  # Get filename without extension

    if behavior_name.endswith('_behaviors'):
        # Remove '_behaviors' suffix and add '_summary'
        base_name = behavior_name[:-10]  # Remove last 10 chars ('_behaviors')
        summary_name = f"{base_name}_summary.csv"
    else:
        # Fallback: just replace .csv with _summary.csv
        raise ValueError(
            f"Unexpected behavior CSV filename format: {behavior_path.name}. "
            f"Expected filename to end with '_behaviors.csv'"
        )

    summary_path = behavior_path.parent / summary_name

    # Check if summary file exists
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Summary CSV file not found: {summary_path}\n"
            f"The summary file is required for LUPE-AMPS analysis.\n"
            f"Please ensure you have run LUPE analysis on this file to generate the summary."
        )

    try:
        # Read the summary CSV
        # Format: Two columns - "Field" and "Value"
        summary_df = pd.read_csv(summary_path)

        # Validate CSV structure
        if 'Field' not in summary_df.columns or 'Value' not in summary_df.columns:
            raise ValueError(
                f"Invalid summary CSV format in {summary_path}. "
                f"Expected columns 'Field' and 'Value'."
            )

        # Create a dictionary for easy lookup
        # Field -> Value mapping
        summary_dict = {}
        for _, row in summary_df.iterrows():
            field = row['Field']
            value = row['Value']
            if pd.notna(field) and pd.notna(value):  # Skip empty rows
                summary_dict[field] = value

        # Extract required parameters
        required_fields = {
            'Duration (minutes)': 'recording_length_min',
            'Framerate (fps)': 'original_fps',
            'Total Frames': 'total_frames'
        }

        params = {}
        missing_fields = []

        for field_name, param_key in required_fields.items():
            if field_name not in summary_dict:
                missing_fields.append(field_name)
                continue

            value_str = str(summary_dict[field_name])

            # Parse the value (remove commas from numbers like "108,000")
            try:
                value_str_clean = value_str.replace(',', '')
                value = float(value_str_clean)

                # Convert to int for total_frames
                if param_key == 'total_frames':
                    value = int(value)

                params[param_key] = value
            except (ValueError, AttributeError) as e:
                raise ValueError(
                    f"Cannot parse value for '{field_name}' in {summary_path}: {value_str}"
                ) from e

        # Check if all required fields were found
        if missing_fields:
            raise ValueError(
                f"Missing required fields in summary CSV {summary_path}: {', '.join(missing_fields)}"
            )

        return params

    except pd.errors.EmptyDataError:
        raise ValueError(f"Summary CSV file is empty: {summary_path}")
    except Exception as e:
        if isinstance(e, (FileNotFoundError, ValueError)):
            raise  # Re-raise our custom errors
        else:
            # Wrap unexpected errors
            raise ValueError(
                f"Error reading summary CSV {summary_path}: {str(e)}"
            ) from e


def get_summary_path_from_behavior_csv(behavior_csv_path: str) -> Path:
    """
    Get the path to the summary CSV file corresponding to a behavior CSV file.

    This is a helper function that constructs the expected summary CSV path
    from a behavior CSV path.

    Args:
        behavior_csv_path (str): Path to behavior CSV file

    Returns:
        Path: Path to the corresponding summary CSV file

    Example:
        >>> get_summary_path_from_behavior_csv('outputs/mouse01/mouse01_behaviors.csv')
        Path('outputs/mouse01/mouse01_summary.csv')
    """
    behavior_path = Path(behavior_csv_path)
    behavior_name = behavior_path.stem

    if behavior_name.endswith('_behaviors'):
        base_name = behavior_name[:-10]
        summary_name = f"{base_name}_summary.csv"
    else:
        # Fallback
        summary_name = f"{behavior_name}_summary.csv"

    return behavior_path.parent / summary_name
