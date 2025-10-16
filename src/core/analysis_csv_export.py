"""
CSV Export Analysis Module

This module exports behavior classifications to CSV files for easy integration
with external analysis tools (Excel, R, Python, etc.).

Exports two formats:
1. Frame-by-frame: Each row is one video frame with its behavior classification
2. Second-by-second: Each row is one second with its behavior classification

Usage:
    from src.core.analysis_csv_export import export_behaviors_to_csv

    # Export behaviors for all files
    export_behaviors_to_csv(behaviors_dict, output_dir='outputs/csv/')
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict
from src.utils.config_manager import get_config


def export_behaviors_to_csv(behaviors: Dict[str, np.ndarray],
                           output_dir: str,
                           framerate: int = None,
                           create_subdirs: bool = True) -> Dict[str, str]:
    """
    Export behavior classifications to CSV files.

    Creates two types of CSV files for each input file:
    - Frame-based: behavior ID for each frame
    - Time-based: behavior ID for each second

    Args:
        behaviors (dict): Dictionary mapping file names to behavior arrays
        output_dir (str): Directory where CSV files will be saved
        framerate (int, optional): Video framerate. If None, uses config default
        create_subdirs (bool): If True, creates 'frames' and 'seconds' subdirectories

    Returns:
        dict: Dictionary mapping file names to their CSV paths

    Example:
        >>> behaviors = {
        >>>     'video1': np.array([0, 0, 1, 1, 2, 2, ...]),
        >>>     'video2': np.array([0, 1, 1, 2, 0, 0, ...])
        >>> }
        >>> paths = export_behaviors_to_csv(behaviors, 'outputs/csv/')
        >>> # Creates: outputs/csv/frames/video1.csv
        >>> #          outputs/csv/frames/video2.csv
        >>> #          outputs/csv/seconds/video1.csv
        >>> #          outputs/csv/seconds/video2.csv
    """
    # Get framerate from config if not provided
    if framerate is None:
        config = get_config()
        framerate = config.get_framerate()

    # Create output directories
    output_path = Path(output_dir)
    if create_subdirs:
        frames_dir = output_path / 'frames'
        seconds_dir = output_path / 'seconds'
        frames_dir.mkdir(parents=True, exist_ok=True)
        seconds_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_path.mkdir(parents=True, exist_ok=True)

    saved_files = {}

    # Export each file
    for file_name, behavior_array in behaviors.items():
        # Clean filename (remove problematic characters)
        clean_name = file_name.replace('/', '_').replace('\\', '_')

        # 1. Export frame-by-frame
        if create_subdirs:
            frames_path = frames_dir / f'{clean_name}.csv'
        else:
            frames_path = output_path / f'{clean_name}_frames.csv'

        # Create DataFrame with frame numbers and behaviors
        df_frames = pd.DataFrame({
            'frame': range(1, len(behavior_array) + 1),
            'behavior_id': behavior_array
        })
        df_frames.to_csv(frames_path, index=False)

        # 2. Export second-by-second
        if create_subdirs:
            seconds_path = seconds_dir / f'{clean_name}.csv'
        else:
            seconds_path = output_path / f'{clean_name}_seconds.csv'

        # Calculate time in seconds for each frame
        time_seconds = [i / framerate for i in range(len(behavior_array))]
        df_seconds = pd.DataFrame({
            'time_seconds': time_seconds,
            'behavior_id': behavior_array
        })
        df_seconds.to_csv(seconds_path, index=False)

        saved_files[file_name] = {
            'frames': str(frames_path),
            'seconds': str(seconds_path)
        }

        print(f'Exported: {clean_name}')

    print(f'\nAll files saved to: {output_dir}')
    return saved_files


def export_behaviors_with_names(behaviors: Dict[str, np.ndarray],
                                output_dir: str,
                                behavior_names: list = None,
                                framerate: int = None) -> Dict[str, str]:
    """
    Export behaviors with human-readable behavior names instead of IDs.

    Args:
        behaviors (dict): Dictionary mapping file names to behavior arrays
        output_dir (str): Output directory
        behavior_names (list, optional): List of behavior names. If None, uses config
        framerate (int, optional): Video framerate

    Returns:
        dict: Dictionary mapping file names to CSV paths

    Example:
        >>> behaviors = {'video1': np.array([0, 0, 1, 2, 2])}
        >>> export_behaviors_with_names(behaviors, 'outputs/')
        >>> # CSV will have: frame, behavior_id, behavior_name
        >>> #                1,     0,           still
        >>> #                2,     0,           still
        >>> #                3,     1,           walking
    """
    # Get behavior names from config if not provided
    if behavior_names is None:
        config = get_config()
        behavior_names = config.get_behavior_names()

    if framerate is None:
        config = get_config()
        framerate = config.get_framerate()

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_files = {}

    for file_name, behavior_array in behaviors.items():
        clean_name = file_name.replace('/', '_').replace('\\', '_')
        csv_path = output_path / f'{clean_name}_with_names.csv'

        # Map behavior IDs to names
        behavior_name_array = [behavior_names[int(b)] for b in behavior_array]

        # Create DataFrame
        df = pd.DataFrame({
            'frame': range(1, len(behavior_array) + 1),
            'time_seconds': [i / framerate for i in range(len(behavior_array))],
            'behavior_id': behavior_array,
            'behavior_name': behavior_name_array
        })

        df.to_csv(csv_path, index=False)
        saved_files[file_name] = str(csv_path)

        print(f'Exported: {clean_name}_with_names.csv')

    return saved_files


def export_behavior_summary(behaviors: Dict[str, np.ndarray],
                           output_path: str,
                           behavior_names: list = None,
                           framerate: int = None) -> str:
    """
    Export a summary CSV with statistics for all files.

    Creates one CSV with rows for each file and columns for each behavior,
    showing the percentage of time spent in each behavior.

    Args:
        behaviors (dict): Dictionary mapping file names to behavior arrays
        output_path (str): Path for the output CSV file
        behavior_names (list, optional): List of behavior names
        framerate (int, optional): Video framerate

    Returns:
        str: Path to the saved summary file

    Example:
        >>> behaviors = {'video1': ..., 'video2': ...}
        >>> path = export_behavior_summary(behaviors, 'outputs/summary.csv')
        >>> # CSV format:
        >>> # file_name, total_frames, duration_sec, still_%, walking_%, ...
        >>> # video1,    1000,         16.67,        45.2,    30.1, ...
        >>> # video2,    1500,         25.00,        50.3,    25.5, ...
    """
    if behavior_names is None:
        config = get_config()
        behavior_names = config.get_behavior_names()

    if framerate is None:
        config = get_config()
        framerate = config.get_framerate()

    # Build summary data
    summary_data = []
    for file_name, behavior_array in behaviors.items():
        row = {
            'file_name': file_name,
            'total_frames': len(behavior_array),
            'duration_seconds': len(behavior_array) / framerate
        }

        # Calculate percentage for each behavior
        for behavior_id, behavior_name in enumerate(behavior_names):
            count = np.sum(behavior_array == behavior_id)
            percentage = (count / len(behavior_array)) * 100
            row[f'{behavior_name}_%'] = round(percentage, 2)
            row[f'{behavior_name}_frames'] = int(count)

        summary_data.append(row)

    # Create DataFrame and save
    df = pd.DataFrame(summary_data)

    # Create output directory if needed
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f'Summary exported to: {output_path}')
    return str(output_path)
