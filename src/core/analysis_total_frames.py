"""
Total Frames Analysis Module

This module analyzes what percentage of total time is spent in each behavior.
Creates pie/donut charts showing the overall behavior distribution.

Usage:
    from src.core.analysis_total_frames import analyze_total_frames

    results = analyze_total_frames(behaviors_dict, output_dir='outputs/')
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
from src.utils.config_manager import get_config
from src.utils.plotting import plot_behavior_pie, save_figure


def calculate_frame_percentages(predictions: np.ndarray,
                                behavior_names: list = None) -> pd.DataFrame:
    """
    Calculate the percentage of frames for each behavior.

    Args:
        predictions (np.ndarray): Behavior predictions array
        behavior_names (list, optional): List of behavior names

    Returns:
        pd.DataFrame: DataFrame with columns: behavior, frames, percentage

    Example:
        >>> predictions = np.array([0, 0, 0, 1, 1, 2])  # 6 frames total
        >>> df = calculate_frame_percentages(predictions)
        >>> # behavior | frames | percentage
        >>> # still    | 3      | 50.0
        >>> # walking  | 2      | 33.3
        >>> # rearing  | 1      | 16.7
    """
    if behavior_names is None:
        config = get_config()
        behavior_names = config.get_behavior_names()

    # Count frames for each behavior
    unique_behaviors, counts = np.unique(predictions, return_counts=True)

    # Calculate percentages
    total_frames = len(predictions)

    # Build result dataframe
    results = []
    for behavior_id, count in zip(unique_behaviors, counts):
        percentage = (count / total_frames) * 100
        results.append({
            'behavior_id': int(behavior_id),
            'behavior': behavior_names[int(behavior_id)],
            'frames': int(count),
            'percentage': round(percentage, 2)
        })

    return pd.DataFrame(results)


def analyze_total_frames(behaviors: Dict[str, np.ndarray],
                         output_dir: str,
                         behavior_names: list = None,
                         behavior_colors: list = None,
                         create_plots: bool = True,
                         save_csv: bool = True,
                         file_prefix: str = None) -> Dict:
    """
    Analyze total frames spent in each behavior for a single file.

    Creates pie chart showing behavior distribution.

    Args:
        behaviors (dict): Dictionary mapping file name to behavior array (single file)
        output_dir (str): Directory for output files
        behavior_names (list, optional): Behavior names
        behavior_colors (list, optional): Colors for visualization
        create_plots (bool): Whether to create plots
        save_csv (bool): Whether to save CSV files
        file_prefix (str, optional): Prefix for output filenames

    Returns:
        dict: Analysis results

    Example:
        >>> behaviors = {'mouse01': array1}
        >>> results = analyze_total_frames(behaviors, 'outputs/mouse01_analysis/', file_prefix='mouse01')
    """
    # Get configuration
    config = get_config()
    if behavior_names is None:
        behavior_names = config.get_behavior_names()
    if behavior_colors is None:
        behavior_colors = config.get_behavior_colors()

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get the single file (should only be one)
    if len(behaviors) != 1:
        raise ValueError(f"Expected single file, got {len(behaviors)} files")

    file_name, behavior_array = next(iter(behaviors.items()))

    # Use file_prefix if provided, otherwise use file_name
    if file_prefix is None:
        file_prefix = file_name

    # Calculate percentages for this file
    percentages_df = calculate_frame_percentages(behavior_array, behavior_names)

    results = {
        'percentages': percentages_df
    }

    # Save CSV
    if save_csv:
        csv_path = output_path / f'{file_prefix}_time_distribution_overall.csv'
        percentages_df.to_csv(csv_path, index=False)
        results['csv_path'] = str(csv_path)

        print(f'Time distribution CSV saved to: {csv_path}')

    # Create plot
    if create_plots:
        plot_path = output_path / f'{file_prefix}_time_distribution.svg'
        fig = plot_behavior_pie(
            percentages_df['frames'].values,
            percentages_df['behavior'].tolist(),
            colors=behavior_colors,
            title='Behavior Distribution (Time %)'
        )
        save_figure(fig, str(plot_path))
        results['plot_path'] = str(plot_path)

        print(f'Plot saved to: {plot_path}')

    return results
