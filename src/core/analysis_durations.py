"""
Behavior Duration Analysis Module

This module analyzes how long each behavior bout lasts (duration analysis).
Provides statistics on bout durations for each behavior.

Usage:
    from src.core.analysis_durations import analyze_bout_durations

    results = analyze_bout_durations(behaviors_dict, output_dir='outputs/')
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
from src.utils.config_manager import get_config
from src.utils.plotting import plot_box_whisker, save_figure
from src.core.classification import get_behavior_bouts


def calculate_bout_durations(predictions: np.ndarray,
                             framerate: int = None) -> Dict[int, List[float]]:
    """
    Calculate durations of all bouts for each behavior.

    Args:
        predictions (np.ndarray): Behavior predictions array
        framerate (int, optional): Video framerate for converting frames to seconds

    Returns:
        dict: Dictionary mapping behavior IDs to lists of bout durations (in seconds)

    Example:
        >>> predictions = np.array([0,0,0, 1,1, 0,0,0,0])
        >>> durations = calculate_bout_durations(predictions, framerate=60)
        >>> # {0: [0.05, 0.067], 1: [0.033]}
        >>> # Behavior 0 had bouts of 3 frames and 4 frames
        >>> # Behavior 1 had one bout of 2 frames
    """
    if framerate is None:
        config = get_config()
        framerate = config.get_framerate()

    # Get bout information
    starts, ends, labels, durations_frames = get_behavior_bouts(predictions)

    # Convert frame durations to seconds
    durations_seconds = durations_frames / framerate

    # Group by behavior
    behavior_durations = {}
    for label, duration in zip(labels, durations_seconds):
        label = int(label)
        if label not in behavior_durations:
            behavior_durations[label] = []
        behavior_durations[label].append(duration)

    return behavior_durations


def analyze_bout_durations(behaviors: Dict[str, np.ndarray],
                           output_dir: str,
                           behavior_names: list = None,
                           framerate: int = None,
                           create_plots: bool = True,
                           save_csv: bool = True) -> Dict:
    """
    Analyze bout durations across all files.

    Calculates statistics (mean, median, std, min, max) for bout durations
    of each behavior.

    Args:
        behaviors (dict): Dictionary mapping file names to behavior arrays
        output_dir (str): Directory for output files
        behavior_names (list, optional): Behavior names
        framerate (int, optional): Video framerate
        create_plots (bool): Whether to create plots
        save_csv (bool): Whether to save CSV files

    Returns:
        dict: Analysis results

    Example:
        >>> behaviors = {'file1': array1, 'file2': array2}
        >>> results = analyze_bout_durations(behaviors, 'outputs/durations/')
        >>> print(results['statistics'])
    """
    # Get configuration
    config = get_config()
    if behavior_names is None:
        behavior_names = config.get_behavior_names()
    if framerate is None:
        framerate = config.get_framerate()

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect all durations
    all_durations = {i: [] for i in range(len(behavior_names))}

    for file_name, behavior_array in behaviors.items():
        file_durations = calculate_bout_durations(behavior_array, framerate)
        for behavior_id, durations in file_durations.items():
            all_durations[behavior_id].extend(durations)

    # Calculate statistics
    stats_data = []
    for behavior_id in range(len(behavior_names)):
        durations = all_durations[behavior_id]
        if len(durations) > 0:
            stats_data.append({
                'behavior': behavior_names[behavior_id],
                'mean_duration_sec': np.mean(durations),
                'median_duration_sec': np.median(durations),
                'std_duration_sec': np.std(durations),
                'min_duration_sec': np.min(durations),
                'max_duration_sec': np.max(durations),
                'total_bouts': len(durations)
            })

    stats_df = pd.DataFrame(stats_data)

    results = {
        'statistics': stats_df,
        'all_durations': all_durations
    }

    # Save CSV
    if save_csv:
        # Statistics summary
        stats_path = output_path / 'bout_durations_statistics.csv'
        stats_df.to_csv(stats_path, index=False)
        results['csv_stats_path'] = str(stats_path)

        # Raw durations (long format for plotting)
        raw_data = []
        for behavior_id, durations in all_durations.items():
            for duration in durations:
                raw_data.append({
                    'behavior': behavior_names[behavior_id],
                    'duration_sec': duration
                })
        raw_df = pd.DataFrame(raw_data)
        raw_path = output_path / 'bout_durations_raw.csv'
        raw_df.to_csv(raw_path, index=False)
        results['csv_raw_path'] = str(raw_path)

        print(f'CSV files saved to: {output_dir}')

    # Create plot
    if create_plots and len(raw_data) > 0:
        plot_path = output_path / 'bout_durations_boxplot.svg'
        fig = plot_box_whisker(
            raw_df,
            x_col='behavior',
            y_col='duration_sec',
            title='Behavior Bout Durations',
            ylabel='Duration (seconds)'
        )
        save_figure(fig, str(plot_path))
        results['plot_path'] = str(plot_path)

        print(f'Plot saved to: {plot_path}')

    return results
