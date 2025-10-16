"""
Binned Timeline Analysis Module

This module analyzes how behavior distribution changes over time by dividing
the recording into time bins and calculating behavior percentages in each bin.

Useful for seeing temporal patterns (e.g., activity increasing/decreasing over time).

Usage:
    from src.core.analysis_binned_timeline import analyze_binned_timeline

    results = analyze_binned_timeline(behaviors_dict, bin_minutes=1, output_dir='outputs/')
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
from src.utils.config_manager import get_config
from src.utils.plotting import plot_binned_timeline, save_figure


def bin_behavior_timeline(predictions: np.ndarray,
                          bin_size_minutes: float = 1.0,
                          framerate: int = None) -> Tuple[np.ndarray, Dict]:
    """
    Bin behavior predictions into time windows.

    Args:
        predictions (np.ndarray): Behavior predictions array
        bin_size_minutes (float): Size of each time bin in minutes
        framerate (int, optional): Video framerate

    Returns:
        tuple: (time_bins, behavior_ratios)
            - time_bins: Array of bin indices
            - behavior_ratios: Dict mapping behavior IDs to arrays of percentages per bin

    Example:
        >>> predictions = np.array([0]*3600 + [1]*3600)  # 2 min of data at 60fps
        >>> bins, ratios = bin_behavior_timeline(predictions, bin_size_minutes=1)
        >>> # First bin: 100% behavior 0
        >>> # Second bin: 100% behavior 1
    """
    if framerate is None:
        config = get_config()
        framerate = config.get_framerate()

    # Calculate bin size in frames
    bin_size_frames = int(bin_size_minutes * 60 * framerate)

    # Number of complete bins
    n_bins = len(predictions) // bin_size_frames

    # Calculate behavior ratios for each bin
    behavior_ratios = {}
    for bin_idx in range(n_bins):
        start_frame = bin_idx * bin_size_frames
        end_frame = (bin_idx + 1) * bin_size_frames
        bin_data = predictions[start_frame:end_frame]

        # Count each behavior in this bin
        unique, counts = np.unique(bin_data, return_counts=True)
        total = len(bin_data)

        for behavior_id, count in zip(unique, counts):
            behavior_id = int(behavior_id)
            if behavior_id not in behavior_ratios:
                behavior_ratios[behavior_id] = []
            ratio = count / total
            behavior_ratios[behavior_id].append(ratio)

    # Ensure all behaviors have values for all bins (fill missing with 0)
    for behavior_id in behavior_ratios:
        if len(behavior_ratios[behavior_id]) < n_bins:
            # Pad with zeros for bins where behavior didn't occur
            behavior_ratios[behavior_id] = [
                behavior_ratios[behavior_id][i] if i < len(behavior_ratios[behavior_id]) else 0
                for i in range(n_bins)
            ]

    time_bins = np.arange(n_bins)
    return time_bins, behavior_ratios


def analyze_binned_timeline(behaviors: Dict[str, np.ndarray],
                            output_dir: str,
                            bin_size_minutes: float = 1.0,
                            behavior_names: list = None,
                            behavior_colors: list = None,
                            framerate: int = None,
                            create_plots: bool = True,
                            save_csv: bool = True) -> Dict:
    """
    Analyze behavior distribution over time for all files.

    Creates timeline plots showing how behavior percentages change over time.

    Args:
        behaviors (dict): Dictionary mapping file names to behavior arrays
        output_dir (str): Directory for output files
        bin_size_minutes (float): Size of time bins in minutes
        behavior_names (list, optional): Behavior names
        behavior_colors (list, optional): Colors for visualization
        framerate (int, optional): Video framerate
        create_plots (bool): Whether to create plots
        save_csv (bool): Whether to save CSV files

    Returns:
        dict: Analysis results

    Example:
        >>> behaviors = {'file1': array1, 'file2': array2}
        >>> results = analyze_binned_timeline(
        >>>     behaviors,
        >>>     output_dir='outputs/timeline/',
        >>>     bin_size_minutes=1.0
        >>> )
    """
    # Get configuration
    config = get_config()
    if behavior_names is None:
        behavior_names = config.get_behavior_names()
    if behavior_colors is None:
        behavior_colors = config.get_behavior_colors()
    if framerate is None:
        framerate = config.get_framerate()

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Bin all files and calculate means/SEMs
    all_binned_data = {i: [] for i in range(len(behavior_names))}

    for file_name, behavior_array in behaviors.items():
        time_bins, ratios = bin_behavior_timeline(behavior_array, bin_size_minutes, framerate)

        # Store ratios for each behavior
        for behavior_id, ratio_array in ratios.items():
            all_binned_data[behavior_id].append(ratio_array)

    # Calculate mean and SEM across files
    n_bins = len(time_bins)
    behavior_stats = {}

    for behavior_id in range(len(behavior_names)):
        if len(all_binned_data[behavior_id]) > 0:
            # Convert to 2D array (files × bins)
            data_array = np.array(all_binned_data[behavior_id])

            # Calculate statistics
            mean_vals = np.mean(data_array, axis=0)
            sem_vals = np.std(data_array, axis=0) / np.sqrt(len(data_array))

            behavior_stats[behavior_names[behavior_id]] = (mean_vals, sem_vals)

    results = {
        'time_bins': time_bins,
        'behavior_stats': behavior_stats,
        'bin_size_minutes': bin_size_minutes
    }

    # Save CSV
    if save_csv:
        # Create DataFrame with all data
        csv_data = {'time_bin': time_bins}
        for behavior_name, (mean_vals, sem_vals) in behavior_stats.items():
            csv_data[f'{behavior_name}_mean'] = mean_vals
            csv_data[f'{behavior_name}_sem'] = sem_vals

        df = pd.DataFrame(csv_data)
        csv_path = output_path / f'binned_timeline_{bin_size_minutes}min.csv'
        df.to_csv(csv_path, index=False)
        results['csv_path'] = str(csv_path)

        print(f'CSV saved to: {csv_path}')

    # Create plot
    if create_plots:
        plot_path = output_path / f'binned_timeline_{bin_size_minutes}min.svg'
        fig = plot_binned_timeline(
            time_bins,
            behavior_stats,
            colors=behavior_colors,
            title=f'Behavior Timeline ({bin_size_minutes}-minute bins)',
            xlabel=f'Time bin ({bin_size_minutes} min each)',
            ylabel='Proportion of time'
        )
        save_figure(fig, str(plot_path))
        results['plot_path'] = str(plot_path)

        print(f'Plot saved to: {plot_path}')

    return results
