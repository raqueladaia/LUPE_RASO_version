"""
Instance Counts Analysis Module

This module counts how many times each behavior occurs (bout frequency).
A bout is a continuous sequence of the same behavior.

For example, if a mouse is still, then walks, then is still again, that's
2 bouts of "still" and 1 bout of "walking".

Usage:
    from src.core.analysis_instance_counts import analyze_instance_counts

    # Analyze bout counts
    results = analyze_instance_counts(behaviors_dict, output_dir='outputs/')
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
from src.utils.config_manager import get_config
from src.utils.plotting import plot_behavior_bars, save_figure
from src.core.classification import get_behavior_bouts


def count_behavior_bouts(predictions: np.ndarray, behavior_names: list = None) -> list:
    """
    Count the number of bouts for each behavior.

    Args:
        predictions (np.ndarray): Behavior predictions array
        behavior_names (list, optional): List of behavior names

    Returns:
        list: List of bout counts, one per behavior

    Example:
        >>> predictions = np.array([0, 0, 1, 1, 0, 2, 2, 2])
        >>> # Behavior 0 appears 2 times, behavior 1 once, behavior 2 once
        >>> counts = count_behavior_bouts(predictions)
        >>> # Returns: [2, 1, 1]
    """
    if behavior_names is None:
        config = get_config()
        behavior_names = config.get_behavior_names()

    # Get bout information
    bout_starts, bout_ends, bout_labels, bout_durations = get_behavior_bouts(predictions)

    # Count bouts for each behavior
    bout_counts = []
    for behavior_id in range(len(behavior_names)):
        count = np.sum(bout_labels == behavior_id)
        bout_counts.append(count if count > 0 else np.nan)

    return bout_counts


def analyze_instance_counts(behaviors: Dict[str, np.ndarray],
                           output_dir: str,
                           behavior_names: list = None,
                           behavior_colors: list = None,
                           create_plots: bool = True,
                           save_csv: bool = True) -> Dict:
    """
    Analyze and visualize instance counts for all files.

    This function:
    1. Counts behavior bouts for each file
    2. Calculates mean and standard deviation across files
    3. Creates bar charts showing results
    4. Saves data to CSV

    Args:
        behaviors (dict): Dictionary mapping file names to behavior arrays
        output_dir (str): Directory for output files
        behavior_names (list, optional): Behavior names
        behavior_colors (list, optional): Colors for visualization
        create_plots (bool): Whether to create plots
        save_csv (bool): Whether to save CSV files

    Returns:
        dict: Analysis results containing:
            - 'mean_counts': Mean bout count per behavior
            - 'std_counts': Standard deviation of bout counts
            - 'all_counts': Bout counts for each file
            - 'csv_path': Path to saved CSV (if save_csv=True)
            - 'plot_path': Path to saved plot (if create_plots=True)

    Example:
        >>> behaviors = {'file1': array1, 'file2': array2}
        >>> results = analyze_instance_counts(behaviors, 'outputs/instance_counts/')
        >>> print(f"Mean walking bouts: {results['mean_counts'][1]}")
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

    # Count bouts for all files
    all_bout_counts = []
    for file_name, behavior_array in behaviors.items():
        bout_counts = count_behavior_bouts(behavior_array, behavior_names)
        all_bout_counts.append(bout_counts)

    # Convert to numpy array for easy statistics
    all_bout_counts = np.array(all_bout_counts)

    # Calculate statistics
    mean_counts = np.nanmean(all_bout_counts, axis=0)
    std_counts = np.nanstd(all_bout_counts, axis=0)

    results = {
        'mean_counts': mean_counts,
        'std_counts': std_counts,
        'all_counts': all_bout_counts
    }

    # Save CSV
    if save_csv:
        # Summary statistics CSV
        summary_df = pd.DataFrame({
            'behavior': behavior_names,
            'mean_count': mean_counts,
            'std_count': std_counts,
            'color': behavior_colors
        })
        summary_path = output_path / 'instance_counts_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        results['csv_summary_path'] = str(summary_path)

        # Raw counts CSV
        raw_df = pd.DataFrame(all_bout_counts, columns=behavior_names)
        raw_df.insert(0, 'file', list(behaviors.keys()))
        raw_path = output_path / 'instance_counts_raw.csv'
        raw_df.to_csv(raw_path, index=False)
        results['csv_raw_path'] = str(raw_path)

        print(f'CSV files saved to: {output_dir}')

    # Create plot
    if create_plots:
        plot_path = output_path / 'instance_counts.svg'
        fig = plot_behavior_bars(
            mean_counts, std_counts, behavior_names, behavior_colors,
            title='Behavior Instance Counts',
            xlabel='Instance counts',
            ylabel='Behavior',
            horizontal=True
        )
        save_figure(fig, str(plot_path))
        results['plot_path'] = str(plot_path)

        print(f'Plot saved to: {plot_path}')

    return results


def compare_instance_counts(behaviors_dict1: Dict[str, np.ndarray],
                           behaviors_dict2: Dict[str, np.ndarray],
                           label1: str = "Dataset 1",
                           label2: str = "Dataset 2",
                           output_dir: str = None) -> pd.DataFrame:
    """
    Compare instance counts between two datasets.

    Useful for comparing different experimental conditions, time points, etc.

    Args:
        behaviors_dict1 (dict): First dataset
        behaviors_dict2 (dict): Second dataset
        label1 (str): Label for first dataset
        label2 (str): Label for second dataset
        output_dir (str, optional): If provided, saves comparison to CSV

    Returns:
        pd.DataFrame: Comparison results

    Example:
        >>> before_treatment = {'mouse1': ..., 'mouse2': ...}
        >>> after_treatment = {'mouse1': ..., 'mouse2': ...}
        >>> comparison = compare_instance_counts(
        >>>     before_treatment, after_treatment,
        >>>     label1="Before", label2="After",
        >>>     output_dir='outputs/comparison/'
        >>> )
    """
    config = get_config()
    behavior_names = config.get_behavior_names()

    # Count bouts for both datasets
    counts1 = []
    for behavior_array in behaviors_dict1.values():
        counts1.append(count_behavior_bouts(behavior_array, behavior_names))
    counts1 = np.array(counts1)

    counts2 = []
    for behavior_array in behaviors_dict2.values():
        counts2.append(count_behavior_bouts(behavior_array, behavior_names))
    counts2 = np.array(counts2)

    # Calculate statistics
    mean1 = np.nanmean(counts1, axis=0)
    std1 = np.nanstd(counts1, axis=0)
    mean2 = np.nanmean(counts2, axis=0)
    std2 = np.nanstd(counts2, axis=0)

    # Create comparison DataFrame
    comparison_df = pd.DataFrame({
        'behavior': behavior_names,
        f'{label1}_mean': mean1,
        f'{label1}_std': std1,
        f'{label2}_mean': mean2,
        f'{label2}_std': std2,
        'difference': mean2 - mean1,
        'percent_change': ((mean2 - mean1) / mean1) * 100
    })

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        csv_path = output_path / 'instance_counts_comparison.csv'
        comparison_df.to_csv(csv_path, index=False)
        print(f'Comparison saved to: {csv_path}')

    return comparison_df
