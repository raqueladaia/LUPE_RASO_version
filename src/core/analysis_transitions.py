"""
Behavior Transitions Analysis Module

This module analyzes how behaviors transition from one to another.
Creates transition matrices showing the probability of switching between behaviors.

Usage:
    from src.core.analysis_transitions import analyze_transitions

    results = analyze_transitions(behaviors_dict, output_dir='outputs/')
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
from src.utils.config_manager import get_config
from src.utils.plotting import plot_heatmap, save_figure
from src.core.classification import behavior_transition_matrix


def analyze_transitions(behaviors: Dict[str, np.ndarray],
                       output_dir: str,
                       behavior_names: list = None,
                       create_plots: bool = True,
                       save_csv: bool = True,
                       file_prefix: str = None) -> Dict:
    """
    Analyze behavior transitions for a single file.

    Calculates transition probabilities: how likely is each behavior to
    follow each other behavior?

    Args:
        behaviors (dict): Dictionary mapping file name to behavior array (single file)
        output_dir (str): Directory for output files
        behavior_names (list, optional): Behavior names
        create_plots (bool): Whether to create plots
        save_csv (bool): Whether to save CSV files
        file_prefix (str, optional): Prefix for output filenames

    Returns:
        dict: Analysis results containing transition matrix

    Example:
        >>> behaviors = {'mouse01': array1}
        >>> results = analyze_transitions(behaviors, 'outputs/mouse01_analysis/', file_prefix='mouse01')
        >>> print(results['transition_matrix'])
        >>> # Shows: probability of transitioning from each behavior (rows)
        >>> #        to each behavior (columns)
    """
    # Get configuration
    config = get_config()
    if behavior_names is None:
        behavior_names = config.get_behavior_names()

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

    # Calculate transition matrix for this file
    transition_df = behavior_transition_matrix(behavior_array, behavior_names)

    results = {
        'transition_matrix': transition_df
    }

    # Save CSV
    if save_csv:
        csv_path = output_path / f'{file_prefix}_transitions_matrix.csv'
        transition_df.to_csv(csv_path)
        results['csv_path'] = str(csv_path)

        print(f'Transition matrix saved to: {csv_path}')

    # Create heatmap
    if create_plots:
        plot_path = output_path / f'{file_prefix}_transitions_heatmap.svg'
        fig = plot_heatmap(
            transition_df.values,
            row_labels=behavior_names,
            col_labels=behavior_names,
            title='Behavior Transition Probabilities',
            xlabel='To Behavior',
            ylabel='From Behavior',
            cmap='YlOrRd',
            annot=True
        )
        save_figure(fig, str(plot_path))
        results['plot_path'] = str(plot_path)

        print(f'Heatmap saved to: {plot_path}')

    return results
