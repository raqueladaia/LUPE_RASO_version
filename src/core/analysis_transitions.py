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
                       save_csv: bool = True) -> Dict:
    """
    Analyze behavior transitions across all files.

    Calculates transition probabilities: how likely is each behavior to
    follow each other behavior?

    Args:
        behaviors (dict): Dictionary mapping file names to behavior arrays
        output_dir (str): Directory for output files
        behavior_names (list, optional): Behavior names
        create_plots (bool): Whether to create plots
        save_csv (bool): Whether to save CSV files

    Returns:
        dict: Analysis results containing transition matrices

    Example:
        >>> behaviors = {'file1': array1, 'file2': array2}
        >>> results = analyze_transitions(behaviors, 'outputs/transitions/')
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

    # Calculate transition matrix for each file
    all_transition_matrices = []

    for file_name, behavior_array in behaviors.items():
        trans_matrix = behavior_transition_matrix(behavior_array, behavior_names)
        all_transition_matrices.append(trans_matrix.values)

    # Average across all files
    mean_transition_matrix = np.mean(all_transition_matrices, axis=0)

    # Create DataFrame
    transition_df = pd.DataFrame(
        mean_transition_matrix,
        index=behavior_names,
        columns=behavior_names
    )

    results = {
        'transition_matrix': transition_df
    }

    # Save CSV
    if save_csv:
        csv_path = output_path / 'transition_matrix.csv'
        transition_df.to_csv(csv_path)
        results['csv_path'] = str(csv_path)

        print(f'Transition matrix saved to: {csv_path}')

    # Create heatmap
    if create_plots:
        plot_path = output_path / 'transition_matrix_heatmap.svg'
        fig = plot_heatmap(
            mean_transition_matrix,
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
