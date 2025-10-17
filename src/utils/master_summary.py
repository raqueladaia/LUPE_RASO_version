"""
Master Summary Generator

This module creates a master ANALYSIS_SUMMARY.csv file that consolidates
all key metrics from individual analysis outputs into a single, easy-to-read file.

The master summary includes:
- Bout counts per behavior
- Time distribution percentages
- Bout duration statistics (mean, median)
- Top behavior transitions

Usage:
    from src.utils.master_summary import create_analysis_summary

    create_analysis_summary(
        file_prefix='mouse01',
        analysis_dir='outputs/mouse01/mouse01_analysis',
        behavior_names=['still', 'walking', 'rearing', 'grooming', 'sniffing', 'other']
    )
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict


def create_analysis_summary(file_prefix: str,
                            analysis_dir: str,
                            behavior_names: List[str]) -> str:
    """
    Create a master summary CSV consolidating all analysis results.

    Reads individual analysis outputs and creates a single ANALYSIS_SUMMARY.csv
    with all key metrics in an easy-to-read format.

    Args:
        file_prefix (str): File prefix (e.g., 'mouse01')
        analysis_dir (str): Directory containing analysis outputs
        behavior_names (list): List of behavior names

    Returns:
        str: Path to the created summary file

    Example:
        >>> create_analysis_summary(
        ...     file_prefix='mouse01',
        ...     analysis_dir='outputs/mouse01/mouse01_analysis',
        ...     behavior_names=['still', 'walking', 'rearing', 'grooming', 'sniffing', 'other']
        ... )
        'outputs/mouse01/mouse01_analysis/mouse01_ANALYSIS_SUMMARY.csv'
    """
    analysis_path = Path(analysis_dir)

    # Dictionary to store all summary data
    summary_data = {}

    # 1. Read bout counts
    try:
        bout_counts_path = analysis_path / f'{file_prefix}_bout_counts_summary.csv'
        if bout_counts_path.exists():
            df = pd.read_csv(bout_counts_path)
            for _, row in df.iterrows():
                behavior = row['behavior']
                summary_data[f'{behavior}_bout_count'] = int(row['bout_count']) if pd.notna(row['bout_count']) else 0
    except Exception as e:
        print(f"Warning: Could not read bout counts: {e}")

    # 2. Read time distribution (percentages)
    try:
        time_dist_path = analysis_path / f'{file_prefix}_time_distribution_overall.csv'
        if time_dist_path.exists():
            df = pd.read_csv(time_dist_path)
            for _, row in df.iterrows():
                behavior = row['behavior']
                summary_data[f'{behavior}_time_pct'] = round(row['percentage'], 2)
                summary_data[f'{behavior}_frames'] = int(row['frames'])
    except Exception as e:
        print(f"Warning: Could not read time distribution: {e}")

    # 3. Read bout duration statistics
    try:
        durations_path = analysis_path / f'{file_prefix}_bout_durations_statistics.csv'
        if durations_path.exists():
            df = pd.read_csv(durations_path)
            for _, row in df.iterrows():
                behavior = row['behavior']
                summary_data[f'{behavior}_mean_duration_sec'] = round(row['mean_duration_sec'], 3)
                summary_data[f'{behavior}_median_duration_sec'] = round(row['median_duration_sec'], 3)
    except Exception as e:
        print(f"Warning: Could not read bout durations: {e}")

    # 4. Read transition matrix and extract top transitions
    try:
        transitions_path = analysis_path / f'{file_prefix}_transitions_matrix.csv'
        if transitions_path.exists():
            df = pd.read_csv(transitions_path, index_col=0)

            # Find top 5 transitions (excluding self-transitions)
            transitions_list = []
            for from_behavior in df.index:
                for to_behavior in df.columns:
                    if from_behavior != to_behavior:  # Exclude self-transitions
                        prob = df.loc[from_behavior, to_behavior]
                        if pd.notna(prob) and prob > 0:
                            transitions_list.append({
                                'from': from_behavior,
                                'to': to_behavior,
                                'probability': prob
                            })

            # Sort by probability and get top 5
            transitions_list.sort(key=lambda x: x['probability'], reverse=True)
            top_5 = transitions_list[:5]

            # Add to summary
            for i, trans in enumerate(top_5, 1):
                summary_data[f'top_transition_{i}'] = f"{trans['from']} → {trans['to']}"
                summary_data[f'top_transition_{i}_prob'] = round(trans['probability'], 3)
    except Exception as e:
        print(f"Warning: Could not read transitions: {e}")

    # Create DataFrame with organized sections
    summary_rows = []

    # Section 1: Bout Counts
    summary_rows.append({'Metric': '=== BOUT COUNTS ===', 'Value': ''})
    for behavior in behavior_names:
        key = f'{behavior}_bout_count'
        if key in summary_data:
            summary_rows.append({'Metric': f'{behavior} (bouts)', 'Value': summary_data[key]})

    summary_rows.append({'Metric': '', 'Value': ''})  # Empty row

    # Section 2: Time Distribution
    summary_rows.append({'Metric': '=== TIME DISTRIBUTION ===', 'Value': ''})
    for behavior in behavior_names:
        key_pct = f'{behavior}_time_pct'
        key_frames = f'{behavior}_frames'
        if key_pct in summary_data:
            summary_rows.append({
                'Metric': f'{behavior} (%)',
                'Value': f"{summary_data[key_pct]}% ({summary_data.get(key_frames, 'N/A')} frames)"
            })

    summary_rows.append({'Metric': '', 'Value': ''})  # Empty row

    # Section 3: Bout Durations
    summary_rows.append({'Metric': '=== BOUT DURATIONS (seconds) ===', 'Value': ''})
    for behavior in behavior_names:
        key_mean = f'{behavior}_mean_duration_sec'
        key_median = f'{behavior}_median_duration_sec'
        if key_mean in summary_data:
            summary_rows.append({
                'Metric': f'{behavior} (mean)',
                'Value': summary_data[key_mean]
            })
            summary_rows.append({
                'Metric': f'{behavior} (median)',
                'Value': summary_data[key_median]
            })

    summary_rows.append({'Metric': '', 'Value': ''})  # Empty row

    # Section 4: Top Transitions
    summary_rows.append({'Metric': '=== TOP TRANSITIONS ===', 'Value': ''})
    for i in range(1, 6):
        key = f'top_transition_{i}'
        key_prob = f'top_transition_{i}_prob'
        if key in summary_data:
            summary_rows.append({
                'Metric': summary_data[key],
                'Value': summary_data[key_prob]
            })

    # Create DataFrame and save
    summary_df = pd.DataFrame(summary_rows)
    output_path = analysis_path / f'{file_prefix}_ANALYSIS_SUMMARY.csv'
    summary_df.to_csv(output_path, index=False)

    print(f'Master summary saved to: {output_path}')

    return str(output_path)
