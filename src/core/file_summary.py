"""
File Summary Module for LUPE Analysis Tool

This module generates comprehensive summaries for DeepLabCut CSV files
processed through the LUPE pipeline. The summary includes general file
information (frames, duration, framerate) and behavior statistics.

The summary is saved as a CSV file in the output directory after
classification is complete.

Usage:
    from src.core.file_summary import generate_dlc_summary

    # Generate summary after classification
    generate_dlc_summary(
        dlc_df=dlc_dataframe,
        predictions=behavior_predictions,
        framerate=60,
        file_path='path/to/video.csv',
        output_dir='outputs/',
        behavior_names=['Still', 'Walking', 'Rearing']
    )

    # Result: Creates '{filename}_summary.csv' in output directory
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, List
from pathlib import Path


def generate_dlc_summary(dlc_df: pd.DataFrame,
                        predictions: np.ndarray,
                        framerate: float,
                        file_path: str,
                        output_dir: str,
                        behavior_names: Optional[List[str]] = None) -> str:
    """
    Generate a comprehensive summary for a processed DLC CSV file.

    This function creates a detailed summary containing:
    - File information (name, path, processing timestamp)
    - Frame and timing data (total frames, duration in sec/min/hr, FPS)
    - Keypoint information (count and names)
    - Behavior statistics (frame counts and percentages per behavior)

    The summary is saved as a CSV file in the output directory with the
    format '{filename}_summary.csv'.

    Args:
        dlc_df (pd.DataFrame): Original DeepLabCut DataFrame with multi-level headers
        predictions (np.ndarray): Array of behavior predictions (one ID per frame)
        framerate (float): Video framerate in frames per second
        file_path (str): Original path to the DLC CSV file
        output_dir (str): Directory where summary file will be saved
        behavior_names (list, optional): List of behavior names. If None,
                                        uses generic names (Behavior 0, 1, 2, etc.)

    Returns:
        str: Path to the generated summary CSV file

    Raises:
        ValueError: If predictions length doesn't match DataFrame rows
        OSError: If output directory cannot be created or written to

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> # Load DLC data
        >>> dlc_df = pd.read_csv('mouse_video.csv', header=[0,1,2])
        >>>
        >>> # After classification
        >>> predictions = np.array([0, 0, 1, 1, 2, 2, 0, 0])  # behavior IDs
        >>>
        >>> # Generate summary
        >>> summary_path = generate_dlc_summary(
        ...     dlc_df=dlc_df,
        ...     predictions=predictions,
        ...     framerate=60.0,
        ...     file_path='data/mouse_video.csv',
        ...     output_dir='outputs/mouse_video',
        ...     behavior_names=['Still', 'Walking', 'Rearing']
        ... )
        >>>
        >>> print(f"Summary saved to: {summary_path}")
        # Summary saved to: outputs/mouse_video/mouse_video_summary.csv
    """
    # Validate inputs
    # Note: dlc_df might only have headers (nrows=0), so use predictions length for frame count
    if dlc_df.shape[0] > 0 and len(predictions) != dlc_df.shape[0]:
        raise ValueError(
            f"Predictions length ({len(predictions)}) does not match "
            f"DataFrame rows ({dlc_df.shape[0]})"
        )

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Extract filename without extension
    filename = Path(file_path).stem

    # Calculate basic file statistics
    # Use predictions length since dlc_df might only contain headers (nrows=0)
    total_frames = len(predictions)
    duration_seconds = total_frames / framerate
    duration_minutes = duration_seconds / 60
    duration_hours = duration_minutes / 60

    # Get keypoint information from DLC DataFrame headers
    keypoint_names = _extract_keypoint_names(dlc_df)
    num_keypoints = len(keypoint_names)

    # Get current timestamp
    processing_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Calculate behavior statistics
    behavior_stats = _calculate_behavior_statistics(predictions, behavior_names)

    # Build summary data as list of (field, value) tuples
    summary_data = []

    # File information
    summary_data.append(("Filename", filename))
    summary_data.append(("Original Path", file_path))
    summary_data.append(("Processing Date", processing_time))
    summary_data.append(("", ""))  # Empty row for readability

    # Frame and timing information
    summary_data.append(("Total Frames", f"{total_frames:,}"))
    summary_data.append(("Duration (seconds)", f"{duration_seconds:.2f}"))
    summary_data.append(("Duration (minutes)", f"{duration_minutes:.2f}"))
    summary_data.append(("Duration (hours)", f"{duration_hours:.4f}"))
    summary_data.append(("Framerate (fps)", f"{framerate:.1f}"))
    summary_data.append(("", ""))  # Empty row

    # Keypoint information
    summary_data.append(("Number of Keypoints", num_keypoints))
    summary_data.append(("Keypoint Names", ", ".join(keypoint_names)))
    summary_data.append(("", ""))  # Empty row

    # Behavior statistics
    summary_data.append(("Behavior Statistics", ""))
    for behavior_id, behavior_name, frame_count, percentage in behavior_stats:
        summary_data.append((f"  {behavior_name} - Frames", f"{frame_count:,}"))
        summary_data.append((f"  {behavior_name} - Percentage", f"{percentage:.2f}%"))

    # Create DataFrame with Field and Value columns
    summary_df = pd.DataFrame(summary_data, columns=["Field", "Value"])

    # Save to CSV
    partial_name = extract_partial_filename(file_path)
    summary_filename = f"{partial_name}_summary.csv"
    summary_path = os.path.join(output_dir, summary_filename)
    summary_df.to_csv(summary_path, index=False)


    return summary_path


def _extract_keypoint_names(dlc_df: pd.DataFrame) -> List[str]:
    """
    Extract keypoint names from DeepLabCut DataFrame headers.

    DLC CSVs have multi-level headers where the second level contains
    body part (keypoint) names. This function extracts unique keypoint
    names from the column structure.

    Args:
        dlc_df (pd.DataFrame): DeepLabCut DataFrame with multi-level columns

    Returns:
        list: Unique keypoint names in order of appearance

    Example:
        >>> # DLC DataFrame columns:
        >>> # Level 0: scorer, scorer, scorer, ...
        >>> # Level 1: nose, nose, nose, ear_left, ear_left, ear_left, ...
        >>> # Level 2: x, y, likelihood, x, y, likelihood, ...
        >>> keypoints = _extract_keypoint_names(dlc_df)
        >>> print(keypoints)
        # ['nose', 'ear_left', 'ear_right', 'neck', ...]
    """
    # Check if columns are multi-level
    if isinstance(dlc_df.columns, pd.MultiIndex):
        # Get the second level (body part names)
        # Typically this is at index 1, but may vary
        if dlc_df.columns.nlevels >= 2:
            bodypart_level = dlc_df.columns.get_level_values(1)
        else:
            bodypart_level = dlc_df.columns.get_level_values(0)
    else:
        # Single-level columns - try to parse from column names
        # Format might be: scorer_bodypart_x, scorer_bodypart_y, etc.
        bodypart_level = dlc_df.columns

    # Extract unique keypoint names while preserving order
    # Remove duplicates (x, y, likelihood all have same keypoint name)
    keypoints = []
    seen = set()
    for name in bodypart_level:
        if name not in seen:
            keypoints.append(name)
            seen.add(name)

    # Filter out coordinate type labels (x, y, likelihood)
    # These might appear in the level we're looking at
    coordinate_labels = {'x', 'y', 'likelihood', 'coords'}
    keypoints = [kp for kp in keypoints if kp.lower() not in coordinate_labels]

    return keypoints


def _calculate_behavior_statistics(predictions: np.ndarray,
                                   behavior_names: Optional[List[str]] = None) -> List[tuple]:
    """
    Calculate frame counts and percentages for each behavior.

    Args:
        predictions (np.ndarray): Array of behavior IDs for each frame
        behavior_names (list, optional): List of behavior names. If None,
                                        uses generic names like "Behavior 0"

    Returns:
        list: List of tuples (behavior_id, behavior_name, frame_count, percentage)

    Example:
        >>> predictions = np.array([0, 0, 0, 1, 1, 2, 2, 2])
        >>> stats = _calculate_behavior_statistics(predictions, ['Still', 'Walk', 'Rear'])
        >>> for id, name, count, pct in stats:
        ...     print(f"{name}: {count} frames ({pct:.1f}%)")
        # Still: 3 frames (37.5%)
        # Walk: 2 frames (25.0%)
        # Rear: 3 frames (37.5%)
    """
    # Get unique behavior IDs and their counts
    unique_behaviors, counts = np.unique(predictions, return_counts=True)

    # Calculate total frames for percentage
    total_frames = len(predictions)

    # Build statistics list
    stats = []
    for behavior_id, count in zip(unique_behaviors, counts):
        # Get behavior name
        if behavior_names is not None and behavior_id < len(behavior_names):
            behavior_name = behavior_names[int(behavior_id)]
        else:
            behavior_name = f"Behavior {int(behavior_id)}"

        # Calculate percentage
        percentage = (count / total_frames) * 100

        stats.append((
            int(behavior_id),
            behavior_name,
            int(count),
            float(percentage)
        ))

    return stats


def generate_batch_summary(summary_files: List[str], output_path: str) -> str:
    """
    Aggregate multiple individual summaries into one master summary table.

    This function is useful when processing multiple DLC files in batch mode.
    It combines the key metrics from each file into a single CSV for comparison.

    Note: This function is currently not called automatically, but is available
    for users who want to create aggregate summaries manually.

    Args:
        summary_files (list): List of paths to individual summary CSV files
        output_path (str): Path where the aggregate summary should be saved

    Returns:
        str: Path to the generated aggregate summary file

    Example:
        >>> summary_files = [
        ...     'outputs/video1/video1_summary.csv',
        ...     'outputs/video2/video2_summary.csv',
        ...     'outputs/video3/video3_summary.csv'
        ... ]
        >>> batch_summary = generate_batch_summary(summary_files, 'batch_summary.csv')
        >>> print(f"Batch summary saved to: {batch_summary}")
    """
    # This function is a placeholder for future batch summary functionality
    # Currently not implemented as per user requirements (individual summaries only)
    raise NotImplementedError(
        "Batch summary generation is not currently implemented. "
        "Only individual file summaries are generated automatically."
    )
