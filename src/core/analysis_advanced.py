"""
Advanced Analysis Module

This module contains advanced analyses including:
- Location/trajectory tracking
- Kinematics (speed, acceleration)
- Distance traveled heatmaps

These analyses require pose data in addition to behavior classifications.

Usage:
    from src.core.analysis_advanced import analyze_location, analyze_kinematics

    # Analyze movement trajectories
    results = analyze_location(pose_data, behaviors, output_dir='outputs/')

    # Analyze movement kinematics
    results = analyze_kinematics(pose_data, behaviors, output_dir='outputs/')
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
from src.utils.config_manager import get_config
from src.utils.plotting import plot_trajectory, plot_heatmap, save_figure


def analyze_location(pose_data: Dict[str, np.ndarray],
                     behaviors: Dict[str, np.ndarray],
                     output_dir: str,
                     keypoint_name: str = "nose",
                     create_plots: bool = True,
                     save_csv: bool = True) -> Dict:
    """
    Analyze location and movement trajectories.

    Tracks the position of a specific keypoint (e.g., nose) over time
    and visualizes movement patterns.

    Args:
        pose_data (dict): Dictionary mapping file names to pose arrays
                         Shape: (frames, keypoints*2) where keypoints*2 = x,y coords
        behaviors (dict): Dictionary mapping file names to behavior arrays
        output_dir (str): Directory for output files
        keypoint_name (str): Name of keypoint to track (default: "nose")
        create_plots (bool): Whether to create plots
        save_csv (bool): Whether to save CSV files

    Returns:
        dict: Analysis results

    Example:
        >>> pose_data = {'file1': pose_array}  # Shape: (1000, 40) for 20 keypoints
        >>> behaviors = {'file1': behavior_array}
        >>> results = analyze_location(pose_data, behaviors, 'outputs/location/')
    """
    # Get keypoint index
    config = get_config()
    keypoints = config.get_keypoints()

    if keypoint_name not in keypoints:
        raise ValueError(f"Keypoint '{keypoint_name}' not found. Available: {keypoints}")

    keypoint_idx = keypoints.index(keypoint_name)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {}

    for file_name in pose_data.keys():
        if file_name not in behaviors:
            continue

        pose = pose_data[file_name]
        behavior = behaviors[file_name]

        # Extract x,y coordinates for the keypoint
        x_coords = pose[:, keypoint_idx * 2]
        y_coords = pose[:, keypoint_idx * 2 + 1]

        # Save trajectory CSV
        if save_csv:
            traj_df = pd.DataFrame({
                'frame': range(len(x_coords)),
                'x': x_coords,
                'y': y_coords,
                'behavior_id': behavior
            })
            csv_path = output_path / f'{file_name}_trajectory.csv'
            traj_df.to_csv(csv_path, index=False)

        # Create trajectory plot
        if create_plots:
            plot_path = output_path / f'{file_name}_trajectory.svg'
            fig = plot_trajectory(
                x_coords, y_coords,
                title=f'{file_name} - {keypoint_name} Trajectory'
            )
            save_figure(fig, str(plot_path))

    print(f'Location analysis saved to: {output_dir}')
    return results


def analyze_kinematics(pose_data: Dict[str, np.ndarray],
                       behaviors: Dict[str, np.ndarray],
                       output_dir: str,
                       keypoint_name: str = "nose",
                       framerate: int = None,
                       create_plots: bool = True,
                       save_csv: bool = True) -> Dict:
    """
    Analyze movement kinematics (speed, distance traveled).

    Calculates movement metrics for each behavior.

    Args:
        pose_data (dict): Dictionary mapping file names to pose arrays
        behaviors (dict): Dictionary mapping file names to behavior arrays
        output_dir (str): Directory for output files
        keypoint_name (str): Name of keypoint to track
        framerate (int, optional): Video framerate
        create_plots (bool): Whether to create plots
        save_csv (bool): Whether to save CSV files

    Returns:
        dict: Analysis results with movement statistics

    Example:
        >>> pose_data = {'file1': pose_array}
        >>> behaviors = {'file1': behavior_array}
        >>> results = analyze_kinematics(pose_data, behaviors, 'outputs/kinematics/')
        >>> print(results['statistics'])
    """
    # Get configuration
    config = get_config()
    keypoints = config.get_keypoints()
    behavior_names = config.get_behavior_names()
    pixel_to_cm = config.get_pixel_to_cm()

    if framerate is None:
        framerate = config.get_framerate()

    if keypoint_name not in keypoints:
        raise ValueError(f"Keypoint '{keypoint_name}' not found")

    keypoint_idx = keypoints.index(keypoint_name)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_kinematics_data = []

    for file_name in pose_data.keys():
        if file_name not in behaviors:
            continue

        pose = pose_data[file_name]
        behavior = behaviors[file_name]

        # Extract coordinates
        x_coords = pose[:, keypoint_idx * 2]
        y_coords = pose[:, keypoint_idx * 2 + 1]

        # Calculate frame-by-frame displacement
        dx = np.diff(x_coords)
        dy = np.diff(y_coords)
        displacement_pixels = np.sqrt(dx**2 + dy**2)
        displacement_cm = displacement_pixels * pixel_to_cm

        # Calculate speed (cm/s)
        speed_cm_per_sec = displacement_cm * framerate

        # Pad to match behavior array length
        speed_cm_per_sec = np.pad(speed_cm_per_sec, (0, 1), constant_values=0)

        # Group by behavior
        for behavior_id in range(len(behavior_names)):
            behavior_mask = behavior == behavior_id
            behavior_speeds = speed_cm_per_sec[behavior_mask]

            if len(behavior_speeds) > 0:
                all_kinematics_data.append({
                    'file': file_name,
                    'behavior': behavior_names[behavior_id],
                    'mean_speed_cm_s': np.mean(behavior_speeds),
                    'max_speed_cm_s': np.max(behavior_speeds),
                    'total_distance_cm': np.sum(displacement_cm[behavior_mask[:-1]])
                })

    # Create statistics DataFrame
    kinematics_df = pd.DataFrame(all_kinematics_data)

    results = {'statistics': kinematics_df}

    # Save CSV
    if save_csv:
        csv_path = output_path / 'kinematics_statistics.csv'
        kinematics_df.to_csv(csv_path, index=False)
        results['csv_path'] = str(csv_path)
        print(f'Kinematics statistics saved to: {csv_path}')

    return results


def create_distance_heatmap(pose_data: np.ndarray,
                            output_path: str,
                            keypoint_name: str = "nose",
                            grid_size: int = 50) -> str:
    """
    Create a heatmap showing where the animal spent most time.

    Args:
        pose_data (np.ndarray): Pose array for one file
        output_path (str): Path to save the heatmap
        keypoint_name (str): Keypoint to track
        grid_size (int): Number of grid cells per dimension

    Returns:
        str: Path to saved heatmap

    Example:
        >>> pose = np.load('pose_data.npy')
        >>> path = create_distance_heatmap(pose, 'outputs/heatmap.svg')
    """
    config = get_config()
    keypoints = config.get_keypoints()
    keypoint_idx = keypoints.index(keypoint_name)

    # Extract coordinates
    x_coords = pose_data[:, keypoint_idx * 2]
    y_coords = pose_data[:, keypoint_idx * 2 + 1]

    # Create 2D histogram (heatmap)
    heatmap, xedges, yedges = np.histogram2d(
        x_coords, y_coords,
        bins=grid_size
    )

    # Create plot
    fig = plot_heatmap(
        heatmap.T,
        title=f'Location Heatmap - {keypoint_name}',
        xlabel='X Position',
        ylabel='Y Position',
        cmap='hot'
    )

    save_figure(fig, output_path)
    print(f'Heatmap saved to: {output_path}')

    return output_path
