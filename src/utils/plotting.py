"""
Plotting Utilities for LUPE Analysis Tool

This module provides consistent visualization functions for all analysis types.
It handles creating plots with proper styling, colors, and formatting based
on the configuration settings.

All plots use behavior-specific colors from the metadata configuration and
can be saved in various formats (SVG, PNG, PDF).

Usage:
    from src.utils.plotting import plot_behavior_pie, plot_behavior_bars
    from src.utils.config_manager import get_config

    config = get_config()
    plot_behavior_pie(values, labels, output_path='plot.svg')
"""

# Configure matplotlib to use thread-safe backend for Windows compatibility
# Must be set BEFORE importing pyplot to avoid threading deadlocks
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend prevents threading issues in background threads

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from src.utils.config_manager import get_config


def setup_plot_style():
    """
    Set up consistent plot styling for all visualizations.

    This applies a clean, professional style to all matplotlib plots.
    """
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['figure.titlesize'] = 14


def save_figure(fig: plt.Figure, output_path: str, dpi: Optional[int] = None):
    """
    Save a matplotlib figure to file.

    Args:
        fig (plt.Figure): The figure to save
        output_path (str): Path where the figure will be saved
        dpi (int, optional): Resolution in dots per inch. If None, uses config default

    Example:
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3], [1, 4, 9])
        >>> save_figure(fig, 'outputs/my_plot.svg')
    """
    if dpi is None:
        config = get_config()
        dpi = config.get_plot_dpi()

    # Create output directory if needed
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')

    # Close the figure to free memory (critical for multi-file processing)
    plt.close(fig)


def plot_behavior_pie(values: np.ndarray,
                      labels: List[str],
                      colors: Optional[List[str]] = None,
                      title: str = "Behavior Distribution",
                      output_path: Optional[str] = None) -> plt.Figure:
    """
    Create a donut/pie chart showing behavior distribution.

    Args:
        values (np.ndarray): Array of values (frame counts or percentages)
        labels (list): List of behavior names
        colors (list, optional): List of colors for each behavior. If None, uses config
        title (str): Plot title
        output_path (str, optional): If provided, saves the plot to this path

    Returns:
        plt.Figure: The created figure

    Example:
        >>> values = np.array([1000, 500, 300])
        >>> labels = ['still', 'walking', 'rearing']
        >>> fig = plot_behavior_pie(values, labels, title='Mouse Behavior')
    """
    if colors is None:
        config = get_config()
        behavior_names = config.get_behavior_names()
        all_colors = config.get_behavior_colors()
        # Map provided labels to colors
        colors = [all_colors[behavior_names.index(label)] for label in labels]

    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create pie chart
    ax.pie(values, colors=colors, labels=labels, autopct='%1.1f%%', pctdistance=0.85)

    # Draw center circle for donut effect
    centre_circle = plt.Circle((0, 0), 0.50, fc='white')
    ax.add_artist(centre_circle)

    ax.set_title(title)

    if output_path:
        save_figure(fig, output_path)

    return fig


def plot_behavior_bars(mean_values: np.ndarray,
                       std_values: np.ndarray,
                       labels: List[str],
                       colors: Optional[List[str]] = None,
                       title: str = "Behavior Counts",
                       xlabel: str = "Behavior",
                       ylabel: str = "Count",
                       output_path: Optional[str] = None,
                       horizontal: bool = True) -> plt.Figure:
    """
    Create a bar chart with error bars for behavior analysis.

    Args:
        mean_values (np.ndarray): Mean values for each behavior
        std_values (np.ndarray): Standard deviation/error for each behavior
        labels (list): List of behavior names
        colors (list, optional): Colors for each bar
        title (str): Plot title
        xlabel (str): X-axis label
        ylabel (str): Y-axis label
        output_path (str, optional): If provided, saves the plot
        horizontal (bool): If True, creates horizontal bars

    Returns:
        plt.Figure: The created figure

    Example:
        >>> means = np.array([50, 30, 20])
        >>> stds = np.array([5, 3, 2])
        >>> labels = ['still', 'walking', 'rearing']
        >>> fig = plot_behavior_bars(means, stds, labels, ylabel='Instance Count')
    """
    if colors is None:
        config = get_config()
        behavior_names = config.get_behavior_names()
        all_colors = config.get_behavior_colors()
        colors = [all_colors[behavior_names.index(label)] for label in labels]

    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    if horizontal:
        ax.barh(labels, mean_values, xerr=std_values, color=colors, zorder=3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    else:
        ax.bar(labels, mean_values, yerr=std_values, color=colors, zorder=3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.xticks(rotation=45, ha='right')

    ax.set_title(title)
    ax.grid(True, zorder=0)

    # Style axis spines
    for spine in ax.spines.values():
        spine.set_color('#D3D3D3')

    if output_path:
        save_figure(fig, output_path)

    return fig


def plot_binned_timeline(time_bins: np.ndarray,
                         behavior_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
                         colors: Optional[List[str]] = None,
                         title: str = "Behavior Timeline",
                         xlabel: str = "Time bin",
                         ylabel: str = "Percent",
                         output_path: Optional[str] = None) -> plt.Figure:
    """
    Create a timeline plot showing behavior percentages over time.

    Args:
        time_bins (np.ndarray): Array of time bin indices
        behavior_data (dict): Dictionary mapping behavior names to (mean, sem) tuples
        colors (list, optional): Colors for each behavior line
        title (str): Plot title
        xlabel (str): X-axis label
        ylabel (str): Y-axis label
        output_path (str, optional): If provided, saves the plot

    Returns:
        plt.Figure: The created figure

    Example:
        >>> time_bins = np.arange(30)  # 30 minutes
        >>> behavior_data = {
        >>>     'still': (mean_array, sem_array),
        >>>     'walking': (mean_array, sem_array)
        >>> }
        >>> fig = plot_binned_timeline(time_bins, behavior_data)
    """
    config = get_config()
    all_behavior_names = config.get_behavior_names()
    all_colors = config.get_behavior_colors()

    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    for behavior_name, (mean_vals, sem_vals) in behavior_data.items():
        color = all_colors[all_behavior_names.index(behavior_name)]
        ax.plot(time_bins, mean_vals, color=color, label=behavior_name, linewidth=2)
        ax.fill_between(time_bins, mean_vals - sem_vals, mean_vals + sem_vals,
                       color=color, alpha=0.2)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc='best')
    ax.spines[['top', 'right']].set_visible(False)

    if output_path:
        save_figure(fig, output_path)

    return fig


def plot_heatmap(data: np.ndarray,
                row_labels: Optional[List[str]] = None,
                col_labels: Optional[List[str]] = None,
                title: str = "Heatmap",
                xlabel: str = "",
                ylabel: str = "",
                cmap: str = 'viridis',
                output_path: Optional[str] = None,
                annot: bool = False) -> plt.Figure:
    """
    Create a heatmap visualization.

    Args:
        data (np.ndarray): 2D array of values to plot
        row_labels (list, optional): Labels for rows
        col_labels (list, optional): Labels for columns
        title (str): Plot title
        xlabel (str): X-axis label
        ylabel (str): Y-axis label
        cmap (str): Colormap name
        output_path (str, optional): If provided, saves the plot
        annot (bool): If True, annotates cells with values

    Returns:
        plt.Figure: The created figure

    Example:
        >>> transition_matrix = np.array([[0.7, 0.2, 0.1],
        >>>                               [0.3, 0.5, 0.2],
        >>>                               [0.2, 0.3, 0.5]])
        >>> behaviors = ['still', 'walking', 'rearing']
        >>> fig = plot_heatmap(transition_matrix, behaviors, behaviors,
        >>>                   title='Behavior Transitions')
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(data, annot=annot, fmt='.2f', cmap=cmap,
                xticklabels=col_labels, yticklabels=row_labels,
                ax=ax, cbar_kws={'label': 'Value'})

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if output_path:
        save_figure(fig, output_path)

    return fig


def plot_box_whisker(data: pd.DataFrame,
                    x_col: str,
                    y_col: str,
                    hue_col: Optional[str] = None,
                    title: str = "Box Plot",
                    xlabel: str = "",
                    ylabel: str = "",
                    output_path: Optional[str] = None) -> plt.Figure:
    """
    Create a box-and-whisker plot.

    Args:
        data (pd.DataFrame): DataFrame containing the data
        x_col (str): Column name for x-axis categories
        y_col (str): Column name for y-axis values
        hue_col (str, optional): Column name for grouping/coloring
        title (str): Plot title
        xlabel (str): X-axis label
        ylabel (str): Y-axis label
        output_path (str, optional): If provided, saves the plot

    Returns:
        plt.Figure: The created figure

    Example:
        >>> df = pd.DataFrame({
        >>>     'Behavior': ['still', 'walking', 'rearing'] * 10,
        >>>     'Duration': np.random.rand(30) * 10
        >>> })
        >>> fig = plot_box_whisker(df, 'Behavior', 'Duration',
        >>>                        ylabel='Duration (s)')
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.boxplot(data=data, x=x_col, y=y_col, hue=hue_col, ax=ax)

    ax.set_title(title)
    ax.set_xlabel(xlabel if xlabel else x_col)
    ax.set_ylabel(ylabel if ylabel else y_col)
    plt.xticks(rotation=45, ha='right')

    if output_path:
        save_figure(fig, output_path)

    return fig


def plot_trajectory(x_coords: np.ndarray,
                   y_coords: np.ndarray,
                   title: str = "Movement Trajectory",
                   colormap: str = 'viridis',
                   output_path: Optional[str] = None,
                   show_start_end: bool = True) -> plt.Figure:
    """
    Plot movement trajectory with color showing progression over time.

    Args:
        x_coords (np.ndarray): X coordinates over time
        y_coords (np.ndarray): Y coordinates over time
        title (str): Plot title
        colormap (str): Colormap for showing time progression
        output_path (str, optional): If provided, saves the plot
        show_start_end (bool): If True, marks start (green) and end (red) points

    Returns:
        plt.Figure: The created figure

    Example:
        >>> x = pose_data[:, 0]  # Nose x-coordinate
        >>> y = pose_data[:, 1]  # Nose y-coordinate
        >>> fig = plot_trajectory(x, y, title='Nose Movement')
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 10))

    # Create color array based on time
    colors = np.arange(len(x_coords))

    # Plot trajectory with color gradient
    scatter = ax.scatter(x_coords, y_coords, c=colors, cmap=colormap,
                        s=1, alpha=0.5)

    if show_start_end:
        ax.scatter(x_coords[0], y_coords[0], c='green', s=100,
                  marker='o', label='Start', zorder=10)
        ax.scatter(x_coords[-1], y_coords[-1], c='red', s=100,
                  marker='X', label='End', zorder=10)
        ax.legend()

    ax.set_title(title)
    ax.set_xlabel('X Position (pixels)')
    ax.set_ylabel('Y Position (pixels)')
    ax.set_aspect('equal')

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Time (frames)')

    if output_path:
        save_figure(fig, output_path)

    return fig


def create_multi_subplot(n_rows: int, n_cols: int,
                         figsize: Optional[Tuple[int, int]] = None,
                         sharex: bool = False,
                         sharey: bool = False) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create a figure with multiple subplots.

    Args:
        n_rows (int): Number of subplot rows
        n_cols (int): Number of subplot columns
        figsize (tuple, optional): Figure size (width, height)
        sharex (bool): Share x-axis across subplots
        sharey (bool): Share y-axis across subplots

    Returns:
        tuple: (figure, axes_array)

    Example:
        >>> fig, axes = create_multi_subplot(2, 2, figsize=(12, 10))
        >>> axes[0, 0].plot([1, 2, 3], [1, 4, 9])
        >>> axes[0, 1].bar(['A', 'B'], [10, 20])
    """
    if figsize is None:
        figsize = (6 * n_cols, 4 * n_rows)

    setup_plot_style()
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize,
                            sharex=sharex, sharey=sharey)

    return fig, axes


def close_all_plots():
    """
    Close all open matplotlib figures to free memory.

    Use this after creating many plots to avoid memory issues.

    Example:
        >>> # Create many plots
        >>> for i in range(100):
        >>>     fig = plot_behavior_pie(...)
        >>> # Free memory
        >>> close_all_plots()
    """
    plt.close('all')
