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
# NOTE: The backend should already be set by the entry point (main_lupe_gui.py)
# before this module is imported. This is a safety fallback.
import matplotlib

# Verify/set backend - warn if it's wrong (indicates import order problem)
current_backend = matplotlib.get_backend()
if current_backend != 'agg':
    # Try to set it (will only work if pyplot hasn't been imported yet)
    try:
        matplotlib.use('Agg')
        print(f"[PLOTTING] Backend changed from '{current_backend}' to 'Agg'")
    except Exception:
        # Backend is locked - pyplot was already imported elsewhere
        print(f"[WARNING] matplotlib backend is '{current_backend}', expected 'agg'.")
        print("[WARNING] This may cause threading issues. Ensure backend is set at entry point.")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Final backend verification (for debugging)
_final_backend = matplotlib.get_backend()
if _final_backend.lower() != 'agg':
    print(f"[WARNING] Final backend is '{_final_backend}', threading deadlocks may occur.")
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

    This function includes safeguards against threading issues:
    - Verifies Agg backend is in use (thread-safe)
    - Wraps operations in try/finally to ensure figure is closed
    - Provides detailed error messages for debugging

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

    try:
        # Verify backend before save (helps debug threading issues)
        current_backend = matplotlib.get_backend()
        if current_backend.lower() != 'agg':
            print(f"[WARNING] Saving figure with backend '{current_backend}' - may hang in threads")

        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')

    except Exception as e:
        print(f"[ERROR] Failed to save figure to {output_path}: {str(e)}")
        raise

    finally:
        # Always close the figure to free memory, even if save failed
        # Critical for multi-file processing to prevent memory leaks
        try:
            plt.close(fig)
        except Exception:
            pass  # Ignore errors during cleanup


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
    Close all open figures to free memory.

    Use this after creating many plots to avoid memory issues.

    Example:
        >>> # Create many plots
        >>> for i in range(100):
        >>>     fig = plot_behavior_pie(...)
        >>> # Free memory
        >>> close_all_plots()
    """
    plt.close('all')


# =============================================================================
# New utility functions for enhanced visualization (Groups x Timepoints grids)
# =============================================================================

def calculate_grouped_stats(
    df: pd.DataFrame,
    groupby_cols: List[str],
    value_col: str = 'value'
) -> pd.DataFrame:
    """
    Calculate mean, SEM, std, and count grouped by specified columns.

    Args:
        df: Input DataFrame containing the data
        groupby_cols: List of column names to group by
        value_col: Column containing values to aggregate (default: 'value')

    Returns:
        DataFrame with columns: [groupby_cols], mean, sem, std, n

    Example:
        >>> stats = calculate_grouped_stats(
        ...     df, ['group', 'timepoint', 'behavior'], 'value'
        ... )
    """
    if df.empty:
        return pd.DataFrame()

    # Filter to only existing columns
    valid_cols = [c for c in groupby_cols if c in df.columns]
    if not valid_cols:
        return pd.DataFrame()

    grouped = df.groupby(valid_cols)[value_col]

    stats = grouped.agg(['mean', 'std', 'count']).reset_index()
    stats.columns = valid_cols + ['mean', 'std', 'n']

    # Calculate SEM (standard error of the mean)
    stats['sem'] = stats['std'] / np.sqrt(stats['n'])

    # Replace NaN SEM with 0 (for n=1 cases)
    stats['sem'] = stats['sem'].fillna(0)

    return stats


def create_group_timepoint_grid(
    n_groups: int,
    n_timepoints: int,
    figsize_per_subplot: Tuple[float, float] = (4, 3),
    sharex: bool = True,
    sharey: bool = True
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create a figure with groups x timepoints subplot grid.

    Args:
        n_groups: Number of rows (groups)
        n_timepoints: Number of columns (timepoints)
        figsize_per_subplot: (width, height) per subplot in inches
        sharex: Share x-axis across subplots
        sharey: Share y-axis across subplots

    Returns:
        (figure, axes_array) where axes_array is 2D: [group_idx, timepoint_idx]

    Example:
        >>> fig, axes = create_group_timepoint_grid(2, 3)
        >>> axes[0, 0].plot(...)  # Group 0, Timepoint 0
    """
    setup_plot_style()

    # Ensure at least 1x1 grid
    n_groups = max(1, n_groups)
    n_timepoints = max(1, n_timepoints)

    figsize = (figsize_per_subplot[0] * n_timepoints,
               figsize_per_subplot[1] * n_groups)

    fig, axes = plt.subplots(
        n_groups, n_timepoints,
        figsize=figsize,
        sharex=sharex,
        sharey=sharey,
        squeeze=False  # Always return 2D array
    )

    return fig, axes


def plot_mean_sem_bars(
    ax: plt.Axes,
    categories: List[str],
    means: np.ndarray,
    sems: np.ndarray,
    colors: Optional[Dict[str, str]] = None,
    capsize: int = 3,
    bar_width: float = 0.7
) -> None:
    """
    Plot bars with mean values and SEM error bars on given axes.

    Args:
        ax: Matplotlib axes to plot on
        categories: List of category labels for x-axis
        means: Array of mean values
        sems: Array of SEM values for error bars
        colors: Optional dict mapping category names to colors
        capsize: Error bar cap size in points
        bar_width: Width of bars (0-1)

    Example:
        >>> fig, ax = plt.subplots()
        >>> plot_mean_sem_bars(ax, ['A', 'B', 'C'], [1, 2, 3], [0.1, 0.2, 0.3])
    """
    x = np.arange(len(categories))

    # Get colors for each category
    if colors is None:
        config = get_config()
        behavior_names = config.get_behavior_names()
        all_colors = config.get_behavior_colors()
        bar_colors = []
        for cat in categories:
            if cat in behavior_names:
                bar_colors.append(all_colors[behavior_names.index(cat)])
            else:
                bar_colors.append('#1f77b4')  # Default blue
    else:
        bar_colors = [colors.get(cat, '#1f77b4') for cat in categories]

    ax.bar(x, means, bar_width, yerr=sems, color=bar_colors,
           capsize=capsize, error_kw={'elinewidth': 1, 'capthick': 1})

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.spines[['top', 'right']].set_visible(False)


def plot_mean_sem_line(
    ax: plt.Axes,
    x_values: List[str],
    means: np.ndarray,
    sems: np.ndarray,
    color: str = '#1f77b4',
    label: Optional[str] = None,
    marker: str = 'o',
    fill_alpha: float = 0.2
) -> None:
    """
    Plot a line with mean values and SEM shaded region on given axes.

    Args:
        ax: Matplotlib axes to plot on
        x_values: List of x-axis labels (e.g., timepoint names)
        means: Array of mean values
        sems: Array of SEM values for shaded region
        color: Line and fill color
        label: Legend label
        marker: Marker style for data points
        fill_alpha: Transparency of SEM shaded region

    Example:
        >>> fig, ax = plt.subplots()
        >>> plot_mean_sem_line(ax, ['Day0', 'Day7'], [1.5, 2.3], [0.2, 0.3])
    """
    x = np.arange(len(x_values))

    ax.plot(x, means, color=color, marker=marker, label=label, linewidth=2)
    ax.fill_between(x, means - sems, means + sems, color=color, alpha=fill_alpha)

    ax.set_xticks(x)
    ax.set_xticklabels(x_values, rotation=45, ha='right')
    ax.spines[['top', 'right']].set_visible(False)


def plot_transition_heatmap_subplot(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    title: str = '',
    cmap: str = 'YlOrRd',
    annotate: bool = True,
    vmin: float = 0,
    vmax: float = 1,
    fmt: str = '.2f'
) -> plt.cm.ScalarMappable:
    """
    Plot a single transition heatmap on given axes.

    Args:
        ax: Matplotlib axes to plot on
        matrix: Square DataFrame with transition probabilities
        title: Subplot title
        cmap: Colormap name
        annotate: Whether to show values in cells
        vmin: Minimum value for color scale
        vmax: Maximum value for color scale
        fmt: Format string for annotations

    Returns:
        ScalarMappable for colorbar creation

    Example:
        >>> fig, ax = plt.subplots()
        >>> sm = plot_transition_heatmap_subplot(ax, trans_matrix, 'Group A')
        >>> plt.colorbar(sm, ax=ax)
    """
    # Use seaborn heatmap for consistent styling
    hm = sns.heatmap(
        matrix,
        ax=ax,
        annot=annotate,
        fmt=fmt,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar=False,  # We'll add a shared colorbar later
        square=True
    )

    ax.set_title(title, fontsize=10)

    # Rotate labels for readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

    # Create ScalarMappable for colorbar
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    return sm


def create_behavior_subplot_grid(
    n_behaviors: int,
    max_cols: int = 3,
    figsize_per_subplot: Tuple[float, float] = (4, 3)
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create a figure with a grid of subplots for behaviors.

    Args:
        n_behaviors: Number of behaviors to plot
        max_cols: Maximum number of columns
        figsize_per_subplot: (width, height) per subplot in inches

    Returns:
        (figure, axes_array) flattened 1D array of axes

    Example:
        >>> fig, axes = create_behavior_subplot_grid(5)
        >>> for i, ax in enumerate(axes[:5]):
        ...     ax.plot(...)
    """
    setup_plot_style()

    n_cols = min(n_behaviors, max_cols)
    n_rows = int(np.ceil(n_behaviors / max_cols))

    figsize = (figsize_per_subplot[0] * n_cols,
               figsize_per_subplot[1] * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    # Flatten axes array for easier iteration
    axes_flat = axes.flatten()

    # Hide unused subplots
    for i in range(n_behaviors, len(axes_flat)):
        axes_flat[i].set_visible(False)

    return fig, axes_flat
