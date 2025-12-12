"""
Report Generator Module

This module generates analysis reports in multiple formats:
- CSV files: Summary statistics, statistical test results, pairwise comparisons
- PNG figures: Visualizations of LUPE and AMPS metrics by experimental factors
- TXT report: Comprehensive statistical analysis report

Output files generated:
    {output_dir}/
    ├── {project}_summary_statistics.csv
    ├── {project}_statistical_tests.csv
    ├── {project}_pairwise_comparisons.csv
    ├── {project}_raw_data_long.csv
    ├── {project}_statistical_report.txt
    └── figures/
        ├── {project}_lupe_bout_counts.png
        ├── {project}_lupe_time_distribution.png
        ├── {project}_lupe_bout_durations.png
        ├── {project}_lupe_timeline.png
        ├── {project}_amps_pain_scale.png
        ├── {project}_amps_metrics.png
        └── ...

Usage:
    from src.core.report_generator import ReportGenerator

    generator = ReportGenerator(config, aggregated_data, test_results)
    generator.create_summary_csv()
    generator.create_figures()
    generator.write_statistical_report()
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

# Use non-interactive backend for thread safety
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def sort_timepoints_numerically(timepoints: List[str]) -> List[str]:
    """
    Sort timepoints by their numeric value rather than alphabetically.

    Handles formats like: -1d, +3d, +7d, +14d, +21d, d7, day14, hab_-1d, etc.

    Args:
        timepoints: List of timepoint strings

    Returns:
        List of timepoints sorted numerically
    """
    def extract_numeric(tp: str) -> float:
        """Extract numeric value from timepoint string for sorting."""
        if not tp or tp == 'all':
            return float('inf')  # Put 'all' at the end

        tp_lower = tp.lower().strip()

        # Try to find a number with optional sign
        # Patterns: +14d, -1d, 14d, d14, day14, day_14, hab_-1d, etc.
        patterns = [
            r'([+-]?\d+)\s*d',      # +14d, -1d, 14d
            r'd\s*([+-]?\d+)',      # d14, d-1
            r'day\s*[_-]?\s*([+-]?\d+)',  # day14, day_14, day-14
            r'hab[_-]?\s*([+-]?\d+)',     # hab_-1d, hab-1
            r'([+-]?\d+)',          # Just a number (fallback)
        ]

        for pattern in patterns:
            match = re.search(pattern, tp_lower)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue

        # If no number found, use alphabetical order by returning a large value
        return float('inf')

    # Sort with numeric key, keeping original strings
    return sorted(timepoints, key=extract_numeric)


class ReportGenerator:
    """
    Generates reports and visualizations from statistical analysis results.

    Attributes:
        config (dict): Project configuration
        aggregated_data (dict): Output from DataAggregator.aggregate_by_factors()
        test_results (list): List of TestResult objects from StatisticalAnalyzer
        output_dir (Path): Directory for output files
        project_name (str): Project name for file prefixes
    """

    # Color palettes for consistent visualization
    GROUP_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    BEHAVIOR_COLORS = {
        'still': '#DC143C',           # crimson
        'walking': '#008B8B',         # darkcyan
        'rearing': '#DAA520',         # goldenrod
        'grooming': '#4169E1',        # royalblue
        'licking hindpaw L': '#663399',  # rebeccapurple
        'licking hindpaw R': '#BA55D3'   # mediumorchid
    }

    def __init__(self, config: Dict, aggregated_data: Dict,
                 test_results: List = None):
        """
        Initialize the ReportGenerator.

        Args:
            config: Project configuration dictionary
            aggregated_data: Output from DataAggregator.aggregate_by_factors()
            test_results: List of TestResult objects (optional)
        """
        self.config = config
        self.aggregated_data = aggregated_data
        self.test_results = test_results or []

        self.output_dir = Path(config.get('output_dir', 'outputs'))
        self.project_name = config.get('project_name', 'project')

        # Ensure output directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir = self.output_dir / 'figures'
        self.figures_dir.mkdir(exist_ok=True)

        # Extract experimental design info
        self.groups = [g.get('name', '') for g in config.get('groups', [])]
        self.conditions = [c.get('name', '') for c in config.get('conditions', [])]
        self.include_sex = config.get('include_sex', False)
        self.has_timepoints = config.get('has_timepoints', False)

        # Extract group colors from config, falling back to defaults
        self.group_colors = []
        for i, g in enumerate(config.get('groups', [])):
            if isinstance(g, dict) and 'color' in g:
                self.group_colors.append(g['color'])
            else:
                # Use default color based on index
                self.group_colors.append(
                    self.GROUP_COLORS[i % len(self.GROUP_COLORS)]
                )

    def create_summary_csv(self) -> List[str]:
        """
        Create summary statistics CSV files.

        Returns:
            list: Paths to created files
        """
        created_files = []

        # Get summary statistics from aggregated data
        lupe_metrics = self.aggregated_data.get('lupe_metrics')
        amps_metrics = self.aggregated_data.get('amps_metrics')

        if lupe_metrics is not None and not lupe_metrics.empty:
            # Summary by factors
            summary = self._calculate_summary_stats(lupe_metrics)
            if not summary.empty:
                path = self.output_dir / f"{self.project_name}_lupe_summary_statistics.csv"
                summary.to_csv(path, index=False)
                created_files.append(str(path))
                logger.info(f"Created: {path}")

            # Raw data long format
            raw_path = self.output_dir / f"{self.project_name}_lupe_raw_data.csv"
            lupe_metrics.to_csv(raw_path, index=False)
            created_files.append(str(raw_path))

        if amps_metrics is not None and not amps_metrics.empty:
            summary = self._calculate_summary_stats(amps_metrics, metric_col='metric_name')
            if not summary.empty:
                path = self.output_dir / f"{self.project_name}_amps_summary_statistics.csv"
                summary.to_csv(path, index=False)
                created_files.append(str(path))
                logger.info(f"Created: {path}")

            raw_path = self.output_dir / f"{self.project_name}_amps_raw_data.csv"
            amps_metrics.to_csv(raw_path, index=False)
            created_files.append(str(raw_path))

        # Statistical test results
        if self.test_results:
            from src.core.statistical_tests import StatisticalAnalyzer

            # Create a temporary analyzer just to use the conversion method
            analyzer = StatisticalAnalyzer(pd.DataFrame(), self.config)
            results_df = analyzer.get_results_dataframe(self.test_results)

            if not results_df.empty:
                path = self.output_dir / f"{self.project_name}_statistical_tests.csv"
                results_df.to_csv(path, index=False)
                created_files.append(str(path))
                logger.info(f"Created: {path}")

            # Post-hoc comparisons
            posthoc_df = analyzer.get_post_hoc_dataframe(self.test_results)
            if not posthoc_df.empty:
                path = self.output_dir / f"{self.project_name}_pairwise_comparisons.csv"
                posthoc_df.to_csv(path, index=False)
                created_files.append(str(path))
                logger.info(f"Created: {path}")

        return created_files

    def _calculate_summary_stats(self, df: pd.DataFrame,
                                 metric_col: str = 'behavior') -> pd.DataFrame:
        """
        Calculate summary statistics grouped by factors and metric.

        Args:
            df: Long-format DataFrame with 'value' column
            metric_col: Column name for metric grouping

        Returns:
            pd.DataFrame: Summary statistics
        """
        if df.empty or 'value' not in df.columns:
            return pd.DataFrame()

        # Determine grouping columns
        groupby_cols = []
        for col in ['group', 'condition', 'sex', 'metric_type', metric_col]:
            if col in df.columns:
                groupby_cols.append(col)

        if not groupby_cols:
            return pd.DataFrame()

        # Calculate statistics
        try:
            summary = df.groupby(groupby_cols)['value'].agg([
                ('n', 'count'),
                ('mean', 'mean'),
                ('std', 'std'),
                ('sem', lambda x: x.std() / np.sqrt(len(x)) if len(x) > 0 else np.nan),
                ('median', 'median'),
                ('min', 'min'),
                ('max', 'max')
            ]).reset_index()

            return summary

        except Exception as e:
            logger.warning(f"Failed to calculate summary stats: {e}")
            return pd.DataFrame()

    def create_figures(self) -> List[str]:
        """
        Create all visualization figures.

        Returns:
            list: Paths to created figure files
        """
        created_files = []

        # LUPE figures
        lupe_metrics = self.aggregated_data.get('lupe_metrics')
        if lupe_metrics is not None and not lupe_metrics.empty:
            files = self._create_lupe_figures(lupe_metrics)
            created_files.extend(files)

        # LUPE timeline (loaded for grid figures below)
        lupe_timeline = self.aggregated_data.get('lupe_timeline')

        # AMPS figures
        amps_metrics = self.aggregated_data.get('amps_metrics')
        if amps_metrics is not None and not amps_metrics.empty:
            files = self._create_amps_figures(amps_metrics)
            created_files.extend(files)

        # Transition data (loaded for grid figures below)
        transitions = self.aggregated_data.get('lupe_transitions')

        # =====================================================================
        # Enhanced visualizations
        # =====================================================================

        # Sex-separated figures (when include_sex=True)
        if self.include_sex and lupe_metrics is not None and not lupe_metrics.empty:
            files = self._create_all_sex_separated_figures(lupe_metrics)
            created_files.extend(files)
            logger.info(f"Created {len(files)} sex-separated LUPE figures")

        # New figures when separate_timepoints=True
        separate_timepoints = self.config.get('separate_timepoints', False)

        if separate_timepoints:
            # Timeline grid (groups x timepoints)
            if lupe_timeline is not None and not lupe_timeline.empty:
                files = self._create_timeline_grid_with_sex(lupe_timeline)
                created_files.extend(files)
                logger.info(f"Created {len(files)} timeline grid figures")

            # Behavior vs Timepoint comparisons
            if lupe_metrics is not None and not lupe_metrics.empty:
                files = self._create_behavior_timepoint_comparison_figures(lupe_metrics)
                created_files.extend(files)
                logger.info(f"Created {len(files)} behavior-timepoint comparison figures")

            # Transition grids (groups x timepoints)
            if transitions:
                files = self._create_transition_grid_figures(transitions)
                created_files.extend(files)
                logger.info(f"Created {len(files)} transition grid figures")

            # Transition delta heatmaps (change from baseline)
            if transitions:
                files = self._create_transition_delta_figures(transitions)
                created_files.extend(files)
                logger.info(f"Created {len(files)} transition delta figures")

        return created_files

    def _create_lupe_figures(self, df: pd.DataFrame) -> List[str]:
        """Create LUPE metric figures."""
        created_files = []

        # Bout counts comparison
        bout_counts = df[df['metric_type'] == 'bout_count']
        if not bout_counts.empty:
            path = self._plot_grouped_bars(
                bout_counts,
                title="LUPE: Bout Counts by Behavior",
                ylabel="Number of Bouts",
                filename=f"{self.project_name}_lupe_bout_counts.png"
            )
            if path:
                created_files.append(path)

        # Time distribution
        time_dist = df[df['metric_type'] == 'time_percentage']
        if not time_dist.empty:
            path = self._plot_grouped_bars(
                time_dist,
                title="LUPE: Time Distribution by Behavior",
                ylabel="Time (%)",
                filename=f"{self.project_name}_lupe_time_distribution.png"
            )
            if path:
                created_files.append(path)

        # Bout durations
        bout_dur = df[df['metric_type'] == 'bout_mean_duration_sec']
        if not bout_dur.empty:
            path = self._plot_grouped_bars(
                bout_dur,
                title="LUPE: Mean Bout Duration by Behavior",
                ylabel="Duration (seconds)",
                filename=f"{self.project_name}_lupe_bout_durations.png"
            )
            if path:
                created_files.append(path)

        return created_files

    def _create_amps_figures(self, df: pd.DataFrame) -> List[str]:
        """Create AMPS metric figures."""
        created_files = []

        # Pain scale comparison
        pain_scale = df[df['metric_type'] == 'pain_scale']
        if not pain_scale.empty:
            path = self._plot_pain_scale(pain_scale)
            if path:
                created_files.append(path)

        # AMPS state metrics
        for metric_type in ['amps_fraction_occupancy', 'amps_number_of_bouts', 'amps_bout_duration']:
            metric_df = df[df['metric_type'] == metric_type]
            if not metric_df.empty:
                title_map = {
                    'amps_fraction_occupancy': 'AMPS: Fraction Occupancy by State',
                    'amps_number_of_bouts': 'AMPS: Number of Bouts by State',
                    'amps_bout_duration': 'AMPS: Bout Duration by State'
                }
                ylabel_map = {
                    'amps_fraction_occupancy': 'Fraction',
                    'amps_number_of_bouts': 'Number of Bouts',
                    'amps_bout_duration': 'Duration (seconds)'
                }

                path = self._plot_amps_states(
                    metric_df,
                    title=title_map.get(metric_type, metric_type),
                    ylabel=ylabel_map.get(metric_type, 'Value'),
                    filename=f"{self.project_name}_{metric_type}.png"
                )
                if path:
                    created_files.append(path)

        return created_files

    def _plot_grouped_bars(self, df: pd.DataFrame, title: str,
                           ylabel: str, filename: str) -> Optional[str]:
        """
        Create grouped bar chart for behavior metrics.

        Args:
            df: DataFrame with columns: group, behavior, value
            title: Plot title
            ylabel: Y-axis label
            filename: Output filename

        Returns:
            str: Path to created file, or None if failed
        """
        try:
            fig, ax = plt.subplots(figsize=(12, 6))

            # Get unique behaviors and groups
            behaviors = df['behavior'].unique()
            groups = df['group'].unique() if 'group' in df.columns else ['all']

            # Calculate means and SEMs
            x = np.arange(len(behaviors))
            width = 0.8 / len(groups)

            for i, group in enumerate(groups):
                if 'group' in df.columns:
                    group_data = df[df['group'] == group]
                else:
                    group_data = df

                means = []
                sems = []
                for behavior in behaviors:
                    behavior_data = group_data[group_data['behavior'] == behavior]['value']
                    means.append(behavior_data.mean())
                    sems.append(behavior_data.sem() if len(behavior_data) > 1 else 0)

                offset = (i - len(groups) / 2 + 0.5) * width
                bars = ax.bar(x + offset, means, width, label=group,
                              color=self.group_colors[i % len(self.group_colors)],
                              yerr=sems, capsize=3)

            ax.set_xlabel('Behavior')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels(behaviors, rotation=45, ha='right')

            if len(groups) > 1:
                ax.legend(title='Group')

            plt.tight_layout()

            path = self.figures_dir / filename
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            logger.info(f"Created figure: {path}")
            return str(path)

        except Exception as e:
            logger.error(f"Failed to create figure {filename}: {e}")
            plt.close('all')
            return None

    def _plot_pain_scale(self, df: pd.DataFrame) -> Optional[str]:
        """Create pain scale box/violin plot."""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            for idx, pc_name in enumerate(['PC1_Behavior_Scale', 'PC2_Pain_Scale']):
                ax = axes[idx]
                pc_data = df[df['metric_name'] == pc_name]

                if pc_data.empty:
                    continue

                groups = pc_data['group'].unique() if 'group' in pc_data.columns else ['all']

                # Box plot
                positions = []
                data_to_plot = []

                for i, group in enumerate(groups):
                    if 'group' in pc_data.columns:
                        group_vals = pc_data[pc_data['group'] == group]['value'].dropna()
                    else:
                        group_vals = pc_data['value'].dropna()

                    if len(group_vals) > 0:
                        data_to_plot.append(group_vals.values)
                        positions.append(i + 1)

                if data_to_plot:
                    bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6,
                                    patch_artist=True)

                    for patch, color in zip(bp['boxes'], self.group_colors):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)

                    ax.set_xticks(positions)
                    ax.set_xticklabels(groups[:len(positions)], rotation=45, ha='right')
                    ax.set_ylabel(pc_name.replace('_', ' '))
                    ax.set_title(pc_name.replace('_', ' '))

            plt.suptitle('AMPS: Pain Scale Projection', fontsize=14)
            plt.tight_layout()

            path = self.figures_dir / f"{self.project_name}_amps_pain_scale.png"
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            logger.info(f"Created figure: {path}")
            return str(path)

        except Exception as e:
            logger.error(f"Failed to create pain scale figure: {e}")
            plt.close('all')
            return None

    def _plot_amps_states(self, df: pd.DataFrame, title: str,
                          ylabel: str, filename: str) -> Optional[str]:
        """Create AMPS state metrics figure."""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            states = df['metric_name'].unique()
            groups = df['group'].unique() if 'group' in df.columns else ['all']

            x = np.arange(len(states))
            width = 0.8 / len(groups)

            for i, group in enumerate(groups):
                if 'group' in df.columns:
                    group_data = df[df['group'] == group]
                else:
                    group_data = df

                means = []
                sems = []
                for state in states:
                    state_data = group_data[group_data['metric_name'] == state]['value']
                    means.append(state_data.mean())
                    sems.append(state_data.sem() if len(state_data) > 1 else 0)

                offset = (i - len(groups) / 2 + 0.5) * width
                ax.bar(x + offset, means, width, label=group,
                       color=self.group_colors[i % len(self.group_colors)],
                       yerr=sems, capsize=3)

            ax.set_xlabel('State')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels(states, rotation=45, ha='right')

            if len(groups) > 1:
                ax.legend(title='Group')

            plt.tight_layout()

            path = self.figures_dir / filename
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            logger.info(f"Created figure: {path}")
            return str(path)

        except Exception as e:
            logger.error(f"Failed to create AMPS states figure: {e}")
            plt.close('all')
            return None

    def _create_timeline_figure(self, df: pd.DataFrame) -> Optional[str]:
        """Create timeline plot showing behavior proportions over time bins."""
        try:
            fig, ax = plt.subplots(figsize=(14, 6))

            # Get behavior columns
            prop_cols = [c for c in df.columns if c.endswith('_proportion')]

            if not prop_cols:
                return None

            # If multiple groups, create subplots
            groups = df['group'].unique() if 'group' in df.columns else ['all']

            if len(groups) == 1:
                # Single group - plot directly
                group_df = df if groups[0] == 'all' else df[df['group'] == groups[0]]
                mean_timeline = group_df.groupby('time_bin')[prop_cols].mean()

                for col in prop_cols:
                    behavior = col.replace('_proportion', '')
                    color = self.BEHAVIOR_COLORS.get(behavior, None)
                    ax.plot(mean_timeline.index, mean_timeline[col],
                            label=behavior, color=color, linewidth=2)

                ax.set_xlabel('Time Bin')
                ax.set_ylabel('Proportion')
                ax.set_title(f'LUPE: Behavior Timeline')
                ax.legend(loc='upper right', fontsize=8)
            else:
                # Multiple groups - show one behavior or use subplots
                fig, axes = plt.subplots(len(groups), 1, figsize=(14, 4 * len(groups)),
                                         sharex=True)
                if len(groups) == 1:
                    axes = [axes]

                for ax_idx, group in enumerate(groups):
                    group_df = df[df['group'] == group]
                    mean_timeline = group_df.groupby('time_bin')[prop_cols].mean()

                    for col in prop_cols:
                        behavior = col.replace('_proportion', '')
                        color = self.BEHAVIOR_COLORS.get(behavior, None)
                        axes[ax_idx].plot(mean_timeline.index, mean_timeline[col],
                                          label=behavior, color=color, linewidth=2)

                    axes[ax_idx].set_ylabel('Proportion')
                    axes[ax_idx].set_title(f'Group: {group}')
                    axes[ax_idx].legend(loc='upper right', fontsize=8)

                axes[-1].set_xlabel('Time Bin')
                plt.suptitle('LUPE: Behavior Timeline by Group', fontsize=14)

            plt.tight_layout()

            path = self.figures_dir / f"{self.project_name}_lupe_timeline.png"
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            logger.info(f"Created figure: {path}")
            return str(path)

        except Exception as e:
            logger.error(f"Failed to create timeline figure: {e}")
            plt.close('all')
            return None

    def _create_transition_heatmaps(self, transitions: Dict) -> Optional[str]:
        """Create transition matrix heatmaps."""
        try:
            n_groups = len(transitions)
            if n_groups == 0:
                return None

            fig, axes = plt.subplots(1, n_groups, figsize=(6 * n_groups, 5))
            if n_groups == 1:
                axes = [axes]

            for ax, (key, matrix) in zip(axes, transitions.items()):
                im = ax.imshow(matrix.values, cmap='YlOrRd', aspect='auto',
                               vmin=0, vmax=1)

                ax.set_xticks(range(len(matrix.columns)))
                ax.set_yticks(range(len(matrix.index)))
                ax.set_xticklabels(matrix.columns, rotation=45, ha='right', fontsize=8)
                ax.set_yticklabels(matrix.index, fontsize=8)

                # Add text annotations
                for i in range(len(matrix.index)):
                    for j in range(len(matrix.columns)):
                        val = matrix.iloc[i, j]
                        if not np.isnan(val) and val > 0.01:
                            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                                    fontsize=7, color='white' if val > 0.5 else 'black')

                if isinstance(key, tuple):
                    title = ', '.join(str(k) for k in key if k != 'all')
                else:
                    title = str(key)
                ax.set_title(f'Transitions: {title}' if title else 'Transitions')
                ax.set_xlabel('To Behavior')
                ax.set_ylabel('From Behavior')

            # Add colorbar
            cbar = fig.colorbar(im, ax=axes, shrink=0.8)
            cbar.set_label('Transition Probability')

            plt.suptitle('LUPE: Transition Matrices', fontsize=14)
            plt.tight_layout()

            path = self.figures_dir / f"{self.project_name}_lupe_transitions.png"
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            logger.info(f"Created figure: {path}")
            return str(path)

        except Exception as e:
            logger.error(f"Failed to create transition heatmaps: {e}")
            plt.close('all')
            return None

    # =========================================================================
    # New methods for enhanced visualization (sex separation, timepoint grids)
    # =========================================================================

    def _filter_data_by_sex(self, df: pd.DataFrame, sex_filter: str) -> pd.DataFrame:
        """
        Filter DataFrame by sex for generating sex-specific figures.

        Args:
            df: DataFrame with 'sex' column
            sex_filter: 'Male', 'Female', or 'All' (pools all data, ignoring sex)

        Returns:
            Filtered DataFrame
        """
        if df is None or df.empty:
            return df

        if sex_filter == 'All' or 'sex' not in df.columns:
            return df

        return df[df['sex'] == sex_filter].copy()

    def _get_sex_filters(self) -> List[str]:
        """
        Get list of sex filters to use based on config.

        Returns:
            List of sex filter values: ['All'] or ['Male', 'Female', 'All']
        """
        if self.include_sex:
            return ['Male', 'Female', 'All']
        return ['All']

    def _create_all_sex_separated_figures(self, df: pd.DataFrame) -> List[str]:
        """
        Generate Male/Female/All versions of bout counts, durations, time distribution.

        Args:
            df: LUPE metrics DataFrame

        Returns:
            List of paths to created files
        """
        created_files = []

        if df is None or df.empty:
            return created_files

        # Define metrics to create sex-separated figures for
        metrics_config = [
            {
                'metric_type': 'bout_count',
                'title_base': 'LUPE: Bout Counts by Behavior',
                'ylabel': 'Number of Bouts',
                'filename_base': f"{self.project_name}_lupe_bout_counts"
            },
            {
                'metric_type': 'time_percentage',
                'title_base': 'LUPE: Time Distribution by Behavior',
                'ylabel': 'Time (%)',
                'filename_base': f"{self.project_name}_lupe_time_distribution"
            },
            {
                'metric_type': 'bout_mean_duration_sec',
                'title_base': 'LUPE: Mean Bout Duration by Behavior',
                'ylabel': 'Duration (seconds)',
                'filename_base': f"{self.project_name}_lupe_bout_durations"
            }
        ]

        for sex_filter in self._get_sex_filters():
            if sex_filter == 'All':
                continue  # Skip 'All' as original figures handle this

            sex_suffix = f"_{sex_filter.lower()}"

            for config in metrics_config:
                metric_df = df[df['metric_type'] == config['metric_type']]
                filtered_df = self._filter_data_by_sex(metric_df, sex_filter)

                if filtered_df.empty:
                    continue

                title = f"{config['title_base']} ({sex_filter})"
                filename = f"{config['filename_base']}{sex_suffix}.png"

                path = self._plot_grouped_bars(
                    filtered_df,
                    title=title,
                    ylabel=config['ylabel'],
                    filename=filename
                )
                if path:
                    created_files.append(path)

        return created_files

    def _create_timeline_grid_figure(self, df: pd.DataFrame,
                                     sex_filter: str = 'All') -> Optional[str]:
        """
        Create timeline subplots: rows = groups, columns = timepoints.
        Each subplot shows behavior proportions over time bins.

        Args:
            df: Timeline DataFrame
            sex_filter: 'Male', 'Female', or 'All'

        Returns:
            Path to created file, or None
        """
        try:
            from src.utils.plotting import (
                create_group_timepoint_grid, setup_plot_style, save_figure
            )

            filtered_df = self._filter_data_by_sex(df, sex_filter)
            if filtered_df.empty:
                return None

            # Get unique groups and timepoints
            groups = filtered_df['group'].unique() if 'group' in filtered_df.columns else ['all']
            timepoints = (filtered_df['timepoint'].unique()
                         if 'timepoint' in filtered_df.columns else ['all'])

            # Remove 'all' if there are actual values and sort timepoints numerically
            groups = [g for g in groups if g != 'all'] or ['all']
            timepoints = sort_timepoints_numerically([t for t in timepoints if t != 'all']) or ['all']

            n_groups = len(groups)
            n_timepoints = len(timepoints)

            # Get behavior columns
            prop_cols = [c for c in filtered_df.columns if c.endswith('_proportion')]
            if not prop_cols:
                return None

            # Create grid
            fig, axes = create_group_timepoint_grid(n_groups, n_timepoints,
                                                    figsize_per_subplot=(5, 3))

            for row_idx, group in enumerate(groups):
                for col_idx, timepoint in enumerate(timepoints):
                    ax = axes[row_idx, col_idx]

                    # Filter data for this group and timepoint
                    mask = pd.Series(True, index=filtered_df.index)
                    if 'group' in filtered_df.columns and group != 'all':
                        mask &= filtered_df['group'] == group
                    if 'timepoint' in filtered_df.columns and timepoint != 'all':
                        mask &= filtered_df['timepoint'] == timepoint

                    subplot_df = filtered_df[mask]

                    if subplot_df.empty:
                        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                                transform=ax.transAxes)
                        continue

                    # Calculate mean timeline
                    mean_timeline = subplot_df.groupby('time_bin')[prop_cols].mean()

                    for col in prop_cols:
                        behavior = col.replace('_proportion', '')
                        color = self.BEHAVIOR_COLORS.get(behavior, None)
                        ax.plot(mean_timeline.index, mean_timeline[col],
                                label=behavior, color=color, linewidth=1.5)

                    # Labels
                    if row_idx == 0:
                        ax.set_title(f'{timepoint}', fontsize=10)
                    if col_idx == 0:
                        ax.set_ylabel(f'{group}\nProportion', fontsize=9)
                    if row_idx == n_groups - 1:
                        ax.set_xlabel('Time Bin', fontsize=9)

                    # Add legend only to first subplot
                    if row_idx == 0 and col_idx == n_timepoints - 1:
                        ax.legend(loc='upper left', fontsize=7, bbox_to_anchor=(1, 1))

            # Add overall title
            sex_suffix = f" ({sex_filter})" if sex_filter != 'All' else ""
            fig.suptitle(f'LUPE: Behavior Timeline - Groups x Timepoints{sex_suffix}',
                        fontsize=14, y=1.02)
            plt.tight_layout()

            # Save
            suffix = f"_{sex_filter.lower()}" if sex_filter != 'All' else ""
            path = self.figures_dir / f"{self.project_name}_lupe_timeline_grid{suffix}.png"
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            logger.info(f"Created figure: {path}")
            return str(path)

        except Exception as e:
            logger.error(f"Failed to create timeline grid figure: {e}")
            plt.close('all')
            return None

    def _create_timeline_grid_with_sex(self, df: pd.DataFrame) -> List[str]:
        """
        Create timeline grid figures with sex separation if enabled.

        Args:
            df: Timeline DataFrame

        Returns:
            List of paths to created files
        """
        created_files = []

        for sex_filter in self._get_sex_filters():
            path = self._create_timeline_grid_figure(df, sex_filter)
            if path:
                created_files.append(path)

        return created_files

    def _is_licking_behavior(self, behavior: str) -> bool:
        """
        Check if a behavior is a licking hindpaw behavior.

        Licking behaviors are rare events that need separate y-axis scaling
        to visualize group differences properly.

        Args:
            behavior: Behavior name string

        Returns:
            True if behavior is a licking hindpaw behavior
        """
        behavior_lower = behavior.lower()
        return 'licking' in behavior_lower and 'hindpaw' in behavior_lower

    def _create_behavior_vs_timepoint_by_behavior(
        self, df: pd.DataFrame, metric_type: str, title: str,
        ylabel: str, filename: str, sex_filter: str = 'All'
    ) -> Optional[str]:
        """
        Create figure with subplots per behavior, x-axis = timepoints, y-axis = mean +/- SEM.
        Data is separated by groups within each subplot (one line per group).

        Y-axis scaling:
        - Regular behaviors (still, walking, rearing, grooming) share the same y-axis
        - Licking hindpaw behaviors share a separate y-axis scale (they are rare events)

        Args:
            df: Long-format metrics DataFrame
            metric_type: The metric being plotted
            title: Figure title
            ylabel: Y-axis label
            filename: Output filename
            sex_filter: 'Male', 'Female', or 'All'

        Returns:
            Path to created file, or None
        """
        try:
            from src.utils.plotting import (
                create_behavior_subplot_grid, plot_mean_sem_line,
                calculate_grouped_stats, setup_plot_style
            )

            # Filter data
            metric_df = df[df['metric_type'] == metric_type]
            filtered_df = self._filter_data_by_sex(metric_df, sex_filter)

            if filtered_df.empty:
                return None

            # Get unique behaviors, groups, and timepoints
            behaviors = filtered_df['behavior'].unique()
            groups = filtered_df['group'].unique() if 'group' in filtered_df.columns else ['all']
            groups = [g for g in groups if g != 'all'] or ['all']
            timepoints = (filtered_df['timepoint'].unique()
                         if 'timepoint' in filtered_df.columns else ['all'])

            timepoints = sort_timepoints_numerically([t for t in timepoints if t != 'all']) or ['all']

            if len(timepoints) < 2:
                return None  # Need multiple timepoints for this figure

            # Create subplot grid
            fig, axes = create_behavior_subplot_grid(len(behaviors), max_cols=3)

            # First pass: calculate y-axis limits separately for regular and licking behaviors
            regular_ymin = float('inf')
            regular_ymax = float('-inf')
            licking_ymin = float('inf')
            licking_ymax = float('-inf')

            for behavior in behaviors:
                behavior_df = filtered_df[filtered_df['behavior'] == behavior]
                is_licking = self._is_licking_behavior(behavior)

                for group in groups:
                    if 'group' in behavior_df.columns and group != 'all':
                        group_df = behavior_df[behavior_df['group'] == group]
                    else:
                        group_df = behavior_df

                    stats = calculate_grouped_stats(group_df, ['timepoint'], 'value')
                    if not stats.empty:
                        # Consider mean +/- SEM for range
                        ymin = (stats['mean'] - stats['sem'].fillna(0)).min()
                        ymax = (stats['mean'] + stats['sem'].fillna(0)).max()

                        if is_licking:
                            if ymin < licking_ymin:
                                licking_ymin = ymin
                            if ymax > licking_ymax:
                                licking_ymax = ymax
                        else:
                            if ymin < regular_ymin:
                                regular_ymin = ymin
                            if ymax > regular_ymax:
                                regular_ymax = ymax

            # Add padding to y-axis limits
            def add_padding(ymin, ymax):
                if ymin != float('inf') and ymax != float('-inf'):
                    y_range = ymax - ymin
                    padding = y_range * 0.1 if y_range > 0 else 0.1
                    return max(0, ymin - padding), ymax + padding
                return ymin, ymax

            regular_ymin, regular_ymax = add_padding(regular_ymin, regular_ymax)
            licking_ymin, licking_ymax = add_padding(licking_ymin, licking_ymax)

            # Second pass: plot data
            for idx, behavior in enumerate(behaviors):
                ax = axes[idx]
                behavior_df = filtered_df[filtered_df['behavior'] == behavior]
                is_licking = self._is_licking_behavior(behavior)

                has_data = False
                for group_idx, group in enumerate(groups):
                    if 'group' in behavior_df.columns and group != 'all':
                        group_df = behavior_df[behavior_df['group'] == group]
                    else:
                        group_df = behavior_df

                    # Calculate stats per timepoint for this group
                    stats = calculate_grouped_stats(group_df, ['timepoint'], 'value')

                    if stats.empty:
                        continue

                    # Sort by timepoint order
                    stats = stats.set_index('timepoint').reindex(timepoints).reset_index()
                    stats = stats.dropna(subset=['mean'])

                    if stats.empty:
                        continue

                    has_data = True
                    color = self.group_colors[group_idx % len(self.group_colors)]
                    plot_mean_sem_line(
                        ax,
                        stats['timepoint'].tolist(),
                        stats['mean'].values,
                        stats['sem'].values,
                        color=color,
                        label=group
                    )

                if not has_data:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                            transform=ax.transAxes)
                else:
                    # Set y-axis limits based on behavior type
                    if is_licking:
                        if licking_ymin != float('inf') and licking_ymax != float('-inf'):
                            ax.set_ylim(licking_ymin, licking_ymax)
                    else:
                        if regular_ymin != float('inf') and regular_ymax != float('-inf'):
                            ax.set_ylim(regular_ymin, regular_ymax)

                ax.set_title(behavior, fontsize=10)
                ax.set_ylabel(ylabel, fontsize=9)

                # Add legend to first subplot only
                if idx == 0 and len(groups) > 1:
                    ax.legend(loc='upper right', fontsize=8)

            # Overall title
            fig.suptitle(title, fontsize=12, y=1.02)
            plt.tight_layout()

            path = self.figures_dir / filename
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            logger.info(f"Created figure: {path}")
            return str(path)

        except Exception as e:
            logger.error(f"Failed to create behavior vs timepoint figure: {e}")
            plt.close('all')
            return None

    def _create_behavior_percent_change_by_behavior(
        self, df: pd.DataFrame, metric_type: str, title: str,
        ylabel: str, filename: str, sex_filter: str = 'All'
    ) -> Optional[str]:
        """
        Create figure with subplots per behavior showing percent change from baseline.

        Shows ((value - baseline) / baseline * 100) for each timepoint.
        Data is separated by groups within each subplot (one line per group).

        Handles zero baseline cases:
        - If baseline = 0 and all values = 0: shows 0% everywhere
        - If baseline = 0 and some values > 0: shows capped value with triangle markers

        Y-axis scaling:
        - Regular behaviors share the same y-axis
        - Licking hindpaw behaviors share a separate y-axis scale

        Args:
            df: Long-format metrics DataFrame
            metric_type: The metric being plotted
            title: Figure title
            ylabel: Y-axis label (e.g., "% Change from Baseline")
            filename: Output filename
            sex_filter: 'Male', 'Female', or 'All'

        Returns:
            Path to created file, or None
        """
        try:
            from src.utils.plotting import (
                create_behavior_subplot_grid, calculate_grouped_stats
            )

            # Cap value for "from zero" increases (can't calculate real %)
            FROM_ZERO_CAP = 200.0

            # Filter data
            metric_df = df[df['metric_type'] == metric_type]
            filtered_df = self._filter_data_by_sex(metric_df, sex_filter)

            if filtered_df.empty:
                return None

            # Get unique behaviors, groups, and timepoints
            behaviors = filtered_df['behavior'].unique()
            groups = filtered_df['group'].unique() if 'group' in filtered_df.columns else ['all']
            groups = [g for g in groups if g != 'all'] or ['all']
            timepoints = (filtered_df['timepoint'].unique()
                         if 'timepoint' in filtered_df.columns else ['all'])

            timepoints = sort_timepoints_numerically([t for t in timepoints if t != 'all']) or ['all']

            if len(timepoints) < 2:
                return None  # Need multiple timepoints for percent change

            baseline_tp = timepoints[0]

            # Create subplot grid
            fig, axes = create_behavior_subplot_grid(len(behaviors), max_cols=3)

            # First pass: calculate percent changes and y-axis limits
            # Separate limits for regular and licking behaviors
            regular_ymin = float('inf')
            regular_ymax = float('-inf')
            licking_ymin = float('inf')
            licking_ymax = float('-inf')

            # Store calculated percent changes for plotting
            # Format: {(behavior, group): {timepoint: (mean_pct, sem_pct, is_from_zero)}}
            percent_changes = {}

            # Track behaviors with zero baseline for annotation
            behaviors_with_zero_baseline = set()

            for behavior in behaviors:
                behavior_df = filtered_df[filtered_df['behavior'] == behavior]
                is_licking = self._is_licking_behavior(behavior)

                for group in groups:
                    if 'group' in behavior_df.columns and group != 'all':
                        group_df = behavior_df[behavior_df['group'] == group]
                    else:
                        group_df = behavior_df

                    # Get baseline values for this behavior/group
                    if 'timepoint' in group_df.columns:
                        baseline_df = group_df[group_df['timepoint'] == baseline_tp]
                    else:
                        baseline_df = group_df

                    baseline_mean = baseline_df['value'].mean() if not baseline_df.empty else 0

                    key = (behavior, group)
                    percent_changes[key] = {}

                    # Check if baseline is zero, NaN, or effectively zero
                    baseline_is_zero = (
                        baseline_mean == 0 or
                        np.isnan(baseline_mean) or
                        (isinstance(baseline_mean, (int, float)) and abs(baseline_mean) < 1e-10)
                    )

                    if baseline_is_zero:
                        # Handle zero/NaN baseline case
                        behaviors_with_zero_baseline.add(behavior)

                        for tp in timepoints:
                            if 'timepoint' in group_df.columns:
                                tp_df = group_df[group_df['timepoint'] == tp]
                            else:
                                tp_df = group_df

                            if tp_df.empty:
                                continue

                            tp_mean = tp_df['value'].mean()

                            # Check if tp_mean is zero, NaN, or effectively zero
                            tp_is_zero = (
                                tp_mean == 0 or
                                np.isnan(tp_mean) or
                                (isinstance(tp_mean, (int, float)) and abs(tp_mean) < 1e-10)
                            )

                            if tp_is_zero:
                                # 0 to 0 = no change
                                percent_changes[key][tp] = (0.0, 0.0, True)
                                plot_value = 0.0
                            else:
                                # 0 to non-zero = capped increase
                                capped_value = FROM_ZERO_CAP if tp_mean > 0 else -FROM_ZERO_CAP
                                percent_changes[key][tp] = (capped_value, 0.0, True)
                                plot_value = capped_value

                            # Update y-axis limits for all zero baseline cases
                            if is_licking:
                                if plot_value < licking_ymin:
                                    licking_ymin = plot_value
                                if plot_value > licking_ymax:
                                    licking_ymax = plot_value
                            else:
                                if plot_value < regular_ymin:
                                    regular_ymin = plot_value
                                if plot_value > regular_ymax:
                                    regular_ymax = plot_value
                    else:
                        # Normal case: baseline > 0
                        for tp in timepoints:
                            if 'timepoint' in group_df.columns:
                                tp_df = group_df[group_df['timepoint'] == tp]
                            else:
                                tp_df = group_df

                            if tp_df.empty:
                                continue

                            # Calculate percent change for each animal, then get mean/SEM
                            tp_values = tp_df['value'].values

                            # Filter out NaN values
                            valid_values = tp_values[~np.isnan(tp_values)]

                            if len(valid_values) == 0:
                                # No valid data at this timepoint
                                continue

                            pct_changes = ((valid_values - baseline_mean) / baseline_mean) * 100

                            mean_pct = np.mean(pct_changes)
                            sem_pct = np.std(pct_changes) / np.sqrt(len(pct_changes)) if len(pct_changes) > 1 else 0

                            # Skip if result is NaN
                            if np.isnan(mean_pct):
                                continue

                            percent_changes[key][tp] = (mean_pct, sem_pct, False)

                            # Update y-axis limits
                            ymin = mean_pct - sem_pct
                            ymax = mean_pct + sem_pct

                            if is_licking:
                                if ymin < licking_ymin:
                                    licking_ymin = ymin
                                if ymax > licking_ymax:
                                    licking_ymax = ymax
                            else:
                                if ymin < regular_ymin:
                                    regular_ymin = ymin
                                if ymax > regular_ymax:
                                    regular_ymax = ymax

            # Add padding to y-axis limits
            def add_padding(ymin, ymax):
                if ymin != float('inf') and ymax != float('-inf'):
                    y_range = ymax - ymin
                    padding = y_range * 0.1 if y_range > 0 else 10
                    return ymin - padding, ymax + padding
                return ymin, ymax

            regular_ymin, regular_ymax = add_padding(regular_ymin, regular_ymax)
            licking_ymin, licking_ymax = add_padding(licking_ymin, licking_ymax)

            # Second pass: plot data
            for idx, behavior in enumerate(behaviors):
                ax = axes[idx]
                is_licking = self._is_licking_behavior(behavior)
                has_zero_baseline = behavior in behaviors_with_zero_baseline

                has_data = False
                for group_idx, group in enumerate(groups):
                    key = (behavior, group)
                    if key not in percent_changes or not percent_changes[key]:
                        continue

                    # Extract data for plotting, separating normal and from-zero points
                    plot_timepoints = []
                    means = []
                    sems = []
                    from_zero_flags = []

                    for tp in timepoints:
                        if tp in percent_changes[key]:
                            plot_timepoints.append(tp)
                            mean_pct, sem_pct, is_from_zero = percent_changes[key][tp]
                            means.append(mean_pct)
                            sems.append(sem_pct)
                            from_zero_flags.append(is_from_zero)

                    if not plot_timepoints:
                        continue

                    has_data = True
                    color = self.group_colors[group_idx % len(self.group_colors)]

                    # Check if this group has any from-zero points
                    has_from_zero = any(from_zero_flags)

                    if has_from_zero:
                        # Plot with triangle markers and dashed line for from-zero data
                        x_positions = list(range(len(plot_timepoints)))

                        # Plot line
                        ax.plot(x_positions, means, color=color, linestyle='--',
                                linewidth=1.5, alpha=0.8)

                        # Plot markers: triangles for from-zero, circles for normal
                        for i, (x, y, sem, is_fz) in enumerate(zip(x_positions, means, sems, from_zero_flags)):
                            marker = '^' if is_fz else 'o'
                            markersize = 8 if is_fz else 6
                            ax.plot(x, y, marker=marker, color=color, markersize=markersize,
                                    label=f"{group}" if i == 0 else "")

                            # Add error bars for non-from-zero points
                            if not is_fz and sem > 0:
                                ax.errorbar(x, y, yerr=sem, color=color, capsize=3, fmt='none')

                        ax.set_xticks(x_positions)
                        ax.set_xticklabels(plot_timepoints, fontsize=8)
                    else:
                        # Normal plotting with SEM shading
                        x_positions = list(range(len(plot_timepoints)))
                        means_arr = np.array(means)
                        sems_arr = np.array(sems)

                        ax.plot(x_positions, means_arr, color=color, marker='o',
                                linewidth=1.5, markersize=6, label=group)
                        ax.fill_between(x_positions, means_arr - sems_arr, means_arr + sems_arr,
                                        color=color, alpha=0.2)

                        ax.set_xticks(x_positions)
                        ax.set_xticklabels(plot_timepoints, fontsize=8)

                # Add horizontal line at y=0 (no change)
                ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

                if not has_data:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                            transform=ax.transAxes)
                else:
                    # Set y-axis limits based on behavior type
                    if is_licking:
                        if licking_ymin != float('inf') and licking_ymax != float('-inf'):
                            ax.set_ylim(licking_ymin, licking_ymax)
                    else:
                        if regular_ymin != float('inf') and regular_ymax != float('-inf'):
                            ax.set_ylim(regular_ymin, regular_ymax)

                # Title with indicator for zero baseline
                title_text = behavior
                if has_zero_baseline:
                    title_text += " *"
                ax.set_title(title_text, fontsize=10)
                ax.set_ylabel(ylabel, fontsize=9)

                # Add legend to first subplot only
                if idx == 0 and len(groups) > 1:
                    ax.legend(loc='upper right', fontsize=8)

            # Overall title with note about zero baseline
            full_title = title
            if behaviors_with_zero_baseline:
                full_title += "\n(* = baseline was 0; triangle markers show change from zero)"
            fig.suptitle(full_title, fontsize=11, y=1.02)
            plt.tight_layout()

            path = self.figures_dir / filename
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            logger.info(f"Created figure: {path}")
            return str(path)

        except Exception as e:
            logger.error(f"Failed to create percent change figure: {e}")
            plt.close('all')
            return None

    def _create_behavior_vs_timepoint_by_timepoint(
        self, df: pd.DataFrame, metric_type: str, title: str,
        ylabel: str, filename: str, sex_filter: str = 'All'
    ) -> Optional[str]:
        """
        Create figure with subplots per timepoint, grouped bars comparing behaviors with mean +/- SEM.
        Data is separated by groups within each subplot (grouped bars per behavior).
        All subplots share the same y-axis scale.

        Args:
            df: Long-format metrics DataFrame
            metric_type: The metric being plotted
            title: Figure title
            ylabel: Y-axis label
            filename: Output filename
            sex_filter: 'Male', 'Female', or 'All'

        Returns:
            Path to created file, or None
        """
        try:
            from src.utils.plotting import (
                create_behavior_subplot_grid, calculate_grouped_stats, setup_plot_style
            )

            # Filter data
            metric_df = df[df['metric_type'] == metric_type]
            filtered_df = self._filter_data_by_sex(metric_df, sex_filter)

            if filtered_df.empty:
                return None

            # Get unique behaviors, groups, and timepoints
            behaviors = list(filtered_df['behavior'].unique())
            groups = filtered_df['group'].unique() if 'group' in filtered_df.columns else ['all']
            groups = [g for g in groups if g != 'all'] or ['all']
            timepoints = (filtered_df['timepoint'].unique()
                         if 'timepoint' in filtered_df.columns else ['all'])

            timepoints = sort_timepoints_numerically([t for t in timepoints if t != 'all']) or ['all']

            if len(timepoints) < 2:
                return None

            # Create subplot grid
            fig, axes = create_behavior_subplot_grid(len(timepoints), max_cols=3)

            # First pass: calculate global y-axis limits
            global_ymin = float('inf')
            global_ymax = float('-inf')

            for timepoint in timepoints:
                if 'timepoint' in filtered_df.columns:
                    tp_df = filtered_df[filtered_df['timepoint'] == timepoint]
                else:
                    tp_df = filtered_df

                for group in groups:
                    if 'group' in tp_df.columns and group != 'all':
                        group_df = tp_df[tp_df['group'] == group]
                    else:
                        group_df = tp_df

                    stats = calculate_grouped_stats(group_df, ['behavior'], 'value')
                    if not stats.empty:
                        ymin = (stats['mean'] - stats['sem'].fillna(0)).min()
                        ymax = (stats['mean'] + stats['sem'].fillna(0)).max()
                        if ymin < global_ymin:
                            global_ymin = ymin
                        if ymax > global_ymax:
                            global_ymax = ymax

            # Add padding to y-axis limits
            if global_ymin != float('inf') and global_ymax != float('-inf'):
                y_range = global_ymax - global_ymin
                padding = y_range * 0.1 if y_range > 0 else 0.1
                global_ymin = max(0, global_ymin - padding)
                global_ymax = global_ymax + padding

            # Second pass: plot data
            for idx, timepoint in enumerate(timepoints):
                ax = axes[idx]

                if 'timepoint' in filtered_df.columns:
                    tp_df = filtered_df[filtered_df['timepoint'] == timepoint]
                else:
                    tp_df = filtered_df

                # Calculate positions for grouped bars
                x = np.arange(len(behaviors))
                n_groups = len(groups)
                width = 0.8 / n_groups

                has_data = False
                for group_idx, group in enumerate(groups):
                    if 'group' in tp_df.columns and group != 'all':
                        group_df = tp_df[tp_df['group'] == group]
                    else:
                        group_df = tp_df

                    # Calculate stats per behavior for this group
                    stats = calculate_grouped_stats(group_df, ['behavior'], 'value')

                    if stats.empty:
                        continue

                    # Reorder to match behaviors list
                    stats = stats.set_index('behavior').reindex(behaviors).reset_index()

                    means = stats['mean'].fillna(0).values
                    sems = stats['sem'].fillna(0).values

                    has_data = True
                    offset = (group_idx - n_groups / 2 + 0.5) * width
                    color = self.group_colors[group_idx % len(self.group_colors)]

                    ax.bar(x + offset, means, width, label=group,
                           color=color, yerr=sems, capsize=3)

                if not has_data:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                            transform=ax.transAxes)
                else:
                    # Set shared y-axis limits
                    if global_ymin != float('inf') and global_ymax != float('-inf'):
                        ax.set_ylim(global_ymin, global_ymax)

                ax.set_xticks(x)
                ax.set_xticklabels(behaviors, rotation=45, ha='right', fontsize=8)
                ax.set_title(timepoint, fontsize=10)
                ax.set_ylabel(ylabel, fontsize=9)

                # Add legend to first subplot only
                if idx == 0 and len(groups) > 1:
                    ax.legend(loc='upper right', fontsize=8)

            # Overall title
            fig.suptitle(title, fontsize=12, y=1.02)
            plt.tight_layout()

            path = self.figures_dir / filename
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            logger.info(f"Created figure: {path}")
            return str(path)

        except Exception as e:
            logger.error(f"Failed to create timepoint comparison figure: {e}")
            plt.close('all')
            return None

    def _create_behavior_timepoint_comparison_figures(self, df: pd.DataFrame) -> List[str]:
        """
        Create both comparison figure types for each LUPE metric.

        Args:
            df: LUPE metrics DataFrame

        Returns:
            List of paths to created files
        """
        created_files = []

        if df is None or df.empty:
            return created_files

        # Metrics to generate comparison figures for
        metrics_config = [
            {
                'metric_type': 'bout_count',
                'title_base': 'Bout Counts',
                'ylabel': 'Number of Bouts'
            },
            {
                'metric_type': 'time_percentage',
                'title_base': 'Time Distribution',
                'ylabel': 'Time (%)'
            },
            {
                'metric_type': 'bout_mean_duration_sec',
                'title_base': 'Mean Bout Duration',
                'ylabel': 'Duration (s)'
            }
        ]

        for sex_filter in self._get_sex_filters():
            sex_suffix = f"_{sex_filter.lower()}" if sex_filter != 'All' else ""
            sex_title = f" ({sex_filter})" if sex_filter != 'All' else ""

            for config in metrics_config:
                # Figure A: By behavior (subplots per behavior, x=timepoints)
                path = self._create_behavior_vs_timepoint_by_behavior(
                    df,
                    metric_type=config['metric_type'],
                    title=f"LUPE: {config['title_base']} Across Timepoints{sex_title}",
                    ylabel=config['ylabel'],
                    filename=f"{self.project_name}_lupe_{config['metric_type']}_by_behavior{sex_suffix}.png",
                    sex_filter=sex_filter
                )
                if path:
                    created_files.append(path)

                # Figure B: By timepoint (subplots per timepoint, comparing behaviors)
                path = self._create_behavior_vs_timepoint_by_timepoint(
                    df,
                    metric_type=config['metric_type'],
                    title=f"LUPE: {config['title_base']} by Timepoint{sex_title}",
                    ylabel=config['ylabel'],
                    filename=f"{self.project_name}_lupe_{config['metric_type']}_by_timepoint{sex_suffix}.png",
                    sex_filter=sex_filter
                )
                if path:
                    created_files.append(path)

                # Figure C: Percent change from baseline (subplots per behavior)
                path = self._create_behavior_percent_change_by_behavior(
                    df,
                    metric_type=config['metric_type'],
                    title=f"LUPE: {config['title_base']} - % Change from Baseline{sex_title}",
                    ylabel='% Change',
                    filename=f"{self.project_name}_lupe_{config['metric_type']}_pct_change{sex_suffix}.png",
                    sex_filter=sex_filter
                )
                if path:
                    created_files.append(path)

        return created_files

    def _create_delta_from_baseline_heatmap(
        self, df: pd.DataFrame, metric_type: str, title: str,
        colorbar_label: str, filename: str, sex_filter: str = 'All'
    ) -> Optional[str]:
        """
        Create a heatmap showing the difference between each timepoint and the first timepoint.

        The heatmap shows delta values (change from baseline) with:
        - Rows: behaviors
        - Columns: timepoints (excluding the first/baseline)
        - Subplots: one per group
        - Color: diverging colormap showing negative (blue) to positive (red) changes

        Args:
            df: Long-format metrics DataFrame
            metric_type: The metric type to analyze
            title: Figure title
            colorbar_label: Label for the colorbar
            filename: Output filename
            sex_filter: 'Male', 'Female', or 'All'

        Returns:
            Path to created file, or None
        """
        try:
            from src.utils.plotting import calculate_grouped_stats

            # Filter data
            metric_df = df[df['metric_type'] == metric_type]
            filtered_df = self._filter_data_by_sex(metric_df, sex_filter)

            if filtered_df.empty:
                return None

            # Get unique values
            behaviors = list(filtered_df['behavior'].unique())
            groups = filtered_df['group'].unique() if 'group' in filtered_df.columns else ['all']
            groups = [g for g in groups if g != 'all'] or ['all']
            timepoints = (filtered_df['timepoint'].unique()
                         if 'timepoint' in filtered_df.columns else ['all'])

            timepoints = sort_timepoints_numerically([t for t in timepoints if t != 'all']) or ['all']

            if len(timepoints) < 2:
                return None  # Need at least 2 timepoints for delta

            baseline_tp = timepoints[0]
            delta_timepoints = timepoints[1:]  # Timepoints to compare against baseline

            n_groups = len(groups)

            # Create figure with subplots for each group
            # Add extra width for colorbar on the right
            fig_width = max(10, len(delta_timepoints) * 1.5 + 4)
            fig_height = max(4, len(behaviors) * 0.5 + 1) * n_groups
            fig, axes = plt.subplots(n_groups, 1, figsize=(fig_width, fig_height),
                                     squeeze=False)

            # Calculate delta values for each group
            all_deltas = []  # For determining color scale

            group_data = {}
            for group in groups:
                if 'group' in filtered_df.columns and group != 'all':
                    group_df = filtered_df[filtered_df['group'] == group]
                else:
                    group_df = filtered_df

                # Calculate mean for each behavior at each timepoint
                delta_matrix = np.zeros((len(behaviors), len(delta_timepoints)))

                for b_idx, behavior in enumerate(behaviors):
                    behavior_df = group_df[group_df['behavior'] == behavior]

                    # Get baseline mean
                    if 'timepoint' in behavior_df.columns:
                        baseline_df = behavior_df[behavior_df['timepoint'] == baseline_tp]
                    else:
                        baseline_df = behavior_df
                    baseline_mean = baseline_df['value'].mean() if not baseline_df.empty else 0

                    # Calculate delta for each subsequent timepoint
                    for t_idx, tp in enumerate(delta_timepoints):
                        if 'timepoint' in behavior_df.columns:
                            tp_df = behavior_df[behavior_df['timepoint'] == tp]
                        else:
                            tp_df = behavior_df
                        tp_mean = tp_df['value'].mean() if not tp_df.empty else 0
                        delta = tp_mean - baseline_mean
                        delta_matrix[b_idx, t_idx] = delta
                        all_deltas.append(delta)

                group_data[group] = delta_matrix

            # Determine symmetric color scale
            if all_deltas:
                max_abs = max(abs(min(all_deltas)), abs(max(all_deltas)))
                if max_abs == 0:
                    max_abs = 1
                vmin, vmax = -max_abs, max_abs
            else:
                vmin, vmax = -1, 1

            # Plot heatmaps
            for g_idx, group in enumerate(groups):
                ax = axes[g_idx, 0]
                delta_matrix = group_data[group]

                # Create heatmap
                im = ax.imshow(delta_matrix, cmap='RdBu_r', aspect='auto',
                               vmin=vmin, vmax=vmax)

                # Set tick labels
                ax.set_xticks(range(len(delta_timepoints)))
                ax.set_xticklabels([f"{tp}\nvs {baseline_tp}" for tp in delta_timepoints],
                                   fontsize=8)
                ax.set_yticks(range(len(behaviors)))
                ax.set_yticklabels(behaviors, fontsize=9)

                # Add value annotations
                for i in range(len(behaviors)):
                    for j in range(len(delta_timepoints)):
                        val = delta_matrix[i, j]
                        # Choose text color based on background
                        text_color = 'white' if abs(val) > max_abs * 0.6 else 'black'
                        ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                                fontsize=7, color=text_color)

                # Labels
                if n_groups > 1:
                    ax.set_ylabel(f'{group}', fontsize=10, fontweight='bold')
                if g_idx == n_groups - 1:
                    ax.set_xlabel('Timepoint Comparison', fontsize=10)

            # Title
            fig.suptitle(title, fontsize=12)

            # Adjust layout to make room for colorbar on the right
            plt.tight_layout()
            fig.subplots_adjust(right=0.85)

            # Add colorbar in a dedicated axes on the right side
            cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label(colorbar_label, fontsize=10)

            path = self.figures_dir / filename
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            logger.info(f"Created figure: {path}")
            return str(path)

        except Exception as e:
            logger.error(f"Failed to create delta heatmap figure: {e}")
            plt.close('all')
            return None

    def _create_delta_heatmap_figures(self, df: pd.DataFrame) -> List[str]:
        """
        Create delta from baseline heatmap figures for all LUPE metrics.

        Args:
            df: LUPE metrics DataFrame

        Returns:
            List of paths to created files
        """
        created_files = []

        if df is None or df.empty:
            return created_files

        # Metrics to generate delta heatmaps for
        metrics_config = [
            {
                'metric_type': 'bout_count',
                'title_base': 'Bout Counts',
                'colorbar_label': 'Delta (counts)'
            },
            {
                'metric_type': 'time_percentage',
                'title_base': 'Time Distribution',
                'colorbar_label': 'Delta (%)'
            },
            {
                'metric_type': 'bout_mean_duration_sec',
                'title_base': 'Mean Bout Duration',
                'colorbar_label': 'Delta (seconds)'
            }
        ]

        for sex_filter in self._get_sex_filters():
            sex_suffix = f"_{sex_filter.lower()}" if sex_filter != 'All' else ""
            sex_title = f" ({sex_filter})" if sex_filter != 'All' else ""

            for config in metrics_config:
                path = self._create_delta_from_baseline_heatmap(
                    df,
                    metric_type=config['metric_type'],
                    title=f"LUPE: {config['title_base']} - Change from Baseline{sex_title}",
                    colorbar_label=config['colorbar_label'],
                    filename=f"{self.project_name}_lupe_{config['metric_type']}_delta_heatmap{sex_suffix}.png",
                    sex_filter=sex_filter
                )
                if path:
                    created_files.append(path)

        return created_files

    def _create_transition_grid_heatmaps(self, transitions: Dict,
                                         sex_filter: str = 'All') -> Optional[str]:
        """
        Create transition matrix heatmaps in groups x timepoints grid.

        Args:
            transitions: Dictionary keyed by (group, condition, sex, timepoint) tuples
            sex_filter: 'Male', 'Female', or 'All' (pools transitions)

        Returns:
            Path to created file, or None
        """
        try:
            from src.utils.plotting import (
                create_group_timepoint_grid, plot_transition_heatmap_subplot,
                setup_plot_style
            )

            if not transitions:
                return None

            # Filter/pool transitions by sex
            filtered_transitions = {}

            for key, matrix in transitions.items():
                if not isinstance(key, tuple) or len(key) < 4:
                    # Use as-is if key format is unexpected
                    filtered_transitions[key] = matrix
                    continue

                group, condition, sex, timepoint = key

                if sex_filter != 'All' and sex != sex_filter and sex != 'all':
                    continue

                # Create new key without sex
                new_key = (group, timepoint)

                if new_key not in filtered_transitions:
                    filtered_transitions[new_key] = []

                filtered_transitions[new_key].append(matrix)

            # Average matrices with same (group, timepoint) key
            averaged_transitions = {}
            for key, matrices in filtered_transitions.items():
                if isinstance(matrices, list):
                    # Average the matrices
                    stacked = np.stack([m.values for m in matrices])
                    avg_matrix = pd.DataFrame(
                        np.nanmean(stacked, axis=0),
                        index=matrices[0].index,
                        columns=matrices[0].columns
                    )
                    averaged_transitions[key] = avg_matrix
                else:
                    averaged_transitions[key] = matrices

            if not averaged_transitions:
                return None

            # Extract unique groups and timepoints
            groups = sorted(set(k[0] for k in averaged_transitions.keys() if k[0] != 'all'))
            groups = groups or ['all']
            timepoints = sort_timepoints_numerically(
                list(set(k[1] for k in averaged_transitions.keys() if k[1] != 'all'))
            )
            timepoints = timepoints or ['all']

            n_groups = len(groups)
            n_timepoints = len(timepoints)

            # Create grid
            fig, axes = create_group_timepoint_grid(
                n_groups, n_timepoints,
                figsize_per_subplot=(4, 4),
                sharex=False, sharey=False
            )

            sm = None  # ScalarMappable for colorbar

            for row_idx, group in enumerate(groups):
                for col_idx, timepoint in enumerate(timepoints):
                    ax = axes[row_idx, col_idx]
                    key = (group, timepoint)

                    if key not in averaged_transitions:
                        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                                transform=ax.transAxes)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        continue

                    matrix = averaged_transitions[key]

                    # Plot heatmap
                    title = f"{group}" if n_groups > 1 else ""
                    if n_timepoints > 1 and row_idx == 0:
                        title = f"{timepoint}"

                    sm = plot_transition_heatmap_subplot(
                        ax, matrix, title=title,
                        annotate=(n_groups * n_timepoints <= 6)  # Annotate only for small grids
                    )

                    # Row labels
                    if col_idx == 0 and n_groups > 1:
                        ax.set_ylabel(f'{group}\nFrom', fontsize=9)
                    elif col_idx == 0:
                        ax.set_ylabel('From', fontsize=9)

                    # Column labels
                    if row_idx == n_groups - 1:
                        ax.set_xlabel('To', fontsize=9)

            # Add shared colorbar
            if sm is not None:
                cbar = fig.colorbar(sm, ax=axes, shrink=0.8, pad=0.02)
                cbar.set_label('Transition Probability')

            # Title
            sex_suffix = f" ({sex_filter})" if sex_filter != 'All' else ""
            fig.suptitle(f'LUPE: Transition Matrices - Groups x Timepoints{sex_suffix}',
                        fontsize=14, y=1.02)
            plt.tight_layout()

            # Save
            suffix = f"_{sex_filter.lower()}" if sex_filter != 'All' else ""
            path = self.figures_dir / f"{self.project_name}_lupe_transitions_grid{suffix}.png"
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            logger.info(f"Created figure: {path}")
            return str(path)

        except Exception as e:
            logger.error(f"Failed to create transition grid heatmaps: {e}")
            plt.close('all')
            return None

    def _create_transition_grid_figures(self, transitions: Dict) -> List[str]:
        """
        Create transition grid figures with sex separation if enabled.

        Args:
            transitions: Dictionary of transition matrices

        Returns:
            List of paths to created files
        """
        created_files = []

        if not transitions:
            return created_files

        for sex_filter in self._get_sex_filters():
            path = self._create_transition_grid_heatmaps(transitions, sex_filter)
            if path:
                created_files.append(path)

        return created_files

    def _create_transition_delta_heatmaps(self, transitions: Dict,
                                           sex_filter: str = 'All') -> Optional[str]:
        """
        Create heatmaps showing the change in transition probabilities from baseline.

        Shows delta (difference) between each timepoint's transition matrix and
        the first timepoint (baseline) transition matrix.

        Args:
            transitions: Dictionary keyed by (group, condition, sex, timepoint) tuples
            sex_filter: 'Male', 'Female', or 'All' (pools transitions)

        Returns:
            Path to created file, or None
        """
        try:
            if not transitions:
                return None

            # Filter/pool transitions by sex (same logic as _create_transition_grid_heatmaps)
            filtered_transitions = {}

            for key, matrix in transitions.items():
                if not isinstance(key, tuple) or len(key) < 4:
                    filtered_transitions[key] = matrix
                    continue

                group, condition, sex, timepoint = key

                if sex_filter != 'All' and sex != sex_filter and sex != 'all':
                    continue

                new_key = (group, timepoint)

                if new_key not in filtered_transitions:
                    filtered_transitions[new_key] = []

                filtered_transitions[new_key].append(matrix)

            # Average matrices with same (group, timepoint) key
            averaged_transitions = {}
            for key, matrices in filtered_transitions.items():
                if isinstance(matrices, list):
                    stacked = np.stack([m.values for m in matrices])
                    avg_matrix = pd.DataFrame(
                        np.nanmean(stacked, axis=0),
                        index=matrices[0].index,
                        columns=matrices[0].columns
                    )
                    averaged_transitions[key] = avg_matrix
                else:
                    averaged_transitions[key] = matrices

            if not averaged_transitions:
                return None

            # Extract unique groups and timepoints
            groups = sorted(set(k[0] for k in averaged_transitions.keys() if k[0] != 'all'))
            groups = groups or ['all']
            timepoints = sort_timepoints_numerically(
                list(set(k[1] for k in averaged_transitions.keys() if k[1] != 'all'))
            )
            timepoints = timepoints or ['all']

            if len(timepoints) < 2:
                return None  # Need at least 2 timepoints for delta

            baseline_tp = timepoints[0]
            delta_timepoints = timepoints[1:]

            n_groups = len(groups)
            n_delta_timepoints = len(delta_timepoints)

            # Get behavior labels from first matrix
            first_key = list(averaged_transitions.keys())[0]
            behaviors = list(averaged_transitions[first_key].index)
            n_behaviors = len(behaviors)

            # Create figure with grid: rows = groups, columns = delta timepoints
            fig_width = max(10, n_delta_timepoints * 4 + 3)
            fig_height = max(4, n_groups * 4)
            fig, axes = plt.subplots(n_groups, n_delta_timepoints,
                                     figsize=(fig_width, fig_height),
                                     squeeze=False)

            # Calculate all delta matrices and find global min/max for color scale
            delta_matrices = {}
            all_deltas = []

            for group in groups:
                baseline_key = (group, baseline_tp)
                if baseline_key not in averaged_transitions:
                    continue

                baseline_matrix = averaged_transitions[baseline_key]

                for tp in delta_timepoints:
                    tp_key = (group, tp)
                    if tp_key not in averaged_transitions:
                        continue

                    tp_matrix = averaged_transitions[tp_key]
                    delta_matrix = tp_matrix - baseline_matrix
                    delta_matrices[(group, tp)] = delta_matrix
                    all_deltas.extend(delta_matrix.values.flatten())

            # Determine symmetric color scale
            if all_deltas:
                all_deltas = [d for d in all_deltas if not np.isnan(d)]
                if all_deltas:
                    max_abs = max(abs(min(all_deltas)), abs(max(all_deltas)))
                    if max_abs == 0:
                        max_abs = 0.1
                    vmin, vmax = -max_abs, max_abs
                else:
                    vmin, vmax = -0.1, 0.1
            else:
                vmin, vmax = -0.1, 0.1

            # Plot delta heatmaps
            im = None
            for row_idx, group in enumerate(groups):
                for col_idx, tp in enumerate(delta_timepoints):
                    ax = axes[row_idx, col_idx]
                    key = (group, tp)

                    if key not in delta_matrices:
                        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                                transform=ax.transAxes)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        continue

                    delta_matrix = delta_matrices[key]

                    # Create heatmap with diverging colormap
                    im = ax.imshow(delta_matrix.values, cmap='RdBu_r', aspect='auto',
                                   vmin=vmin, vmax=vmax)

                    # Set tick labels
                    ax.set_xticks(range(n_behaviors))
                    ax.set_xticklabels(behaviors, rotation=45, ha='right', fontsize=7)
                    ax.set_yticks(range(n_behaviors))
                    ax.set_yticklabels(behaviors, fontsize=7)

                    # Add value annotations for small grids
                    if n_groups * n_delta_timepoints <= 4:
                        for i in range(n_behaviors):
                            for j in range(n_behaviors):
                                val = delta_matrix.iloc[i, j]
                                if not np.isnan(val):
                                    text_color = 'white' if abs(val) > max_abs * 0.6 else 'black'
                                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                                            fontsize=6, color=text_color)

                    # Title for column (timepoint comparison)
                    if row_idx == 0:
                        ax.set_title(f'{tp} vs {baseline_tp}', fontsize=9)

                    # Row label (group)
                    if col_idx == 0 and n_groups > 1:
                        ax.set_ylabel(f'{group}\nFrom', fontsize=9, fontweight='bold')
                    elif col_idx == 0:
                        ax.set_ylabel('From', fontsize=9)

                    # X-axis label
                    if row_idx == n_groups - 1:
                        ax.set_xlabel('To', fontsize=9)

            # Title
            sex_suffix = f" ({sex_filter})" if sex_filter != 'All' else ""
            fig.suptitle(f'LUPE: Transition Probability Changes from Baseline{sex_suffix}',
                        fontsize=12)

            # Adjust layout and add colorbar
            plt.tight_layout()
            fig.subplots_adjust(right=0.88)

            # Add colorbar on the right
            if im is not None:
                cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
                cbar = fig.colorbar(im, cax=cbar_ax)
                cbar.set_label('Delta (probability)', fontsize=10)

            # Save
            suffix = f"_{sex_filter.lower()}" if sex_filter != 'All' else ""
            path = self.figures_dir / f"{self.project_name}_lupe_transitions_delta{suffix}.png"
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            logger.info(f"Created figure: {path}")
            return str(path)

        except Exception as e:
            logger.error(f"Failed to create transition delta heatmaps: {e}")
            plt.close('all')
            return None

    def _create_transition_delta_figures(self, transitions: Dict) -> List[str]:
        """
        Create transition delta heatmap figures with sex separation if enabled.

        Args:
            transitions: Dictionary of transition matrices

        Returns:
            List of paths to created files
        """
        created_files = []

        if not transitions:
            return created_files

        for sex_filter in self._get_sex_filters():
            path = self._create_transition_delta_heatmaps(transitions, sex_filter)
            if path:
                created_files.append(path)

        return created_files

    def write_statistical_report(self) -> str:
        """
        Write comprehensive statistical report as text file.

        Returns:
            str: Path to created report file
        """
        report_lines = []

        # Header
        report_lines.append("=" * 80)
        report_lines.append("                    STATISTICAL ANALYSIS REPORT")
        report_lines.append(f"                    Project: {self.project_name}")
        report_lines.append(f"                    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Experimental design
        report_lines.append("1. EXPERIMENTAL DESIGN")
        report_lines.append("-" * 40)
        report_lines.append(f"   Groups: {', '.join(self.groups) if self.groups else 'None'}")
        report_lines.append(f"   Conditions: {', '.join(self.conditions) if self.conditions else 'None'}")
        report_lines.append(f"   Sex as factor: {'Yes' if self.include_sex else 'No'}")
        report_lines.append(f"   Timepoints: {'Yes' if self.has_timepoints else 'No'}")

        # Count animals
        animal_count = 0
        assignments = self.config.get('animal_assignments', {})
        def count_animals(obj):
            nonlocal animal_count
            if isinstance(obj, list):
                animal_count += len([a for a in obj if isinstance(a, str)])
            elif isinstance(obj, dict):
                for v in obj.values():
                    count_animals(v)
        count_animals(assignments)
        report_lines.append(f"   Total animals: {animal_count}")
        report_lines.append("")

        # Data summary
        report_lines.append("2. DATA SUMMARY")
        report_lines.append("-" * 40)

        lupe_metrics = self.aggregated_data.get('lupe_metrics')
        if lupe_metrics is not None and not lupe_metrics.empty:
            n_animals = lupe_metrics['animal_id'].nunique() if 'animal_id' in lupe_metrics.columns else 0
            n_metrics = lupe_metrics['metric_type'].nunique() if 'metric_type' in lupe_metrics.columns else 0
            report_lines.append(f"   LUPE data: {n_animals} animals, {n_metrics} metric types")

        amps_metrics = self.aggregated_data.get('amps_metrics')
        if amps_metrics is not None and not amps_metrics.empty:
            n_animals = amps_metrics['animal_id'].nunique() if 'animal_id' in amps_metrics.columns else 0
            report_lines.append(f"   AMPS data: {n_animals} animals")
        report_lines.append("")

        # Statistical test results
        report_lines.append("3. STATISTICAL TEST RESULTS")
        report_lines.append("-" * 40)

        if not self.test_results:
            report_lines.append("   No statistical tests performed.")
        else:
            for result in self.test_results:
                report_lines.append("")
                report_lines.append(f"   Metric: {result.metric_name}")
                report_lines.append(f"   Test: {result.test_name}")

                if not np.isnan(result.statistic):
                    report_lines.append(f"   Statistic: {result.statistic:.4f}")
                    report_lines.append(f"   p-value: {result.p_value:.4f}")

                    # Add significance stars
                    if result.p_value < 0.001:
                        sig = " ***"
                    elif result.p_value < 0.01:
                        sig = " **"
                    elif result.p_value < 0.05:
                        sig = " *"
                    else:
                        sig = " (ns)"
                    report_lines.append(f"   Significance: p {'<' if result.p_value < 0.05 else '>'} 0.05{sig}")

                if not np.isnan(result.effect_size):
                    report_lines.append(f"   Effect size ({result.effect_size_name}): {result.effect_size:.4f} ({result.effect_interpretation})")

                # Sample sizes
                if result.sample_sizes:
                    sizes = ', '.join(f"{k}: n={v}" for k, v in result.sample_sizes.items())
                    report_lines.append(f"   Sample sizes: {sizes}")

                # Assumptions
                if result.assumptions_met:
                    assump = ', '.join(f"{k}={v}" for k, v in result.assumptions_met.items())
                    report_lines.append(f"   Assumptions: {assump}")

                # Post-hoc results
                if result.post_hoc_results is not None and not result.post_hoc_results.empty:
                    report_lines.append("   Post-hoc comparisons:")
                    for _, row in result.post_hoc_results.iterrows():
                        g1 = row.get('group1', '')
                        g2 = row.get('group2', '')
                        p = row.get('p_corrected', row.get('p_value', np.nan))
                        sig = '*' if row.get('significant', False) else ''
                        report_lines.append(f"      {g1} vs {g2}: p = {p:.4f} {sig}")

        report_lines.append("")

        # Methodology notes
        report_lines.append("4. METHODOLOGY NOTES")
        report_lines.append("-" * 40)
        report_lines.append("""
   Test selection criteria:
   - 2 groups: Independent t-test (parametric) or Mann-Whitney U (non-parametric)
   - 3+ groups: One-way ANOVA (parametric) or Kruskal-Wallis (non-parametric)

   Assumptions tested:
   - Normality: Shapiro-Wilk (n < 50) or Kolmogorov-Smirnov (n >= 50)
   - Homogeneity of variance: Levene's test

   Post-hoc tests:
   - Parametric: Tukey HSD
   - Non-parametric: Dunn's test with Bonferroni correction

   Effect size interpretation:
   - Cohen's d: 0.2 = small, 0.5 = medium, 0.8 = large
   - Eta-squared: 0.01 = small, 0.06 = medium, 0.14 = large

   Significance levels:
   - * p < 0.05
   - ** p < 0.01
   - *** p < 0.001
""")

        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("                         END OF REPORT")
        report_lines.append("=" * 80)

        # Write to file
        report_text = '\n'.join(report_lines)
        path = self.output_dir / f"{self.project_name}_statistical_report.txt"

        with open(path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        logger.info(f"Created report: {path}")
        return str(path)

    def generate_all_outputs(self) -> Dict[str, List[str]]:
        """
        Generate all outputs: CSVs, figures, and report.

        Returns:
            dict: Dictionary with keys 'csv', 'figures', 'report' containing file paths
        """
        outputs = {
            'csv': self.create_summary_csv(),
            'figures': self.create_figures(),
            'report': [self.write_statistical_report()]
        }

        logger.info(f"Generated {len(outputs['csv'])} CSV files, "
                    f"{len(outputs['figures'])} figures, "
                    f"{len(outputs['report'])} report")

        return outputs
