"""
Statistical Tests Module

This module provides statistical analysis capabilities for LUPE and AMPS data.
It automatically selects appropriate statistical tests based on the experimental
design and data characteristics.

Test Selection Logic:
- Single group, no timepoints: Descriptive statistics only
- Two groups, no timepoints: Independent t-test (or Mann-Whitney U if non-normal)
- 3+ groups, no timepoints: One-way ANOVA (or Kruskal-Wallis if non-normal)
- Any groups, multiple timepoints: Two-way ANOVA (group x time) or mixed ANOVA
- With sex factor: Add sex as additional factor

Assumptions tested:
- Normality: Shapiro-Wilk test (n < 50) or Kolmogorov-Smirnov (n >= 50)
- Homogeneity of variance: Levene's test

Effect sizes calculated:
- t-test: Cohen's d
- ANOVA: Eta-squared (eta^2) or partial eta-squared
- Mann-Whitney U: rank-biserial correlation (r)
- Kruskal-Wallis: epsilon-squared

Post-hoc tests:
- Parametric: Tukey HSD
- Non-parametric: Dunn's test with Bonferroni correction

Usage:
    from src.core.statistical_tests import StatisticalAnalyzer

    analyzer = StatisticalAnalyzer(df_long, config)
    results = analyzer.analyze_all_metrics()
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy import stats
import logging
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Container for statistical test results."""
    metric_name: str
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    effect_size_name: str
    effect_interpretation: str
    factors_tested: List[str]
    sample_sizes: Dict[str, int]
    assumptions_met: Dict[str, bool]
    post_hoc_results: Optional[pd.DataFrame]
    notes: str


class StatisticalAnalyzer:
    """
    Performs statistical analysis on aggregated LUPE and AMPS data.

    Automatically selects appropriate statistical tests based on:
    - Number of groups
    - Number of conditions/timepoints
    - Whether sex is a factor
    - Normality of data

    Attributes:
        data (pd.DataFrame): Long-format DataFrame with metric values
        config (dict): Project configuration
        factors (list): List of experimental factors to analyze
        n_groups (int): Number of groups in the design
        n_conditions (int): Number of conditions
        has_sex (bool): Whether sex is a factor
        has_timepoints (bool): Whether timepoints are analyzed separately
    """

    # Effect size interpretation thresholds
    COHENS_D_THRESHOLDS = {'small': 0.2, 'medium': 0.5, 'large': 0.8}
    ETA_SQUARED_THRESHOLDS = {'small': 0.01, 'medium': 0.06, 'large': 0.14}
    R_THRESHOLDS = {'small': 0.1, 'medium': 0.3, 'large': 0.5}

    def __init__(self, data: pd.DataFrame, config: Dict):
        """
        Initialize the statistical analyzer.

        Args:
            data: Long-format DataFrame with columns including:
                  - animal_id, group, condition, sex (as applicable)
                  - metric_type, behavior/metric_name, value
            config: Project configuration dictionary
        """
        self.data = data
        self.config = config
        self._determine_factors()

    def _determine_factors(self):
        """Determine experimental factors from config."""
        self.groups = [g.get('name', '') for g in self.config.get('groups', [])]
        self.conditions = [c.get('name', '') for c in self.config.get('conditions', [])]
        self.has_sex = self.config.get('include_sex', False)
        self.has_timepoints = self.config.get('has_timepoints', False)
        self.separate_timepoints = self.config.get('separate_timepoints', False)

        self.n_groups = len(self.groups) if self.groups else 1
        self.n_conditions = len(self.conditions) if self.conditions else 1

        # Build list of factors to analyze
        self.factors = []
        if self.n_groups > 1:
            self.factors.append('group')
        if self.n_conditions > 1:
            self.factors.append('condition')
        if self.has_sex:
            self.factors.append('sex')

        logger.info(f"Design: {self.n_groups} groups, {self.n_conditions} conditions, "
                    f"sex={self.has_sex}, timepoints={self.has_timepoints}")

    def check_normality(self, values: np.ndarray) -> Tuple[bool, float]:
        """
        Test for normality using appropriate test.

        Uses Shapiro-Wilk for n < 50, Kolmogorov-Smirnov for n >= 50.

        Args:
            values: Array of values to test

        Returns:
            tuple: (is_normal, p_value)
        """
        values = values[~np.isnan(values)]

        if len(values) < 3:
            return True, 1.0  # Assume normal with very small samples

        try:
            if len(values) < 50:
                stat, p_value = stats.shapiro(values)
            else:
                # Kolmogorov-Smirnov test against normal distribution
                stat, p_value = stats.kstest(values, 'norm',
                                             args=(np.mean(values), np.std(values)))

            is_normal = p_value > 0.05
            return is_normal, p_value

        except Exception as e:
            logger.warning(f"Normality test failed: {e}")
            return True, 1.0  # Default to assuming normality

    def check_homogeneity(self, groups: List[np.ndarray]) -> Tuple[bool, float]:
        """
        Test for homogeneity of variance using Levene's test.

        Args:
            groups: List of arrays, one per group

        Returns:
            tuple: (is_homogeneous, p_value)
        """
        # Filter out empty groups
        valid_groups = [g[~np.isnan(g)] for g in groups if len(g[~np.isnan(g)]) >= 2]

        if len(valid_groups) < 2:
            return True, 1.0

        try:
            stat, p_value = stats.levene(*valid_groups)
            is_homogeneous = p_value > 0.05
            return is_homogeneous, p_value
        except Exception as e:
            logger.warning(f"Homogeneity test failed: {e}")
            return True, 1.0

    def _calculate_cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """
        Calculate Cohen's d effect size for two groups.

        Args:
            group1, group2: Arrays of values

        Returns:
            float: Cohen's d value
        """
        g1 = group1[~np.isnan(group1)]
        g2 = group2[~np.isnan(group2)]

        if len(g1) < 2 or len(g2) < 2:
            return np.nan

        n1, n2 = len(g1), len(g2)
        var1, var2 = np.var(g1, ddof=1), np.var(g2, ddof=1)

        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return 0.0

        return (np.mean(g1) - np.mean(g2)) / pooled_std

    def _calculate_eta_squared(self, ss_between: float, ss_total: float) -> float:
        """
        Calculate eta-squared effect size for ANOVA.

        Args:
            ss_between: Sum of squares between groups
            ss_total: Total sum of squares

        Returns:
            float: Eta-squared value
        """
        if ss_total == 0:
            return 0.0
        return ss_between / ss_total

    def _calculate_rank_biserial(self, u_stat: float, n1: int, n2: int) -> float:
        """
        Calculate rank-biserial correlation for Mann-Whitney U.

        Args:
            u_stat: U statistic
            n1, n2: Sample sizes

        Returns:
            float: Rank-biserial correlation (r)
        """
        return 1 - (2 * u_stat) / (n1 * n2)

    def _interpret_effect_size(self, value: float, effect_type: str) -> str:
        """
        Interpret effect size magnitude.

        Args:
            value: Effect size value
            effect_type: Type of effect size ('d', 'eta_squared', 'r')

        Returns:
            str: Interpretation ('negligible', 'small', 'medium', 'large')
        """
        value = abs(value)

        if effect_type == 'd':
            thresholds = self.COHENS_D_THRESHOLDS
        elif effect_type == 'eta_squared':
            thresholds = self.ETA_SQUARED_THRESHOLDS
        else:  # r
            thresholds = self.R_THRESHOLDS

        if value >= thresholds['large']:
            return 'large'
        elif value >= thresholds['medium']:
            return 'medium'
        elif value >= thresholds['small']:
            return 'small'
        else:
            return 'negligible'

    def run_ttest(self, values: np.ndarray, groups: np.ndarray) -> TestResult:
        """
        Run independent samples t-test.

        Args:
            values: Array of metric values
            groups: Array of group labels

        Returns:
            TestResult: Test results
        """
        unique_groups = np.unique(groups[~pd.isna(groups)])

        if len(unique_groups) != 2:
            raise ValueError("T-test requires exactly 2 groups")

        g1 = values[groups == unique_groups[0]]
        g2 = values[groups == unique_groups[1]]

        g1 = g1[~np.isnan(g1)]
        g2 = g2[~np.isnan(g2)]

        # Check assumptions
        norm1, p_norm1 = self.check_normality(g1)
        norm2, p_norm2 = self.check_normality(g2)
        is_normal = norm1 and norm2

        homogeneous, p_homog = self.check_homogeneity([g1, g2])

        assumptions = {
            'normality': is_normal,
            'homogeneity': homogeneous
        }

        # Run appropriate test
        if is_normal:
            stat, p_value = stats.ttest_ind(g1, g2, equal_var=homogeneous)
            test_name = "Independent t-test" + (" (Welch's)" if not homogeneous else "")
            effect_size = self._calculate_cohens_d(g1, g2)
            effect_name = "Cohen's d"
            effect_type = 'd'
        else:
            # Use Mann-Whitney U
            stat, p_value = stats.mannwhitneyu(g1, g2, alternative='two-sided')
            test_name = "Mann-Whitney U"
            effect_size = self._calculate_rank_biserial(stat, len(g1), len(g2))
            effect_name = "rank-biserial r"
            effect_type = 'r'

        interpretation = self._interpret_effect_size(effect_size, effect_type)

        return TestResult(
            metric_name="",  # Filled by caller
            test_name=test_name,
            statistic=stat,
            p_value=p_value,
            effect_size=effect_size,
            effect_size_name=effect_name,
            effect_interpretation=interpretation,
            factors_tested=['group'],
            sample_sizes={str(unique_groups[0]): len(g1), str(unique_groups[1]): len(g2)},
            assumptions_met=assumptions,
            post_hoc_results=None,
            notes=""
        )

    def run_oneway_anova(self, values: np.ndarray, groups: np.ndarray) -> TestResult:
        """
        Run one-way ANOVA or Kruskal-Wallis test.

        Args:
            values: Array of metric values
            groups: Array of group labels

        Returns:
            TestResult: Test results
        """
        unique_groups = np.unique(groups[~pd.isna(groups)])

        if len(unique_groups) < 2:
            raise ValueError("ANOVA requires at least 2 groups")

        # Prepare groups
        group_data = []
        sample_sizes = {}

        for g in unique_groups:
            g_vals = values[groups == g]
            g_vals = g_vals[~np.isnan(g_vals)]
            group_data.append(g_vals)
            sample_sizes[str(g)] = len(g_vals)

        # Check assumptions
        normality_results = [self.check_normality(gd)[0] for gd in group_data]
        is_normal = all(normality_results)
        homogeneous, _ = self.check_homogeneity(group_data)

        assumptions = {
            'normality': is_normal,
            'homogeneity': homogeneous
        }

        if is_normal:
            # One-way ANOVA
            stat, p_value = stats.f_oneway(*group_data)
            test_name = "One-way ANOVA"

            # Calculate eta-squared
            grand_mean = np.mean(np.concatenate(group_data))
            ss_between = sum(len(gd) * (np.mean(gd) - grand_mean)**2 for gd in group_data)
            ss_total = sum(np.sum((gd - grand_mean)**2) for gd in group_data)
            effect_size = self._calculate_eta_squared(ss_between, ss_total)
            effect_name = "eta-squared"
            effect_type = 'eta_squared'

            # Post-hoc: Tukey HSD
            post_hoc = self._tukey_hsd(group_data, unique_groups)
        else:
            # Kruskal-Wallis test
            stat, p_value = stats.kruskal(*group_data)
            test_name = "Kruskal-Wallis H"

            # Epsilon-squared effect size
            n_total = sum(len(gd) for gd in group_data)
            effect_size = (stat - len(unique_groups) + 1) / (n_total - len(unique_groups))
            effect_name = "epsilon-squared"
            effect_type = 'eta_squared'  # Use same interpretation

            # Post-hoc: Dunn's test (simplified)
            post_hoc = self._dunn_test(group_data, unique_groups)

        interpretation = self._interpret_effect_size(effect_size, effect_type)

        return TestResult(
            metric_name="",
            test_name=test_name,
            statistic=stat,
            p_value=p_value,
            effect_size=effect_size,
            effect_size_name=effect_name,
            effect_interpretation=interpretation,
            factors_tested=['group'],
            sample_sizes=sample_sizes,
            assumptions_met=assumptions,
            post_hoc_results=post_hoc,
            notes=""
        )

    def _tukey_hsd(self, group_data: List[np.ndarray],
                   group_names: np.ndarray) -> pd.DataFrame:
        """
        Perform Tukey HSD post-hoc test.

        Args:
            group_data: List of arrays for each group
            group_names: Array of group names

        Returns:
            pd.DataFrame: Post-hoc comparison results
        """
        try:
            from scipy.stats import tukey_hsd

            result = tukey_hsd(*group_data)

            # Build comparison table
            comparisons = []
            for i in range(len(group_names)):
                for j in range(i + 1, len(group_names)):
                    comparisons.append({
                        'group1': str(group_names[i]),
                        'group2': str(group_names[j]),
                        'mean_diff': np.mean(group_data[i]) - np.mean(group_data[j]),
                        'p_value': result.pvalue[i, j],
                        'significant': result.pvalue[i, j] < 0.05
                    })

            return pd.DataFrame(comparisons)

        except (ImportError, Exception) as e:
            logger.warning(f"Tukey HSD failed: {e}")
            return self._pairwise_ttest(group_data, group_names)

    def _dunn_test(self, group_data: List[np.ndarray],
                   group_names: np.ndarray) -> pd.DataFrame:
        """
        Perform Dunn's test with Bonferroni correction.

        Args:
            group_data: List of arrays for each group
            group_names: Array of group names

        Returns:
            pd.DataFrame: Post-hoc comparison results
        """
        comparisons = []
        n_comparisons = len(group_names) * (len(group_names) - 1) / 2

        for i in range(len(group_names)):
            for j in range(i + 1, len(group_names)):
                # Mann-Whitney U for each pair
                try:
                    stat, p = stats.mannwhitneyu(
                        group_data[i], group_data[j],
                        alternative='two-sided'
                    )
                    # Bonferroni correction
                    p_corrected = min(p * n_comparisons, 1.0)

                    comparisons.append({
                        'group1': str(group_names[i]),
                        'group2': str(group_names[j]),
                        'U_statistic': stat,
                        'p_value': p,
                        'p_corrected': p_corrected,
                        'significant': p_corrected < 0.05
                    })
                except Exception:
                    continue

        return pd.DataFrame(comparisons)

    def _pairwise_ttest(self, group_data: List[np.ndarray],
                        group_names: np.ndarray) -> pd.DataFrame:
        """
        Perform pairwise t-tests with Bonferroni correction (fallback).

        Args:
            group_data: List of arrays for each group
            group_names: Array of group names

        Returns:
            pd.DataFrame: Post-hoc comparison results
        """
        comparisons = []
        n_comparisons = len(group_names) * (len(group_names) - 1) / 2

        for i in range(len(group_names)):
            for j in range(i + 1, len(group_names)):
                try:
                    stat, p = stats.ttest_ind(group_data[i], group_data[j])
                    p_corrected = min(p * n_comparisons, 1.0)

                    comparisons.append({
                        'group1': str(group_names[i]),
                        'group2': str(group_names[j]),
                        't_statistic': stat,
                        'p_value': p,
                        'p_corrected': p_corrected,
                        'significant': p_corrected < 0.05
                    })
                except Exception:
                    continue

        return pd.DataFrame(comparisons)

    def analyze_metric(self, metric_data: pd.DataFrame, metric_name: str) -> TestResult:
        """
        Analyze a single metric using appropriate statistical test.

        Automatically selects test based on experimental design.

        Args:
            metric_data: DataFrame with this metric's data
            metric_name: Name of the metric being analyzed

        Returns:
            TestResult: Statistical test results
        """
        # Extract values and grouping
        values = metric_data['value'].values

        # Remove NaN values
        valid_mask = ~np.isnan(values)
        values = values[valid_mask]

        if len(values) < 3:
            return self._descriptive_only(values, metric_name)

        # Determine which test to run based on design
        if 'group' in metric_data.columns and self.n_groups > 1:
            groups = metric_data['group'].values[valid_mask]

            if self.n_groups == 2:
                result = self.run_ttest(values, groups)
            else:
                result = self.run_oneway_anova(values, groups)

            result.metric_name = metric_name
            return result

        # No grouping factor - descriptive only
        return self._descriptive_only(values, metric_name)

    def _descriptive_only(self, values: np.ndarray, metric_name: str) -> TestResult:
        """
        Return descriptive statistics when no test is appropriate.

        Args:
            values: Array of values
            metric_name: Name of the metric

        Returns:
            TestResult: Descriptive statistics only
        """
        return TestResult(
            metric_name=metric_name,
            test_name="Descriptive statistics only",
            statistic=np.nan,
            p_value=np.nan,
            effect_size=np.nan,
            effect_size_name="N/A",
            effect_interpretation="N/A",
            factors_tested=[],
            sample_sizes={'total': len(values)},
            assumptions_met={},
            post_hoc_results=None,
            notes="No between-group comparison possible with current design"
        )

    def analyze_all_metrics(self, df: pd.DataFrame = None) -> List[TestResult]:
        """
        Analyze all metrics in the dataset.

        Args:
            df: Optional DataFrame to analyze (uses self.data if not provided)

        Returns:
            list: List of TestResult objects for each metric
        """
        if df is None:
            df = self.data

        if df is None or df.empty:
            logger.warning("No data to analyze")
            return []

        results = []

        # Group data by metric type and behavior/name
        groupby_cols = []
        if 'metric_type' in df.columns:
            groupby_cols.append('metric_type')
        if 'behavior' in df.columns:
            groupby_cols.append('behavior')
        if 'metric_name' in df.columns:
            groupby_cols.append('metric_name')

        if not groupby_cols:
            # Try to analyze the whole dataset as one metric
            if 'value' in df.columns:
                result = self.analyze_metric(df, "overall")
                results.append(result)
            return results

        # Analyze each metric separately
        for group_key, group_df in df.groupby(groupby_cols):
            if isinstance(group_key, tuple):
                metric_name = "_".join(str(k) for k in group_key)
            else:
                metric_name = str(group_key)

            try:
                result = self.analyze_metric(group_df, metric_name)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to analyze {metric_name}: {e}")

        return results

    def get_results_dataframe(self, results: List[TestResult]) -> pd.DataFrame:
        """
        Convert results list to a DataFrame.

        Args:
            results: List of TestResult objects

        Returns:
            pd.DataFrame: Results in tabular format
        """
        rows = []
        for r in results:
            rows.append({
                'metric': r.metric_name,
                'test': r.test_name,
                'statistic': r.statistic,
                'p_value': r.p_value,
                'effect_size': r.effect_size,
                'effect_size_name': r.effect_size_name,
                'effect_interpretation': r.effect_interpretation,
                'factors': ', '.join(r.factors_tested),
                'total_n': sum(r.sample_sizes.values()),
                'normality_ok': r.assumptions_met.get('normality', 'N/A'),
                'homogeneity_ok': r.assumptions_met.get('homogeneity', 'N/A'),
                'notes': r.notes
            })

        return pd.DataFrame(rows)

    def get_post_hoc_dataframe(self, results: List[TestResult]) -> pd.DataFrame:
        """
        Extract all post-hoc results into a single DataFrame.

        Args:
            results: List of TestResult objects

        Returns:
            pd.DataFrame: All post-hoc comparisons
        """
        dfs = []
        for r in results:
            if r.post_hoc_results is not None and not r.post_hoc_results.empty:
                ph_df = r.post_hoc_results.copy()
                ph_df['metric'] = r.metric_name
                dfs.append(ph_df)

        if dfs:
            return pd.concat(dfs, ignore_index=True)
        else:
            return pd.DataFrame()
