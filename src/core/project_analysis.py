"""
Project Analysis Orchestrator

This module provides the main orchestrator class that coordinates the entire
statistical analysis pipeline. It manages the workflow from data loading
through statistical analysis to report generation.

The orchestrator:
1. Validates project configuration and data sources
2. Loads and aggregates LUPE and AMPS data
3. Runs appropriate statistical tests
4. Generates summary CSVs, figures, and reports

Usage:
    from src.core.project_analysis import ProjectAnalyzer

    analyzer = ProjectAnalyzer(config)
    success = analyzer.run_complete_analysis(progress_callback)

Or step-by-step:
    analyzer = ProjectAnalyzer(config)
    analyzer.validate()
    analyzer.load_data()
    analyzer.run_statistics()
    analyzer.generate_outputs()
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime

from src.core.data_aggregator import DataAggregator
from src.core.statistical_tests import StatisticalAnalyzer
from src.core.report_generator import ReportGenerator

logger = logging.getLogger(__name__)


@dataclass
class AnalysisProgress:
    """Container for analysis progress information."""
    stage: str
    message: str
    progress: float  # 0.0 to 1.0
    details: Optional[str] = None


class ProjectAnalyzer:
    """
    Main orchestrator for project statistical analysis.

    Coordinates the entire analysis pipeline:
    - Data validation and loading
    - Statistical analysis
    - Output generation

    Attributes:
        config (dict): Project configuration
        output_dir (Path): Directory for output files
        aggregator (DataAggregator): Data loading/aggregation component
        all_data (dict): Loaded data from all animals
        aggregated_data (dict): Data aggregated by experimental factors
        test_results (list): Statistical test results
    """

    def __init__(self, config: Dict):
        """
        Initialize the ProjectAnalyzer.

        Args:
            config: Project configuration dictionary containing:
                - project_name: Name of the project
                - data_source_dir: Path to LUPE output folders
                - amps_output_dir: Path to AMPS outputs (optional)
                - output_dir: Path for analysis outputs
                - animal_assignments: Mapping of animals to factors
                - groups, conditions, include_sex, etc.
        """
        self.config = config
        self.output_dir = Path(config.get('output_dir', 'outputs'))

        # Initialize components (data aggregator created lazily)
        self.aggregator = None
        self.all_data = None
        self.aggregated_data = None
        self.test_results = None

        # Track progress
        self._progress_callback = None

    def _report_progress(self, stage: str, message: str, progress: float,
                         details: str = None):
        """Report progress to callback if set."""
        if self._progress_callback:
            progress_info = AnalysisProgress(
                stage=stage,
                message=message,
                progress=progress,
                details=details
            )
            self._progress_callback(progress_info)

        logger.info(f"[{progress*100:.0f}%] {stage}: {message}")

    def validate(self) -> tuple:
        """
        Validate configuration and data sources.

        Returns:
            tuple: (is_valid, list of error messages)
        """
        errors = []

        # Check required fields
        if not self.config.get('project_name'):
            errors.append("Project name is required")

        if not self.config.get('data_source_dir'):
            errors.append("Data source directory is required")

        if not self.config.get('output_dir'):
            errors.append("Output directory is required")

        if not self.config.get('animal_assignments'):
            errors.append("Animal assignments are required (use 'Distribute Animals' button)")

        # Check data source exists
        data_source = Path(self.config.get('data_source_dir', ''))
        if data_source and not data_source.exists():
            errors.append(f"Data source directory not found: {data_source}")

        # Check AMPS source if specified
        amps_source = self.config.get('amps_output_dir', '')
        if amps_source and not Path(amps_source).exists():
            errors.append(f"AMPS output directory not found: {amps_source}")

        # Validate via data aggregator
        if not errors:
            self.aggregator = DataAggregator(self.config)
            is_valid, agg_errors = self.aggregator.validate_data_sources()
            errors.extend(agg_errors)

        return len(errors) == 0, errors

    def load_data(self) -> bool:
        """
        Load and aggregate all LUPE and AMPS data.

        Returns:
            bool: True if data was loaded successfully
        """
        self._report_progress("Loading", "Initializing data aggregator...", 0.1)

        if self.aggregator is None:
            self.aggregator = DataAggregator(self.config)

        # Load all data
        self._report_progress("Loading", "Loading LUPE and AMPS data...", 0.15)
        self.all_data = self.aggregator.load_all_data()

        if not self.all_data:
            logger.error("No data could be loaded")
            return False

        self._report_progress("Loading", f"Loaded data for {len(self.all_data)} animals", 0.25)

        # Aggregate by factors
        self._report_progress("Aggregating", "Aggregating data by experimental factors...", 0.30)
        self.aggregated_data = self.aggregator.aggregate_by_factors(self.all_data)

        return True

    def run_statistics(self) -> bool:
        """
        Run statistical analyses on aggregated data.

        Returns:
            bool: True if analysis completed successfully
        """
        if self.aggregated_data is None:
            logger.error("No aggregated data available for analysis")
            return False

        self._report_progress("Statistics", "Running statistical tests...", 0.40)

        # Analyze LUPE metrics
        lupe_metrics = self.aggregated_data.get('lupe_metrics')
        amps_metrics = self.aggregated_data.get('amps_metrics')

        self.test_results = []

        if lupe_metrics is not None and not lupe_metrics.empty:
            self._report_progress("Statistics", "Analyzing LUPE metrics...", 0.45)
            analyzer = StatisticalAnalyzer(lupe_metrics, self.config)
            lupe_results = analyzer.analyze_all_metrics(lupe_metrics)
            self.test_results.extend(lupe_results)
            self._report_progress("Statistics",
                                  f"Completed {len(lupe_results)} LUPE analyses", 0.55)

        if amps_metrics is not None and not amps_metrics.empty:
            self._report_progress("Statistics", "Analyzing AMPS metrics...", 0.60)
            analyzer = StatisticalAnalyzer(amps_metrics, self.config)
            amps_results = analyzer.analyze_all_metrics(amps_metrics)
            self.test_results.extend(amps_results)
            self._report_progress("Statistics",
                                  f"Completed {len(amps_results)} AMPS analyses", 0.70)

        self._report_progress("Statistics",
                              f"Total: {len(self.test_results)} statistical tests completed", 0.75)
        return True

    def generate_outputs(self) -> Dict[str, List[str]]:
        """
        Generate all output files (CSVs, figures, report).

        Returns:
            dict: Dictionary with 'csv', 'figures', 'report' keys containing file paths
        """
        self._report_progress("Outputs", "Initializing report generator...", 0.80)

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        generator = ReportGenerator(
            self.config,
            self.aggregated_data or {},
            self.test_results or []
        )

        # Generate CSVs
        self._report_progress("Outputs", "Creating summary CSV files...", 0.82)
        csv_files = generator.create_summary_csv()

        # Generate figures
        self._report_progress("Outputs", "Creating figures...", 0.88)
        figure_files = generator.create_figures()

        # Generate report
        self._report_progress("Outputs", "Writing statistical report...", 0.95)
        report_file = generator.write_statistical_report()

        outputs = {
            'csv': csv_files,
            'figures': figure_files,
            'report': [report_file]
        }

        total_files = len(csv_files) + len(figure_files) + 1
        self._report_progress("Complete",
                              f"Generated {total_files} output files", 1.0,
                              f"Output directory: {self.output_dir}")

        return outputs

    def run_complete_analysis(self,
                              progress_callback: Callable[[AnalysisProgress], None] = None
                              ) -> tuple:
        """
        Run the complete analysis pipeline.

        This is the main entry point for running the full analysis.
        It validates, loads data, runs statistics, and generates outputs.

        Args:
            progress_callback: Optional callback for progress updates.
                               Called with AnalysisProgress objects.

        Returns:
            tuple: (success: bool, result: dict or error_message: str)
                   On success, result contains output file paths.
                   On failure, error_message describes what went wrong.
        """
        self._progress_callback = progress_callback
        start_time = datetime.now()

        try:
            # Step 1: Validate
            self._report_progress("Validating", "Validating configuration...", 0.05)
            is_valid, errors = self.validate()

            if not is_valid:
                error_msg = "Validation failed:\n" + "\n".join(f"- {e}" for e in errors)
                logger.error(error_msg)
                return False, error_msg

            # Step 2: Load data
            self._report_progress("Loading", "Loading data...", 0.10)
            if not self.load_data():
                return False, "Failed to load data. Check data source directory and animal assignments."

            # Step 3: Run statistics
            self._report_progress("Analyzing", "Running statistical analyses...", 0.40)
            if not self.run_statistics():
                return False, "Failed to run statistical analyses."

            # Step 4: Generate outputs
            self._report_progress("Generating", "Generating output files...", 0.80)
            outputs = self.generate_outputs()

            # Calculate elapsed time
            elapsed = datetime.now() - start_time
            elapsed_str = f"{elapsed.total_seconds():.1f} seconds"

            self._report_progress("Complete",
                                  f"Analysis complete in {elapsed_str}", 1.0)

            # Build result summary
            result = {
                'outputs': outputs,
                'elapsed_time': elapsed_str,
                'n_animals': len(self.all_data) if self.all_data else 0,
                'n_tests': len(self.test_results) if self.test_results else 0,
                'output_dir': str(self.output_dir)
            }

            return True, result

        except Exception as e:
            logger.exception("Analysis failed with exception")
            return False, f"Analysis failed: {str(e)}"

        finally:
            # Clean up
            if self.aggregator:
                self.aggregator.clear_cache()

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the analysis results.

        Returns:
            dict: Summary information
        """
        summary = {
            'project_name': self.config.get('project_name', 'Unknown'),
            'n_animals_loaded': len(self.all_data) if self.all_data else 0,
            'n_statistical_tests': len(self.test_results) if self.test_results else 0,
            'significant_results': 0,
            'output_dir': str(self.output_dir)
        }

        # Count significant results
        if self.test_results:
            summary['significant_results'] = sum(
                1 for r in self.test_results
                if hasattr(r, 'p_value') and r.p_value < 0.05
            )

        # Add data info
        if self.aggregated_data:
            lupe = self.aggregated_data.get('lupe_metrics')
            amps = self.aggregated_data.get('amps_metrics')

            if lupe is not None:
                summary['n_lupe_metrics'] = lupe['metric_type'].nunique() if 'metric_type' in lupe.columns else 0
            if amps is not None:
                summary['n_amps_metrics'] = amps['metric_type'].nunique() if 'metric_type' in amps.columns else 0

        return summary


def run_analysis(config: Dict,
                 progress_callback: Callable[[AnalysisProgress], None] = None
                 ) -> tuple:
    """
    Convenience function to run complete analysis.

    Args:
        config: Project configuration dictionary
        progress_callback: Optional progress callback

    Returns:
        tuple: (success, result_or_error)
    """
    analyzer = ProjectAnalyzer(config)
    return analyzer.run_complete_analysis(progress_callback)
