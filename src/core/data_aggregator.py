"""
Data Aggregator for Project Analysis

This module loads and aggregates LUPE and AMPS analysis outputs for multiple animals,
organizing them by experimental factors (groups, conditions, sex, timepoints).

The aggregator reads CSV files produced by LUPE and AMPS analyses and combines them
into pandas DataFrames suitable for statistical analysis.

Supported folder structures:

1. Nested structure (folder_structure="nested"):
   data_source_dir/
   ├── {animal_id}/
   │   ├── {timepoint}/                          # If has_timepoints=True
   │   │   ├── {animal_id}_{timepoint}_analysis/
   │   │   │   ├── {animal_id}_{timepoint}_timeline_*.csv
   │   │   │   └── ...
   │   └── {animal_id}_analysis/                 # If has_timepoints=False
   │       └── ...

2. Flat structure (folder_structure="flat"):
   data_source_dir/
   ├── {animal_id}_{timepoint}/                  # Combined naming
   │   ├── {animal_id}_{timepoint}_analysis/
   │   │   └── ...

3. No timepoints (default):
   data_source_dir/
   ├── {animal_id}/
   │   ├── {animal_id}_analysis/
   │   │   ├── {animal_id}_timeline_*.csv
   │   │   └── ...

Expected file structure for AMPS outputs (if exists):
    amps_output_dir/
    ├── Section2_pain_scale/
    │   └── pain_scale_projection.csv
    ├── Section3_behavior_metrics/
    │   ├── {prefix}_fraction_occupancy.csv
    │   ├── {prefix}_number_of_bouts.csv
    │   └── {prefix}_bout_duration.csv
    └── Section4_model_fit/
        └── {prefix}_feature_importance.csv

Usage:
    from src.core.data_aggregator import DataAggregator

    aggregator = DataAggregator(project_config)
    all_data = aggregator.load_all_data()
    df_long = aggregator.aggregate_by_factors(all_data)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
import glob

# Set up logging for this module
logger = logging.getLogger(__name__)


class DataAggregator:
    """
    Loads and aggregates LUPE and AMPS analysis outputs for statistical analysis.

    This class handles the complexity of loading multiple CSV files per animal,
    organizing them by experimental factors, and producing long-format DataFrames
    suitable for statistical analysis.

    Attributes:
        config (dict): Project configuration containing animal assignments
        data_source_dir (Path): Directory containing LUPE output folders
        amps_output_dir (Path): Directory containing AMPS analysis outputs (optional)
        animal_assignments (dict): Mapping of groups/conditions/sex to animal IDs
        include_sex (bool): Whether sex is a factor in the experiment
        has_timepoints (bool): Whether timepoints are a factor
        separate_timepoints (bool): Whether to keep timepoints separated in analysis
    """

    def __init__(self, config: Dict):
        """
        Initialize the DataAggregator with project configuration.

        Args:
            config (dict): Project configuration dictionary containing:
                - data_source_dir: Path to LUPE output directory
                - amps_output_dir: Path to AMPS output directory (optional)
                - animal_assignments: Mapping of experimental factors to animal IDs
                - include_sex: Whether sex is included as a factor
                - has_timepoints: Whether timepoints exist
                - separate_timepoints: Whether to analyze timepoints separately
                - timepoints: List of timepoint definitions with 'name' keys
                - folder_structure: 'nested' (animal/timepoint/) or 'flat' (animal_timepoint/)
                - groups: List of group definitions
                - conditions: List of condition definitions
        """
        self.config = config
        self.data_source_dir = Path(config.get('data_source_dir', ''))
        self.amps_output_dir = Path(config.get('amps_output_dir', '')) if config.get('amps_output_dir') else None
        self.animal_assignments = config.get('animal_assignments', {})
        self.include_sex = config.get('include_sex', False)
        self.has_timepoints = config.get('has_timepoints', False)
        self.separate_timepoints = config.get('separate_timepoints', False)

        # Folder structure: 'nested' = animal/timepoint/, 'flat' = animal_timepoint/
        self.folder_structure = config.get('folder_structure', 'nested')

        # Extract group and condition names from config
        self.groups = [g.get('name', '') for g in config.get('groups', [])]
        self.conditions = [c.get('name', '') for c in config.get('conditions', [])]

        # Extract timepoint names from config
        self.timepoints = [t.get('name', '') for t in config.get('timepoints', []) if t.get('name')]

        # Cache for loaded data to avoid repeated file reads
        self._data_cache = {}

        # Cache for discovered folders (lazy-loaded)
        self._discovered_folders = None

    def _ensure_folders_discovered(self):
        """
        Ensure folders have been discovered. Lazy-loads folder discovery.

        Uses the folder_discovery module to scan the data source directory
        and find all folders containing LUPE analysis outputs.
        """
        if self._discovered_folders is None:
            from src.utils.folder_discovery import discover_lupe_folders
            self._discovered_folders = discover_lupe_folders(self.data_source_dir)
            logger.info(f"Discovered {len(self._discovered_folders)} folders with LUPE data")

    def validate_data_sources(self) -> Tuple[bool, List[str]]:
        """
        Validate that all expected data files exist.

        Uses folder discovery to find folders containing LUPE data,
        matching animal IDs and timepoints by substring within folder names.

        Returns:
            tuple: (is_valid, list of error messages)
        """
        errors = []

        # Check data source directory exists
        if not self.data_source_dir or not self.data_source_dir.exists():
            errors.append(f"Data source directory not found: {self.data_source_dir}")
            return False, errors

        # Discover folders with LUPE data
        self._ensure_folders_discovered()

        if not self._discovered_folders:
            errors.append(f"No folders with LUPE data found in: {self.data_source_dir}")
            return False, errors

        # Get all animal IDs from assignments
        all_animals = self._get_all_animal_ids()

        if not all_animals:
            errors.append("No animal IDs found in assignments")
            return False, errors

        # Determine timepoints to check
        timepoints_to_check = self.timepoints if self.has_timepoints and self.timepoints else [None]

        # Check each animal has a matching folder
        for animal_id in all_animals:
            for timepoint in timepoints_to_check:
                # Try to find a folder for this animal/timepoint
                analysis_dir = self._get_analysis_dir(animal_id, timepoint)

                if analysis_dir is None:
                    tp_str = f" at timepoint '{timepoint}'" if timepoint else ""
                    errors.append(f"Could not find folder containing '{animal_id}'{tp_str}")

        return len(errors) == 0, errors

    def _get_analysis_dir(self, animal_id: str, timepoint: Optional[str] = None) -> Optional[Path]:
        """
        Get the analysis directory path for an animal, optionally at a specific timepoint.

        Uses folder discovery with substring matching to find folders where the animal_id
        (and optionally timepoint) appear within the folder name. This supports complex
        naming conventions like "RASO_2270_1438_20240204_hab_-1d".

        Args:
            animal_id: The animal identifier (will be searched as substring)
            timepoint: The timepoint name (optional, also searched as substring)

        Returns:
            Path to analysis directory, or None if not found
        """
        # Ensure folders have been discovered
        self._ensure_folders_discovered()

        # Use folder discovery to find matching folder
        from src.utils.folder_discovery import find_folder_for_animal

        folder_info = find_folder_for_animal(
            self._discovered_folders,
            animal_id,
            timepoint
        )

        if folder_info:
            return folder_info.get('analysis_dir')

        return None

    def _get_file_prefix(self, animal_id: str, timepoint: Optional[str] = None) -> str:
        """
        Get the file prefix for LUPE output files.

        Tries to determine the actual file prefix used in the folder by examining
        existing files. Falls back to folder name or animal_id if needed.

        Args:
            animal_id: The animal identifier
            timepoint: The timepoint name (optional)

        Returns:
            String prefix for files
        """
        # Ensure folders have been discovered
        self._ensure_folders_discovered()

        # Find the folder for this animal/timepoint
        from src.utils.folder_discovery import find_folder_for_animal, get_file_prefix_from_folder

        folder_info = find_folder_for_animal(
            self._discovered_folders,
            animal_id,
            timepoint
        )

        if folder_info:
            # Try to extract prefix from existing files in the folder
            prefix = get_file_prefix_from_folder(folder_info)
            if prefix:
                return prefix

            # Fall back to folder name (often the prefix matches the folder name)
            return folder_info['folder_name']

        # Fall back to simple construction
        if self.has_timepoints and timepoint:
            return f"{animal_id}_{timepoint}"
        return animal_id

    def _get_all_animal_ids(self) -> List[str]:
        """
        Extract all unique animal IDs from the assignments structure.

        Returns:
            list: List of unique animal IDs
        """
        all_ids = set()

        def extract_ids(obj):
            """Recursively extract animal IDs from nested dict/list structure."""
            if isinstance(obj, list):
                # This is a list of animal IDs
                for item in obj:
                    if isinstance(item, str):
                        all_ids.add(item.strip())
            elif isinstance(obj, dict):
                for value in obj.values():
                    extract_ids(value)

        extract_ids(self.animal_assignments)
        return list(all_ids)

    def _get_animal_factors(self, animal_id: str) -> Dict[str, str]:
        """
        Get the experimental factors (group, condition, sex) for an animal.

        Args:
            animal_id: The animal identifier

        Returns:
            dict: Dictionary with keys 'group', 'condition', 'sex' (if applicable)
        """
        factors = {'group': 'all', 'condition': 'all', 'sex': 'all'}

        def search_assignments(obj, path=None):
            """Recursively search for animal_id in assignments structure."""
            if path is None:
                path = []

            if isinstance(obj, list):
                if animal_id in [a.strip() for a in obj if isinstance(a, str)]:
                    return path
            elif isinstance(obj, dict):
                for key, value in obj.items():
                    result = search_assignments(value, path + [key])
                    if result is not None:
                        return result
            return None

        path = search_assignments(self.animal_assignments)

        if path:
            # Path structure depends on config:
            # With groups+conditions+sex: [group, condition, sex]
            # With groups+conditions: [group, condition]
            # With groups only: [group]
            # With 'all' only: ['all']
            if path[0] != 'all':
                if len(path) >= 1 and path[0] in self.groups:
                    factors['group'] = path[0]
                if len(path) >= 2 and path[1] in self.conditions:
                    factors['condition'] = path[1]
                if len(path) >= 3 and path[2] in ['Male', 'Female']:
                    factors['sex'] = path[2]
                elif len(path) >= 2 and self.include_sex and path[1] in ['Male', 'Female']:
                    factors['sex'] = path[1]

        return factors

    def load_lupe_data(self, animal_id: str, timepoint: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Load all LUPE analysis outputs for a single animal at a specific timepoint.

        Args:
            animal_id: The animal identifier (folder name)
            timepoint: The timepoint name (optional, required if has_timepoints=True)

        Returns:
            dict: Dictionary containing:
                - timeline: DataFrame with binned behavior proportions
                - bout_counts: DataFrame with bout counts per behavior
                - time_distribution: DataFrame with time percentages per behavior
                - bout_durations: DataFrame with duration statistics per behavior
                - transitions: DataFrame with transition probability matrix
            Returns None if data cannot be loaded.
        """
        cache_key = f"lupe_{animal_id}_{timepoint}" if timepoint else f"lupe_{animal_id}"
        if cache_key in self._data_cache:
            return self._data_cache[cache_key]

        # Get the analysis directory using folder structure logic
        analysis_dir = self._get_analysis_dir(animal_id, timepoint)

        if analysis_dir is None:
            tp_str = f" at timepoint {timepoint}" if timepoint else ""
            logger.warning(f"Analysis directory not found for {animal_id}{tp_str}")
            return None

        # Determine file prefix
        file_prefix = self._get_file_prefix(animal_id, timepoint)

        data = {}

        try:
            # Load timeline (find the file with pattern)
            # Try with full prefix first, then with just animal_id
            timeline_files = list(analysis_dir.glob(f"{file_prefix}_timeline_*.csv"))
            if not timeline_files:
                timeline_files = list(analysis_dir.glob(f"{animal_id}_timeline_*.csv"))
            if not timeline_files:
                timeline_files = list(analysis_dir.glob("*_timeline_*.csv"))

            if timeline_files:
                data['timeline'] = pd.read_csv(timeline_files[0])
            else:
                logger.warning(f"Timeline file not found for {animal_id}")
                data['timeline'] = None

            # Load bout counts
            data['bout_counts'] = self._load_csv_with_fallback(
                analysis_dir, 'bout_counts_summary', file_prefix, animal_id
            )

            # Load time distribution
            data['time_distribution'] = self._load_csv_with_fallback(
                analysis_dir, 'time_distribution_overall', file_prefix, animal_id
            )

            # Load bout durations
            data['bout_durations'] = self._load_csv_with_fallback(
                analysis_dir, 'bout_durations_statistics', file_prefix, animal_id
            )

            # Load transitions matrix
            trans_df = self._load_csv_with_fallback(
                analysis_dir, 'transitions_matrix', file_prefix, animal_id,
                index_col=0
            )
            data['transitions'] = trans_df

            self._data_cache[cache_key] = data
            return data

        except Exception as e:
            logger.error(f"Error loading LUPE data for {animal_id}: {str(e)}")
            return None

    def _load_csv_with_fallback(
        self,
        analysis_dir: Path,
        file_suffix: str,
        file_prefix: str,
        animal_id: str,
        **read_csv_kwargs
    ) -> Optional[pd.DataFrame]:
        """
        Load a CSV file, trying multiple filename patterns.

        Tries in order:
        1. {file_prefix}_{suffix}.csv
        2. {animal_id}_{suffix}.csv
        3. *_{suffix}.csv (any matching file)

        Args:
            analysis_dir: Directory containing analysis files
            file_suffix: File suffix without leading underscore
            file_prefix: Primary file prefix to try
            animal_id: Animal ID for fallback
            **read_csv_kwargs: Additional kwargs for pd.read_csv

        Returns:
            DataFrame if found, None otherwise
        """
        # Try primary pattern
        file_path = analysis_dir / f"{file_prefix}_{file_suffix}.csv"
        if file_path.exists():
            return pd.read_csv(file_path, **read_csv_kwargs)

        # Try animal_id pattern
        file_path = analysis_dir / f"{animal_id}_{file_suffix}.csv"
        if file_path.exists():
            return pd.read_csv(file_path, **read_csv_kwargs)

        # Try any matching file
        matches = list(analysis_dir.glob(f"*_{file_suffix}.csv"))
        if matches:
            return pd.read_csv(matches[0], **read_csv_kwargs)

        logger.warning(f"File not found: {file_prefix}_{file_suffix}.csv in {analysis_dir}")
        return None

    def load_amps_data(self, animal_id: str) -> Optional[Dict[str, Any]]:
        """
        Load AMPS analysis outputs for a single animal.

        AMPS outputs are typically in a shared analysis folder, with each file
        containing rows for multiple animals identified by filename.

        Args:
            animal_id: The animal identifier

        Returns:
            dict: Dictionary containing:
                - pain_scale: Series with PC1 and PC2 values
                - fraction_occupancy: Series with state fractions
                - number_of_bouts: Series with bout counts per state
                - bout_duration: Series with mean duration per state
                - feature_importance: Series with feature importance scores
            Returns None if AMPS data is not available.
        """
        if not self.amps_output_dir or not self.amps_output_dir.exists():
            return None

        cache_key = f"amps_{animal_id}"
        if cache_key in self._data_cache:
            return self._data_cache[cache_key]

        data = {}

        try:
            # Construct expected filename pattern for AMPS CSVs
            # AMPS files use the behaviors.csv filename as identifier
            behavior_filename = f"{animal_id}_behaviors.csv"

            # Load pain scale projection
            pain_scale_file = self.amps_output_dir / "Section2_pain_scale" / "pain_scale_projection.csv"
            if pain_scale_file.exists():
                df = pd.read_csv(pain_scale_file)
                row = df[df['filename'] == behavior_filename]
                if not row.empty:
                    data['pain_scale'] = row.iloc[0]
                else:
                    # Try partial match
                    row = df[df['filename'].str.contains(animal_id, na=False)]
                    if not row.empty:
                        data['pain_scale'] = row.iloc[0]
            else:
                data['pain_scale'] = None

            # Load behavior metrics from Section3
            section3_dir = self.amps_output_dir / "Section3_behavior_metrics"
            if section3_dir.exists():
                # Find the metrics files (they have a prefix)
                for metric_type in ['fraction_occupancy', 'number_of_bouts', 'bout_duration']:
                    metric_files = list(section3_dir.glob(f"*_{metric_type}.csv"))
                    if metric_files:
                        df = pd.read_csv(metric_files[0])
                        row = df[df['filename'] == behavior_filename]
                        if row.empty:
                            row = df[df['filename'].str.contains(animal_id, na=False)]
                        if not row.empty:
                            data[metric_type] = row.iloc[0]
                        else:
                            data[metric_type] = None
                    else:
                        data[metric_type] = None
            else:
                data['fraction_occupancy'] = None
                data['number_of_bouts'] = None
                data['bout_duration'] = None

            # Load feature importance from Section4
            section4_dir = self.amps_output_dir / "Section4_model_fit"
            if section4_dir.exists():
                feature_files = list(section4_dir.glob("*_feature_importance.csv"))
                if feature_files:
                    df = pd.read_csv(feature_files[0])
                    row = df[df['filename'] == behavior_filename]
                    if row.empty:
                        row = df[df['filename'].str.contains(animal_id, na=False)]
                    if not row.empty:
                        data['feature_importance'] = row.iloc[0]
                    else:
                        data['feature_importance'] = None
                else:
                    data['feature_importance'] = None
            else:
                data['feature_importance'] = None

            self._data_cache[cache_key] = data
            return data

        except Exception as e:
            logger.error(f"Error loading AMPS data for {animal_id}: {str(e)}")
            return None

    def load_all_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Load all LUPE and AMPS data for all animals in the project.

        If has_timepoints is True, loads data for each animal at each timepoint.
        The data key format is 'animal_id' (no timepoints) or 'animal_id_timepoint'.

        Returns:
            dict: Dictionary keyed by animal_id (or animal_id_timepoint) containing:
                - factors: Experimental factors (group, condition, sex, timepoint)
                - lupe: LUPE analysis data
                - amps: AMPS analysis data (if available)
        """
        all_data = {}
        all_animals = self._get_all_animal_ids()

        # Determine timepoints to load
        timepoints_to_load = self.timepoints if self.has_timepoints and self.timepoints else [None]

        total_combinations = len(all_animals) * len(timepoints_to_load)
        logger.info(f"Loading data for {len(all_animals)} animals across {len(timepoints_to_load)} timepoint(s)...")

        for animal_id in all_animals:
            factors = self._get_animal_factors(animal_id)

            for timepoint in timepoints_to_load:
                # Create data key - include timepoint if applicable
                data_key = f"{animal_id}_{timepoint}" if timepoint else animal_id

                # Load LUPE data
                lupe_data = self.load_lupe_data(animal_id, timepoint)

                # Load AMPS data (currently not timepoint-aware)
                amps_data = self.load_amps_data(animal_id)

                if lupe_data is not None:
                    # Copy factors and add timepoint
                    entry_factors = factors.copy()
                    if timepoint:
                        entry_factors['timepoint'] = timepoint

                    all_data[data_key] = {
                        'animal_id': animal_id,
                        'timepoint': timepoint,
                        'factors': entry_factors,
                        'lupe': lupe_data,
                        'amps': amps_data
                    }
                else:
                    tp_str = f" at {timepoint}" if timepoint else ""
                    logger.warning(f"Skipping {animal_id}{tp_str}: no LUPE data found")

        logger.info(f"Successfully loaded data for {len(all_data)} animal-timepoint combinations")
        return all_data

    def aggregate_lupe_metrics(self, all_data: Dict) -> pd.DataFrame:
        """
        Aggregate LUPE metrics into a long-format DataFrame.

        Creates a DataFrame with columns for experimental factors and all
        LUPE metrics, suitable for statistical analysis.

        Args:
            all_data: Output from load_all_data()

        Returns:
            pd.DataFrame: Long-format DataFrame with columns:
                - animal_id, group, condition, sex, timepoint
                - Behavior-specific columns for each metric type
        """
        rows = []

        for data_key, data in all_data.items():
            factors = data['factors']
            lupe = data.get('lupe', {})
            # Get the actual animal_id (not the data_key which may include timepoint)
            animal_id = data.get('animal_id', data_key)
            timepoint = data.get('timepoint')

            if not lupe:
                continue

            base_row = {
                'animal_id': animal_id,
                'group': factors.get('group', 'all'),
                'condition': factors.get('condition', 'all'),
                'sex': factors.get('sex', 'all'),
                'timepoint': timepoint if timepoint else 'all'
            }

            # Extract bout counts per behavior
            if lupe.get('bout_counts') is not None:
                for _, row in lupe['bout_counts'].iterrows():
                    behavior = row.get('behavior', 'unknown')
                    rows.append({
                        **base_row,
                        'metric_type': 'bout_count',
                        'behavior': behavior,
                        'value': row.get('bout_count', np.nan)
                    })

            # Extract time distribution per behavior
            if lupe.get('time_distribution') is not None:
                for _, row in lupe['time_distribution'].iterrows():
                    behavior = row.get('behavior', 'unknown')
                    rows.append({
                        **base_row,
                        'metric_type': 'time_percentage',
                        'behavior': behavior,
                        'value': row.get('percentage', np.nan)
                    })

            # Extract bout duration statistics per behavior
            if lupe.get('bout_durations') is not None:
                for _, row in lupe['bout_durations'].iterrows():
                    behavior = row.get('behavior', 'unknown')
                    for stat in ['mean_duration_sec', 'median_duration_sec', 'std_duration_sec']:
                        if stat in row:
                            rows.append({
                                **base_row,
                                'metric_type': f'bout_{stat}',
                                'behavior': behavior,
                                'value': row[stat]
                            })

        return pd.DataFrame(rows)

    def aggregate_lupe_timeline(self, all_data: Dict) -> pd.DataFrame:
        """
        Aggregate LUPE binned timeline data.

        Creates a DataFrame with time bin information for each animal.

        Args:
            all_data: Output from load_all_data()

        Returns:
            pd.DataFrame: DataFrame with columns:
                - animal_id, group, condition, sex, timepoint, time_bin
                - Proportion columns for each behavior
        """
        rows = []

        for data_key, data in all_data.items():
            factors = data['factors']
            lupe = data.get('lupe', {})
            animal_id = data.get('animal_id', data_key)
            timepoint = data.get('timepoint')

            if not lupe or lupe.get('timeline') is None:
                continue

            timeline = lupe['timeline']

            for _, row in timeline.iterrows():
                row_dict = {
                    'animal_id': animal_id,
                    'group': factors.get('group', 'all'),
                    'condition': factors.get('condition', 'all'),
                    'sex': factors.get('sex', 'all'),
                    'timepoint': timepoint if timepoint else 'all',
                    'time_bin': row.get('time_bin', np.nan)
                }

                # Add proportion columns for each behavior
                for col in timeline.columns:
                    if col.endswith('_proportion'):
                        behavior = col.replace('_proportion', '')
                        row_dict[f'{behavior}_proportion'] = row[col]

                rows.append(row_dict)

        return pd.DataFrame(rows)

    def aggregate_lupe_transitions(self, all_data: Dict) -> Dict[str, pd.DataFrame]:
        """
        Aggregate transition matrices by experimental factors.

        Args:
            all_data: Output from load_all_data()

        Returns:
            dict: Dictionary with factor combinations as keys and averaged
                  transition matrices as values. Keys include timepoint if applicable.
        """
        # Group animals by factors
        factor_groups = {}

        for data_key, data in all_data.items():
            factors = data['factors']
            lupe = data.get('lupe', {})
            timepoint = data.get('timepoint')

            if not lupe or lupe.get('transitions') is None:
                continue

            # Include timepoint in the key if applicable
            key = (
                factors.get('group', 'all'),
                factors.get('condition', 'all'),
                factors.get('sex', 'all'),
                timepoint if timepoint else 'all'
            )

            if key not in factor_groups:
                factor_groups[key] = []
            factor_groups[key].append(lupe['transitions'])

        # Average transitions within each group
        averaged_transitions = {}
        for key, matrices in factor_groups.items():
            if matrices:
                # Stack and average
                stacked = np.stack([m.values for m in matrices])
                averaged = np.nanmean(stacked, axis=0)
                averaged_df = pd.DataFrame(
                    averaged,
                    index=matrices[0].index,
                    columns=matrices[0].columns
                )
                averaged_transitions[key] = averaged_df

        return averaged_transitions

    def aggregate_amps_metrics(self, all_data: Dict) -> pd.DataFrame:
        """
        Aggregate AMPS metrics into a long-format DataFrame.

        Args:
            all_data: Output from load_all_data()

        Returns:
            pd.DataFrame: Long-format DataFrame with AMPS metrics including timepoint
        """
        rows = []

        for data_key, data in all_data.items():
            factors = data['factors']
            amps = data.get('amps')
            animal_id = data.get('animal_id', data_key)
            timepoint = data.get('timepoint')

            if not amps:
                continue

            base_row = {
                'animal_id': animal_id,
                'group': factors.get('group', 'all'),
                'condition': factors.get('condition', 'all'),
                'sex': factors.get('sex', 'all'),
                'timepoint': timepoint if timepoint else 'all'
            }

            # Pain scale projection
            if amps.get('pain_scale') is not None:
                ps = amps['pain_scale']
                if 'PC1_Behavior_Scale' in ps.index:
                    rows.append({
                        **base_row,
                        'metric_type': 'pain_scale',
                        'metric_name': 'PC1_Behavior_Scale',
                        'value': ps['PC1_Behavior_Scale']
                    })
                if 'PC2_Pain_Scale' in ps.index:
                    rows.append({
                        **base_row,
                        'metric_type': 'pain_scale',
                        'metric_name': 'PC2_Pain_Scale',
                        'value': ps['PC2_Pain_Scale']
                    })

            # State-based metrics
            for metric_type in ['fraction_occupancy', 'number_of_bouts', 'bout_duration']:
                if amps.get(metric_type) is not None:
                    metric_data = amps[metric_type]
                    for col in metric_data.index:
                        if col.startswith('State '):
                            rows.append({
                                **base_row,
                                'metric_type': f'amps_{metric_type}',
                                'metric_name': col,
                                'value': metric_data[col]
                            })

            # Feature importance
            if amps.get('feature_importance') is not None:
                fi = amps['feature_importance']
                for col in fi.index:
                    if col != 'filename':
                        rows.append({
                            **base_row,
                            'metric_type': 'feature_importance',
                            'metric_name': col,
                            'value': fi[col]
                        })

        return pd.DataFrame(rows)

    def aggregate_by_factors(self, all_data: Dict) -> Dict[str, pd.DataFrame]:
        """
        Aggregate all data by experimental factors into analysis-ready DataFrames.

        This is the main aggregation method that produces all DataFrames needed
        for statistical analysis.

        Args:
            all_data: Output from load_all_data()

        Returns:
            dict: Dictionary containing:
                - 'lupe_metrics': Long-format LUPE metrics DataFrame
                - 'lupe_timeline': Binned timeline DataFrame
                - 'lupe_transitions': Dict of averaged transition matrices
                - 'amps_metrics': Long-format AMPS metrics DataFrame
        """
        logger.info("Aggregating data by experimental factors...")

        result = {
            'lupe_metrics': self.aggregate_lupe_metrics(all_data),
            'lupe_timeline': self.aggregate_lupe_timeline(all_data),
            'lupe_transitions': self.aggregate_lupe_transitions(all_data),
            'amps_metrics': self.aggregate_amps_metrics(all_data)
        }

        # Log summary statistics
        logger.info(f"  LUPE metrics: {len(result['lupe_metrics'])} rows")
        logger.info(f"  LUPE timeline: {len(result['lupe_timeline'])} rows")
        logger.info(f"  LUPE transitions: {len(result['lupe_transitions'])} groups")
        logger.info(f"  AMPS metrics: {len(result['amps_metrics'])} rows")

        return result

    def get_summary_statistics(self, aggregated_data: Dict) -> pd.DataFrame:
        """
        Calculate summary statistics for all metrics by factor combinations.

        Args:
            aggregated_data: Output from aggregate_by_factors()

        Returns:
            pd.DataFrame: Summary statistics with columns:
                - group, condition, sex, timepoint, metric_type, behavior/metric_name
                - n, mean, sem, std, median, min, max
        """
        summary_rows = []

        # Process LUPE metrics
        lupe_metrics = aggregated_data.get('lupe_metrics')
        if lupe_metrics is not None and not lupe_metrics.empty:
            groupby_cols = ['group', 'condition', 'sex', 'timepoint', 'metric_type', 'behavior']
            existing_cols = [c for c in groupby_cols if c in lupe_metrics.columns]

            if existing_cols and 'value' in lupe_metrics.columns:
                grouped = lupe_metrics.groupby(existing_cols)['value']
                stats = grouped.agg(['count', 'mean', 'std', 'median', 'min', 'max'])
                stats['sem'] = grouped.sem()
                stats = stats.reset_index()
                stats['data_source'] = 'lupe'
                summary_rows.append(stats)

        # Process AMPS metrics
        amps_metrics = aggregated_data.get('amps_metrics')
        if amps_metrics is not None and not amps_metrics.empty:
            groupby_cols = ['group', 'condition', 'sex', 'timepoint', 'metric_type', 'metric_name']
            existing_cols = [c for c in groupby_cols if c in amps_metrics.columns]

            if existing_cols and 'value' in amps_metrics.columns:
                grouped = amps_metrics.groupby(existing_cols)['value']
                stats = grouped.agg(['count', 'mean', 'std', 'median', 'min', 'max'])
                stats['sem'] = grouped.sem()
                stats = stats.reset_index()
                stats['data_source'] = 'amps'
                summary_rows.append(stats)

        if summary_rows:
            return pd.concat(summary_rows, ignore_index=True)
        else:
            return pd.DataFrame()

    def clear_cache(self):
        """Clear the data cache to free memory."""
        self._data_cache.clear()
        logger.info("Data cache cleared")
