"""
Metadata Loader for Excel Files

This module provides functionality to load experiment metadata from Excel files.
The metadata file contains information about animals and their experimental assignments.

Expected Excel columns:
- animal_id (required): Unique identifier for each animal
- group (optional): Treatment group assignment
- condition (optional): Experimental condition
- sex (optional): Male or Female
- timepoint (optional): Timepoint name (e.g., Day0, Day7)

The loader validates the metadata against the project configuration and converts
it to the animal_assignments structure used by the analysis pipeline.

Usage:
    from src.utils.metadata_loader import MetadataLoader

    loader = MetadataLoader(config)
    success, result = loader.load_excel(file_path)
    if success:
        assignments = result['animal_assignments']
    else:
        errors = result  # List of error messages
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class MetadataLoader:
    """
    Loads and validates experiment metadata from Excel files.

    Attributes:
        config (dict): Project configuration for validation
        groups (list): Expected group names from config
        conditions (list): Expected condition names from config
        timepoints (list): Expected timepoint names from config
        include_sex (bool): Whether sex is a factor
    """

    # Possible column name variations (case-insensitive matching)
    ANIMAL_ID_COLUMNS = ['animal_id', 'animalid', 'animal', 'id', 'subject', 'subject_id']
    GROUP_COLUMNS = ['group', 'treatment', 'treatment_group', 'grp']
    CONDITION_COLUMNS = ['condition', 'cond', 'day', 'timepoint_condition']
    SEX_COLUMNS = ['sex', 'gender']
    TIMEPOINT_COLUMNS = ['timepoint', 'time_point', 'time', 'date', 'day', 'tp']

    def __init__(self, config: Dict):
        """
        Initialize the MetadataLoader with project configuration.

        Args:
            config: Project configuration dictionary
        """
        self.config = config
        self.groups = [g.get('name', '') for g in config.get('groups', [])]
        self.conditions = [c.get('name', '') for c in config.get('conditions', [])]
        self.timepoints = [t.get('name', '') for t in config.get('timepoints', [])]
        self.include_sex = config.get('include_sex', False)
        self.has_timepoints = config.get('has_timepoints', False)

    def load_excel(self, file_path: str) -> Tuple[bool, Any]:
        """
        Load and validate metadata from an Excel file.

        Args:
            file_path: Path to the Excel file

        Returns:
            tuple: (success, result)
                - On success: (True, dict with 'animal_assignments' and 'metadata_df')
                - On failure: (False, list of error messages)
        """
        errors = []

        # Check file exists
        path = Path(file_path)
        if not path.exists():
            return False, [f"File not found: {file_path}"]

        # Load Excel file
        try:
            df = pd.read_excel(file_path)
        except Exception as e:
            return False, [f"Failed to read Excel file: {str(e)}"]

        if df.empty:
            return False, ["Excel file is empty"]

        # Normalize column names (lowercase, strip whitespace)
        df.columns = df.columns.str.strip().str.lower()

        # Find required columns
        animal_col = self._find_column(df.columns, self.ANIMAL_ID_COLUMNS)
        if not animal_col:
            return False, ["Could not find animal ID column. Expected columns like: " +
                          ", ".join(self.ANIMAL_ID_COLUMNS)]

        # Find optional columns
        group_col = self._find_column(df.columns, self.GROUP_COLUMNS)
        condition_col = self._find_column(df.columns, self.CONDITION_COLUMNS)
        sex_col = self._find_column(df.columns, self.SEX_COLUMNS)
        timepoint_col = self._find_column(df.columns, self.TIMEPOINT_COLUMNS)

        # Validate data
        validation_errors = self._validate_data(
            df, animal_col, group_col, condition_col, sex_col, timepoint_col
        )
        if validation_errors:
            errors.extend(validation_errors)

        # Check for discrepancies with config
        discrepancy_warnings = self._check_discrepancies(
            df, group_col, condition_col, timepoint_col
        )

        if errors:
            return False, errors

        # Build animal assignments structure
        assignments = self._build_assignments(
            df, animal_col, group_col, condition_col, sex_col, timepoint_col
        )

        # Get unique animals list
        unique_animals = df[animal_col].dropna().unique().tolist()

        result = {
            'animal_assignments': assignments,
            'metadata_df': df,
            'unique_animals': unique_animals,
            'warnings': discrepancy_warnings,
            'columns_found': {
                'animal_id': animal_col,
                'group': group_col,
                'condition': condition_col,
                'sex': sex_col,
                'timepoint': timepoint_col
            }
        }

        return True, result

    def _find_column(self, columns: pd.Index, possible_names: List[str]) -> Optional[str]:
        """
        Find a column by checking possible name variations.

        Args:
            columns: DataFrame column index
            possible_names: List of possible column names

        Returns:
            str: Found column name, or None if not found
        """
        columns_lower = [c.lower() for c in columns]

        for name in possible_names:
            if name.lower() in columns_lower:
                # Return the original column name
                idx = columns_lower.index(name.lower())
                return columns[idx]

        return None

    def _validate_data(self, df: pd.DataFrame, animal_col: str,
                       group_col: Optional[str], condition_col: Optional[str],
                       sex_col: Optional[str], timepoint_col: Optional[str]) -> List[str]:
        """
        Validate the data in the DataFrame.

        Args:
            df: DataFrame with metadata
            *_col: Column names

        Returns:
            list: List of validation error messages
        """
        errors = []

        # Check for missing animal IDs
        missing_ids = df[animal_col].isna().sum()
        if missing_ids > 0:
            errors.append(f"{missing_ids} rows have missing animal IDs")

        # Check for duplicate animal IDs (unless timepoints exist)
        if not timepoint_col:
            duplicates = df[animal_col].dropna().duplicated()
            if duplicates.any():
                dup_ids = df.loc[duplicates, animal_col].unique()
                errors.append(f"Duplicate animal IDs found: {', '.join(str(x) for x in dup_ids[:5])}" +
                            ("..." if len(dup_ids) > 5 else ""))

        # Check sex values if column exists and sex is a factor
        if sex_col and self.include_sex:
            valid_sex = ['male', 'female', 'm', 'f']
            sex_values = df[sex_col].dropna().str.lower().unique()
            invalid_sex = [s for s in sex_values if s not in valid_sex]
            if invalid_sex:
                errors.append(f"Invalid sex values: {', '.join(invalid_sex)}. "
                            f"Expected: Male/Female or M/F")

        return errors

    def _check_discrepancies(self, df: pd.DataFrame,
                            group_col: Optional[str],
                            condition_col: Optional[str],
                            timepoint_col: Optional[str]) -> List[str]:
        """
        Check for discrepancies between Excel data and project configuration.

        Args:
            df: DataFrame with metadata
            *_col: Column names

        Returns:
            list: List of warning messages
        """
        warnings = []

        # Check groups
        if group_col and self.groups:
            excel_groups = set(df[group_col].dropna().unique())
            config_groups = set(self.groups)

            extra_in_excel = excel_groups - config_groups
            missing_in_excel = config_groups - excel_groups

            if extra_in_excel:
                warnings.append(f"Groups in Excel not in config: {', '.join(str(x) for x in extra_in_excel)}")
            if missing_in_excel:
                warnings.append(f"Groups in config not in Excel: {', '.join(missing_in_excel)}")

        # Check conditions
        if condition_col and self.conditions:
            excel_conditions = set(df[condition_col].dropna().unique())
            config_conditions = set(self.conditions)

            extra_in_excel = excel_conditions - config_conditions
            missing_in_excel = config_conditions - excel_conditions

            if extra_in_excel:
                warnings.append(f"Conditions in Excel not in config: {', '.join(str(x) for x in extra_in_excel)}")
            if missing_in_excel:
                warnings.append(f"Conditions in config not in Excel: {', '.join(missing_in_excel)}")

        # Check timepoints
        if timepoint_col and self.timepoints:
            excel_timepoints = set(df[timepoint_col].dropna().unique())
            config_timepoints = set(self.timepoints)

            extra_in_excel = excel_timepoints - config_timepoints
            missing_in_excel = config_timepoints - excel_timepoints

            if extra_in_excel:
                warnings.append(f"Timepoints in Excel not in config: {', '.join(str(x) for x in extra_in_excel)}")
            if missing_in_excel:
                warnings.append(f"Timepoints in config not in Excel: {', '.join(missing_in_excel)}")

        return warnings

    def _build_assignments(self, df: pd.DataFrame, animal_col: str,
                          group_col: Optional[str], condition_col: Optional[str],
                          sex_col: Optional[str], timepoint_col: Optional[str]) -> Dict:
        """
        Build the animal_assignments structure from DataFrame.

        The structure is hierarchical based on available factors:
        - With groups, conditions, sex: {group: {condition: {sex: [animals]}}}
        - With groups, conditions: {group: {condition: [animals]}}
        - With groups only: {group: [animals]}
        - No factors: {'all': [animals]}

        Args:
            df: DataFrame with metadata
            *_col: Column names

        Returns:
            dict: Animal assignments structure
        """
        # Normalize sex values
        if sex_col:
            df = df.copy()
            df[sex_col] = df[sex_col].str.lower().map(
                lambda x: 'Male' if x in ['male', 'm'] else ('Female' if x in ['female', 'f'] else x)
            )

        # Determine nesting structure based on available columns
        has_groups = group_col is not None and len(df[group_col].dropna().unique()) > 0
        has_conditions = condition_col is not None and len(df[condition_col].dropna().unique()) > 0
        has_sex = sex_col is not None and self.include_sex
        has_tp = timepoint_col is not None and self.has_timepoints

        assignments = {}

        if not has_groups and not has_conditions:
            # Simple case: all animals in one list (possibly split by sex)
            if has_sex:
                assignments['all'] = {}
                for sex in ['Male', 'Female']:
                    mask = df[sex_col] == sex
                    animals = df.loc[mask, animal_col].dropna().unique().tolist()
                    if animals:
                        # Convert to strings
                        assignments['all'][sex] = [str(a) for a in animals]
            else:
                assignments['all'] = [str(a) for a in df[animal_col].dropna().unique().tolist()]
        else:
            # Complex case: nested structure
            for _, row in df.iterrows():
                animal_id = str(row[animal_col]) if pd.notna(row[animal_col]) else None
                if not animal_id:
                    continue

                group = str(row[group_col]) if group_col and pd.notna(row[group_col]) else 'all'
                condition = str(row[condition_col]) if condition_col and pd.notna(row[condition_col]) else 'all'
                sex = row[sex_col] if sex_col and pd.notna(row[sex_col]) else 'all'
                timepoint = str(row[timepoint_col]) if timepoint_col and pd.notna(row[timepoint_col]) else None

                # Build nested structure
                if group not in assignments:
                    assignments[group] = {}

                if has_conditions:
                    if condition not in assignments[group]:
                        assignments[group][condition] = {} if has_sex else []

                    if has_sex:
                        if sex not in assignments[group][condition]:
                            assignments[group][condition][sex] = []
                        if animal_id not in assignments[group][condition][sex]:
                            assignments[group][condition][sex].append(animal_id)
                    else:
                        if animal_id not in assignments[group][condition]:
                            assignments[group][condition].append(animal_id)
                else:
                    if has_sex:
                        if not isinstance(assignments[group], dict):
                            assignments[group] = {}
                        if sex not in assignments[group]:
                            assignments[group][sex] = []
                        if animal_id not in assignments[group][sex]:
                            assignments[group][sex].append(animal_id)
                    else:
                        if not isinstance(assignments[group], list):
                            assignments[group] = []
                        if animal_id not in assignments[group]:
                            assignments[group].append(animal_id)

        return assignments

    def get_summary(self, file_path: str) -> Optional[Dict]:
        """
        Get a summary of the metadata file without full processing.

        Args:
            file_path: Path to the Excel file

        Returns:
            dict: Summary information, or None if file cannot be read
        """
        try:
            df = pd.read_excel(file_path)
            df.columns = df.columns.str.strip().str.lower()

            animal_col = self._find_column(df.columns, self.ANIMAL_ID_COLUMNS)

            return {
                'n_rows': len(df),
                'n_animals': df[animal_col].nunique() if animal_col else 0,
                'columns': list(df.columns),
                'animal_column_found': animal_col is not None
            }
        except Exception:
            return None


def load_metadata(file_path: str, config: Dict) -> Tuple[bool, Any]:
    """
    Convenience function to load metadata from Excel.

    Args:
        file_path: Path to the Excel file
        config: Project configuration

    Returns:
        tuple: (success, result_or_errors)
    """
    loader = MetadataLoader(config)
    return loader.load_excel(file_path)
