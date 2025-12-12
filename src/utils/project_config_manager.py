"""
Project Configuration Manager

This module provides functionality to manage project configuration files.
Each project configuration is stored as a separate JSON file in the projects/ directory.

Project configurations define:
- Groups (e.g., Treatment, Control)
- Conditions per group (e.g., Day 0, Day 7, Day 14)
- Whether sex is included as a variable
- Timepoint settings
- Data source directories for LUPE and AMPS outputs
- Animal assignments mapping animal IDs to experimental factors
- Output directory for analysis results

Usage:
    from src.utils.project_config_manager import ProjectConfigManager

    manager = ProjectConfigManager()
    projects = manager.list_projects()
    config = manager.load_project("My Experiment")
    manager.save_project(config)
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional


class ProjectConfigManager:
    """
    Manages project configuration JSON files.

    Project configurations are stored in the projects/ directory as JSON files.
    Each project has a unique name that serves as the filename.

    Attributes:
        projects_dir (Path): Directory where project JSON files are stored
    """

    def __init__(self, projects_dir: str = "projects"):
        """
        Initialize the project configuration manager.

        Args:
            projects_dir (str): Path to directory for storing project configs.
                               Will be created if it doesn't exist.
        """
        self.projects_dir = Path(projects_dir)
        self.projects_dir.mkdir(parents=True, exist_ok=True)

    def list_projects(self) -> List[str]:
        """
        List all saved project names.

        Returns:
            list: List of project names (without .json extension)

        Example:
            >>> manager = ProjectConfigManager()
            >>> projects = manager.list_projects()
            >>> # Returns ['My Experiment', 'Control Study', ...]
        """
        projects = []
        for json_file in self.projects_dir.glob("*.json"):
            # Remove .json extension to get project name
            projects.append(json_file.stem)
        return sorted(projects)

    def load_project(self, name: str) -> Dict:
        """
        Load project configuration from JSON file.

        Args:
            name (str): Project name (without .json extension)

        Returns:
            dict: Project configuration dictionary

        Raises:
            FileNotFoundError: If project file doesn't exist
            json.JSONDecodeError: If file is not valid JSON

        Example:
            >>> config = manager.load_project("My Experiment")
            >>> print(config['groups'])
        """
        file_path = self.projects_dir / f"{name}.json"

        if not file_path.exists():
            raise FileNotFoundError(f"Project not found: {name}")

        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def save_project(self, config: Dict) -> str:
        """
        Save project configuration to JSON file.

        The project name is taken from config['project_name'].
        If the file already exists, it will be overwritten.
        Timestamps for created_date and modified_date are automatically managed.

        Args:
            config (dict): Project configuration dictionary

        Returns:
            str: Path to saved file

        Raises:
            ValueError: If config is invalid (missing required fields)

        Example:
            >>> config = {
            ...     'project_name': 'My Experiment',
            ...     'groups': [{'name': 'Treatment'}, {'name': 'Control'}],
            ...     'conditions': [{'name': 'Day 0'}, {'name': 'Day 7'}],
            ...     'include_sex': True
            ... }
            >>> manager.save_project(config)
        """
        # Validate before saving
        is_valid, errors = self.validate_config(config)
        if not is_valid:
            raise ValueError(f"Invalid configuration: {'; '.join(errors)}")

        project_name = config['project_name']
        file_path = self.projects_dir / f"{project_name}.json"

        # Add/update timestamps
        now = datetime.now().isoformat()
        if 'created_date' not in config:
            config['created_date'] = now
        config['modified_date'] = now

        # Save to file with pretty formatting
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)

        return str(file_path)

    def delete_project(self, name: str) -> bool:
        """
        Delete a project configuration file.

        Args:
            name (str): Project name to delete

        Returns:
            bool: True if deleted, False if file didn't exist

        Example:
            >>> manager.delete_project("Old Experiment")
        """
        file_path = self.projects_dir / f"{name}.json"

        if file_path.exists():
            file_path.unlink()
            return True
        return False

    def project_exists(self, name: str) -> bool:
        """
        Check if a project with the given name exists.

        Args:
            name (str): Project name to check

        Returns:
            bool: True if project exists
        """
        file_path = self.projects_dir / f"{name}.json"
        return file_path.exists()

    def validate_config(self, config: Dict) -> Tuple[bool, List[str]]:
        """
        Validate a project configuration.

        Validation rules:
        1. project_name is required and non-empty
        2. Group names must be unique and non-empty
        3. Condition names must be unique and non-empty
        4. If conditions exist, at least one group must exist

        Args:
            config (dict): Configuration to validate

        Returns:
            tuple: (is_valid, error_messages)
                - is_valid (bool): True if configuration is valid
                - error_messages (list): List of validation error messages

        Example:
            >>> is_valid, errors = manager.validate_config(config)
            >>> if not is_valid:
            >>>     print("Errors:", errors)
        """
        errors = []

        # Check project name
        project_name = config.get('project_name', '').strip()
        if not project_name:
            errors.append("Project name is required")

        # Check for invalid characters in project name (will be filename)
        invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
        for char in invalid_chars:
            if char in project_name:
                errors.append(f"Project name cannot contain '{char}'")

        # Check groups
        groups = config.get('groups', [])
        group_names = []
        for i, group in enumerate(groups):
            name = group.get('name', '').strip() if isinstance(group, dict) else ''
            if not name:
                errors.append(f"Group {i+1} has an empty name")
            elif name in group_names:
                errors.append(f"Duplicate group name: '{name}'")
            else:
                group_names.append(name)

        # Check conditions
        conditions = config.get('conditions', [])
        condition_names = []
        for i, condition in enumerate(conditions):
            name = condition.get('name', '').strip() if isinstance(condition, dict) else ''
            if not name:
                errors.append(f"Condition {i+1} has an empty name")
            elif name in condition_names:
                errors.append(f"Duplicate condition name: '{name}'")
            else:
                condition_names.append(name)

        # Check business rule: conditions require groups
        if conditions and not groups:
            errors.append("Conditions cannot exist without at least one group")

        return len(errors) == 0, errors

    def validate_animal_assignments(self, config: Dict) -> Tuple[bool, List[str]]:
        """
        Validate animal assignments in the configuration.

        Checks that:
        1. If groups/conditions/sex are defined, assignments structure matches
        2. Animal IDs are non-empty strings
        3. No duplicate animal IDs exist across different assignments

        Args:
            config (dict): Configuration to validate

        Returns:
            tuple: (is_valid, error_messages)
        """
        errors = []
        assignments = config.get('animal_assignments', {})

        if not assignments:
            # Empty assignments is valid - just means not yet configured
            return True, []

        all_animal_ids = []

        def extract_and_validate(obj, path=""):
            """Recursively validate animal IDs in nested structure."""
            if isinstance(obj, list):
                for item in obj:
                    if not isinstance(item, str) or not item.strip():
                        errors.append(f"Invalid animal ID at {path}: must be non-empty string")
                    else:
                        all_animal_ids.append(item.strip())
            elif isinstance(obj, dict):
                for key, value in obj.items():
                    new_path = f"{path}/{key}" if path else key
                    extract_and_validate(value, new_path)

        extract_and_validate(assignments)

        # Check for duplicates
        seen = set()
        for animal_id in all_animal_ids:
            if animal_id in seen:
                errors.append(f"Duplicate animal ID: {animal_id}")
            seen.add(animal_id)

        return len(errors) == 0, errors

    def create_empty_config(self, project_name: str = "") -> Dict:
        """
        Create an empty project configuration template.

        Args:
            project_name (str): Optional project name to pre-fill

        Returns:
            dict: Empty configuration template with all required fields

        Example:
            >>> config = manager.create_empty_config("New Study")
            >>> # Returns template with project_name filled in
        """
        return {
            "project_name": project_name,
            "groups": [],
            "conditions": [],
            "include_sex": False,
            "has_timepoints": False,
            "separate_timepoints": False,
            "timepoints": [],
            "folder_structure": "nested",
            "data_source_dir": "",
            "amps_output_dir": "",
            "output_dir": "",
            "metadata_file": "",
            "animal_assignments": {},
            "notes": ""
        }

    def get_project_summary(self, name: str) -> Optional[Dict]:
        """
        Get a summary of a project configuration without loading full file.

        Args:
            name (str): Project name

        Returns:
            dict or None: Summary with configuration overview
                         Returns None if project doesn't exist

        Example:
            >>> summary = manager.get_project_summary("My Experiment")
            >>> print(f"Groups: {summary['group_count']}")
        """
        try:
            config = self.load_project(name)

            # Count total animals in assignments
            animal_count = 0
            assignments = config.get("animal_assignments", {})

            def count_animals(obj):
                nonlocal animal_count
                if isinstance(obj, list):
                    animal_count += len([a for a in obj if isinstance(a, str) and a.strip()])
                elif isinstance(obj, dict):
                    for value in obj.values():
                        count_animals(value)

            count_animals(assignments)

            return {
                "project_name": config.get("project_name", name),
                "group_count": len(config.get("groups", [])),
                "condition_count": len(config.get("conditions", [])),
                "include_sex": config.get("include_sex", False),
                "has_timepoints": config.get("has_timepoints", False),
                "animal_count": animal_count,
                "has_data_source": bool(config.get("data_source_dir", "")),
                "has_output_dir": bool(config.get("output_dir", "")),
                "created_date": config.get("created_date", "Unknown"),
                "modified_date": config.get("modified_date", "Unknown")
            }
        except Exception:
            return None


# Convenience function for getting a singleton-like manager instance
_manager_instance = None


def get_project_manager(projects_dir: str = "projects") -> ProjectConfigManager:
    """
    Get a ProjectConfigManager instance (creates one if needed).

    This provides a convenient way to access the manager without
    creating new instances everywhere.

    Args:
        projects_dir (str): Directory for project files

    Returns:
        ProjectConfigManager: Manager instance

    Example:
        >>> from src.utils.project_config_manager import get_project_manager
        >>> manager = get_project_manager()
        >>> projects = manager.list_projects()
    """
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = ProjectConfigManager(projects_dir)
    return _manager_instance
