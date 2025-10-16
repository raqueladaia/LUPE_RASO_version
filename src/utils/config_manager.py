"""
Configuration Manager for LUPE Analysis Tool

This module handles loading and accessing configuration settings from JSON files.
It provides a centralized way to manage metadata (behavior definitions, keypoints, etc.)
and user settings (paths, preferences, etc.).

The configuration files are stored in the 'config/' directory:
- metadata.json: Contains behavior definitions, colors, keypoints, and physical parameters
- settings.json: Contains user preferences and default paths

Usage:
    from src.utils.config_manager import ConfigManager

    # Initialize the configuration manager
    config = ConfigManager()

    # Access metadata
    behavior_names = config.get_behavior_names()
    behavior_colors = config.get_behavior_colors()

    # Access settings
    output_path = config.get_setting('paths', 'default_output_path')

    # Update settings
    config.update_setting('paths', 'default_output_path', 'new/path')
    config.save_settings()
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


class ConfigManager:
    """
    Manages configuration files for the LUPE Analysis Tool.

    This class provides methods to load, access, and update configuration settings
    from JSON files. It handles both metadata (static configuration) and user settings
    (dynamic configuration that can be modified).

    Attributes:
        config_dir (Path): Path to the configuration directory
        metadata_path (Path): Path to the metadata.json file
        settings_path (Path): Path to the settings.json file
        metadata (dict): Loaded metadata configuration
        settings (dict): Loaded user settings
    """

    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the ConfigManager.

        Args:
            config_dir (str, optional): Path to the configuration directory.
                If not provided, defaults to the 'config/' directory in the project root.

        Raises:
            FileNotFoundError: If configuration files are not found
            json.JSONDecodeError: If configuration files contain invalid JSON
        """
        # Determine the configuration directory path
        if config_dir is None:
            # Get the project root directory (3 levels up from this file)
            # src/utils/config_manager.py -> src/utils/ -> src/ -> project_root/
            project_root = Path(__file__).parent.parent.parent
            self.config_dir = project_root / 'config'
        else:
            self.config_dir = Path(config_dir)

        # Set paths to configuration files
        self.metadata_path = self.config_dir / 'metadata.json'
        self.settings_path = self.config_dir / 'settings.json'

        # Load configuration files
        self.metadata = self._load_json(self.metadata_path)
        self.settings = self._load_json(self.settings_path)

    def _load_json(self, file_path: Path) -> Dict[str, Any]:
        """
        Load a JSON file and return its contents as a dictionary.

        Args:
            file_path (Path): Path to the JSON file

        Returns:
            dict: Parsed JSON content

        Raises:
            FileNotFoundError: If the file doesn't exist
            json.JSONDecodeError: If the file contains invalid JSON
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _save_json(self, data: Dict[str, Any], file_path: Path) -> None:
        """
        Save a dictionary to a JSON file with proper formatting.

        Args:
            data (dict): Data to save
            file_path (Path): Path where the JSON file will be saved
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # ==================== Metadata Access Methods ====================

    def get_behavior_names(self) -> List[str]:
        """
        Get the list of behavior names.

        Returns:
            list: List of behavior names (e.g., ['still', 'walking', ...])
        """
        return self.metadata['behaviors']['names']

    def get_behavior_colors(self) -> List[str]:
        """
        Get the list of behavior colors for visualization.

        Returns:
            list: List of color names (e.g., ['crimson', 'darkcyan', ...])
        """
        return self.metadata['behaviors']['colors']

    def get_keypoints(self) -> List[str]:
        """
        Get the list of body keypoint names used in pose estimation.

        Returns:
            list: List of keypoint names (e.g., ['nose', 'mouth', ...])
        """
        return self.metadata['keypoints']

    def get_pixel_to_cm(self) -> float:
        """
        Get the pixel-to-centimeter conversion factor.

        Returns:
            float: Conversion factor (pixels to cm)
        """
        return self.metadata['physical_parameters']['pixel_to_cm']

    def get_framerate(self) -> int:
        """
        Get the default framerate for video analysis.

        Returns:
            int: Framerate in frames per second
        """
        return self.metadata['analysis_parameters']['framerate']

    def get_smoothing_window(self) -> int:
        """
        Get the smoothing window size for behavior predictions.

        Returns:
            int: Window size for smoothing
        """
        return self.metadata['analysis_parameters']['smoothing_window']

    def get_repeat_factor(self) -> int:
        """
        Get the repeat factor for upsampling predictions.

        Returns:
            int: Repeat factor
        """
        return self.metadata['analysis_parameters']['repeat_factor']

    def get_likelihood_threshold(self) -> float:
        """
        Get the minimum likelihood threshold for pose estimation.

        Returns:
            float: Likelihood threshold (0-1)
        """
        return self.metadata['analysis_parameters']['likelihood_threshold']

    # ==================== Settings Access Methods ====================

    def get_setting(self, section: str, key: str) -> Any:
        """
        Get a specific setting value from the settings file.

        Args:
            section (str): The section name (e.g., 'paths', 'output_preferences')
            key (str): The setting key within the section

        Returns:
            Any: The setting value

        Raises:
            KeyError: If the section or key doesn't exist
        """
        return self.settings[section][key]

    def update_setting(self, section: str, key: str, value: Any) -> None:
        """
        Update a specific setting value.

        Note: This only updates the in-memory settings. Call save_settings()
        to persist changes to the file.

        Args:
            section (str): The section name
            key (str): The setting key within the section
            value (Any): The new value

        Raises:
            KeyError: If the section doesn't exist
        """
        if section not in self.settings:
            raise KeyError(f"Settings section '{section}' not found")

        self.settings[section][key] = value

    def save_settings(self) -> None:
        """
        Save the current settings to the settings.json file.

        This persists any changes made via update_setting() to disk.
        """
        self._save_json(self.settings, self.settings_path)

    def reload_settings(self) -> None:
        """
        Reload settings from the settings.json file.

        This discards any unsaved changes and reloads from disk.
        """
        self.settings = self._load_json(self.settings_path)

    def get_all_settings(self) -> Dict[str, Any]:
        """
        Get all settings as a dictionary.

        Returns:
            dict: Complete settings dictionary
        """
        return self.settings.copy()

    def get_default_output_path(self) -> str:
        """
        Get the default output path for analysis results.

        Returns:
            str: Default output path
        """
        return self.get_setting('paths', 'default_output_path')

    def get_plot_format(self) -> str:
        """
        Get the preferred plot output format.

        Returns:
            str: Plot format (e.g., 'svg', 'png', 'pdf')
        """
        return self.get_setting('output_preferences', 'plot_format')

    def get_plot_dpi(self) -> int:
        """
        Get the DPI (resolution) for saved plots.

        Returns:
            int: DPI value
        """
        return self.get_setting('output_preferences', 'plot_dpi')

    def should_save_csv(self) -> bool:
        """
        Check if CSV output should be saved.

        Returns:
            bool: True if CSV should be saved, False otherwise
        """
        return self.get_setting('output_preferences', 'save_csv')

    def should_save_plots(self) -> bool:
        """
        Check if plot figures should be saved.

        Returns:
            bool: True if plots should be saved, False otherwise
        """
        return self.get_setting('output_preferences', 'save_plots')


# Singleton instance for easy access throughout the application
_config_instance = None


def get_config() -> ConfigManager:
    """
    Get the singleton ConfigManager instance.

    This ensures that only one ConfigManager is created and reused throughout
    the application, avoiding redundant file loading.

    Returns:
        ConfigManager: The singleton configuration manager instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigManager()
    return _config_instance
