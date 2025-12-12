"""
Folder Discovery for LUPE Analysis

This module provides utilities for discovering LUPE analysis outputs in data directories
regardless of the folder naming convention used.

Supports folder names like:
- Simple: "1438/" or "animal_1438/"
- Complex: "RASO_2270_1438_20240204_hab_-1d/"
- Nested: "1438/hab_-1d/"

The discovery system scans directories for folders containing LUPE analysis CSV files
and matches them to animal IDs and timepoints using flexible substring matching.

Usage:
    from src.utils.folder_discovery import discover_lupe_folders, find_folder_for_animal

    # Discover all folders with LUPE data
    folders = discover_lupe_folders(Path("/path/to/data"))

    # Find folder for a specific animal and timepoint
    folder = find_folder_for_animal(folders, "1438", "hab_-1d")
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# LUPE analysis files that indicate a valid analysis folder
LUPE_INDICATOR_FILES = [
    '*_bout_counts_summary.csv',
    '*_time_distribution_overall.csv',
    '*_bout_durations_statistics.csv',
    '*_transitions_matrix.csv',
    '*_timeline_*.csv'
]


def discover_lupe_folders(data_source_dir: Path, max_depth: int = 3) -> List[Dict]:
    """
    Recursively scan directory for folders containing LUPE analysis outputs.

    Searches for folders that contain LUPE CSV files either:
    - In an *_analysis subfolder
    - Directly in the folder itself

    Args:
        data_source_dir: Root directory to scan
        max_depth: Maximum folder depth to search (default 3)

    Returns:
        List of dicts, each containing:
        - folder_path: Path to the main folder
        - analysis_dir: Path where LUPE CSV files are located
        - folder_name: Name of the main folder
        - lupe_files: List of LUPE CSV files found
    """
    discovered = []
    data_source_dir = Path(data_source_dir)

    if not data_source_dir.exists():
        logger.warning(f"Data source directory does not exist: {data_source_dir}")
        return discovered

    def scan_folder(folder: Path, depth: int):
        """Recursively scan a folder for LUPE data."""
        if depth > max_depth:
            return

        # Skip hidden folders
        if folder.name.startswith('.'):
            return

        # Check if this folder has an _analysis subfolder
        analysis_subfolders = list(folder.glob('*_analysis'))

        for analysis_dir in analysis_subfolders:
            if analysis_dir.is_dir():
                lupe_files = find_lupe_files(analysis_dir)
                if lupe_files:
                    discovered.append({
                        'folder_path': folder,
                        'analysis_dir': analysis_dir,
                        'folder_name': folder.name,
                        'lupe_files': lupe_files
                    })
                    return  # Found data, don't recurse deeper

        # Check if LUPE files are directly in this folder
        lupe_files = find_lupe_files(folder)
        if lupe_files:
            discovered.append({
                'folder_path': folder,
                'analysis_dir': folder,
                'folder_name': folder.name,
                'lupe_files': lupe_files
            })
            return  # Found data, don't recurse deeper

        # Recurse into subfolders
        for subfolder in folder.iterdir():
            if subfolder.is_dir():
                scan_folder(subfolder, depth + 1)

    # Start scanning from the data source directory
    for item in data_source_dir.iterdir():
        if item.is_dir():
            scan_folder(item, 1)

    logger.info(f"Discovered {len(discovered)} folders with LUPE data in {data_source_dir}")
    return discovered


def find_lupe_files(folder: Path) -> List[Path]:
    """
    Find LUPE analysis CSV files in a folder.

    Args:
        folder: Directory to search

    Returns:
        List of Paths to LUPE CSV files found
    """
    files = []
    for pattern in LUPE_INDICATOR_FILES:
        files.extend(folder.glob(pattern))
    return files


def find_folder_for_animal(
    discovered_folders: List[Dict],
    animal_id: str,
    timepoint: Optional[str] = None
) -> Optional[Dict]:
    """
    Find the folder containing data for a specific animal and optionally timepoint.

    Uses substring matching to find animal_id within folder names.
    If timepoint is specified, also checks for timepoint in the folder name.

    Matching priority:
    1. Both animal_id AND timepoint appear in folder name (if timepoint specified)
    2. Animal_id appears as substring in folder name

    Args:
        discovered_folders: List of folder info dicts from discover_lupe_folders()
        animal_id: Animal identifier to search for
        timepoint: Optional timepoint to also match

    Returns:
        Dict with folder info if found, None otherwise
    """
    animal_id_lower = str(animal_id).lower()
    timepoint_lower = timepoint.lower() if timepoint else None

    # Normalize timepoint for matching (handle variations like hab_-1d, hab-1d)
    timepoint_normalized = normalize_timepoint(timepoint) if timepoint else None

    best_match = None
    best_score = 0

    for folder_info in discovered_folders:
        folder_name = folder_info['folder_name'].lower()
        folder_normalized = normalize_for_matching(folder_name)

        # Check if animal_id appears in folder name
        if animal_id_lower not in folder_name and animal_id_lower not in folder_normalized:
            continue

        # Calculate match score
        score = 1  # Base score for animal_id match

        # If timepoint required, check for it
        if timepoint_lower:
            if timepoint_lower in folder_name or timepoint_normalized in folder_normalized:
                score = 2  # Higher score for both matches
            else:
                continue  # Skip if timepoint required but not found

        # Keep best match
        if score > best_score:
            best_score = score
            best_match = folder_info

    return best_match


def normalize_timepoint(timepoint: str) -> str:
    """
    Normalize a timepoint string for flexible matching.

    Handles variations like:
    - "hab_-1d" -> "hab-1d"
    - "Day 7" -> "day7"
    - "d7" stays "d7"

    Args:
        timepoint: Original timepoint string

    Returns:
        Normalized timepoint string (lowercase, underscores removed, spaces removed)
    """
    if not timepoint:
        return ""

    normalized = timepoint.lower()
    # Remove underscores and spaces
    normalized = normalized.replace('_', '').replace(' ', '')
    return normalized


def normalize_for_matching(text: str) -> str:
    """
    Normalize text for flexible matching.

    Args:
        text: Original text

    Returns:
        Normalized text (lowercase, underscores/hyphens/spaces removed)
    """
    if not text:
        return ""

    normalized = text.lower()
    normalized = normalized.replace('_', '').replace('-', '').replace(' ', '')
    return normalized


def extract_info_from_folder_name(folder_name: str) -> Dict:
    """
    Try to extract metadata from a folder name.

    Attempts to identify components like animal ID, date, timepoint from
    folder names like "RASO_2270_1438_20240204_hab_-1d".

    Args:
        folder_name: Name of the folder

    Returns:
        Dict with extracted info:
        - parts: List of underscore-separated parts
        - potential_animal_ids: Parts that look like animal IDs (numeric)
        - potential_dates: Parts that look like dates (YYYYMMDD format)
        - potential_timepoints: Parts that might be timepoints
    """
    info = {
        'parts': [],
        'potential_animal_ids': [],
        'potential_dates': [],
        'potential_timepoints': []
    }

    # Split by underscores
    parts = folder_name.split('_')
    info['parts'] = parts

    for part in parts:
        # Check for numeric parts (potential animal IDs)
        if part.isdigit() and len(part) >= 3:
            info['potential_animal_ids'].append(part)

        # Check for date pattern (YYYYMMDD)
        if re.match(r'^\d{8}$', part):
            info['potential_dates'].append(part)

        # Check for timepoint patterns
        # Common patterns: d7, day7, hab, baseline, -1d, etc.
        if re.match(r'^(d\d+|day\d+|hab|baseline|\-?\d+d)$', part.lower()):
            info['potential_timepoints'].append(part)

    return info


def get_file_prefix_from_folder(folder_info: Dict) -> Optional[str]:
    """
    Determine the file prefix used in a folder's LUPE files.

    Looks at the CSV files in the folder to determine what prefix they use.

    Args:
        folder_info: Dict with folder information from discover_lupe_folders()

    Returns:
        File prefix string if found, None otherwise
    """
    lupe_files = folder_info.get('lupe_files', [])

    if not lupe_files:
        return None

    # Look at a file to extract the prefix
    # Files are named like: PREFIX_bout_counts_summary.csv
    for f in lupe_files:
        name = f.stem  # Filename without extension

        # Try to extract prefix from known suffixes
        for suffix in ['_bout_counts_summary', '_time_distribution_overall',
                       '_bout_durations_statistics', '_transitions_matrix']:
            if suffix in name:
                prefix = name.replace(suffix, '')
                return prefix

        # Try timeline pattern
        match = re.match(r'(.+)_timeline_', name)
        if match:
            return match.group(1)

    return None


def match_folders_to_animals(
    discovered_folders: List[Dict],
    animal_ids: List[str],
    timepoints: Optional[List[str]] = None
) -> Dict[Tuple[str, Optional[str]], Dict]:
    """
    Match discovered folders to a list of animal IDs and optional timepoints.

    Args:
        discovered_folders: List from discover_lupe_folders()
        animal_ids: List of animal IDs to match
        timepoints: Optional list of timepoint names

    Returns:
        Dict mapping (animal_id, timepoint) tuples to folder info dicts.
        If no timepoints, timepoint in key will be None.
    """
    matches = {}

    timepoints_to_check = timepoints if timepoints else [None]

    for animal_id in animal_ids:
        for timepoint in timepoints_to_check:
            folder_info = find_folder_for_animal(discovered_folders, animal_id, timepoint)
            if folder_info:
                matches[(str(animal_id), timepoint)] = folder_info

    return matches


def summarize_discovered_folders(discovered_folders: List[Dict]) -> str:
    """
    Create a human-readable summary of discovered folders.

    Args:
        discovered_folders: List from discover_lupe_folders()

    Returns:
        Multi-line string summary
    """
    if not discovered_folders:
        return "No folders with LUPE data discovered."

    lines = [f"Discovered {len(discovered_folders)} folders with LUPE data:"]
    lines.append("")

    for i, folder_info in enumerate(discovered_folders[:20]):  # Limit to first 20
        folder_name = folder_info['folder_name']
        analysis_dir = folder_info['analysis_dir']
        n_files = len(folder_info.get('lupe_files', []))

        lines.append(f"  {i+1}. {folder_name}")
        lines.append(f"      Analysis dir: {analysis_dir.name}")
        lines.append(f"      LUPE files: {n_files}")

        # Try to extract info
        info = extract_info_from_folder_name(folder_name)
        if info['potential_animal_ids']:
            lines.append(f"      Potential animal IDs: {', '.join(info['potential_animal_ids'])}")
        if info['potential_timepoints']:
            lines.append(f"      Potential timepoints: {', '.join(info['potential_timepoints'])}")

    if len(discovered_folders) > 20:
        lines.append(f"  ... and {len(discovered_folders) - 20} more")

    return "\n".join(lines)
