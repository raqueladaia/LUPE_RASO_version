"""
LUPE-AMPS Helper Utilities

This module provides utility functions for LUPE-AMPS analysis, including
CSV validation, data downsampling, bout calculations, transition matrix operations,
and batch file discovery.

Functions:
    - validate_behavior_csv: Check if CSV has correct format
    - downsample_sequence: Downsample behavior sequence using mode
    - calculate_bouts: Count bouts for a specific behavior
    - calculate_bout_durations: Calculate mean bout duration
    - safe_load_pca_model: Safely load PCA model from pickle file
    - calculate_transition_matrix: Calculate behavior transition probability matrix
    - apply_model_condition: Apply feature ablation condition to transition matrix
    - discover_behavior_files: Recursively find all behavior CSV files in a folder
    - extract_animal_date_from_path: Extract animal and date from folder structure
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from scipy.stats import mode
from scipy.ndimage import label
from typing import Tuple, List, Dict, Optional


def validate_behavior_csv(file_path: str) -> Tuple[bool, str]:
    """
    Validate that a CSV file has the correct format for LUPE-AMPS analysis.

    The CSV must have:
    - Two columns: 'frame' and 'behavior_id'
    - behavior_id values should be integers (typically 0-5)
    - At least some data rows

    Args:
        file_path (str): Path to the CSV file

    Returns:
        tuple: (is_valid, error_message)
            - is_valid (bool): True if file is valid
            - error_message (str): Empty if valid, otherwise description of error

    Example:
        >>> is_valid, error = validate_behavior_csv('behaviors.csv')
        >>> if not is_valid:
        >>>     print(f"Invalid file: {error}")
    """
    try:
        # Check file exists
        if not Path(file_path).exists():
            return False, f"File does not exist: {file_path}"

        # Try to read the CSV
        df = pd.read_csv(file_path)

        # Check columns
        required_columns = ['frame', 'behavior_id']
        if not all(col in df.columns for col in required_columns):
            return False, f"CSV must have columns: {required_columns}. Found: {list(df.columns)}"

        # Check if there's data
        if len(df) == 0:
            return False, "CSV file is empty (no data rows)"

        # Check behavior_id is numeric
        if not pd.api.types.is_numeric_dtype(df['behavior_id']):
            return False, "behavior_id column must contain numeric values"

        # Check for NaN values
        if df['behavior_id'].isna().any():
            return False, "behavior_id column contains NaN values"

        return True, ""

    except Exception as e:
        return False, f"Error reading CSV: {str(e)}"


def downsample_sequence(behavior_array: np.ndarray,
                       original_fps: int,
                       target_fps: int) -> np.ndarray:
    """
    Downsample behavior sequence using mode (most common value in windows).

    This function groups frames into windows and takes the most common
    behavior in each window as the downsampled value.

    Args:
        behavior_array (np.ndarray): Original behavior sequence
        original_fps (int): Original framerate (e.g., 60)
        target_fps (int): Target framerate (e.g., 20)

    Returns:
        np.ndarray: Downsampled behavior sequence

    Example:
        >>> original = np.array([0,0,0,1,1,1,2,2,2])
        >>> downsampled = downsample_sequence(original, 60, 20)
        >>> # Groups every 3 frames, returns [0, 1, 2]
    """
    # Calculate window size (how many frames to group)
    window_size = int(original_fps / target_fps)

    if window_size <= 1:
        # No downsampling needed
        return behavior_array.copy()

    downsampled = []

    # Process in windows
    for i in range(0, len(behavior_array), window_size):
        window = behavior_array[i:i+window_size]

        if len(window) == 0:
            break

        # Get most common value in window
        m = mode(window, keepdims=False)
        # Handle both old and new scipy versions
        value = m.mode[0] if hasattr(m.mode, '__iter__') else m.mode
        downsampled.append(value)

    return np.array(downsampled)


def calculate_bouts(behavior_array: np.ndarray, state_id: int) -> int:
    """
    Count the number of bouts (continuous sequences) for a specific behavior state.

    A bout is defined as a continuous sequence of the same behavior.
    For example, [0,0,1,1,0,0] has 2 bouts of behavior 0.

    Args:
        behavior_array (np.ndarray): Behavior sequence
        state_id (int): The behavior state to count (e.g., 0, 1, 2, ...)

    Returns:
        int: Number of bouts for this state

    Example:
        >>> behaviors = np.array([0, 0, 1, 1, 0, 2, 2, 2])
        >>> num_bouts = calculate_bouts(behaviors, 0)
        >>> # Returns 2 (two separate bouts of behavior 0)
    """
    # Create binary array: 1 where state matches, 0 elsewhere
    state_binary = (behavior_array == state_id).astype(int)

    # Use scipy.ndimage.label to identify connected components
    labeled_array, num_features = label(state_binary)

    return num_features


def calculate_bout_durations(behavior_array: np.ndarray,
                            state_id: int,
                            fps: int) -> List[float]:
    """
    Calculate durations (in seconds) of all bouts for a specific behavior state.

    Args:
        behavior_array (np.ndarray): Behavior sequence
        state_id (int): The behavior state to analyze
        fps (int): Framerate (frames per second)

    Returns:
        list: List of bout durations in seconds (empty list if no bouts)

    Example:
        >>> behaviors = np.array([0, 0, 0, 1, 1, 0])
        >>> durations = calculate_bout_durations(behaviors, 0, 20)
        >>> # Returns [0.15, 0.05] (3 frames and 1 frame at 20fps)
    """
    # Create binary array
    state_binary = (behavior_array == state_id).astype(int)

    # Label connected components
    labeled_array, num_features = label(state_binary)

    # Calculate duration of each bout
    durations = []
    for bout_id in range(1, num_features + 1):
        # Count frames in this bout
        bout_length_frames = np.sum(labeled_array == bout_id)
        # Convert to seconds
        bout_duration_sec = bout_length_frames / fps
        durations.append(bout_duration_sec)

    return durations


def safe_load_pca_model(model_path: str):
    """
    Safely load a PCA model from a pickle file with error handling.

    Args:
        model_path (str): Path to the pickle file containing PCA model

    Returns:
        object: Loaded PCA model (scikit-learn PCA object)

    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If file is not a valid pickle or doesn't contain PCA model

    Example:
        >>> model = safe_load_pca_model('models/model_AMPS.pkl')
        >>> transformed = model.transform(data)
    """
    model_path = Path(model_path)

    # Check file exists
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Try to load pickle
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load pickle file: {str(e)}")

    # Verify it has a transform method (duck typing for PCA-like objects)
    if not hasattr(model, 'transform'):
        raise ValueError("Loaded object does not have a 'transform' method. "
                        "Expected a PCA model.")

    # Optionally check for n_components attribute (typical for PCA)
    if not hasattr(model, 'n_components_'):
        print(f"Warning: Loaded model may not be a standard PCA object. "
              f"Type: {type(model)}")

    return model


def calculate_transition_matrix(sequence: np.ndarray,
                                n_behaviors: int = 6) -> np.ndarray:
    """
    Calculate normalized transition probability matrix from behavior sequence.

    A transition matrix shows the probability of transitioning from one behavior
    state to another. Entry [i,j] represents P(next_state=j | current_state=i).

    The matrix is row-normalized so that each row sums to 1.0 (or 0 if that
    behavior never occurred).

    Args:
        sequence (np.ndarray): Array of behavior IDs (integers 0 to n_behaviors-1)
        n_behaviors (int): Number of behavior states (default: 6)

    Returns:
        np.ndarray: Normalized transition matrix of shape (n_behaviors, n_behaviors)
                   Entry [i,j] = probability of transitioning from state i to state j

    Example:
        >>> seq = np.array([0, 0, 1, 1, 0, 2, 2, 0])
        >>> matrix = calculate_transition_matrix(seq, n_behaviors=3)
        >>> # Row 0 shows: P(0->0), P(0->1), P(0->2)
        >>> # Each row sums to 1.0

    Note:
        This function counts all transitions including self-transitions
        (staying in the same state). For model fit analysis, the diagonal
        can be zeroed using apply_model_condition(matrix, condition=2).
    """
    # Initialize count matrix
    count_matrix = np.zeros((n_behaviors, n_behaviors), dtype=np.float64)

    # Count transitions
    # For each consecutive pair of frames, count the transition
    for i in range(len(sequence) - 1):
        from_state = int(sequence[i])
        to_state = int(sequence[i + 1])

        # Ensure states are within valid range
        if 0 <= from_state < n_behaviors and 0 <= to_state < n_behaviors:
            count_matrix[from_state, to_state] += 1

    # Normalize rows to get probabilities
    # Each row should sum to 1.0 (probability distribution)
    row_sums = count_matrix.sum(axis=1, keepdims=True)

    # Avoid division by zero for behaviors that never occurred
    row_sums[row_sums == 0] = 1.0

    probability_matrix = count_matrix / row_sums

    return probability_matrix


def apply_model_condition(matrix: np.ndarray, condition: int) -> np.ndarray:
    """
    Apply feature ablation condition to a transition matrix.

    This function modifies a transition matrix according to one of 9 conditions
    used in the LUPE-AMPS model fit analysis. Each condition tests the importance
    of different features by removing them from the analysis.

    Conditions:
        1: Full model - no modifications (all transitions included)
        2: No self-transitions - diagonal set to zero
        3: No State 1 (behavior 0) - row and column 0 zeroed
        4: No State 2 (behavior 1) - row and column 1 zeroed
        5: No State 3 (behavior 2) - row and column 2 zeroed
        6: No State 4 (behavior 3) - row and column 3 zeroed
        7: No State 5 (behavior 4) - row and column 4 zeroed
        8: No State 6 (behavior 5) - row and column 5 zeroed
        9: Control condition - no modification (shuffling done separately)

    Args:
        matrix (np.ndarray): Transition matrix of shape (n_behaviors, n_behaviors)
        condition (int): Condition number (1-9)

    Returns:
        np.ndarray: Modified transition matrix (copy of original, not in-place)

    Raises:
        ValueError: If condition is not in range 1-9

    Example:
        >>> matrix = calculate_transition_matrix(sequence)
        >>> # Test importance of self-transitions
        >>> matrix_no_self = apply_model_condition(matrix, condition=2)
        >>> # Test importance of behavior 0
        >>> matrix_no_state1 = apply_model_condition(matrix, condition=3)

    Note:
        The function always returns a copy, never modifies the original matrix.
        For condition 9 (shuffled control), the shuffling of behavior labels
        should be done before calculating the transition matrix, not here.
    """
    if condition < 1 or condition > 9:
        raise ValueError(f"Condition must be 1-9, got {condition}")

    # Always work on a copy to avoid modifying original
    result = matrix.copy()

    if condition == 1:
        # Full model - no modifications
        pass

    elif condition == 2:
        # No self-transitions - zero the diagonal
        np.fill_diagonal(result, 0)

    elif 3 <= condition <= 8:
        # Remove specific behavior (conditions 3-8 remove behaviors 0-5)
        behavior_to_remove = condition - 3

        # Zero out the entire row (transitions FROM this behavior)
        result[behavior_to_remove, :] = 0

        # Zero out the entire column (transitions TO this behavior)
        result[:, behavior_to_remove] = 0

    elif condition == 9:
        # Control condition - matrix used as-is
        # Shuffling is done at the sequence level before matrix calculation
        pass

    return result


def extract_animal_date_from_path(csv_path: str) -> Tuple[str, str]:
    """
    Extract animal name and date from folder structure.

    Assumes folder structure: .../Animal/Date/file_behaviors.csv
    where Animal is 2 levels up and Date is 1 level up from the file.

    Args:
        csv_path (str): Full path to a behavior CSV file

    Returns:
        tuple: (animal_name, date_string)
            - animal_name (str): Name of the animal (grandparent folder)
            - date_string (str): Date identifier (parent folder)

    Raises:
        ValueError: If path doesn't have enough parent directories

    Example:
        >>> path = 'C:/data/Mouse01/2024-01-15/recording_behaviors.csv'
        >>> animal, date = extract_animal_date_from_path(path)
        >>> # Returns: ('Mouse01', '2024-01-15')
    """
    path = Path(csv_path).resolve()

    # Get parent folders
    # parent = date folder (1 level up)
    # parent.parent = animal folder (2 levels up)
    date_folder = path.parent
    animal_folder = date_folder.parent

    # Validate that we have actual folder names (not root)
    if date_folder.name == '' or animal_folder.name == '':
        raise ValueError(
            f"Cannot extract animal/date from path: {csv_path}\n"
            f"Expected structure: .../Animal/Date/file_behaviors.csv"
        )

    return animal_folder.name, date_folder.name


def discover_behavior_files(root_folder: str,
                           check_summary: bool = True) -> List[Dict]:
    """
    Recursively find all behavior CSV files in a folder and extract metadata.

    Searches for files matching the pattern '*_behaviors.csv' and extracts
    animal name and date from the folder structure.

    Args:
        root_folder (str): Root directory to search in
        check_summary (bool): If True, only include files that have a
                             corresponding *_summary.csv file (default: True)

    Returns:
        list: List of dictionaries, each containing:
            - 'path' (str): Full path to the behavior CSV file
            - 'animal' (str): Animal name (from grandparent folder)
            - 'date' (str): Date identifier (from parent folder)
            - 'filename' (str): Just the filename without path
            - 'has_summary' (bool): Whether corresponding summary file exists

    Example:
        >>> files = discover_behavior_files('C:/data/experiment1/')
        >>> for f in files:
        >>>     print(f"{f['animal']}/{f['date']}: {f['filename']}")
        >>> # Output:
        >>> # Mouse01/2024-01-15: recording_behaviors.csv
        >>> # Mouse01/2024-01-20: recording_behaviors.csv
        >>> # Mouse02/2024-01-18: recording_behaviors.csv

    Note:
        Files are returned sorted by (animal, date, filename) for consistent ordering.
        Files that don't match the expected folder structure are skipped with a warning.
    """
    root_path = Path(root_folder)

    if not root_path.exists():
        raise FileNotFoundError(f"Root folder does not exist: {root_folder}")

    if not root_path.is_dir():
        raise ValueError(f"Path is not a directory: {root_folder}")

    # Find all behavior CSV files recursively
    behavior_files = list(root_path.rglob('*_behaviors.csv'))

    discovered = []
    skipped_count = 0

    for csv_path in behavior_files:
        try:
            # Extract animal and date from path
            animal, date = extract_animal_date_from_path(str(csv_path))

            # Check for corresponding summary file
            summary_path = csv_path.parent / csv_path.name.replace('_behaviors.csv', '_summary.csv')
            has_summary = summary_path.exists()

            # Skip if no summary and check_summary is True
            if check_summary and not has_summary:
                print(f"Warning: Skipping {csv_path.name} - no corresponding summary file")
                skipped_count += 1
                continue

            discovered.append({
                'path': str(csv_path),
                'animal': animal,
                'date': date,
                'filename': csv_path.name,
                'has_summary': has_summary
            })

        except ValueError as e:
            # Path structure doesn't match expected format
            print(f"Warning: Skipping {csv_path} - {str(e)}")
            skipped_count += 1
            continue

    # Sort by animal, date, filename for consistent ordering
    discovered.sort(key=lambda x: (x['animal'], x['date'], x['filename']))

    # Print summary
    print(f"Discovered {len(discovered)} behavior files in {root_folder}")
    if skipped_count > 0:
        print(f"  ({skipped_count} files skipped due to missing summary or invalid structure)")

    return discovered


def get_summary_path_for_behavior_file(behavior_csv_path: str) -> Optional[Path]:
    """
    Get the path to the summary CSV file corresponding to a behavior CSV file.

    Args:
        behavior_csv_path (str): Path to behavior CSV file

    Returns:
        Path or None: Path to summary file if it exists, None otherwise

    Example:
        >>> summary = get_summary_path_for_behavior_file('data/mouse_behaviors.csv')
        >>> if summary:
        >>>     print(f"Summary file: {summary}")
    """
    behavior_path = Path(behavior_csv_path)

    # Replace _behaviors.csv with _summary.csv
    if '_behaviors.csv' in behavior_path.name:
        summary_name = behavior_path.name.replace('_behaviors.csv', '_summary.csv')
        summary_path = behavior_path.parent / summary_name

        if summary_path.exists():
            return summary_path

    return None
