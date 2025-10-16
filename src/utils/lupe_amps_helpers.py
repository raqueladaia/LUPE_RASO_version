"""
LUPE-AMPS Helper Utilities

This module provides utility functions for LUPE-AMPS analysis, including
CSV validation, data downsampling, and bout calculations.

Functions:
    - validate_behavior_csv: Check if CSV has correct format
    - downsample_sequence: Downsample behavior sequence using mode
    - calculate_bouts: Count bouts for a specific behavior
    - calculate_bout_durations: Calculate mean bout duration
    - safe_load_pca_model: Safely load PCA model from pickle file
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from scipy.stats import mode
from scipy.ndimage import label
from typing import Tuple, List


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
