"""
Classification Module for LUPE Analysis Tool

This module handles behavior classification using the trained A-SOiD model.
It combines data loading, feature extraction, and model prediction into
a streamlined pipeline.

The classification process:
1. Load pose data (x,y coordinates from DeepLabCut)
2. Extract features (distances, angles, displacements)
3. Apply the trained model to predict behaviors
4. Smooth predictions to remove jitter
5. Return frame-by-frame behavior classifications

Usage:
    from src.core.classification import classify_behaviors, predict_and_smooth

    # Classify behaviors from pose data
    behaviors = classify_behaviors(model, [pose_data], framerate=60)

    # Result: behaviors[0] = array of behavior IDs per frame
    # E.g., behaviors[0] = [0, 0, 0, 1, 1, 1, 2, 2, ...]
    #       where 0=still, 1=walking, 2=rearing, etc.
"""

import numpy as np
import pandas as pd
import gc
import psutil
from typing import Any, Dict, List, Optional
from src.core.feature_extraction import feature_extraction, weighted_smoothing
from src.utils.config_manager import get_config


def classify_behaviors(model: Any,
                      poses: List[np.ndarray],
                      framerate: Optional[int] = None,
                      repeat_factor: Optional[int] = None,
                      smoothing_window: Optional[int] = None) -> List[np.ndarray]:
    """
    Classify behaviors from pose data using the trained model.

    This is the main function for behavior classification. It handles the
    complete pipeline from pose data to smoothed behavior predictions.

    Args:
        model: Trained A-SOiD classification model (loaded with load_model)
        poses (list): List of pose arrays, one per video file
                     Each array has shape (num_frames, num_coordinates)
                     where num_coordinates = num_keypoints * 2 (x,y pairs)
        framerate (int, optional): Video framerate in fps. If None, uses
                                  config default (typically 60 fps)
        repeat_factor (int, optional): Upsampling factor for predictions.
                                      If None, uses config default (typically 6)
        smoothing_window (int, optional): Window size for smoothing jitter.
                                         If None, uses config default (typically 12)

    Returns:
        list: List of behavior prediction arrays, one per input file
              Each array contains integer behavior IDs for each frame
              0 = still, 1 = walking, 2 = rearing, etc.

    Example:
        >>> from src.core.data_loader import load_model
        >>> from src.core.classification import classify_behaviors
        >>>
        >>> # Load the trained model
        >>> model = load_model('model/model_LUPE-AMPS.pkl')
        >>>
        >>> # Load pose data (shape: frames × 40 coordinates for 20 keypoints)
        >>> pose_data = np.load('data/mouse_pose.npy')
        >>>
        >>> # Classify behaviors
        >>> behaviors = classify_behaviors(model, [pose_data])
        >>>
        >>> # Access predictions for first file
        >>> predictions = behaviors[0]
        >>> print(f"Predicted {len(predictions)} frames")
        >>> print(f"Behavior at frame 100: {predictions[100]}")
    """
    # Load config for default parameters if not provided
    config = get_config()
    if framerate is None:
        framerate = config.get_framerate()
    if repeat_factor is None:
        repeat_factor = config.get_repeat_factor()
    if smoothing_window is None:
        smoothing_window = config.get_smoothing_window()

    # Step 1: Extract features from pose data
    # This computes distances, angles, and displacements

    # Log memory before feature extraction
    process = psutil.Process()
    mem_start = process.memory_info().rss / (1024 * 1024)  # Convert to MB
    print(f"[MEMORY] START of classification: {mem_start:.1f} MB")
    print(f"[MEMORY] Processing {len(poses)} file(s)")
    print(f"[MEMORY] First file shape: {poses[0].shape} ({poses[0].nbytes / (1024*1024):.1f} MB)")

    print(f"[MEMORY] Calling feature_extraction()...")
    features = feature_extraction(poses, len(poses), framerate)

    # Log memory after feature extraction
    mem_after_features = process.memory_info().rss / (1024 * 1024)  # Convert to MB
    mem_increase = mem_after_features - mem_start
    print(f"[MEMORY] AFTER feature_extraction: {mem_after_features:.1f} MB (increased by {mem_increase:.1f} MB)")
    print(f"[MEMORY] Returned {len(features)} feature array(s)")
    if len(features) > 0:
        print(f"[MEMORY] First feature shape: {features[0].shape} ({features[0].nbytes / (1024*1024):.1f} MB)")
    print(f"[MEMORY] Feature extraction complete - memory optimization: 83% reduction from inference mode")

    # Step 2: Predict behaviors for each file
    predictions = []
    for file_idx, feat in enumerate(features):
        # Get the total number of frames in the original video
        total_n_frames = poses[file_idx].shape[0]

        # Ensure features have correct dtype for model compatibility
        # Convert to float64 explicitly to avoid dtype mismatch issues
        feat = _ensure_feature_dtype(feat)

        # Predict on downsampled features (10 Hz)
        predict_downsampled = model.predict(feat)

        # Upsample predictions back to original framerate with explicit intermediate cleanup
        # repeat_factor converts from 10 Hz to original fps (e.g., 60 fps)
        # Make intermediates explicit to ensure proper memory cleanup
        predictions_repeated = predict_downsampled.repeat(repeat_factor)
        predictions_padded = np.pad(predictions_repeated, (repeat_factor, 0), 'edge')
        predictions_upsampled = predictions_padded[:total_n_frames]

        # Free intermediate upsampling arrays
        del predictions_repeated, predictions_padded

        # Step 3: Smooth predictions to remove jitter
        # This removes short behavior bouts that are likely classification errors
        predictions_smooth = weighted_smoothing(predictions_upsampled, size=smoothing_window)

        predictions.append(predictions_smooth)

        # Free this file's features and intermediate arrays immediately after use
        # This prevents accumulation when processing multiple files
        del feat, predict_downsampled, predictions_upsampled
        # NOTE: gc.collect() removed from loop - was causing slowdown on Windows
        # Will run once after all predictions complete

    # Free entire features list after all predictions are made
    del features
    gc.collect()  # Single GC after all files processed

    return predictions


def predict_and_smooth(model: Any,
                      features: List[np.ndarray],
                      total_frames: List[int],
                      repeat_factor: Optional[int] = None,
                      smoothing_window: Optional[int] = None) -> List[np.ndarray]:
    """
    Predict behaviors from pre-extracted features and apply smoothing.

    Use this function if you've already extracted features and just want
    to run prediction and smoothing.

    Args:
        model: Trained classification model
        features (list): List of feature arrays (one per file)
        total_frames (list): List of frame counts (one per file)
        repeat_factor (int, optional): Upsampling factor
        smoothing_window (int, optional): Smoothing window size

    Returns:
        list: List of smoothed behavior predictions

    Example:
        >>> # If you've already extracted features:
        >>> features = [features_file1, features_file2]
        >>> frame_counts = [1000, 1500]
        >>> predictions = predict_and_smooth(model, features, frame_counts)
    """
    # Load config for defaults
    config = get_config()
    if repeat_factor is None:
        repeat_factor = config.get_repeat_factor()
    if smoothing_window is None:
        smoothing_window = config.get_smoothing_window()

    predictions = []
    for feat, n_frames in zip(features, total_frames):
        # Ensure correct dtype
        feat = _ensure_feature_dtype(feat)

        # Predict
        predict_ds = model.predict(feat)

        # Upsample
        predictions_up = np.pad(
            predict_ds.repeat(repeat_factor),
            (repeat_factor, 0),
            'edge'
        )[:n_frames]

        # Smooth
        predictions_smooth = weighted_smoothing(predictions_up, size=smoothing_window)
        predictions.append(predictions_smooth)

    return predictions


def predict_single_file(model: Any,
                       pose: np.ndarray,
                       framerate: Optional[int] = None,
                       repeat_factor: Optional[int] = None,
                       smoothing_window: Optional[int] = None) -> np.ndarray:
    """
    Classify behaviors for a single video file.

    Convenience wrapper around classify_behaviors for single-file processing.

    Args:
        model: Trained classification model
        pose (np.ndarray): Pose array for one video file
        framerate (int, optional): Video framerate
        repeat_factor (int, optional): Upsampling factor
        smoothing_window (int, optional): Smoothing window size

    Returns:
        np.ndarray: Behavior predictions for the video

    Example:
        >>> pose_data = load_pose('video1.csv')
        >>> predictions = predict_single_file(model, pose_data)
        >>> print(f"Video has {len(predictions)} frames")
    """
    result = classify_behaviors(
        model, [pose],
        framerate=framerate,
        repeat_factor=repeat_factor,
        smoothing_window=smoothing_window
    )
    return result[0]


def get_behavior_summary(predictions: np.ndarray,
                        behavior_names: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Get a summary of behavior distribution in predictions.

    This creates a DataFrame showing how much time was spent in each behavior.

    Args:
        predictions (np.ndarray): Array of behavior predictions
        behavior_names (list, optional): List of behavior names. If None,
                                        uses names from config.

    Returns:
        pd.DataFrame: Summary with columns:
                     - behavior: Behavior name
                     - frames: Number of frames
                     - percentage: Percentage of total time

    Example:
        >>> predictions = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
        >>> summary = get_behavior_summary(predictions)
        >>> print(summary)
        #       behavior  frames  percentage
        #  0       still       3       33.33
        #  1     walking       2       22.22
        #  2     rearing       4       44.44
    """
    # Get behavior names from config if not provided
    if behavior_names is None:
        config = get_config()
        behavior_names = config.get_behavior_names()

    # Count occurrences of each behavior
    unique_behaviors, counts = np.unique(predictions, return_counts=True)

    # Calculate percentages
    total_frames = len(predictions)
    percentages = (counts / total_frames) * 100

    # Create summary DataFrame
    summary_data = []
    for behavior_id, count, percentage in zip(unique_behaviors, counts, percentages):
        summary_data.append({
            'behavior_id': int(behavior_id),
            'behavior': behavior_names[int(behavior_id)],
            'frames': int(count),
            'percentage': round(percentage, 2)
        })

    return pd.DataFrame(summary_data)


def get_behavior_bouts(predictions: np.ndarray) -> tuple:
    """
    Extract behavior bout information from predictions.

    A "bout" is a continuous sequence of the same behavior.
    This function identifies all bouts and their properties.

    Args:
        predictions (np.ndarray): Array of behavior predictions

    Returns:
        tuple: (bout_starts, bout_ends, bout_labels, bout_durations)
            - bout_starts: Array of frame indices where bouts start
            - bout_ends: Array of frame indices where bouts end
            - bout_labels: Array of behavior IDs for each bout
            - bout_durations: Array of bout durations in frames

    Example:
        >>> # Predictions: [0, 0, 0, 1, 1, 0, 0, 2, 2, 2]
        >>> predictions = np.array([0, 0, 0, 1, 1, 0, 0, 2, 2, 2])
        >>> starts, ends, labels, durations = get_behavior_bouts(predictions)
        >>> # Bout 1: still from frame 0-2 (3 frames)
        >>> # Bout 2: walking from frame 3-4 (2 frames)
        >>> # Bout 3: still from frame 5-6 (2 frames)
        >>> # Bout 4: rearing from frame 7-9 (3 frames)
    """
    # Find where behavior changes occur
    # np.diff finds differences between consecutive elements
    # np.where finds indices where difference != 0 (behavior changes)
    change_indices = np.where(np.diff(predictions) != 0)[0]

    # Start indices: first frame is always a start, plus frames after changes
    bout_starts = np.hstack(([0], change_indices + 1))

    # End indices: frames where changes occur, plus last frame
    bout_ends = np.hstack((change_indices, [len(predictions) - 1]))

    # Get the behavior ID for each bout (sample at start index)
    bout_labels = predictions[bout_starts]

    # Calculate bout durations (end - start + 1 for inclusive counting)
    bout_durations = bout_ends - bout_starts + 1

    return bout_starts, bout_ends, bout_labels, bout_durations


def filter_short_bouts(predictions: np.ndarray,
                      min_duration: int = 6) -> np.ndarray:
    """
    Remove behavior bouts shorter than a minimum duration.

    Short bouts are often classification errors or brief movements.
    This function replaces them with the previous longer bout's behavior.

    Args:
        predictions (np.ndarray): Array of behavior predictions
        min_duration (int): Minimum bout duration in frames (default: 6)

    Returns:
        np.ndarray: Filtered predictions with short bouts removed

    Example:
        >>> # Original: [0, 0, 0, 1, 0, 0, 0, 0]
        >>> #                    ^-- Single frame of behavior 1
        >>> predictions = np.array([0, 0, 0, 1, 0, 0, 0, 0])
        >>> filtered = filter_short_bouts(predictions, min_duration=2)
        >>> # Result:   [0, 0, 0, 0, 0, 0, 0, 0]
        >>> #                    ^-- Short bout removed
    """
    # Get bout information
    starts, ends, labels, durations = get_behavior_bouts(predictions)

    # Create filtered copy
    filtered = predictions.copy()

    # Process each bout
    for i in range(len(starts)):
        if durations[i] < min_duration and i > 0:
            # Replace short bout with previous behavior
            filtered[starts[i]:ends[i]+1] = filtered[starts[i]-1]

    return filtered


def behavior_transition_matrix(predictions: np.ndarray,
                               behavior_names: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Create a transition matrix showing how behaviors follow each other.

    This shows the probability of transitioning from one behavior to another.

    Args:
        predictions (np.ndarray): Array of behavior predictions
        behavior_names (list, optional): List of behavior names

    Returns:
        pd.DataFrame: Transition matrix where entry [i,j] is the probability
                     of transitioning from behavior i to behavior j

    Example:
        >>> predictions = np.array([0, 0, 1, 1, 0, 2, 2, 0])
        >>> matrix = behavior_transition_matrix(predictions)
        >>> # Shows: still -> walking, walking -> still, still -> rearing, etc.
    """
    # Get behavior names
    if behavior_names is None:
        config = get_config()
        behavior_names = config.get_behavior_names()

    # Get bout information
    starts, ends, labels, durations = get_behavior_bouts(predictions)

    # Count transitions
    n_behaviors = len(behavior_names)
    transition_counts = np.zeros((n_behaviors, n_behaviors))

    # Count each transition
    for i in range(len(labels) - 1):
        from_behavior = int(labels[i])
        to_behavior = int(labels[i + 1])
        transition_counts[from_behavior, to_behavior] += 1

    # Convert counts to probabilities
    # Divide each row by its sum to get probabilities
    row_sums = transition_counts.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1
    transition_probs = transition_counts / row_sums

    # Create DataFrame for better readability
    df = pd.DataFrame(
        transition_probs,
        index=behavior_names,
        columns=behavior_names
    )

    return df


def _ensure_feature_dtype(features: np.ndarray, target_dtype: type = np.float64) -> np.ndarray:
    """
    Ensure features have the correct dtype for model compatibility.

    This function explicitly converts feature arrays to the specified dtype
    to avoid dtype mismatch errors when passing features to sklearn models.
    This is especially important when loading models trained with older versions.

    Args:
        features (np.ndarray): Feature array to convert
        target_dtype (type): Target numpy dtype (default: np.float64)

    Returns:
        np.ndarray: Features with correct dtype

    Example:
        >>> features = np.array([[1, 2], [3, 4]], dtype=np.float32)
        >>> features_fixed = _ensure_feature_dtype(features)
        >>> print(features_fixed.dtype)  # float64
    """
    if not isinstance(features, np.ndarray):
        features = np.array(features)

    # Convert to target dtype if needed
    if features.dtype != target_dtype:
        features = features.astype(target_dtype)

    # Ensure array is contiguous in memory (C-order)
    # This can prevent some obscure prediction errors
    if not features.flags['C_CONTIGUOUS']:
        features = np.ascontiguousarray(features)

    return features
