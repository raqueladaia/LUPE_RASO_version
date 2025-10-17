"""
Data Loading Module for LUPE Analysis Tool

This module provides functions to load various types of data files used in LUPE analysis:
- Machine learning models (A-SOiD behavior classification models)
- Pose estimation data (DeepLabCut CSV files)
- Behavior predictions (pickle files)
- Feature data

All functions include detailed comments to help users understand the data flow.

Usage:
    from src.core.data_loader import load_model, load_data, load_behaviors

    # Load the A-SOiD classification model
    model = load_model('path/to/model.pkl')

    # Load pose data
    data = load_data('path/to/data.pkl')

    # Load behavior predictions
    behaviors = load_behaviors('path/to/behaviors.pkl')
"""

import pickle
import joblib
import os
import warnings
import numpy as np
from typing import Any, Dict, List, Optional
from pathlib import Path


def load_model(path: str) -> Any:
    """
    Load a machine learning model from a pickle file with dtype compatibility handling.

    This function loads the A-SOiD (Active Semi-supervised Clustering) model
    used for behavior classification. The model has been pre-trained on LUPE
    box behavior data.

    The function includes multiple compatibility layers to handle models pickled
    with older versions of scikit-learn or numpy:
    1. Standard pickle loading
    2. Alternative encoding handling for Python 2/3 compatibility
    3. Joblib loading as fallback
    4. Dtype conversion for numpy 2.0 compatibility

    Args:
        path (str): Path to the model file (.pkl format)

    Returns:
        Any: The loaded model object (typically a scikit-learn classifier)

    Raises:
        FileNotFoundError: If the model file doesn't exist
        Exception: If there's an error loading the model with troubleshooting hints

    Example:
        >>> model = load_model('model/model_LUPE-AMPS.pkl')
        >>> # Now you can use model.predict(features) to classify behaviors
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    # Try multiple loading strategies to handle version incompatibilities
    loading_strategies = [
        ("standard pickle", lambda: _load_with_pickle(path)),
        ("pickle with latin1 encoding", lambda: _load_with_pickle(path, encoding='latin1')),
        ("joblib", lambda: _load_with_joblib(path)),
    ]

    last_error = None
    for strategy_name, loader in loading_strategies:
        try:
            print(f"Attempting to load model using {strategy_name}...")
            model = loader()
            print(f"[OK] Successfully loaded model using {strategy_name}")

            # Apply dtype fixes to ensure compatibility
            # model = _fix_model_dtypes(model)

            return model
        except Exception as e:
            last_error = e
            print(f"  [ERROR] {strategy_name} failed: {str(e)}")
            continue

    # If all strategies failed, provide detailed error message
    error_msg = (
        f"Error loading model from {path}.\n"
        f"Last error: {str(last_error)}\n\n"
        f"Troubleshooting suggestions:\n"
        f"1. The model may be incompatible with current scikit-learn version\n"
        f"2. Try re-saving the model with current versions (see docs/TROUBLESHOOTING.md)\n"
        f"3. Check that the file is a valid pickle file\n"
        f"4. Ensure numpy and scikit-learn are up to date: pip install -U numpy scikit-learn"
    )
    raise Exception(error_msg)


def _load_with_pickle(path: str, encoding: str = 'ASCII') -> Any:
    """
    Load model using standard pickle with specified encoding.

    Args:
        path (str): Path to the model file
        encoding (str): Encoding to use ('ASCII', 'latin1', etc.)

    Returns:
        Any: Loaded model
    """
    with open(path, 'rb') as file:
        return pickle.load(file, encoding=encoding)


def _load_with_joblib(path: str) -> Any:
    """
    Load model using joblib (alternative to pickle for sklearn models).

    Args:
        path (str): Path to the model file

    Returns:
        Any: Loaded model
    """
    return joblib.load(path)


def _fix_model_dtypes(model: Any) -> Any:
    """
    Fix dtype incompatibilities in loaded models.

    This function addresses common dtype issues when loading models trained
    with older versions of scikit-learn/numpy:
    - Converts deprecated numpy dtypes to current equivalents
    - Ensures float arrays use float64
    - Handles numpy 2.0 compatibility issues

    Args:
        model: The loaded model object

    Returns:
        Any: Model with fixed dtypes
    """
    try:
        # Import sklearn here to avoid issues if not installed
        from sklearn.utils.validation import check_array

        # If model has a classes_ attribute (common in sklearn classifiers)
        if hasattr(model, 'classes_'):
            if isinstance(model.classes_, np.ndarray):
                # Ensure classes array uses current dtype representations
                model.classes_ = np.asarray(model.classes_)

        # If model has feature_importances_ or coef_ attributes
        for attr_name in ['feature_importances_', 'coef_', 'intercept_']:
            if hasattr(model, attr_name):
                attr_value = getattr(model, attr_name)
                if isinstance(attr_value, np.ndarray):
                    # Convert to explicit float64
                    setattr(model, attr_name, np.asarray(attr_value, dtype=np.float64))

        # For ensemble models or pipelines, recursively fix estimators
        if hasattr(model, 'estimators_'):
            # RandomForest, GradientBoosting, etc.
            if isinstance(model.estimators_, (list, np.ndarray)):
                for estimator in model.estimators_:
                    _fix_model_dtypes(estimator)

        if hasattr(model, 'steps'):
            # Pipeline
            for name, estimator in model.steps:
                _fix_model_dtypes(estimator)

        print("  [OK] Applied dtype compatibility fixes")

    except Exception as e:
        # If dtype fixing fails, log warning but don't fail
        warnings.warn(f"Could not apply all dtype fixes: {str(e)}")

    return model


def load_data(path: str) -> Any:
    """
    Load processed pose data from a pickle file.

    This loads pose estimation data that has been previously processed
    and saved. The data typically contains x,y coordinates for body keypoints
    tracked over time.

    Args:
        path (str): Path to the data file (.pkl format)

    Returns:
        Any: The loaded data (typically a numpy array or list of arrays)

    Raises:
        FileNotFoundError: If the data file doesn't exist
        Exception: If there's an error loading the data

    Example:
        >>> data = load_data('processed_dataset/pose_data.pkl')
        >>> print(f"Loaded data with shape: {data.shape}")
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    try:
        with open(path, 'rb') as file:
            data = pickle.load(file)
        return data
    except Exception as e:
        raise Exception(f"Error loading data from {path}: {str(e)}")


def load_features(path: str) -> Any:
    """
    Load extracted features from a pickle file.

    Features are computed from raw pose data and include:
    - Distances between body parts
    - Angles between body parts
    - Displacement (movement) between frames
    These features are used as input to the behavior classification model.

    Args:
        path (str): Path to the features file (.pkl format)

    Returns:
        Any: The loaded features (typically a numpy array or list of arrays)

    Raises:
        FileNotFoundError: If the features file doesn't exist
        Exception: If there's an error loading the features

    Example:
        >>> features = load_features('processed_dataset/features.pkl')
        >>> print(f"Loaded features with {len(features)} samples")
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Features file not found: {path}")

    try:
        with open(path, 'rb') as file:
            features = pickle.load(file)
        return features
    except Exception as e:
        raise Exception(f"Error loading features from {path}: {str(e)}")


def load_behaviors(path: str) -> Dict[str, Any]:
    """
    Load behavior predictions from a pickle file.

    This loads previously computed behavior classifications. Each file contains
    frame-by-frame behavior predictions (integers representing different behaviors).

    SIMPLIFIED STRUCTURE (no groups/conditions):
    The dictionary structure is: behaviors[file_name] = array_of_predictions

    Args:
        path (str): Path to the behaviors file (.pkl format)

    Returns:
        dict: Dictionary mapping file names to behavior prediction arrays.
              Each array contains integers representing behavior classes per frame.

    Raises:
        FileNotFoundError: If the behaviors file doesn't exist
        Exception: If there's an error loading the behaviors

    Example:
        >>> behaviors = load_behaviors('processed_dataset/behaviors.pkl')
        >>> for file_name, predictions in behaviors.items():
        >>>     print(f"{file_name}: {len(predictions)} frames")
        >>>     # predictions[i] gives the behavior class for frame i
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Behaviors file not found: {path}")

    try:
        with open(path, 'rb') as file:
            behaviors = pickle.load(file)
        return behaviors
    except Exception as e:
        raise Exception(f"Error loading behaviors from {path}: {str(e)}")


def save_behaviors(behaviors: Dict[str, Any], path: str) -> None:
    """
    Save behavior predictions to a pickle file.

    This saves behavior classifications for later use or analysis.

    Args:
        behaviors (dict): Dictionary mapping file names to behavior arrays
        path (str): Path where the behaviors file will be saved (.pkl format)

    Raises:
        Exception: If there's an error saving the behaviors

    Example:
        >>> behaviors = {'file1': np.array([0, 1, 2, ...]), 'file2': ...}
        >>> save_behaviors(behaviors, 'outputs/my_behaviors.pkl')
    """
    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(path, 'wb') as file:
            pickle.dump(behaviors, file)
    except Exception as e:
        raise Exception(f"Error saving behaviors to {path}: {str(e)}")


def load_model_features(path: str, name: str) -> List[Any]:
    """
    Load features and targets from A-SOiD training output.

    This function loads the features and target labels used during
    model training. It's useful for understanding model performance
    or retraining.

    Args:
        path (str): Path to the directory containing the model files
        name (str): Name of the model/experiment subdirectory

    Returns:
        list: List containing features and target data

    Raises:
        FileNotFoundError: If the file doesn't exist
        Exception: If there's an error loading the data

    Example:
        >>> data = load_model_features('model_output/', 'experiment_1')
        >>> features, targets = data[0], data[1]
    """
    file_path = os.path.join(path, name, 'feats_targets.sav')
    return _load_sav(file_path)


def load_embeddings(path: str, name: str) -> List[Any]:
    """
    Load embedding vectors from A-SOiD model output.

    Embeddings are lower-dimensional representations of the feature space
    used for clustering and visualization.

    Args:
        path (str): Path to the directory containing the model files
        name (str): Name of the model/experiment subdirectory

    Returns:
        list: List containing embedding data

    Raises:
        FileNotFoundError: If the file doesn't exist
        Exception: If there's an error loading the data

    Example:
        >>> embeddings = load_embeddings('model_output/', 'experiment_1')
    """
    file_path = os.path.join(path, name, 'embedding_output.sav')
    return _load_sav(file_path)


def _load_sav(file_path: str) -> List[Any]:
    """
    Internal helper function to load .sav files using joblib.

    Joblib is used for efficient serialization of large numpy arrays
    and scikit-learn models.

    Args:
        file_path (str): Path to the .sav file

    Returns:
        list: List of loaded data objects

    Raises:
        FileNotFoundError: If the file doesn't exist
        Exception: If there's an error loading the file
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"SAV file not found: {file_path}")

    try:
        with open(file_path, 'rb') as file:
            data = joblib.load(file)
        # Convert to list if not already
        return [item for item in data]
    except Exception as e:
        raise Exception(f"Error loading SAV file from {file_path}: {str(e)}")


def list_pickle_files(directory: str, pattern: str = "*.pkl") -> List[str]:
    """
    List all pickle files in a directory matching a pattern.

    Useful for batch processing multiple data files.

    Args:
        directory (str): Path to the directory to search
        pattern (str): Glob pattern for matching files (default: "*.pkl")

    Returns:
        list: List of paths to matching pickle files

    Example:
        >>> files = list_pickle_files('data/', '*behaviors*.pkl')
        >>> for file in files:
        >>>     behaviors = load_behaviors(file)
        >>>     # Process each file
    """
    directory_path = Path(directory)
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    files = list(directory_path.glob(pattern))
    return [str(f) for f in files]


def verify_data_integrity(data_path: str) -> bool:
    """
    Verify that a pickle file can be loaded without errors.

    This is useful for checking data integrity before starting
    a long analysis process.

    Args:
        data_path (str): Path to the pickle file to verify

    Returns:
        bool: True if the file loads successfully, False otherwise

    Example:
        >>> if verify_data_integrity('data/my_data.pkl'):
        >>>     print("Data file is valid")
        >>> else:
        >>>     print("Data file is corrupted or invalid")
    """
    try:
        with open(data_path, 'rb') as file:
            _ = pickle.load(file)
        return True
    except Exception as e:
        print(f"Data integrity check failed: {str(e)}")
        return False
