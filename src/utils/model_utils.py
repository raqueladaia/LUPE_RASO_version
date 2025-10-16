"""
Model Utilities for LUPE Analysis Tool

This module provides utilities for handling machine learning models,
particularly for fixing compatibility issues between different versions
of scikit-learn and numpy.

Common use cases:
- Converting legacy models to current versions
- Verifying model compatibility
- Re-pickling models with current library versions

Usage:
    from src.utils.model_utils import convert_legacy_model, verify_model

    # Convert an old model to be compatible with current versions
    convert_legacy_model('old_model.pkl', 'converted_model.pkl')

    # Verify a model can be loaded and used
    is_valid = verify_model('model.pkl')
"""

import pickle
import joblib
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import warnings


def convert_legacy_model(input_path: str,
                        output_path: Optional[str] = None,
                        verbose: bool = True) -> bool:
    """
    Convert a legacy model to be compatible with current sklearn/numpy versions.

    This function loads an old model (potentially with dtype incompatibilities)
    and re-saves it with current library versions, fixing dtype issues along the way.

    Args:
        input_path (str): Path to the legacy model file
        output_path (str, optional): Path for the converted model. If None,
                                    adds '_converted' suffix to input filename
        verbose (bool): Whether to print progress messages

    Returns:
        bool: True if conversion successful, False otherwise

    Example:
        >>> # Convert an old A-SOiD model
        >>> convert_legacy_model('model_LUPE-AMPS.pkl', 'model_LUPE-AMPS_v2.pkl')
        Attempting to load model using standard pickle...
        ✓ Successfully loaded model using standard pickle
        ✓ Applied dtype compatibility fixes
        ✓ Model converted and saved to: model_LUPE-AMPS_v2.pkl
        True
    """
    from src.core.data_loader import load_model

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Model file not found: {input_path}")

    # Determine output path
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_converted{input_path.suffix}"
    else:
        output_path = Path(output_path)

    try:
        if verbose:
            print(f"Converting model: {input_path}")
            print(f"Output will be saved to: {output_path}")
            print()

        # Load model using our enhanced loader with dtype fixes
        model = load_model(str(input_path))

        if verbose:
            print()
            print("Saving converted model...")

        # Save with current versions using joblib (more efficient for sklearn models)
        joblib.dump(model, output_path, compress=3)

        if verbose:
            print(f"✓ Model converted and saved to: {output_path}")
            print()
            print("You can now use the converted model file instead of the original.")

        return True

    except Exception as e:
        if verbose:
            print(f"✗ Conversion failed: {str(e)}")
        return False


def verify_model(model_path: str,
                test_features: Optional[np.ndarray] = None,
                verbose: bool = True) -> Tuple[bool, Dict[str, Any]]:
    """
    Verify that a model can be loaded and used successfully.

    This function attempts to load a model and optionally run a test prediction
    to ensure it works correctly.

    Args:
        model_path (str): Path to the model file
        test_features (np.ndarray, optional): Test features for prediction.
                                             If None, only tests loading.
        verbose (bool): Whether to print diagnostic messages

    Returns:
        tuple: (is_valid, info_dict)
            - is_valid: bool indicating if model is valid
            - info_dict: Dictionary with diagnostic information

    Example:
        >>> # Just verify loading
        >>> is_valid, info = verify_model('model.pkl')
        >>> if is_valid:
        >>>     print("Model is valid!")
        >>>     print(f"Model type: {info['model_type']}")

        >>> # Test with features
        >>> test_data = np.random.randn(10, 100).astype(np.float64)
        >>> is_valid, info = verify_model('model.pkl', test_data)
    """
    from src.core.data_loader import load_model

    info = {
        'model_type': None,
        'can_load': False,
        'can_predict': False,
        'has_classes': False,
        'n_classes': 0,
        'error': None
    }

    try:
        # Test loading
        if verbose:
            print(f"Verifying model: {model_path}")
            print("-" * 60)

        model = load_model(model_path)
        info['can_load'] = True
        info['model_type'] = type(model).__name__

        if verbose:
            print(f"✓ Model loaded successfully")
            print(f"  Type: {info['model_type']}")

        # Check for common attributes
        if hasattr(model, 'classes_'):
            info['has_classes'] = True
            info['n_classes'] = len(model.classes_)
            if verbose:
                print(f"  Classes: {model.classes_}")

        # Test prediction if features provided
        if test_features is not None:
            if verbose:
                print(f"\nTesting prediction with features of shape {test_features.shape}...")

            predictions = model.predict(test_features)
            info['can_predict'] = True

            if verbose:
                print(f"✓ Prediction successful")
                print(f"  Output shape: {predictions.shape}")
                print(f"  Unique predictions: {np.unique(predictions)}")

        if verbose:
            print("-" * 60)
            print("✓ Model verification PASSED")

        return True, info

    except Exception as e:
        info['error'] = str(e)

        if verbose:
            print(f"✗ Model verification FAILED")
            print(f"  Error: {str(e)}")

        return False, info


def create_test_features(n_samples: int = 10,
                        n_features: int = 100,
                        dtype: type = np.float64) -> np.ndarray:
    """
    Create random test features for model verification.

    Useful for testing if a model can make predictions without needing real data.

    Args:
        n_samples (int): Number of samples (frames)
        n_features (int): Number of features per sample
        dtype (type): Numpy dtype for the array

    Returns:
        np.ndarray: Random feature array of shape (n_samples, n_features)

    Example:
        >>> test_data = create_test_features(n_samples=10, n_features=100)
        >>> is_valid, info = verify_model('model.pkl', test_data)
    """
    # Generate random features similar to what the model expects
    # Use a fixed seed for reproducibility
    np.random.seed(42)
    features = np.random.randn(n_samples, n_features).astype(dtype)
    return features


def get_model_info(model: Any, verbose: bool = True) -> Dict[str, Any]:
    """
    Extract detailed information about a loaded model.

    Args:
        model: Loaded sklearn model
        verbose (bool): Whether to print the information

    Returns:
        dict: Dictionary containing model information

    Example:
        >>> from src.core.data_loader import load_model
        >>> model = load_model('model.pkl')
        >>> info = get_model_info(model)
    """
    info = {
        'type': type(model).__name__,
        'module': type(model).__module__,
        'has_classes': hasattr(model, 'classes_'),
        'n_classes': len(model.classes_) if hasattr(model, 'classes_') else 0,
        'classes': list(model.classes_) if hasattr(model, 'classes_') else None,
        'has_feature_importances': hasattr(model, 'feature_importances_'),
        'n_features_in': getattr(model, 'n_features_in_', None),
        'attributes': []
    }

    # Get list of important attributes
    for attr in dir(model):
        if not attr.startswith('_') and attr.endswith('_'):
            info['attributes'].append(attr)

    if verbose:
        print("Model Information:")
        print(f"  Type: {info['type']}")
        print(f"  Module: {info['module']}")
        print(f"  Number of classes: {info['n_classes']}")
        if info['classes']:
            print(f"  Classes: {info['classes']}")
        if info['n_features_in']:
            print(f"  Expected features: {info['n_features_in']}")
        print(f"  Fitted attributes: {', '.join(info['attributes'][:5])}...")

    return info


def batch_convert_models(input_dir: str,
                         output_dir: Optional[str] = None,
                         pattern: str = "*.pkl",
                         verbose: bool = True) -> Dict[str, bool]:
    """
    Convert multiple legacy models in a directory.

    Args:
        input_dir (str): Directory containing legacy models
        output_dir (str, optional): Directory for converted models. If None,
                                   uses input_dir with '_converted' suffix
        pattern (str): Glob pattern for model files
        verbose (bool): Whether to print progress

    Returns:
        dict: Dictionary mapping model names to conversion success status

    Example:
        >>> results = batch_convert_models('models/', 'models_converted/')
        >>> for name, success in results.items():
        >>>     print(f"{name}: {'✓' if success else '✗'}")
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Find all model files
    model_files = list(input_dir.glob(pattern))

    if len(model_files) == 0:
        print(f"No model files matching '{pattern}' found in {input_dir}")
        return {}

    # Determine output directory
    if output_dir is None:
        output_dir = input_dir.parent / f"{input_dir.name}_converted"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 60)
        print(f"Batch Converting {len(model_files)} Models")
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print("=" * 60)
        print()

    results = {}
    for i, model_file in enumerate(model_files, 1):
        if verbose:
            print(f"[{i}/{len(model_files)}] {model_file.name}")
            print("-" * 60)

        output_path = output_dir / model_file.name
        success = convert_legacy_model(str(model_file), str(output_path), verbose=False)
        results[model_file.name] = success

        if verbose:
            status = "✓ Success" if success else "✗ Failed"
            print(f"{status}\n")

    # Summary
    if verbose:
        print("=" * 60)
        print("Conversion Summary:")
        successes = sum(1 for v in results.values() if v)
        print(f"  Successful: {successes}/{len(results)}")
        print(f"  Failed: {len(results) - successes}/{len(results)}")
        print("=" * 60)

    return results


if __name__ == '__main__':
    """
    CLI interface for model utilities.

    Usage:
        python -m src.utils.model_utils convert model.pkl
        python -m src.utils.model_utils verify model.pkl
        python -m src.utils.model_utils info model.pkl
    """
    import sys

    if len(sys.argv) < 2:
        print("Model Utilities")
        print()
        print("Usage:")
        print("  python -m src.utils.model_utils convert <input_model> [output_model]")
        print("  python -m src.utils.model_utils verify <model> [--test]")
        print("  python -m src.utils.model_utils info <model>")
        print()
        sys.exit(1)

    command = sys.argv[1]

    if command == "convert":
        if len(sys.argv) < 3:
            print("Error: Please provide input model path")
            sys.exit(1)

        input_path = sys.argv[2]
        output_path = sys.argv[3] if len(sys.argv) > 3 else None

        success = convert_legacy_model(input_path, output_path)
        sys.exit(0 if success else 1)

    elif command == "verify":
        if len(sys.argv) < 3:
            print("Error: Please provide model path")
            sys.exit(1)

        model_path = sys.argv[2]
        test_pred = "--test" in sys.argv

        test_features = create_test_features() if test_pred else None
        is_valid, info = verify_model(model_path, test_features)

        sys.exit(0 if is_valid else 1)

    elif command == "info":
        if len(sys.argv) < 3:
            print("Error: Please provide model path")
            sys.exit(1)

        from src.core.data_loader import load_model
        model = load_model(sys.argv[2])
        get_model_info(model)

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
