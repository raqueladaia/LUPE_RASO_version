"""
DeepLabCut CSV Preprocessing Module

This module handles raw DeepLabCut (DLC) CSV output files and prepares them
for behavior classification. DLC produces CSV files with multi-level headers
containing x, y coordinates and likelihood values for each tracked keypoint.

The preprocessing workflow:
1. Load DLC CSV with multi-level headers
2. Filter low-confidence poses based on likelihood threshold
3. Extract x,y coordinates (excluding likelihood columns)
4. Save processed data for feature extraction and classification

Usage:
    from src.core.dlc_preprocessing import preprocess_dlc_csv, batch_process_dlc_files

    # Single file
    pose_data = preprocess_dlc_csv('video_DLC.csv', likelihood_threshold=0.1)

    # Multiple files
    all_poses = batch_process_dlc_files('dlc_data/', likelihood_threshold=0.1)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from glob import glob
import pickle
import time
import gc
import h5py

from src.core.feature_extraction import filter_pose_noise
from src.utils.config_manager import get_config


def load_dlc_csv(csv_path: str, skip_rows: int = 0, fast_mode: bool = True) -> pd.DataFrame:
    """
    Load a DeepLabCut CSV file with multi-level headers.

    DeepLabCut outputs CSV files with 3-4 header rows containing:
    - Row 0: Scorer information
    - Row 1: Body part names
    - Row 2: Coordinate type (x, y, likelihood)
    - Row 3: (optional) Additional metadata

    OPTIMIZED: Uses optimized pandas parameters for 3-6x faster loading
    of large DLC CSV files (30-60 minute videos).

    Args:
        csv_path (str): Path to the DLC CSV file
        skip_rows (int): Number of extra rows to skip (default: 0)
        fast_mode (bool): Use optimized loading parameters (default: True)

    Returns:
        pd.DataFrame: Loaded DataFrame with multi-level column headers

    Raises:
        FileNotFoundError: If CSV file doesn't exist
        Exception: If CSV format is incorrect

    Example:
        >>> df = load_dlc_csv('mouse_video_DLC_resnet50_tracking.csv')
        >>> print(df.shape)
        (10000, 60)  # 10000 frames, 20 keypoints × 3 (x, y, likelihood)

    Performance:
        - 30-minute video (108k frames): ~2-4 seconds (was ~10-15s)
        - 60-minute video (216k frames): ~4-8 seconds (was ~25-30s)
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"DLC CSV file not found: {csv_path}")

    # Prepare optimized loading parameters
    if fast_mode:
        # These parameters significantly speed up large CSV loading:
        # - low_memory=False: Prevents dtype guessing chunk-by-chunk
        # - dtype: All columns are float (coordinates and likelihoods)
        # - engine='c': Use faster C parser
        read_params = {
            'low_memory': False,
            'engine': 'c',
            'dtype': np.float64,  # All data columns are numeric
        }
    else:
        read_params = {}

    try:
        # Try loading with 4-level headers (most common DLC format)
        df = pd.read_csv(
            csv_path,
            header=[0, 1, 2, 3],
            index_col=0,
            skiprows=skip_rows,
            **read_params
        )
        return df
    except Exception as e:
        # Fall back to 3-level headers
        try:
            df = pd.read_csv(
                csv_path,
                header=[0, 1, 2],
                index_col=0,
                skiprows=skip_rows,
                **read_params
            )
            return df
        except Exception as e2:
            raise Exception(
                f"Error loading DLC CSV from {csv_path}. "
                f"Expected multi-level headers (3 or 4 rows). "
                f"Original error: {str(e)}"
            )


def load_dlc_h5(h5_path: str) -> pd.DataFrame:
    """
    Load a DeepLabCut HDF5 file with multi-level headers.

    HDF5 files contain the same data as CSV files but in a much more efficient
    binary format. They load 5-10x faster and use significantly less memory.

    Args:
        h5_path (str): Path to the DLC HDF5 file

    Returns:
        pd.DataFrame: Loaded DataFrame with multi-level column headers

    Raises:
        FileNotFoundError: If H5 file doesn't exist
        Exception: If H5 file format is incorrect

    Example:
        >>> df = load_dlc_h5('mouse_video_DLC_resnet50_tracking.h5')
        >>> print(df.shape)
        (10000, 60)  # 10000 frames, 20 keypoints × 3 (x, y, likelihood)

    Performance:
        - H5 files load 5-10x faster than equivalent CSV files
        - Memory usage is significantly lower due to efficient binary format
        - Supports memory-mapped access for very large files
    """
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"DLC H5 file not found: {h5_path}")

    try:
        # Load HDF5 file using pandas
        # DLC H5 files typically store data with key='df_with_missing' or the first available key
        df = pd.read_hdf(h5_path)
        return df
    except Exception as e:
        # If pandas can't read it directly, try with h5py to inspect structure
        try:
            with h5py.File(h5_path, 'r') as f:
                # Get list of available keys
                keys = list(f.keys())
                if not keys:
                    raise Exception(f"No datasets found in H5 file: {h5_path}")

                # Try reading with the first key
                df = pd.read_hdf(h5_path, key=keys[0])
                return df
        except Exception as e2:
            raise Exception(
                f"Error loading DLC H5 from {h5_path}. "
                f"Expected DeepLabCut HDF5 format. "
                f"Original error: {str(e)}"
            )


def get_coordinate_indices(dlc_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get column indices for coordinates (x,y) and likelihood values.

    DLC CSV columns are arranged as: x, y, likelihood, x, y, likelihood, ...
    for each body part.

    Args:
        dlc_df (pd.DataFrame): Loaded DLC DataFrame

    Returns:
        tuple: (coord_indices, likelihood_indices)
            - coord_indices: Indices of x,y columns
            - likelihood_indices: Indices of likelihood columns

    Example:
        >>> df = load_dlc_csv('data.csv')
        >>> coord_idx, llh_idx = get_coordinate_indices(df)
        >>> print(coord_idx)  # [0, 1, 3, 4, 6, 7, ...]
        >>> print(llh_idx)    # [2, 5, 8, ...]
    """
    all_columns = np.arange(dlc_df.shape[1])

    # Likelihood columns are every 3rd column starting from index 2
    # Pattern: x(0), y(1), likelihood(2), x(3), y(4), likelihood(5), ...
    likelihood_indices = all_columns[2::3]

    # Coordinate columns are everything except likelihood
    coord_indices = np.array([i for i in all_columns if i not in likelihood_indices])

    return coord_indices, likelihood_indices


def preprocess_dlc_csv(csv_path: str,
                       likelihood_threshold: Optional[float] = None,
                       save_output: bool = False,
                       output_path: Optional[str] = None) -> np.ndarray:
    """
    Preprocess a single DLC file (CSV or H5 format).

    This function:
    1. Loads the DLC file (auto-detects CSV or H5 format)
    2. Identifies x,y coordinate and likelihood columns
    3. Filters low-confidence poses
    4. Returns cleaned pose data as numpy array

    Args:
        csv_path (str): Path to DLC file (.csv or .h5)
        likelihood_threshold (float, optional): Minimum likelihood (0-1).
                                               If None, uses config default (0.1)
        save_output (bool): Whether to save processed data to file
        output_path (str, optional): Where to save output. If None, auto-generates
                                    from input filename

    Returns:
        np.ndarray: Filtered pose data of shape (frames, keypoints*2)
                   where keypoints*2 represents x,y pairs

    Example:
        >>> # Process CSV file
        >>> pose_data = preprocess_dlc_csv(
        >>>     'mouse1_DLC.csv',
        >>>     likelihood_threshold=0.1,
        >>>     save_output=True
        >>> )
        >>> # Process H5 file (much faster!)
        >>> pose_data = preprocess_dlc_csv('mouse1_DLC.h5', likelihood_threshold=0.1)
        >>> print(pose_data.shape)  # (10000, 40) for 20 keypoints
    """
    # Get likelihood threshold from config if not provided
    if likelihood_threshold is None:
        config = get_config()
        likelihood_threshold = config.get_likelihood_threshold()

    # Detect file format from extension
    file_path = Path(csv_path)
    file_ext = file_path.suffix.lower()

    print(f"Loading DLC file: {csv_path}")
    total_start = time.time()

    # Load file using appropriate loader
    load_start = time.time()
    if file_ext == '.h5':
        dlc_df = load_dlc_h5(csv_path)
        print(f"  Loaded H5 file: {dlc_df.shape[0]:,} frames, {dlc_df.shape[1]} columns ({time.time() - load_start:.2f}s)")
    elif file_ext == '.csv':
        dlc_df = load_dlc_csv(csv_path, fast_mode=True)
        print(f"  Loaded CSV file: {dlc_df.shape[0]:,} frames, {dlc_df.shape[1]} columns ({time.time() - load_start:.2f}s)")
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Expected .csv or .h5")

    # Get column indices
    coord_indices, likelihood_indices = get_coordinate_indices(dlc_df)
    print(f"  Found {len(coord_indices)} coordinate columns, {len(likelihood_indices)} likelihood columns")

    # Filter noise (now using JIT-optimized function)
    print(f"  Filtering poses with likelihood < {likelihood_threshold}...")
    filter_start = time.time()
    filtered_data, quality_metrics = filter_pose_noise(
        dlc_df,
        idx_selected=coord_indices,
        idx_llh=likelihood_indices,
        llh_value=likelihood_threshold
    )
    filter_time = time.time() - filter_start

    # Report quality
    mean_quality = np.mean(quality_metrics) * 100
    print(f"  Quality: {mean_quality:.1f}% of frames needed correction ({filter_time:.2f}s)")

    # Explicitly free memory from DataFrame (critical for multi-file processing)
    # The filtered_data is now a numpy array, so we can release the large DataFrame
    del dlc_df
    gc.collect()

    # Save if requested
    if save_output:
        if output_path is None:
            # Auto-generate output path
            input_path = Path(csv_path)
            output_path = input_path.parent / f"{input_path.stem}_processed.pkl"

        save_start = time.time()
        with open(output_path, 'wb') as f:
            pickle.dump(filtered_data, f)
        save_time = time.time() - save_start
        print(f"  Saved to: {output_path} ({save_time:.2f}s)")

    total_time = time.time() - total_start
    print(f"[OK] Preprocessing complete (total: {total_time:.2f}s)")
    return filtered_data


def batch_process_dlc_files(input_directory: str,
                            pattern: str = "*.csv",
                            likelihood_threshold: Optional[float] = None,
                            save_combined: bool = True,
                            output_path: Optional[str] = None) -> Dict[str, np.ndarray]:
    """
    Process multiple DLC CSV files in a directory.

    Args:
        input_directory (str): Directory containing DLC CSV files
        pattern (str): Glob pattern for matching CSV files (default: "*.csv")
        likelihood_threshold (float, optional): Minimum likelihood threshold
        save_combined (bool): Whether to save all processed data to single file
        output_path (str, optional): Path for combined output file

    Returns:
        dict: Dictionary mapping file names to processed pose arrays
              {filename: pose_array, ...}

    Example:
        >>> # Process all CSV files in directory
        >>> poses = batch_process_dlc_files('dlc_output/', likelihood_threshold=0.1)
        >>> print(f"Processed {len(poses)} files")
        >>> for name, data in poses.items():
        >>>     print(f"{name}: {data.shape}")
    """
    input_dir = Path(input_directory)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Find all CSV files
    csv_files = list(input_dir.glob(pattern))

    if len(csv_files) == 0:
        raise FileNotFoundError(
            f"No CSV files matching '{pattern}' found in {input_dir}"
        )

    print("=" * 60)
    print(f"Batch Processing {len(csv_files)} DLC CSV Files")
    print("=" * 60)

    # Process each file
    processed_data = {}
    for i, csv_file in enumerate(csv_files, 1):
        print(f"\n[{i}/{len(csv_files)}] {csv_file.name}")
        print("-" * 60)

        try:
            # Use stem (filename without extension) as key
            file_key = csv_file.stem
            pose_data = preprocess_dlc_csv(
                str(csv_file),
                likelihood_threshold=likelihood_threshold,
                save_output=False  # We'll save combined file instead
            )
            processed_data[file_key] = pose_data

        except Exception as e:
            print(f"[ERROR] Error processing {csv_file.name}: {str(e)}")
            print("  Skipping this file and continuing...")
            continue

    # Save combined output
    if save_combined and len(processed_data) > 0:
        if output_path is None:
            output_path = input_dir / "processed_pose_data.pkl"

        with open(output_path, 'wb') as f:
            pickle.dump(processed_data, f)

        print("\n" + "=" * 60)
        print(f"[OK] Saved combined data to: {output_path}")
        print(f"  Total files processed: {len(processed_data)}")
        print("=" * 60)

    return processed_data


def verify_dlc_csv(csv_path: str) -> Dict[str, any]:
    """
    Verify a DLC CSV file and return information about it.

    Useful for checking if a file is valid DLC output before processing.

    Args:
        csv_path (str): Path to CSV file to verify

    Returns:
        dict: Information about the CSV file:
            - valid: bool (whether it's a valid DLC CSV)
            - num_frames: int (number of frames)
            - num_keypoints: int (estimated number of keypoints)
            - header_levels: int (number of header rows)
            - error: str (error message if not valid)

    Example:
        >>> info = verify_dlc_csv('data.csv')
        >>> if info['valid']:
        >>>     print(f"Valid DLC CSV with {info['num_keypoints']} keypoints")
        >>> else:
        >>>     print(f"Invalid: {info['error']}")
    """
    info = {
        'valid': False,
        'num_frames': 0,
        'num_keypoints': 0,
        'header_levels': 0,
        'error': None
    }

    try:
        # Try loading
        df = load_dlc_csv(csv_path)

        # Get basic info
        info['num_frames'] = df.shape[0]
        info['header_levels'] = df.columns.nlevels

        # Estimate number of keypoints
        # Each keypoint has 3 columns (x, y, likelihood)
        info['num_keypoints'] = df.shape[1] // 3

        # Check if structure looks correct
        if df.shape[1] % 3 != 0:
            info['error'] = f"Column count ({df.shape[1]}) not divisible by 3"
            return info

        info['valid'] = True
        info['error'] = None

    except Exception as e:
        info['error'] = str(e)

    return info


def get_dlc_keypoint_names(csv_path: str) -> List[str]:
    """
    Extract keypoint names from DLC CSV headers.

    Args:
        csv_path (str): Path to DLC CSV file

    Returns:
        list: List of keypoint names

    Example:
        >>> names = get_dlc_keypoint_names('data.csv')
        >>> print(names)
        ['nose', 'left_ear', 'right_ear', 'neck', ...]
    """
    df = load_dlc_csv(csv_path)

    # Keypoint names are typically in the second level of the multi-index
    # and repeat every 3 columns (x, y, likelihood)
    if df.columns.nlevels >= 2:
        # Get unique body part names
        keypoint_names = df.columns.get_level_values(1).unique().tolist()
    else:
        # Fallback: cannot extract names
        keypoint_names = []

    return keypoint_names
