"""
LUPE-AMPS Analysis Module

This module implements the LUPE-AMPS (Pain Scale) analysis pipeline.
AMPS = Advanced Multivariate Pain Scale

The analysis consists of 4 main sections:
1. Preprocessing: Downsample and calculate behavior metrics
2. PCA Projection: Project onto pain scale (PC1 = behavior, PC2 = pain)
3. Metrics Visualization: Create plots showing behavior patterns
4. Model Fit Analysis: Test feature importance with 9 model variations

Usage:
    from src.core.analysis_lupe_amps import LupeAmpsAnalysis

    analysis = LupeAmpsAnalysis(model_path='models/model_AMPS.pkl')
    results = analysis.run_complete_analysis(csv_files, output_dir)
"""

import numpy as np
import pandas as pd

# NOTE: matplotlib.pyplot is imported inside methods that need it
# This ensures the backend is set before pyplot is loaded
# The entry point (main_lupe_amps_gui.py) must set matplotlib.use('Agg') first

import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.spatial.distance import cdist

from src.utils.lupe_amps_helpers import (
    validate_behavior_csv,
    downsample_sequence,
    calculate_bouts,
    calculate_bout_durations,
    safe_load_pca_model,
    calculate_transition_matrix,
    apply_model_condition,
    discover_behavior_files,
    extract_animal_date_from_path
)


class LupeAmpsAnalysis:
    """
    Main class for LUPE-AMPS pain scale analysis.

    This class handles the complete LUPE-AMPS analysis pipeline, from
    preprocessing behavior CSV files through PCA projection and visualization.

    Attributes:
        model_path (str): Path to the pre-trained PCA model
        num_behaviors (int): Number of behavior states (default: 6)
        original_fps (int): Original framerate of recordings
        target_fps (int): Target framerate after downsampling
        recording_length_min (int): Expected recording length in minutes
    """

    def __init__(self,
                 model_path: str = 'models/model_AMPS.pkl',
                 num_behaviors: int = 6,
                 target_fps: int = 20):
        """
        Initialize LUPE-AMPS analysis.

        Note: Recording length and original framerate are now read automatically
        from the summary CSV files generated during LUPE analysis. Each file can
        have different parameters.

        Args:
            model_path (str): Path to PCA model file
            num_behaviors (int): Number of behavior states (default: 6)
            target_fps (int): Target framerate after downsampling (default: 20)
        """
        self.model_path = model_path
        self.num_behaviors = num_behaviors
        self.target_fps = target_fps

        # PCA model (loaded when needed)
        self.pca_model = None

    def preprocess_single_file(self,
                               csv_path: str,
                               original_fps: float,
                               recording_length_min: float) -> Dict:
        """
        Preprocess a single behavior CSV file (Section 1).

        This function:
        1. Loads the CSV file
        2. Validates format
        3. Pads/truncates to expected length (based on recording_length_min and original_fps)
        4. Downsamples from original_fps to target_fps
        5. Calculates behavior metrics:
           - Fraction occupancy (% time in each state)
           - Number of bouts (count of continuous sequences)
           - Bout duration (mean seconds per bout)

        Args:
            csv_path (str): Path to behavior CSV file
            original_fps (float): Original recording framerate (read from summary CSV)
            recording_length_min (float): Recording duration in minutes (read from summary CSV)

        Returns:
            dict: Dictionary containing:
                - 'filename': Original filename
                - 'occupancy': Array of shape (num_behaviors,) with fraction occupancy
                - 'num_bouts': Array of shape (num_behaviors,) with bout counts
                - 'bout_duration': Array of shape (num_behaviors,) with mean durations
                - 'downsampled_sequence': Full downsampled behavior array

        Raises:
            ValueError: If CSV format is invalid
        """
        # Calculate expected frame counts based on parameters
        expected_frames_original = int(original_fps * recording_length_min * 60)
        expected_frames_downsampled = int(self.target_fps * recording_length_min * 60)

        # Validate CSV
        is_valid, error_msg = validate_behavior_csv(csv_path)
        if not is_valid:
            raise ValueError(f"Invalid CSV file: {error_msg}")

        # Load CSV
        df = pd.read_csv(csv_path)
        behavior_array = df['behavior_id'].to_numpy()

        # Pad or truncate to expected length
        if len(behavior_array) < expected_frames_original:
            # Pad with zeros
            behavior_array = np.pad(
                behavior_array,
                (0, expected_frames_original - len(behavior_array)),
                mode='constant',
                constant_values=0
            )
        else:
            # Truncate
            behavior_array = behavior_array[:expected_frames_original]

        # Downsample
        downsampled = downsample_sequence(
            behavior_array,
            original_fps,
            self.target_fps
        )

        # Ensure exact length after downsampling
        if len(downsampled) < expected_frames_downsampled:
            downsampled = np.pad(
                downsampled,
                (0, expected_frames_downsampled - len(downsampled)),
                mode='constant'
            )
        elif len(downsampled) > expected_frames_downsampled:
            downsampled = downsampled[:expected_frames_downsampled]

        # Calculate metrics for each behavior state
        occupancy = np.zeros(self.num_behaviors)
        num_bouts = np.zeros(self.num_behaviors)
        bout_duration = np.zeros(self.num_behaviors)

        for state_id in range(self.num_behaviors):
            # Fraction occupancy
            occupancy[state_id] = np.sum(downsampled == state_id) / len(downsampled)

            # Number of bouts
            num_bouts[state_id] = calculate_bouts(downsampled, state_id)

            # Bout duration
            durations = calculate_bout_durations(downsampled, state_id, self.target_fps)
            bout_duration[state_id] = np.mean(durations) if durations else 0.0

        return {
            'filename': Path(csv_path).name,
            'occupancy': occupancy,
            'num_bouts': num_bouts,
            'bout_duration': bout_duration,
            'downsampled_sequence': downsampled
        }

    def project_to_pain_scale(self, occupancy_data: np.ndarray) -> np.ndarray:
        """
        Project occupancy data onto the pain scale using PCA (Section 2).

        The LUPE-AMPS model transforms 6-dimensional behavior occupancy
        into a 2-dimensional space:
        - PC1 = "Generalized Behavior Scale"
        - PC2 = "Pain Behavior Scale"

        Args:
            occupancy_data (np.ndarray): Array of shape (n_files, num_behaviors)
                                        containing fraction occupancy values

        Returns:
            np.ndarray: Array of shape (n_files, 2) with PC1 and PC2 coordinates

        Raises:
            ValueError: If model cannot be loaded or data shape is incorrect
        """
        # Load model if not already loaded
        if self.pca_model is None:
            self.pca_model = safe_load_pca_model(self.model_path)

        # Validate input shape
        if occupancy_data.ndim == 1:
            # Single file, reshape
            occupancy_data = occupancy_data.reshape(1, -1)

        if occupancy_data.shape[1] != self.num_behaviors:
            raise ValueError(
                f"Expected {self.num_behaviors} behaviors, "
                f"got {occupancy_data.shape[1]}"
            )

        # Project using PCA
        projected = self.pca_model.transform(occupancy_data)

        # Return first 2 components
        return projected[:, :2]

    def create_pca_scatter_plot(self,
                               pc_coordinates: np.ndarray,
                               filenames: List[str],
                               output_path: str,
                               title: str = "LUPE-AMPS Pain Scale Projection"):
        """
        Create scatter plot of PCA projection (Section 2 visualization).

        Args:
            pc_coordinates (np.ndarray): Array of shape (n_files, 2) with PC1, PC2
            filenames (list): List of filenames for each point
            output_path (str): Path to save plot (without extension)
            title (str): Plot title
        """
        # Import pyplot here to ensure backend is set before use
        import matplotlib.pyplot as plt

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(10, 8))

        # Create scatter plot
        plt.scatter(pc_coordinates[:, 0], pc_coordinates[:, 1],
                   s=100, alpha=0.6, color='steelblue', edgecolors='black', linewidth=0.5)

        # Add labels for each point (optional, can be toggled)
        if len(filenames) <= 20:  # Only label if not too many points
            for i, filename in enumerate(filenames):
                plt.annotate(filename, (pc_coordinates[i, 0], pc_coordinates[i, 1]),
                           fontsize=8, alpha=0.7, xytext=(5, 5),
                           textcoords='offset points')

        plt.xlabel("PC1 (Generalized Behavior Scale)", fontsize=12)
        plt.ylabel("PC2 (Pain Behavior Scale)", fontsize=12, fontweight='bold')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save as PNG and SVG
        plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{output_path}.svg", format='svg', bbox_inches='tight')
        plt.close()

        # Also save coordinates to CSV
        df = pd.DataFrame({
            'filename': filenames,
            'PC1_Behavior_Scale': pc_coordinates[:, 0],
            'PC2_Pain_Scale': pc_coordinates[:, 1]
        })
        df.to_csv(f"{output_path}.csv", index=False)

    def create_metrics_plots(self,
                            all_metrics: List[Dict],
                            output_dir: str,
                            project_name: str = "LUPE-AMPS"):
        """
        Create visualization plots for behavior metrics (Section 3).

        Generates bar plots with mean ± SEM for:
        - Fraction occupancy
        - Number of bouts
        - Bout duration

        Args:
            all_metrics (list): List of metric dictionaries from preprocess_single_file()
            output_dir (str): Directory to save plots
            project_name (str): Project name for file naming
        """
        # Import pyplot here to ensure backend is set before use
        import matplotlib.pyplot as plt

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract metrics into arrays
        n_files = len(all_metrics)
        occupancy_all = np.array([m['occupancy'] for m in all_metrics])
        num_bouts_all = np.array([m['num_bouts'] for m in all_metrics])
        bout_duration_all = np.array([m['bout_duration'] for m in all_metrics])
        filenames = [m['filename'] for m in all_metrics]

        # State labels
        states = [f"State {i+1}" for i in range(self.num_behaviors)]

        # Calculate statistics (mean and SEM)
        if n_files > 1:
            occ_mean = np.mean(occupancy_all, axis=0)
            occ_sem = np.std(occupancy_all, axis=0, ddof=1) / np.sqrt(n_files)

            bouts_mean = np.mean(num_bouts_all, axis=0)
            bouts_sem = np.std(num_bouts_all, axis=0, ddof=1) / np.sqrt(n_files)

            dur_mean = np.mean(bout_duration_all, axis=0)
            dur_sem = np.std(bout_duration_all, axis=0, ddof=1) / np.sqrt(n_files)
        else:
            # Single file - no error bars
            occ_mean = occupancy_all[0]
            occ_sem = np.zeros(self.num_behaviors)

            bouts_mean = num_bouts_all[0]
            bouts_sem = np.zeros(self.num_behaviors)

            dur_mean = bout_duration_all[0]
            dur_sem = np.zeros(self.num_behaviors)

        # Plot 1: Fraction Occupancy
        fig, ax = plt.subplots(figsize=(10, 6))
        x_pos = np.arange(len(states))
        ax.bar(x_pos, occ_mean, yerr=occ_sem if n_files > 1 else None,
               capsize=5, color='skyblue', edgecolor='black', alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(states)
        ax.set_xlabel("Behavior State", fontsize=12)
        ax.set_ylabel("Fraction Occupancy", fontsize=12)
        ax.set_title(f"Fraction Occupancy Across States ({n_files} file(s))", fontsize=14)
        ax.set_ylim(0, max(occ_mean) * 1.2)
        plt.tight_layout()
        plt.savefig(output_dir / f"{project_name}_fraction_occupancy.png", dpi=300)
        plt.savefig(output_dir / f"{project_name}_fraction_occupancy.svg")
        plt.close()

        # Plot 2: Number of Bouts
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x_pos, bouts_mean, yerr=bouts_sem if n_files > 1 else None,
               capsize=5, color='coral', edgecolor='black', alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(states)
        ax.set_xlabel("Behavior State", fontsize=12)
        ax.set_ylabel("Number of Bouts", fontsize=12)
        ax.set_title(f"Number of Bouts per State ({n_files} file(s))", fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / f"{project_name}_number_of_bouts.png", dpi=300)
        plt.savefig(output_dir / f"{project_name}_number_of_bouts.svg")
        plt.close()

        # Plot 3: Bout Duration
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x_pos, dur_mean, yerr=dur_sem if n_files > 1 else None,
               capsize=5, color='lightgreen', edgecolor='black', alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(states)
        ax.set_xlabel("Behavior State", fontsize=12)
        ax.set_ylabel("Mean Bout Duration (seconds)", fontsize=12)
        ax.set_title(f"Bout Duration per State ({n_files} file(s))", fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / f"{project_name}_bout_duration.png", dpi=300)
        plt.savefig(output_dir / f"{project_name}_bout_duration.svg")
        plt.close()

        # Save CSVs
        df_occ = pd.DataFrame(occupancy_all, columns=states)
        df_occ.insert(0, 'filename', filenames)
        df_occ.to_csv(output_dir / f"{project_name}_fraction_occupancy.csv", index=False)

        df_bouts = pd.DataFrame(num_bouts_all, columns=states)
        df_bouts.insert(0, 'filename', filenames)
        df_bouts.to_csv(output_dir / f"{project_name}_number_of_bouts.csv", index=False)

        df_dur = pd.DataFrame(bout_duration_all, columns=states)
        df_dur.insert(0, 'filename', filenames)
        df_dur.to_csv(output_dir / f"{project_name}_bout_duration.csv", index=False)

    def analyze_model_fit(self,
                         behavior_sequences: List[np.ndarray],
                         filenames: List[str],
                         output_dir: str,
                         project_name: str = "LUPE-AMPS",
                         window_size_sec: int = 30,
                         window_slide_sec: int = 10,
                         n_permutations: int = 100):
        """
        Analyze model fit with feature ablation using transition matrices (Section 4).

        This analysis tests 9 model variations to understand which behavioral features
        are important for characterizing behavior patterns. It uses transition matrices
        to capture behavior dynamics and measures model fit using Euclidean distance
        to a reference centroid.

        The 9 model conditions are:
            1. Full model - all transition features included
            2. No self-transitions - diagonal of transition matrix zeroed
            3-8. Removal of each behavior (states 0-5) - row and column zeroed
            9. Shuffled control - permutation baseline for statistical comparison

        Algorithm:
            1. Segment each behavior sequence into overlapping windows
            2. Calculate transition matrix for each window (6x6, normalized)
            3. Flatten matrices to 36-dimensional vectors
            4. Compute centroid (mean vector) across all windows
            5. For each condition, apply feature ablation and calculate distance to centroid
            6. Lower distance = better model fit (features are important)

        Args:
            behavior_sequences (list): List of downsampled behavior arrays (at target_fps)
            filenames (list): Corresponding filenames for labeling
            output_dir (str): Directory to save outputs
            project_name (str): Project name for file naming (default: "LUPE-AMPS")
            window_size_sec (int): Window size in seconds (default: 30)
            window_slide_sec (int): Sliding step in seconds (default: 10)
            n_permutations (int): Number of permutations for condition 9 (default: 100)

        Outputs:
            - {project_name}_feature_importance.png: Line plot of model fit scores
            - {project_name}_feature_importance.svg: Vector version for publication
            - {project_name}_feature_importance.csv: Raw scores per file and condition
        """
        # Import pyplot here to ensure backend is set before use
        import matplotlib.pyplot as plt

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        n_files = len(behavior_sequences)

        # Calculate window parameters in frames
        window_size_frames = self.target_fps * window_size_sec
        window_slide_frames = self.target_fps * window_slide_sec

        # Model condition names (matching reference implementation)
        model_conditions = [
            "Full Model",
            "No Self-Transition",
            "No State 1",
            "No State 2",
            "No State 3",
            "No State 4",
            "No State 5",
            "No State 6",
            "Shuffled (Control)"
        ]

        # Step 1: Calculate transition matrices for all windows across all files
        # Each matrix is flattened to a 36-dimensional vector (6x6)
        all_window_vectors = []
        window_file_indices = []  # Track which file each window came from

        for file_idx, seq in enumerate(behavior_sequences):
            # Determine number of windows for this sequence
            seq_length = len(seq)
            if seq_length < window_size_frames:
                # Sequence too short, use entire sequence as one window
                n_windows = 1
            else:
                n_windows = (seq_length - window_size_frames) // window_slide_frames + 1

            for win_idx in range(n_windows):
                start = win_idx * window_slide_frames
                end = min(start + window_size_frames, seq_length)
                window = seq[start:end]

                # Calculate transition matrix for this window
                trans_matrix = calculate_transition_matrix(window, self.num_behaviors)

                # Flatten to vector (row-major order)
                vector = trans_matrix.flatten()
                all_window_vectors.append(vector)
                window_file_indices.append(file_idx)

        # Convert to numpy array: shape (n_total_windows, 36)
        all_window_vectors = np.array(all_window_vectors)
        window_file_indices = np.array(window_file_indices)

        # Step 2: Calculate centroid (mean vector across all windows)
        # This represents the "average" transition pattern
        centroid = np.mean(all_window_vectors, axis=0)

        # Step 3: For each condition, calculate mean distance to centroid per file
        # Lower distance = better fit = feature is NOT important
        # Higher distance = worse fit = feature IS important for the model
        model_fit_scores = np.zeros((n_files, len(model_conditions)))

        for cond_idx, cond_name in enumerate(model_conditions):
            condition_num = cond_idx + 1  # Conditions are 1-indexed

            if condition_num == 9:
                # Condition 9: Shuffled control (permutation test)
                # Shuffle behavior labels and recalculate distances
                shuffled_distances = []

                for perm in range(n_permutations):
                    perm_vectors = []

                    for file_idx, seq in enumerate(behavior_sequences):
                        # Shuffle the behavior labels within this sequence
                        shuffled_seq = seq.copy()
                        np.random.shuffle(shuffled_seq)

                        # Calculate windows for shuffled sequence
                        seq_length = len(shuffled_seq)
                        if seq_length < window_size_frames:
                            n_windows = 1
                        else:
                            n_windows = (seq_length - window_size_frames) // window_slide_frames + 1

                        for win_idx in range(n_windows):
                            start = win_idx * window_slide_frames
                            end = min(start + window_size_frames, seq_length)
                            window = shuffled_seq[start:end]

                            trans_matrix = calculate_transition_matrix(window, self.num_behaviors)
                            perm_vectors.append(trans_matrix.flatten())

                    perm_vectors = np.array(perm_vectors)
                    perm_centroid = np.mean(perm_vectors, axis=0)

                    # Calculate mean distance for this permutation
                    distances = np.linalg.norm(perm_vectors - perm_centroid, axis=1)
                    shuffled_distances.append(np.mean(distances))

                # Use mean of shuffled distances as the control score
                mean_shuffled_distance = np.mean(shuffled_distances)

                # Assign same control distance to all files (it's a global baseline)
                for file_idx in range(n_files):
                    model_fit_scores[file_idx, cond_idx] = mean_shuffled_distance

            else:
                # Conditions 1-8: Apply feature ablation and calculate distances
                for file_idx in range(n_files):
                    # Get windows for this file
                    file_mask = window_file_indices == file_idx
                    file_vectors = all_window_vectors[file_mask]

                    if len(file_vectors) == 0:
                        model_fit_scores[file_idx, cond_idx] = np.nan
                        continue

                    # Apply condition to each window vector
                    # First reshape back to matrix, apply condition, then flatten
                    modified_vectors = []
                    for vec in file_vectors:
                        matrix = vec.reshape(self.num_behaviors, self.num_behaviors)
                        modified_matrix = apply_model_condition(matrix, condition_num)
                        modified_vectors.append(modified_matrix.flatten())

                    modified_vectors = np.array(modified_vectors)

                    # Also apply condition to centroid
                    centroid_matrix = centroid.reshape(self.num_behaviors, self.num_behaviors)
                    modified_centroid = apply_model_condition(centroid_matrix, condition_num).flatten()

                    # Calculate Euclidean distances to modified centroid
                    distances = np.linalg.norm(modified_vectors - modified_centroid, axis=1)

                    # Store mean distance for this file
                    model_fit_scores[file_idx, cond_idx] = np.mean(distances)

        # Step 4: Create visualization
        fig, ax = plt.subplots(figsize=(12, 6))

        # Calculate overall mean and SEM across files
        mean_scores = np.nanmean(model_fit_scores, axis=0)
        if n_files > 1:
            sem_scores = np.nanstd(model_fit_scores, axis=0, ddof=1) / np.sqrt(n_files)
        else:
            sem_scores = np.zeros(len(model_conditions))

        x_pos = np.arange(len(model_conditions))

        # Plot line with markers
        ax.plot(x_pos, mean_scores, marker='o', linestyle='-', linewidth=2,
               markersize=8, color='steelblue', label='Mean Distance to Centroid')

        # Add error band (SEM)
        if n_files > 1:
            ax.fill_between(x_pos, mean_scores - sem_scores, mean_scores + sem_scores,
                           alpha=0.3, color='steelblue')

        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_conditions, rotation=45, ha='right')
        ax.set_ylabel("Mean Euclidean Distance to Centroid", fontsize=12)
        ax.set_xlabel("Model Condition", fontsize=12)
        ax.set_title(f"Feature Importance Analysis ({n_files} file(s))", fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, linestyle='--')

        # Add interpretation note
        ax.text(0.02, 0.98, "Higher distance = feature removal disrupts model\n(feature is important)",
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        # Save plots
        plt.savefig(output_dir / f"{project_name}_feature_importance.png", dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / f"{project_name}_feature_importance.svg", format='svg', bbox_inches='tight')
        plt.close()

        # Step 5: Save CSV with all results
        df = pd.DataFrame(model_fit_scores, columns=model_conditions)
        df.insert(0, 'filename', filenames)
        df.to_csv(output_dir / f"{project_name}_feature_importance.csv", index=False)

        # Also save summary statistics
        summary_data = {
            'Condition': model_conditions,
            'Mean_Distance': mean_scores,
            'SEM': sem_scores if n_files > 1 else [0] * len(model_conditions),
            'N_Files': [n_files] * len(model_conditions)
        }
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_csv(output_dir / f"{project_name}_feature_importance_summary.csv", index=False)

    def run_complete_analysis(self,
                             csv_files: List[str],
                             output_base_dir: str,
                             project_name: str = "LUPE-AMPS",
                             sections: List[int] = [1, 2, 3, 4],
                             progress_callback=None) -> Dict:
        """
        Run the complete LUPE-AMPS analysis pipeline.

        Args:
            csv_files (list): List of behavior CSV file paths
            output_base_dir (str): Base output directory
            project_name (str): Project name for file naming
            sections (list): Which sections to run (1, 2, 3, 4)
            progress_callback (callable): Optional callback function(message) for progress updates

        Returns:
            dict: Results dictionary with paths to generated files
        """
        output_base = Path(output_base_dir)
        results = {}

        def log(message):
            if progress_callback:
                progress_callback(message)
            print(message)

        # Section 1: Preprocessing
        if 1 in sections:
            log("=" * 60)
            log("Section 1: Preprocessing and Metrics Calculation")
            log("=" * 60)

            # Import summary reader
            from src.utils.amps_summary_reader import read_parameters_from_summary

            all_metrics = []
            for i, csv_file in enumerate(csv_files, 1):
                log(f"  [{i}/{len(csv_files)}] Processing {Path(csv_file).name}...")
                try:
                    # Read parameters from summary CSV
                    log(f"    Reading parameters from summary CSV...")
                    params = read_parameters_from_summary(csv_file)

                    # Log detected parameters
                    log(f"    Recording Length: {params['recording_length_min']:.1f} min")
                    log(f"    Original Framerate: {params['original_fps']:.0f} fps")
                    log(f"    Total Frames: {params['total_frames']:,}")
                    log(f"    Target Framerate: {self.target_fps} fps")

                    # Preprocess with detected parameters
                    metrics = self.preprocess_single_file(
                        csv_file,
                        original_fps=params['original_fps'],
                        recording_length_min=params['recording_length_min']
                    )
                    all_metrics.append(metrics)
                    log(f"    Complete")
                except FileNotFoundError as e:
                    log(f"    Error: {str(e)}")
                    log(f"    Please ensure LUPE analysis was run to generate summary CSV.")
                    continue
                except Exception as e:
                    log(f"    Error: {str(e)}")
                    continue

            results['section1'] = all_metrics

            # Save summary CSV
            section1_dir = output_base / f"{project_name}_LUPE-AMPS" / "Section1_preprocessing"
            section1_dir.mkdir(parents=True, exist_ok=True)

            summary_data = []
            for m in all_metrics:
                row = {'filename': m['filename']}
                for i in range(self.num_behaviors):
                    row[f'state{i+1}_occupancy'] = m['occupancy'][i]
                    row[f'state{i+1}_num_bouts'] = m['num_bouts'][i]
                    row[f'state{i+1}_bout_duration'] = m['bout_duration'][i]
                summary_data.append(row)

            df_summary = pd.DataFrame(summary_data)
            summary_path = section1_dir / "metrics_all_files.csv"
            df_summary.to_csv(summary_path, index=False)
            log(f"[OK] Section 1 complete. Saved: {summary_path}")
            results['section1_csv'] = str(summary_path)

        # Section 2: PCA Projection
        if 2 in sections and 1 in sections:
            log("\n" + "=" * 60)
            log("Section 2: PCA Pain Scale Projection")
            log("=" * 60)

            occupancy_array = np.array([m['occupancy'] for m in all_metrics])
            filenames = [m['filename'] for m in all_metrics]

            log("  Projecting to pain scale...")
            pc_coordinates = self.project_to_pain_scale(occupancy_array)
            log("  [OK] Projection complete")

            section2_dir = output_base / f"{project_name}_LUPE-AMPS" / "Section2_pain_scale"
            section2_dir.mkdir(parents=True, exist_ok=True)

            output_path = section2_dir / "pain_scale_projection"
            self.create_pca_scatter_plot(pc_coordinates, filenames, str(output_path))
            log(f"[OK] Section 2 complete. Saved: {output_path}.png/svg/csv")
            results['section2_plot'] = str(output_path)

        # Section 3: Metrics Visualization
        if 3 in sections and 1 in sections:
            log("\n" + "=" * 60)
            log("Section 3: Behavior Metrics Visualization")
            log("=" * 60)

            section3_dir = output_base / f"{project_name}_LUPE-AMPS" / "Section3_behavior_metrics"
            self.create_metrics_plots(all_metrics, str(section3_dir), project_name)
            log(f"[OK] Section 3 complete. Saved: {section3_dir}/")
            results['section3_dir'] = str(section3_dir)

        # Section 4: Model Fit Analysis
        if 4 in sections and 1 in sections:
            log("\n" + "=" * 60)
            log("Section 4: Model Feature Importance Analysis")
            log("=" * 60)

            sequences = [m['downsampled_sequence'] for m in all_metrics]
            filenames = [m['filename'] for m in all_metrics]

            section4_dir = output_base / f"{project_name}_LUPE-AMPS" / "Section4_model_fit"
            self.analyze_model_fit(sequences, filenames, str(section4_dir), project_name)
            log(f"[OK] Section 4 complete. Saved: {section4_dir}/")
            results['section4_dir'] = str(section4_dir)

        log("\n" + "=" * 60)
        log("LUPE-AMPS Analysis Complete!")
        log(f"All results saved to: {output_base / f'{project_name}_LUPE-AMPS'}")
        log("=" * 60)

        return results

    def run_single_file_analysis(self,
                                 csv_file: str,
                                 output_dir: str,
                                 sections: List[int] = [1, 2, 3, 4],
                                 progress_callback=None) -> Dict:
        """
        Run LUPE-AMPS analysis on a single file without aggregation.

        This method processes a single behavior CSV file and saves all results
        to the specified output directory. Unlike run_complete_analysis(), this
        does not aggregate across multiple files.

        Args:
            csv_file (str): Path to the behavior CSV file
            output_dir (str): Directory to save outputs (will be created if needed)
            sections (list): Which sections to run (1, 2, 3, 4)
            progress_callback (callable): Optional callback function(message) for progress

        Returns:
            dict: Results dictionary containing:
                - 'filename': Name of processed file
                - 'metrics': Preprocessing metrics (occupancy, bouts, duration)
                - 'pc_coordinates': PCA coordinates (PC1, PC2)
                - 'output_files': List of generated output file paths
                - 'success': Boolean indicating if analysis completed

        Example:
            >>> analysis = LupeAmpsAnalysis()
            >>> results = analysis.run_single_file_analysis(
            ...     'data/mouse_behaviors.csv',
            ...     'outputs/Mouse01/2024-01-15/'
            ... )
        """
        # Import pyplot here to ensure backend is set before use
        import matplotlib.pyplot as plt

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = {
            'filename': Path(csv_file).name,
            'success': False,
            'output_files': []
        }

        def log(message):
            if progress_callback:
                progress_callback(message)
            print(message)

        # Import summary reader
        from src.utils.amps_summary_reader import read_parameters_from_summary

        # Section 1: Preprocessing
        try:
            log(f"Processing: {Path(csv_file).name}")

            # Read parameters from summary CSV
            params = read_parameters_from_summary(csv_file)
            log(f"  Recording: {params['recording_length_min']:.1f} min @ {params['original_fps']:.0f} fps")

            # Preprocess
            metrics = self.preprocess_single_file(
                csv_file,
                original_fps=params['original_fps'],
                recording_length_min=params['recording_length_min']
            )
            results['metrics'] = metrics

            if 1 in sections:
                # Save metrics CSV
                metrics_data = {
                    'filename': [metrics['filename']],
                }
                for i in range(self.num_behaviors):
                    metrics_data[f'state{i+1}_occupancy'] = [metrics['occupancy'][i]]
                    metrics_data[f'state{i+1}_num_bouts'] = [metrics['num_bouts'][i]]
                    metrics_data[f'state{i+1}_bout_duration'] = [metrics['bout_duration'][i]]

                df_metrics = pd.DataFrame(metrics_data)
                metrics_path = output_path / "metrics.csv"
                df_metrics.to_csv(metrics_path, index=False)
                results['output_files'].append(str(metrics_path))
                log(f"  [OK] Metrics saved: {metrics_path.name}")

        except Exception as e:
            log(f"  Error in preprocessing: {str(e)}")
            return results

        # Section 2: PCA Projection
        if 2 in sections:
            try:
                occupancy_array = np.array([metrics['occupancy']])
                pc_coordinates = self.project_to_pain_scale(occupancy_array)
                results['pc_coordinates'] = {
                    'PC1': float(pc_coordinates[0, 0]),
                    'PC2': float(pc_coordinates[0, 1])
                }

                # Save PCA results
                pca_data = {
                    'filename': [metrics['filename']],
                    'PC1_Behavior_Scale': [pc_coordinates[0, 0]],
                    'PC2_Pain_Scale': [pc_coordinates[0, 1]]
                }
                df_pca = pd.DataFrame(pca_data)
                pca_csv_path = output_path / "pain_scale.csv"
                df_pca.to_csv(pca_csv_path, index=False)
                results['output_files'].append(str(pca_csv_path))

                # Create simple scatter plot (single point)
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(pc_coordinates[0, 0], pc_coordinates[0, 1],
                          s=100, c='steelblue', alpha=0.8, edgecolors='black')
                ax.set_xlabel("PC1 - Behavior Scale", fontsize=12)
                ax.set_ylabel("PC2 - Pain Scale", fontsize=12)
                ax.set_title(f"Pain Scale Projection\n{metrics['filename']}", fontsize=14)
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()

                pca_png_path = output_path / "pain_scale.png"
                pca_svg_path = output_path / "pain_scale.svg"
                plt.savefig(pca_png_path, dpi=300, bbox_inches='tight')
                plt.savefig(pca_svg_path, format='svg', bbox_inches='tight')
                plt.close()

                results['output_files'].extend([str(pca_png_path), str(pca_svg_path)])
                log(f"  [OK] Pain scale projection saved")

            except Exception as e:
                log(f"  Error in PCA projection: {str(e)}")

        # Section 3: Behavior Metrics Plots
        if 3 in sections:
            try:
                states = [f"State {i+1}" for i in range(self.num_behaviors)]

                # Fraction Occupancy - simple bar plot
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(states, metrics['occupancy'], color='steelblue', alpha=0.8, edgecolor='black')
                ax.set_ylabel("Fraction Occupancy", fontsize=12)
                ax.set_xlabel("Behavior State", fontsize=12)
                ax.set_title(f"Fraction Occupancy\n{metrics['filename']}", fontsize=14)
                ax.set_ylim(0, max(0.5, max(metrics['occupancy']) * 1.1))
                plt.tight_layout()

                occ_png = output_path / "fraction_occupancy.png"
                occ_svg = output_path / "fraction_occupancy.svg"
                occ_csv = output_path / "fraction_occupancy.csv"
                plt.savefig(occ_png, dpi=300, bbox_inches='tight')
                plt.savefig(occ_svg, format='svg', bbox_inches='tight')
                plt.close()

                df_occ = pd.DataFrame({'state': states, 'occupancy': metrics['occupancy']})
                df_occ.to_csv(occ_csv, index=False)
                results['output_files'].extend([str(occ_png), str(occ_svg), str(occ_csv)])

                # Number of Bouts - simple bar plot
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(states, metrics['num_bouts'], color='coral', alpha=0.8, edgecolor='black')
                ax.set_ylabel("Number of Bouts", fontsize=12)
                ax.set_xlabel("Behavior State", fontsize=12)
                ax.set_title(f"Number of Bouts\n{metrics['filename']}", fontsize=14)
                plt.tight_layout()

                bouts_png = output_path / "number_of_bouts.png"
                bouts_svg = output_path / "number_of_bouts.svg"
                bouts_csv = output_path / "number_of_bouts.csv"
                plt.savefig(bouts_png, dpi=300, bbox_inches='tight')
                plt.savefig(bouts_svg, format='svg', bbox_inches='tight')
                plt.close()

                df_bouts = pd.DataFrame({'state': states, 'num_bouts': metrics['num_bouts']})
                df_bouts.to_csv(bouts_csv, index=False)
                results['output_files'].extend([str(bouts_png), str(bouts_svg), str(bouts_csv)])

                # Bout Duration - simple bar plot
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(states, metrics['bout_duration'], color='seagreen', alpha=0.8, edgecolor='black')
                ax.set_ylabel("Mean Bout Duration (s)", fontsize=12)
                ax.set_xlabel("Behavior State", fontsize=12)
                ax.set_title(f"Mean Bout Duration\n{metrics['filename']}", fontsize=14)
                plt.tight_layout()

                dur_png = output_path / "bout_duration.png"
                dur_svg = output_path / "bout_duration.svg"
                dur_csv = output_path / "bout_duration.csv"
                plt.savefig(dur_png, dpi=300, bbox_inches='tight')
                plt.savefig(dur_svg, format='svg', bbox_inches='tight')
                plt.close()

                df_dur = pd.DataFrame({'state': states, 'bout_duration': metrics['bout_duration']})
                df_dur.to_csv(dur_csv, index=False)
                results['output_files'].extend([str(dur_png), str(dur_svg), str(dur_csv)])

                log(f"  [OK] Behavior metrics plots saved")

            except Exception as e:
                log(f"  Error in metrics visualization: {str(e)}")

        # Section 4: Feature Importance
        if 4 in sections:
            try:
                sequences = [metrics['downsampled_sequence']]
                filenames = [metrics['filename']]
                self.analyze_model_fit(sequences, filenames, str(output_path), "AMPS")
                results['output_files'].extend([
                    str(output_path / "AMPS_feature_importance.png"),
                    str(output_path / "AMPS_feature_importance.svg"),
                    str(output_path / "AMPS_feature_importance.csv")
                ])
                log(f"  [OK] Feature importance analysis saved")

            except Exception as e:
                log(f"  Error in feature importance: {str(e)}")

        results['success'] = True
        log(f"  Complete: {len(results['output_files'])} files generated")

        return results

    def run_batch_analysis(self,
                          input_folder: str,
                          output_folder: str,
                          sections: List[int] = [1, 2, 3, 4],
                          progress_callback=None) -> Dict:
        """
        Run LUPE-AMPS analysis on all behavior files in a folder.

        This method automatically discovers all behavior CSV files in the input
        folder, extracts animal and date information from the folder structure,
        and saves results organized by animal/date.

        Expected input folder structure:
            input_folder/
            ├── Animal01/
            │   ├── 2024-01-15/
            │   │   ├── recording_behaviors.csv
            │   │   └── recording_summary.csv
            │   └── 2024-01-20/
            │       └── ...

        Output folder structure:
            output_folder/
            ├── Animal01/
            │   ├── 2024-01-15/
            │   │   ├── metrics.csv
            │   │   ├── pain_scale.csv/png/svg
            │   │   └── ...
            │   └── 2024-01-20/
            │       └── ...

        Args:
            input_folder (str): Root folder containing behavior files
            output_folder (str): Root folder for outputs (organized by animal/date)
            sections (list): Which sections to run (1, 2, 3, 4)
            progress_callback (callable): Optional callback function(message) for progress

        Returns:
            dict: Summary results containing:
                - 'total_files': Number of files found
                - 'processed': Number successfully processed
                - 'failed': Number that failed
                - 'skipped': Number skipped (missing summary)
                - 'results': List of individual file results

        Example:
            >>> analysis = LupeAmpsAnalysis()
            >>> results = analysis.run_batch_analysis(
            ...     'C:/data/experiment1/',
            ...     'C:/outputs/experiment1/'
            ... )
            >>> print(f"Processed {results['processed']} of {results['total_files']} files")
        """
        def log(message):
            if progress_callback:
                progress_callback(message)
            print(message)

        log("=" * 60)
        log("LUPE-AMPS Batch Analysis")
        log("=" * 60)
        log(f"Input folder: {input_folder}")
        log(f"Output folder: {output_folder}")
        log("")

        # Discover all behavior files
        log("Discovering behavior files...")
        try:
            files = discover_behavior_files(input_folder, check_summary=True)
        except Exception as e:
            log(f"Error discovering files: {str(e)}")
            return {
                'total_files': 0,
                'processed': 0,
                'failed': 0,
                'skipped': 0,
                'results': [],
                'error': str(e)
            }

        if len(files) == 0:
            log("No behavior files found. Please check:")
            log("  1. Files are named *_behaviors.csv")
            log("  2. Each file has a corresponding *_summary.csv")
            log("  3. Folder structure is: Animal/Date/file_behaviors.csv")
            return {
                'total_files': 0,
                'processed': 0,
                'failed': 0,
                'skipped': 0,
                'results': []
            }

        log(f"\nFound {len(files)} behavior files:")
        for f in files:
            log(f"  {f['animal']}/{f['date']}: {f['filename']}")

        log("\n" + "-" * 60)
        log("Processing files...")
        log("-" * 60)

        # Process each file
        batch_results = {
            'total_files': len(files),
            'processed': 0,
            'failed': 0,
            'skipped': 0,
            'results': []
        }

        for i, file_info in enumerate(files, 1):
            log(f"\n[{i}/{len(files)}] {file_info['animal']}/{file_info['date']}/{file_info['filename']}")

            # Create output directory: output_folder/animal/date/
            file_output_dir = Path(output_folder) / file_info['animal'] / file_info['date']

            try:
                # Run single file analysis
                result = self.run_single_file_analysis(
                    csv_file=file_info['path'],
                    output_dir=str(file_output_dir),
                    sections=sections,
                    progress_callback=None  # Use simple logging for batch
                )

                result['animal'] = file_info['animal']
                result['date'] = file_info['date']
                batch_results['results'].append(result)

                if result['success']:
                    batch_results['processed'] += 1
                else:
                    batch_results['failed'] += 1

            except Exception as e:
                log(f"  ERROR: {str(e)}")
                batch_results['failed'] += 1
                batch_results['results'].append({
                    'filename': file_info['filename'],
                    'animal': file_info['animal'],
                    'date': file_info['date'],
                    'success': False,
                    'error': str(e)
                })

        # Print summary
        log("\n" + "=" * 60)
        log("Batch Analysis Complete!")
        log("=" * 60)
        log(f"Total files found: {batch_results['total_files']}")
        log(f"Successfully processed: {batch_results['processed']}")
        log(f"Failed: {batch_results['failed']}")
        log(f"Output location: {output_folder}")
        log("=" * 60)

        return batch_results
