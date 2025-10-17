"""
Main GUI Window for LUPE Analysis Tool

This module provides a graphical user interface for LUPE analysis using tkinter.
The GUI allows users to:
- Load DeepLabCut CSV files
- Classify behaviors using a pre-trained A-SOiD model
- Select which analyses to run
- Configure output settings
- Run the complete analysis pipeline with a single click

Output Structure:
    For each input CSV file (e.g., "mouse01DLC_resnet50.csv"), the GUI creates:

    outputs/
    └── mouse01/                          # Partial name extracted before "DLC"
        ├── mouse01_behaviors.csv         # Frame-by-frame behavior classifications
        ├── mouse01_time.csv              # Time vector for each frame
        ├── mouse01_summary.csv           # Recording metadata and behavior statistics
        └── mouse01_analysis/             # Analysis results subfolder
            ├── mouse01_ANALYSIS_SUMMARY.csv
            ├── mouse01_bout_counts_summary.csv
            ├── mouse01_bout_counts.svg
            ├── mouse01_time_distribution_overall.csv
            ├── mouse01_time_distribution.svg
            ├── mouse01_bout_durations_statistics.csv
            ├── mouse01_bout_durations_raw.csv
            ├── mouse01_bout_durations.svg
            ├── mouse01_timeline_1.0min.csv
            ├── mouse01_timeline_1.0min.svg
            ├── mouse01_transitions_matrix.csv
            └── mouse01_transitions_heatmap.svg

Usage:
    from src.gui.main_window import LupeGUI

    app = LupeGUI()
    app.run()
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
from pathlib import Path
import gc

import numpy as np
import pandas as pd
from src.core.data_loader import load_model
from src.core.classification import classify_behaviors
from src.core.dlc_preprocessing import preprocess_dlc_csv
from src.core.analysis_csv_export import export_behaviors_to_csv, export_behavior_summary
from src.core.analysis_instance_counts import analyze_instance_counts
from src.core.analysis_total_frames import analyze_total_frames
from src.core.analysis_durations import analyze_bout_durations
from src.core.analysis_binned_timeline import analyze_binned_timeline
from src.core.analysis_transitions import analyze_transitions
from src.core.file_summary import generate_dlc_summary
from src.utils.config_manager import get_config
from src.utils.master_summary import create_analysis_summary
from src.utils.filename_utils import extract_partial_filename


class LupeGUI:
    """
    Main GUI application for LUPE Analysis Tool.

    This class creates and manages the graphical user interface for running
    the complete LUPE analysis pipeline: from raw DeepLabCut CSV files through
    behavior classification to final analysis outputs.
    """

    def __init__(self):
        """Initialize the GUI application."""
        self.root = tk.Tk()
        self.root.title("LUPE Analysis Tool")
        self.root.geometry("900x560")

        # Application state
        self.dlc_csv_paths = []
        self.model_path = None
        self.behaviors_data = None
        self.output_dir = None

        # Create GUI components
        self._create_widgets()

        # Initialize default model path
        self._initialize_default_model()

        # Load configuration
        self.config = get_config()

    def _create_widgets(self):
        """Create all GUI widgets."""
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights for resizing
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)  # Left column
        main_frame.columnconfigure(1, weight=2)  # Right column (larger for log)

        # ========== Title ==========
        title_label = ttk.Label(
            main_frame,
            text="Process DLC (.csv) files through LUPE A-SOiD model",
            font=('Arial', 16, 'bold')
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=10)

        # ========== File Selection Section (Left Column) ==========
        file_frame = ttk.LabelFrame(main_frame, text="Data Files", padding="10")
        file_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5, padx=(0, 5))

        # Configure column weights to prevent button cutoff
        file_frame.columnconfigure(1, weight=1)

        # DLC CSV files
        ttk.Label(file_frame, text="DLC CSV Files:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.dlc_path_var = tk.StringVar(master=self.root, value="No files selected")
        ttk.Entry(file_frame, textvariable=self.dlc_path_var, width=30).grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(file_frame, text="Browse...", command=self._select_dlc_files).grid(row=0, column=2, sticky=tk.W, padx=(5, 0))

        # Model file
        ttk.Label(file_frame, text="Model File:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.model_path_var = tk.StringVar(master=self.root, value="models/model_LUPE.pkl")
        ttk.Entry(file_frame, textvariable=self.model_path_var, width=30).grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(file_frame, text="Browse...", command=self._select_model_file).grid(row=1, column=2, sticky=tk.W, padx=(5, 0))

        # Likelihood threshold
        ttk.Label(file_frame, text="Likelihood Threshold:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.llh_var = tk.DoubleVar(master=self.root, value=0.1)
        ttk.Spinbox(file_frame, from_=0.0, to=1.0, increment=0.05,
                   textvariable=self.llh_var, width=10).grid(row=2, column=1, sticky=tk.W, padx=5)

        # Output directory
        ttk.Label(file_frame, text="Output Directory:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.output_dir_var = tk.StringVar(master=self.root, value="outputs/")
        ttk.Entry(file_frame, textvariable=self.output_dir_var, width=30).grid(row=3, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(file_frame, text="Browse...", command=self._select_output_dir).grid(row=3, column=2, sticky=tk.W, padx=(5, 0))

        # ========== Analysis Selection Section (Left Column) ==========
        analysis_frame = ttk.LabelFrame(main_frame, text="Select Analyses", padding="10")
        analysis_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5, padx=(0, 5))

        # Analysis checkboxes
        self.analyses = {}
        analyses_list = [
            ("csv_export", "Export to CSV"),
            ("instance_counts", "Behavior Instance Counts"),
            ("total_frames", "Total Frames (Pie Charts)"),
            ("durations", "Bout Durations"),
            ("timeline", "Binned Timeline"),
            ("transitions", "Behavior Transitions")
        ]

        for i, (key, label) in enumerate(analyses_list):
            var = tk.BooleanVar(master=self.root, value=True)
            self.analyses[key] = var
            ttk.Checkbutton(analysis_frame, text=label, variable=var).grid(
                row=i // 2, column=i % 2, sticky=tk.W, padx=10, pady=2
            )

        # Select All / Deselect All buttons
        button_frame = ttk.Frame(analysis_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=10)
        ttk.Button(button_frame, text="Select All", command=self._select_all_analyses).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Deselect All", command=self._deselect_all_analyses).pack(side=tk.LEFT, padx=5)

        # ========== Options Section (Left Column) ==========
        options_frame = ttk.LabelFrame(main_frame, text="Options", padding="10")
        options_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=5, padx=(0, 5))

        ttk.Label(options_frame, text="Timeline Bin Size (minutes):").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.bin_minutes_var = tk.DoubleVar(master=self.root, value=1.0)
        ttk.Spinbox(options_frame, from_=0.5, to=10.0, increment=0.5,
                   textvariable=self.bin_minutes_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5)

        # ========== Action Buttons (Left Column) ==========
        action_frame = ttk.Frame(main_frame)
        action_frame.grid(row=4, column=0, pady=15, padx=(0, 5))

        self.run_button = ttk.Button(
            action_frame,
            text="Run Analysis",
            command=self._run_analysis,
            style="Accent.TButton"
        )
        self.run_button.pack(side=tk.LEFT, padx=5)

        ttk.Button(action_frame, text="Clear Log", command=self._clear_log).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Exit", command=self.root.quit).pack(side=tk.LEFT, padx=5)

        # ========== Progress and Log Section (Right Column) ==========
        log_frame = ttk.LabelFrame(main_frame, text="Progress Log", padding="10")
        log_frame.grid(row=1, column=1, rowspan=4, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        main_frame.rowconfigure(1, weight=1)

        # Progress bar
        self.progress_var = tk.DoubleVar(master=self.root)
        self.progress_bar = ttk.Progressbar(
            log_frame,
            variable=self.progress_var,
            maximum=100,
            mode='determinate'
        )
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        log_frame.columnconfigure(0, weight=1)

        # Log text area
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            height=30,
            width=60,
            state='disabled',
            wrap=tk.WORD
        )
        self.log_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        log_frame.rowconfigure(1, weight=1)
        log_frame.columnconfigure(0, weight=1)

    def _initialize_default_model(self):
        """Initialize default model path if file exists."""
        # Construct full path relative to project root
        default_model_relative = "models/model_LUPE.pkl"
        default_model_path = Path(__file__).parent.parent.parent / default_model_relative

        # Check if model file exists and set path silently
        if default_model_path.exists():
            self.model_path = str(default_model_path)
        else:
            # Model doesn't exist - keep None
            self.model_path = None

    def _select_dlc_files(self):
        """Open file dialog to select DLC CSV files."""
        filenames = filedialog.askopenfilenames(
            title="Select DLC CSV Files",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filenames:
            self.dlc_csv_paths = list(filenames)
            count = len(filenames)
            self.dlc_path_var.set(f"{count} file(s) selected")
            self._log(f"Selected {count} DLC CSV file(s)")

    def _select_model_file(self):
        """Open file dialog to select model file."""
        filename = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        if filename:
            self.model_path = filename
            self.model_path_var.set(Path(filename).name)
            self._log(f"Selected model file: {filename}")

    def _select_output_dir(self):
        """Open directory dialog to select output directory."""
        dirname = filedialog.askdirectory(
            title="Select Output Directory"
        )
        if dirname:
            self.output_dir = dirname
            self.output_dir_var.set(dirname)
            self._log(f"Selected output directory: {dirname}")

    def _select_all_analyses(self):
        """Select all analysis checkboxes."""
        for var in self.analyses.values():
            var.set(True)

    def _deselect_all_analyses(self):
        """Deselect all analysis checkboxes."""
        for var in self.analyses.values():
            var.set(False)

    def _log(self, message):
        """
        Add a message to the log text area.

        Args:
            message (str): Message to log
        """
        self.log_text.configure(state='normal')
        self.log_text.insert(tk.END, message + '\n')
        self.log_text.see(tk.END)
        self.log_text.configure(state='disabled')
        self.root.update_idletasks()

    def _clear_log(self):
        """Clear the log text area."""
        self.log_text.configure(state='normal')
        self.log_text.delete('1.0', tk.END)
        self.log_text.configure(state='disabled')

    def _update_progress(self, value):
        """
        Update the progress bar.

        Args:
            value (float): Progress value (0-100)
        """
        self.progress_var.set(value)
        self.root.update_idletasks()

    def _run_analysis(self):
        """Run the selected analyses in a background thread."""
        # Validate inputs
        if not self.dlc_csv_paths:
            messagebox.showerror("Error", "Please select DLC CSV files first.")
            return
        if not self.model_path:
            messagebox.showerror("Error", "Please select a model file first.")
            return

        # Check if any analysis is selected
        if not any(var.get() for var in self.analyses.values()):
            messagebox.showwarning("Warning", "Please select at least one analysis.")
            return

        # Disable run button during analysis
        self.run_button.configure(state='disabled')

        # Run analysis in background thread to keep GUI responsive
        thread = threading.Thread(target=self._perform_analysis)
        thread.start()

    def _perform_analysis(self):
        """Perform the actual analysis (runs in background thread)."""
        try:
            self._log("=" * 60)
            self._log("Starting LUPE Analysis")
            self._log("=" * 60)

            # Load model once for all files
            self._log(f"\nLoading model from: {self.model_path}")
            model = load_model(self.model_path)
            self._log("[OK] Model loaded")

            # Get base output directory
            base_output_dir = Path(self.output_dir_var.get())
            base_output_dir.mkdir(parents=True, exist_ok=True)

            # Get configuration
            config = get_config()
            framerate = config.get_framerate()

            # Count selected analyses for progress tracking
            selected_analyses = [key for key, var in self.analyses.items() if var.get()]
            total_steps = len(self.dlc_csv_paths) * (2 + len(selected_analyses))  # preprocess + classify + analyses
            current_step = 0

            # Process each CSV file individually
            for file_idx, csv_path in enumerate(self.dlc_csv_paths, 1):
                csv_file = Path(csv_path)
                self._log(f"\n{'='*60}")
                self._log(f"Processing file {file_idx}/{len(self.dlc_csv_paths)}: {csv_file.name}")
                self._log(f"{'='*60}")

                # Step 1: Preprocess DLC CSV
                self._log(f"\n[Step 1] Preprocessing...")
                self._log(f"  Likelihood threshold: {self.llh_var.get()}")
                try:
                    pose_data = preprocess_dlc_csv(
                        str(csv_path),
                        likelihood_threshold=self.llh_var.get(),
                        save_output=False
                    )
                    self._log(f"  [OK] Preprocessed shape: {pose_data.shape}")
                    current_step += 1
                    self._update_progress((current_step / total_steps) * 100)
                except Exception as e:
                    self._log(f"  [ERROR] Preprocessing error: {str(e)}")
                    self._log(f"  Skipping {csv_file.name}")
                    current_step += (2 + len(selected_analyses))  # Skip all steps for this file
                    continue

                # Step 2: Classify behaviors
                self._log(f"\n[Step 2] Classifying behaviors...")
                try:
                    predictions = classify_behaviors(model, [pose_data])[0]
                    self._log(f"  [OK] Classified {len(predictions):,} frames")
                    self._log(f"  [OK] Found {len(np.unique(predictions))} unique behaviors")
                    current_step += 1
                    self._update_progress((current_step / total_steps) * 100)

                    # Explicitly free pose_data to release memory (critical for multi-file processing)
                    del pose_data
                    gc.collect()

                except Exception as e:
                    self._log(f"  [ERROR] Classification error: {str(e)}")
                    self._log(f"  Skipping {csv_file.name}")
                    current_step += (1 + len(selected_analyses))  # Skip remaining steps for this file

                    # Free memory before continuing to next file
                    del pose_data
                    gc.collect()
                    continue

                # Step 3: Create output folder structure
                partial_name = extract_partial_filename(str(csv_path))
                file_output_dir = base_output_dir / partial_name
                analysis_dir = file_output_dir / f"{partial_name}_analysis"
                file_output_dir.mkdir(parents=True, exist_ok=True)
                analysis_dir.mkdir(parents=True, exist_ok=True)

                self._log(f"\n[Step 3] Saving outputs to: {file_output_dir}")

                # Save behaviors CSV (frame, behavior_id)
                behaviors_csv_path = file_output_dir / f"{partial_name}_behaviors.csv"
                df_behaviors = pd.DataFrame({
                    'frame': range(1, len(predictions) + 1),
                    'behavior_id': predictions
                })
                df_behaviors.to_csv(behaviors_csv_path, index=False)
                self._log(f"  [OK] Saved: {behaviors_csv_path.name}")

                # Save time vector CSV (frame, time_seconds)
                time_csv_path = file_output_dir / f"{partial_name}_time.csv"
                time_seconds = np.array([i / framerate for i in range(len(predictions))])
                df_time = pd.DataFrame({
                    'frame': range(1, len(predictions) + 1),
                    'time_seconds': time_seconds
                })
                df_time.to_csv(time_csv_path, index=False)
                self._log(f"  [OK] Saved: {time_csv_path.name}")

                # Generate file summary with metadata and behavior statistics
                try:
                    # Load DLC CSV headers to extract keypoint information
                    # We only need to read the header rows (first 4 rows) for efficiency
                    dlc_df_headers = pd.read_csv(str(csv_path), header=[0, 1, 2], nrows=0)

                    # Get behavior names from config
                    behavior_names = config.get_behavior_names()

                    # Generate summary
                    summary_path = generate_dlc_summary(
                        dlc_df=dlc_df_headers,
                        predictions=predictions,
                        framerate=framerate,
                        file_path=str(csv_path),
                        output_dir=str(file_output_dir),
                        behavior_names=behavior_names
                    )
                    self._log(f"  [OK] Saved: {Path(summary_path).name}")
                except Exception as e:
                    self._log(f"  [WARNING] Could not generate summary: {str(e)}")

                # Create single-file behaviors dictionary for analysis functions
                behaviors_dict = {partial_name: predictions}

                # Run selected analyses
                self._log(f"\n[Step 4] Running analyses...")
                bin_minutes = self.bin_minutes_var.get()

                if self.analyses['instance_counts'].get():
                    try:
                        self._log(f"  - Bout counts...")
                        analyze_instance_counts(behaviors_dict, str(analysis_dir), file_prefix=partial_name)
                        self._log(f"    [OK] Complete")
                    except Exception as e:
                        self._log(f"    [ERROR] {str(e)}")
                    current_step += 1
                    self._update_progress((current_step / total_steps) * 100)

                if self.analyses['total_frames'].get():
                    try:
                        self._log(f"  - Time distribution...")
                        analyze_total_frames(behaviors_dict, str(analysis_dir), file_prefix=partial_name)
                        self._log(f"    [OK] Complete")
                    except Exception as e:
                        self._log(f"    [ERROR] {str(e)}")
                    current_step += 1
                    self._update_progress((current_step / total_steps) * 100)

                if self.analyses['durations'].get():
                    try:
                        self._log(f"  - Bout durations...")
                        analyze_bout_durations(behaviors_dict, str(analysis_dir), file_prefix=partial_name)
                        self._log(f"    [OK] Complete")
                    except Exception as e:
                        self._log(f"    [ERROR] {str(e)}")
                    current_step += 1
                    self._update_progress((current_step / total_steps) * 100)

                if self.analyses['timeline'].get():
                    try:
                        self._log(f"  - Timeline...")
                        analyze_binned_timeline(behaviors_dict, str(analysis_dir),
                                              bin_size_minutes=bin_minutes, file_prefix=partial_name)
                        self._log(f"    [OK] Complete")
                    except Exception as e:
                        self._log(f"    [ERROR] {str(e)}")
                    current_step += 1
                    self._update_progress((current_step / total_steps) * 100)

                if self.analyses['transitions'].get():
                    try:
                        self._log(f"  - Behavior transitions...")
                        analyze_transitions(behaviors_dict, str(analysis_dir), file_prefix=partial_name)
                        self._log(f"    [OK] Complete")
                    except Exception as e:
                        self._log(f"    [ERROR] {str(e)}")
                    current_step += 1
                    self._update_progress((current_step / total_steps) * 100)

                # Skip csv_export since we already saved behaviors and time
                if self.analyses['csv_export'].get():
                    current_step += 1
                    self._update_progress((current_step / total_steps) * 100)

                # Create master summary if any analyses were run
                if any([self.analyses[key].get() for key in ['instance_counts', 'total_frames', 'durations', 'transitions']]):
                    try:
                        self._log(f"\n[Step 5] Creating master summary...")
                        create_analysis_summary(
                            file_prefix=partial_name,
                            analysis_dir=str(analysis_dir),
                            behavior_names=config.get_behavior_names()
                        )
                        self._log(f"  [OK] Master summary created")
                    except Exception as e:
                        self._log(f"  [WARNING] Could not create master summary: {str(e)}")

                self._log(f"[OK] Completed processing: {csv_file.name}")

                # Aggressively free memory after each file completes all processing
                # This ensures garbage collection happens between files
                gc.collect()

            self._log("\n" + "=" * 60)
            self._log("All analyses completed successfully!")
            self._log(f"Results saved to: {base_output_dir}")
            self._log("=" * 60)

            # Show completion message
            self.root.after(0, lambda: messagebox.showinfo(
                "Success",
                f"Analysis completed successfully!\n\nResults saved to:\n{base_output_dir}"
            ))

        except Exception as e:
            error_msg = f"Error during analysis: {str(e)}"
            self._log(f"\n[ERROR] {error_msg}")
            import traceback
            self._log(traceback.format_exc())
            self.root.after(0, lambda: messagebox.showerror("Error", error_msg))

        finally:
            # Re-enable run button
            self.root.after(0, lambda: self.run_button.configure(state='normal'))
            self._update_progress(0)

    def _rename_with_prefix(self, directory: Path, prefix: str, base_name: str):
        """
        Rename files in directory to add prefix.

        Args:
            directory (Path): Directory containing files to rename
            prefix (str): Prefix to add to filenames
            base_name (str): Base name to look for in existing files
        """
        for file_path in directory.iterdir():
            if file_path.is_file() and file_path.name.startswith(base_name):
                new_name = f"{prefix}_{file_path.name}"
                new_path = directory / new_name
                file_path.rename(new_path)

    def run(self):
        """Start the GUI application."""
        self.root.mainloop()


if __name__ == '__main__':
    app = LupeGUI()
    app.run()
