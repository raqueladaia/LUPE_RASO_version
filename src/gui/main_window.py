"""
Main GUI Window for LUPE Analysis Tool

This module provides a graphical user interface for LUPE analysis using tkinter.
The GUI allows users to:
- Load DeepLabCut files (CSV or H5 format)
- Classify behaviors using a pre-trained A-SOiD model
- Select which analyses to run
- Configure output settings
- Run the complete analysis pipeline with a single click

Output Structure:
    For each input file (e.g., "mouse01DLC_resnet50.csv" or "mouse01DLC_resnet50.h5"), the GUI creates:

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
import time
import psutil
import matplotlib.pyplot as plt

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
from src.utils.plotting import close_all_plots


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

        # Framerate configuration state
        self.detected_frames = 0
        self.calculated_framerate = 60.0  # Default fallback
        self.framerate_user_modified = False

        # Create GUI components
        self._create_widgets()

        # Initialize default model path
        self._initialize_default_model()

        # Load configuration
        self.config = get_config()

    def _log_memory_status(self, label: str = ""):
        """
        Log current memory usage and matplotlib figure count for diagnostics.

        Args:
            label (str): Optional label to identify when this measurement was taken
        """
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            fig_count = len(plt.get_fignums())

            msg = f"[MEMORY] {label}: {memory_mb:.1f} MB used, {fig_count} figures open"
            self._log(msg)
        except Exception as e:
            self._log(f"[MEMORY] Could not get memory info: {str(e)}")

    def _format_elapsed_time(self, seconds: float) -> str:
        """
        Format elapsed time in seconds to human-readable string.

        Args:
            seconds (float): Elapsed time in seconds

        Returns:
            str: Formatted time string (e.g., "45.2s", "2m 34s", "1h 5m 23s")

        Examples:
            >>> _format_elapsed_time(45.234)
            "45.2s"
            >>> _format_elapsed_time(154.7)
            "2m 35s"
            >>> _format_elapsed_time(3923.5)
            "1h 5m 24s"
        """
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours}h {minutes}m {secs}s"

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
            text="Process DLC files (.csv or .h5) through LUPE A-SOiD model",
            font=('Arial', 16, 'bold')
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=10)

        # ========== File Selection Section (Left Column) ==========
        file_frame = ttk.LabelFrame(main_frame, text="Data Files", padding="10")
        file_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5, padx=(0, 5))

        # Configure column weights to prevent button cutoff
        file_frame.columnconfigure(1, weight=1)

        # DLC Files (CSV or H5)
        ttk.Label(file_frame, text="DLC Files:").grid(row=0, column=0, sticky=tk.W, pady=5)
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

        # Framerate Configuration
        ttk.Separator(options_frame, orient='horizontal').grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)

        ttk.Label(options_frame, text="Framerate Configuration:", font=('TkDefaultFont', 9, 'bold')).grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(5,2))

        # Radio buttons for mode selection
        self.framerate_mode_var = tk.StringVar(master=self.root, value="fps")

        fps_radio = ttk.Radiobutton(options_frame, text="Specify FPS:",
                                     variable=self.framerate_mode_var, value="fps",
                                     command=self._on_framerate_mode_changed)
        fps_radio.grid(row=3, column=0, sticky=tk.W, pady=2)

        self.fps_var = tk.DoubleVar(master=self.root, value=60.0)
        self.fps_entry = ttk.Spinbox(options_frame, from_=10.0, to=300.0, increment=1.0,
                                     textvariable=self.fps_var, width=10,
                                     command=self._on_fps_changed)
        self.fps_entry.grid(row=3, column=1, sticky=tk.W, padx=5)
        self.fps_entry.bind('<KeyRelease>', lambda e: self._on_fps_changed())

        duration_radio = ttk.Radiobutton(options_frame, text="Specify Duration (min):",
                                          variable=self.framerate_mode_var, value="duration",
                                          command=self._on_framerate_mode_changed)
        duration_radio.grid(row=4, column=0, sticky=tk.W, pady=2)

        self.duration_var = tk.DoubleVar(master=self.root, value=0.0)
        self.duration_entry = ttk.Spinbox(options_frame, from_=0.1, to=1000.0, increment=1.0,
                                          textvariable=self.duration_var, width=10,
                                          command=self._on_duration_changed, state='disabled')
        self.duration_entry.grid(row=4, column=1, sticky=tk.W, padx=5)
        self.duration_entry.bind('<KeyRelease>', lambda e: self._on_duration_changed())

        # Display detected and calculated values
        self.frames_label = ttk.Label(options_frame, text="Detected: Select files to detect frames",
                                      foreground='gray')
        self.frames_label.grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=(5,2))

        self.calculated_label = ttk.Label(options_frame, text="Calculated: --",
                                          foreground='blue')
        self.calculated_label.grid(row=6, column=0, columnspan=2, sticky=tk.W, pady=(0,5))

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
        """Open file dialog to select DLC files (CSV or H5 format)."""
        filenames = filedialog.askopenfilenames(
            title="Select DLC Files (CSV or H5)",
            filetypes=[
                ("DLC files", "*.csv *.h5"),
                ("CSV files", "*.csv"),
                ("H5 files", "*.h5"),
                ("All files", "*.*")
            ]
        )
        if filenames:
            self.dlc_csv_paths = list(filenames)
            count = len(filenames)
            self.dlc_path_var.set(f"{count} file(s) selected")
            self._log(f"Selected {count} DLC file(s)")

            # Detect frame count from first file for framerate calculation
            self._detect_frame_count()

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

    def _on_framerate_mode_changed(self):
        """Handle framerate mode radio button change (FPS vs Duration)."""
        mode = self.framerate_mode_var.get()
        if mode == "fps":
            self.fps_entry.config(state='normal')
            self.duration_entry.config(state='disabled')
        else:  # mode == "duration"
            self.fps_entry.config(state='disabled')
            self.duration_entry.config(state='normal')

        # Mark as user-modified
        self.framerate_user_modified = True
        # Recalculate
        self._calculate_framerate_values()

    def _on_fps_changed(self):
        """Handle FPS value change."""
        self.framerate_user_modified = True
        self._calculate_framerate_values()

    def _on_duration_changed(self):
        """Handle duration value change."""
        self.framerate_user_modified = True
        self._calculate_framerate_values()

    def _calculate_framerate_values(self):
        """Calculate framerate or duration based on user input and detected frames."""
        if self.detected_frames == 0:
            # No files loaded yet
            self.calculated_label.config(text="Calculated: Select files first", foreground='gray')
            return

        mode = self.framerate_mode_var.get()

        try:
            if mode == "fps":
                # User specified FPS, calculate duration
                fps = self.fps_var.get()
                if fps <= 0:
                    self.calculated_label.config(text="Calculated: Invalid FPS (must be > 0)", foreground='red')
                    return

                duration_minutes = self.detected_frames / (fps * 60)
                self.calculated_framerate = fps
                self.calculated_label.config(
                    text=f"Calculated Duration: {duration_minutes:.2f} minutes",
                    foreground='blue'
                )

                # Warn if fps is unusual
                if fps < 20 or fps > 120:
                    self.calculated_label.config(
                        text=f"Calculated Duration: {duration_minutes:.2f} min (⚠️ Unusual FPS)",
                        foreground='orange'
                    )

            else:  # mode == "duration"
                # User specified duration, calculate FPS
                duration_minutes = self.duration_var.get()
                if duration_minutes <= 0:
                    self.calculated_label.config(text="Calculated: Invalid duration (must be > 0)", foreground='red')
                    return

                fps = self.detected_frames / (duration_minutes * 60)
                self.calculated_framerate = fps
                self.calculated_label.config(
                    text=f"Calculated FPS: {fps:.2f} fps",
                    foreground='blue'
                )

                # Warn if calculated fps is unusual
                if fps < 20 or fps > 120:
                    self.calculated_label.config(
                        text=f"Calculated FPS: {fps:.2f} fps (⚠️ Unusual value)",
                        foreground='orange'
                    )

        except tk.TclError:
            # Handle case where spinbox value is being edited
            self.calculated_label.config(text="Calculated: Enter valid number", foreground='gray')

    def _detect_frame_count(self):
        """Detect frame count from the first selected file."""
        if not self.dlc_csv_paths:
            return

        try:
            first_file = Path(self.dlc_csv_paths[0])
            file_ext = first_file.suffix.lower()

            if file_ext == '.h5':
                # Read H5 file to get frame count
                import pandas as pd
                df = pd.read_hdf(str(first_file))
                self.detected_frames = len(df)
            elif file_ext == '.csv':
                # Read CSV file to get frame count
                import pandas as pd
                # Read just to count rows (efficient)
                df = pd.read_csv(str(first_file), header=[0, 1, 2], usecols=[0])
                self.detected_frames = len(df)
            else:
                self._log(f"[WARNING] Unknown file format: {file_ext}")
                return

            # Update display
            self.frames_label.config(
                text=f"Detected: {self.detected_frames:,} frames in first file",
                foreground='black'
            )

            # Calculate values
            self._calculate_framerate_values()

            self._log(f"Detected {self.detected_frames:,} frames from {first_file.name}")

        except Exception as e:
            self._log(f"[WARNING] Could not detect frame count: {str(e)}")
            self.frames_label.config(
                text=f"Detected: Error reading file",
                foreground='red'
            )

    def _log(self, message):
        """
        Add a message to the log text area (thread-safe).

        This method can be called from any thread. It schedules the actual
        GUI update to run on the main thread using root.after().

        Args:
            message (str): Message to log
        """
        # Schedule GUI update on main thread (thread-safe)
        self.root.after(0, self._log_safe, message)

    def _log_safe(self, message):
        """
        Internal method to update log text (runs on main thread only).

        Args:
            message (str): Message to log
        """
        self.log_text.configure(state='normal')
        self.log_text.insert(tk.END, message + '\n')
        self.log_text.see(tk.END)
        self.log_text.configure(state='disabled')

    def _clear_log(self):
        """Clear the log text area."""
        self.log_text.configure(state='normal')
        self.log_text.delete('1.0', tk.END)
        self.log_text.configure(state='disabled')

    def _update_progress(self, value):
        """
        Update the progress bar (thread-safe).

        This method can be called from any thread. It schedules the actual
        GUI update to run on the main thread using root.after().

        Args:
            value (float): Progress value (0-100)
        """
        # Schedule GUI update on main thread (thread-safe)
        self.root.after(0, self._update_progress_safe, value)

    def _update_progress_safe(self, value):
        """
        Internal method to update progress bar (runs on main thread only).

        Args:
            value (float): Progress value (0-100)
        """
        self.progress_var.set(value)

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

        # Check if user has specified framerate (warn if using default)
        if not self.framerate_user_modified:
            response = messagebox.askquestion(
                "Framerate Not Specified",
                "You have not specified the video framerate or duration.\n\n"
                "LUPE will assume 60 fps, but if your videos are recorded\n"
                "at a different framerate, all time-based measurements\n"
                "will be INCORRECT:\n\n"
                "  • Behavior durations\n"
                "  • Timeline binning\n"
                "  • All timestamps in outputs\n\n"
                "Please verify your video framerate before continuing.\n\n"
                "Continue with 60 fps anyway?",
                icon='warning'
            )
            if response == 'no':
                return  # User cancelled

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

            # Record start time for total analysis duration
            analysis_start_time = time.time()

            # Log initial memory state
            self._log_memory_status("Analysis start")

            # Load model once for all files
            self._log(f"\nLoading model from: {self.model_path}")
            model = load_model(self.model_path)
            self._log("[OK] Model loaded")
            self._log_memory_status("After model load")

            # Get base output directory
            base_output_dir = Path(self.output_dir_var.get())
            base_output_dir.mkdir(parents=True, exist_ok=True)

            # Get configuration
            config = get_config()

            # Use user-specified framerate (or default if not modified)
            framerate = self.calculated_framerate
            if not self.framerate_user_modified:
                self._log("[WARNING] Using default 60 fps - verify this matches your video framerate")
            else:
                self._log(f"[INFO] Using framerate: {framerate:.2f} fps")

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

                # Log memory before processing this file
                self._log_memory_status(f"Before file {file_idx}")

                # Record start time for this file
                file_start_time = time.time()

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
                    predictions = classify_behaviors(model, [pose_data], framerate=framerate)[0]
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

                # Explicitly delete DataFrame to free memory (Phase 1 Fix 1B)
                del df_behaviors

                # Save time vector CSV (frame, time_seconds)
                time_csv_path = file_output_dir / f"{partial_name}_time.csv"
                time_seconds = np.array([i / framerate for i in range(len(predictions))])
                df_time = pd.DataFrame({
                    'frame': range(1, len(predictions) + 1),
                    'time_seconds': time_seconds
                })
                df_time.to_csv(time_csv_path, index=False)
                self._log(f"  [OK] Saved: {time_csv_path.name}")

                # Explicitly delete DataFrame and array to free memory (Phase 1 Fix 1B)
                del df_time, time_seconds

                # Generate file summary with metadata and behavior statistics
                try:
                    # Load DLC file headers to extract keypoint information
                    # Support both CSV and H5 formats
                    file_ext = csv_file.suffix.lower()

                    if file_ext == '.h5':
                        # For H5 files, read only first few rows for efficiency
                        dlc_df_headers = pd.read_hdf(str(csv_path))
                        # Keep only first row to get column structure (headers)
                        dlc_df_headers = dlc_df_headers.iloc[:0]
                    elif file_ext == '.csv':
                        # For CSV files, read only header rows (first 4 rows)
                        dlc_df_headers = pd.read_csv(str(csv_path), header=[0, 1, 2], nrows=0)
                    else:
                        raise ValueError(f"Unsupported file format: {file_ext}")

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

                    # Explicitly delete DataFrame to free memory (Phase 1 Fix 1B)
                    del dlc_df_headers

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

                # Calculate elapsed time for this file
                file_elapsed = time.time() - file_start_time
                file_time_str = self._format_elapsed_time(file_elapsed)

                self._log(f"[OK] Completed processing: {csv_file.name} (took {file_time_str})")

                # Phase 1 Fix 1B: Explicitly delete behaviors_dict to free memory
                del behaviors_dict

                # Phase 1 Fix 1A: Close all figures to prevent memory leak
                # This clears matplotlib's internal figure cache that accumulates over multiple files
                self._log_memory_status(f"Before cleanup file {file_idx}")
                close_all_plots()

                # Phase 1 Fix 1C: Aggressive garbage collection (double call for Windows)
                # Windows systems benefit from multiple gc.collect() calls to resolve circular references
                gc.collect()
                gc.collect()

                # Log memory after cleanup
                self._log_memory_status(f"After cleanup file {file_idx}")

            # Calculate total analysis time
            total_elapsed = time.time() - analysis_start_time
            total_time_str = self._format_elapsed_time(total_elapsed)

            self._log("\n" + "=" * 60)
            self._log("All analyses completed successfully!")
            self._log(f"Total time: {total_time_str}")
            self._log(f"Results saved to: {base_output_dir}")
            self._log("=" * 60)

            # Log final memory state
            self._log_memory_status("All files complete")

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
            # Phase 2 Fix: Release model from memory after all processing
            # Model can be 50-200MB depending on complexity
            if 'model' in locals():
                self._log("[CLEANUP] Releasing model from memory")
                del model
                gc.collect()
                self._log_memory_status("After model cleanup")

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
