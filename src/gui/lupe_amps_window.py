"""
LUPE-AMPS Analysis GUI

This module provides a graphical user interface for LUPE-AMPS pain scale analysis.

The GUI allows users to:
- Select multiple behavior CSV files
- Load LUPE-AMPS PCA model
- Configure analysis parameters
- Run comprehensive pain scale analysis
- View progress in real-time

Usage:
    from src.gui.lupe_amps_window import LupeAmpsGUI

    app = LupeAmpsGUI()
    app.run()
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
from pathlib import Path

from src.core.analysis_lupe_amps import LupeAmpsAnalysis


class LupeAmpsGUI:
    """
    GUI application for LUPE-AMPS pain scale analysis.

    This class creates and manages the graphical interface for running
    LUPE-AMPS analysis on multiple behavior CSV files.
    """

    def __init__(self):
        """Initialize the GUI application."""
        self.root = tk.Tk()
        self.root.title("LUPE-AMPS Pain Scale Analysis")
        self.root.geometry("1000x616")

        # Application state
        self.csv_files = []  # List of selected CSV file paths
        self.model_path = "models/model_AMPS.pkl"  # Default model path

        # Create GUI components
        self._create_widgets()

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
            text="LUPE-AMPS Pain Scale Analysis",
            font=('Arial', 16, 'bold')
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=10)

        # ========== Create scrollable left column ==========
        # Canvas for scrolling
        canvas = tk.Canvas(main_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        scrollbar.grid(row=1, column=0, sticky=(tk.N, tk.S, tk.E))
        main_frame.rowconfigure(1, weight=1)

        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # ========== Input Files Section (Scrollable Left Column) ==========
        files_frame = ttk.LabelFrame(scrollable_frame, text="Input Files", padding="10")
        files_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5, padx=5)

        # Single file entry (for direct path input)
        ttk.Label(files_frame, text="Add single file (paste path):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.single_file_var = tk.StringVar()
        ttk.Entry(files_frame, textvariable=self.single_file_var, width=35).grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(files_frame, text="Add", command=self._add_single_file).grid(row=0, column=2, padx=2)
        files_frame.columnconfigure(1, weight=1)

        # OR separator
        ttk.Label(files_frame, text="OR", font=('Arial', 9, 'italic')).grid(row=1, column=0, columnspan=3, pady=5)

        # File list with scrollbar (for multiple files)
        ttk.Label(files_frame, text="Selected files:").grid(row=2, column=0, sticky=tk.W, pady=2)
        list_frame = ttk.Frame(files_frame)
        list_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        files_frame.rowconfigure(3, weight=1)

        self.file_listbox = tk.Listbox(list_frame, height=6, selectmode=tk.EXTENDED)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.file_listbox.yview)
        self.file_listbox.configure(yscrollcommand=scrollbar.set)

        self.file_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)

        # File buttons
        btn_frame = ttk.Frame(files_frame)
        btn_frame.grid(row=4, column=0, columnspan=3, pady=5)

        ttk.Button(btn_frame, text="Browse Multiple...", command=self._add_files).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Remove Selected", command=self._remove_files).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Clear All", command=self._clear_files).pack(side=tk.LEFT, padx=2)

        # File count label
        self.file_count_var = tk.StringVar(value="0 file(s) selected")
        ttk.Label(files_frame, textvariable=self.file_count_var).grid(row=5, column=0, columnspan=3, pady=5)

        # ========== Model Selection (Scrollable Left Column) ==========
        model_frame = ttk.LabelFrame(scrollable_frame, text="Model", padding="10")
        model_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5, padx=5)

        ttk.Label(model_frame, text="LUPE-AMPS Model:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.model_path_var = tk.StringVar(value=self.model_path)
        ttk.Entry(model_frame, textvariable=self.model_path_var, width=30, state='readonly').grid(
            row=0, column=1, sticky=(tk.W, tk.E), padx=5
        )
        ttk.Button(model_frame, text="Browse...", command=self._select_model).grid(row=0, column=2, padx=5)
        model_frame.columnconfigure(1, weight=1)

        # ========== Parameters (Scrollable Left Column) ==========
        params_frame = ttk.LabelFrame(scrollable_frame, text="Parameters", padding="10")
        params_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5, padx=5)

        ttk.Label(params_frame, text="Recording Length (min):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.rec_length_var = tk.IntVar(value=30)
        ttk.Spinbox(params_frame, from_=1, to=120, textvariable=self.rec_length_var, width=10).grid(
            row=0, column=1, sticky=tk.W, padx=5
        )

        ttk.Label(params_frame, text="Original Framerate (fps):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.original_fps_var = tk.IntVar(value=60)
        ttk.Spinbox(params_frame, from_=1, to=240, textvariable=self.original_fps_var, width=10).grid(
            row=1, column=1, sticky=tk.W, padx=5
        )

        ttk.Label(params_frame, text="Target Framerate (fps):").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.target_fps_var = tk.IntVar(value=20)
        ttk.Spinbox(params_frame, from_=1, to=240, textvariable=self.target_fps_var, width=10).grid(
            row=2, column=1, sticky=tk.W, padx=5
        )

        # ========== Output Settings (Scrollable Left Column) ==========
        output_frame = ttk.LabelFrame(scrollable_frame, text="Output Settings", padding="10")
        output_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=5, padx=5)

        ttk.Label(output_frame, text="Output Directory:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.output_dir_var = tk.StringVar(value="outputs/")
        ttk.Entry(output_frame, textvariable=self.output_dir_var, width=25).grid(
            row=0, column=1, sticky=(tk.W, tk.E), padx=5
        )
        ttk.Button(output_frame, text="Browse...", command=self._select_output_dir).grid(row=0, column=2, padx=5)
        output_frame.columnconfigure(1, weight=1)

        ttk.Label(output_frame, text="Project Name:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.project_name_var = tk.StringVar(value="LUPE-AMPS")
        ttk.Entry(output_frame, textvariable=self.project_name_var, width=25).grid(
            row=1, column=1, sticky=(tk.W, tk.E), padx=5
        )

        # ========== Analysis Sections (Scrollable Left Column) ==========
        sections_frame = ttk.LabelFrame(scrollable_frame, text="Analysis Sections", padding="10")
        sections_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=5, padx=5)

        self.sections = {}
        sections_list = [
            (1, "Section 1: Preprocessing & Metrics"),
            (2, "Section 2: PCA Pain Scale Projection"),
            (3, "Section 3: Behavior Metrics Visualization"),
            (4, "Section 4: Model Feature Importance")
        ]

        for i, (section_num, label) in enumerate(sections_list):
            var = tk.BooleanVar(value=True)
            self.sections[section_num] = var
            ttk.Checkbutton(sections_frame, text=label, variable=var).grid(
                row=i, column=0, sticky=tk.W, pady=2
            )

        # Select All / Deselect All buttons
        button_frame = ttk.Frame(sections_frame)
        button_frame.grid(row=len(sections_list), column=0, pady=10)
        ttk.Button(button_frame, text="Select All", command=self._select_all_sections).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Deselect All", command=self._deselect_all_sections).pack(side=tk.LEFT, padx=5)

        # ========== Action Buttons (Scrollable Left Column) ==========
        action_frame = ttk.Frame(scrollable_frame)
        action_frame.grid(row=5, column=0, pady=15, padx=5)

        self.run_button = ttk.Button(
            action_frame,
            text="Run LUPE-AMPS Analysis",
            command=self._run_analysis,
            style="Accent.TButton"
        )
        self.run_button.pack(side=tk.LEFT, padx=5)

        ttk.Button(action_frame, text="Clear Log", command=self._clear_log).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Exit", command=self.root.quit).pack(side=tk.LEFT, padx=5)

        # ========== Progress and Log Section (Right Column) ==========
        log_frame = ttk.LabelFrame(main_frame, text="Progress Log", padding="10")
        log_frame.grid(row=1, column=1, rowspan=6, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        main_frame.rowconfigure(1, weight=1)

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            log_frame,
            variable=self.progress_var,
            maximum=100,
            mode='indeterminate'
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

    def _add_single_file(self):
        """Add a single file from the text entry field."""
        file_path = self.single_file_var.get().strip()

        if not file_path:
            messagebox.showwarning("Warning", "Please enter a file path.")
            return

        # Check if file exists
        if not Path(file_path).exists():
            messagebox.showerror("Error", f"File not found:\n{file_path}")
            return

        # Add to list if not already present
        if file_path not in self.csv_files:
            self.csv_files.append(file_path)
            self.file_listbox.insert(tk.END, Path(file_path).name)
            self._update_file_count()
            self._log(f"Added: {Path(file_path).name}")
            # Clear the entry field
            self.single_file_var.set("")
        else:
            messagebox.showinfo("Info", "File is already in the list.")

    def _add_files(self):
        """Open file dialog to add CSV files."""
        filenames = filedialog.askopenfilenames(
            title="Select Behavior CSV Files",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filenames:
            for filename in filenames:
                if filename not in self.csv_files:
                    self.csv_files.append(filename)
                    self.file_listbox.insert(tk.END, Path(filename).name)

            self._update_file_count()
            self._log(f"Added {len(filenames)} file(s)")

    def _remove_files(self):
        """Remove selected files from the list."""
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            return

        # Remove in reverse order to maintain correct indices
        for index in reversed(selected_indices):
            self.file_listbox.delete(index)
            del self.csv_files[index]

        self._update_file_count()
        self._log(f"Removed {len(selected_indices)} file(s)")

    def _clear_files(self):
        """Clear all files from the list."""
        self.csv_files = []
        self.file_listbox.delete(0, tk.END)
        self._update_file_count()
        self._log("Cleared all files")

    def _update_file_count(self):
        """Update the file count label."""
        count = len(self.csv_files)
        self.file_count_var.set(f"{count} file(s) selected")

    def _select_model(self):
        """Open file dialog to select model file."""
        filename = filedialog.askopenfilename(
            title="Select LUPE-AMPS Model File",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
            initialdir="models"
        )
        if filename:
            self.model_path = filename
            self.model_path_var.set(filename)
            self._log(f"Selected model: {Path(filename).name}")

    def _select_output_dir(self):
        """Open directory dialog to select output directory."""
        dirname = filedialog.askdirectory(title="Select Output Directory")
        if dirname:
            self.output_dir_var.set(dirname)
            self._log(f"Output directory: {dirname}")

    def _select_all_sections(self):
        """Select all analysis sections."""
        for var in self.sections.values():
            var.set(True)

    def _deselect_all_sections(self):
        """Deselect all analysis sections."""
        for var in self.sections.values():
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

    def _run_analysis(self):
        """Run the LUPE-AMPS analysis in a background thread."""
        # Validate inputs
        if not self.csv_files:
            messagebox.showerror("Error", "Please add at least one behavior CSV file.")
            return

        if not Path(self.model_path).exists():
            messagebox.showerror("Error", f"Model file not found: {self.model_path}")
            return

        # Check if any section is selected
        selected_sections = [num for num, var in self.sections.items() if var.get()]
        if not selected_sections:
            messagebox.showwarning("Warning", "Please select at least one analysis section.")
            return

        # Disable run button during analysis
        self.run_button.configure(state='disabled')
        self.progress_bar.start()

        # Run analysis in background thread
        thread = threading.Thread(target=self._perform_analysis)
        thread.start()

    def _perform_analysis(self):
        """Perform the actual analysis (runs in background thread)."""
        try:
            self._log("=" * 60)
            self._log("Starting LUPE-AMPS Pain Scale Analysis")
            self._log("=" * 60)

            # Get selected sections
            selected_sections = [num for num, var in self.sections.items() if var.get()]

            # Create analysis object
            analysis = LupeAmpsAnalysis(
                model_path=self.model_path,
                num_behaviors=6,
                original_fps=self.original_fps_var.get(),
                target_fps=self.target_fps_var.get(),
                recording_length_min=self.rec_length_var.get()
            )

            # Run complete analysis
            results = analysis.run_complete_analysis(
                csv_files=self.csv_files,
                output_base_dir=self.output_dir_var.get(),
                project_name=self.project_name_var.get(),
                sections=selected_sections,
                progress_callback=self._log
            )

            self._log("\n" + "=" * 60)
            self._log("Analysis completed successfully!")
            self._log(f"Results saved to: {self.output_dir_var.get()}")
            self._log("=" * 60)

            # Show completion message
            output_path = Path(self.output_dir_var.get()) / f"{self.project_name_var.get()}_LUPE-AMPS"
            self.root.after(0, lambda: messagebox.showinfo(
                "Success",
                f"LUPE-AMPS analysis completed!\n\nResults saved to:\n{output_path}"
            ))

        except Exception as e:
            error_msg = f"Error during analysis: {str(e)}"
            self._log(f"\n✗ {error_msg}")
            import traceback
            self._log(traceback.format_exc())
            self.root.after(0, lambda: messagebox.showerror("Error", error_msg))

        finally:
            # Re-enable run button and stop progress bar
            self.root.after(0, lambda: self.run_button.configure(state='normal'))
            self.root.after(0, lambda: self.progress_bar.stop())

    def run(self):
        """Start the GUI application."""
        self.root.mainloop()


if __name__ == '__main__':
    app = LupeAmpsGUI()
    app.run()
