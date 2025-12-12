"""
Animal Distribution Window

A popup window for loading animal metadata from an Excel file and assigning animals
to experimental factor combinations (groups, conditions, sex, timepoints).

The window allows users to:
- Specify data source directories for LUPE and AMPS outputs
- Load animal metadata from an Excel file
- View a summary of loaded assignments
- Validate that data files exist for all animals
- See warnings about discrepancies between Excel and project configuration

The Excel file should contain columns for:
- animal_id (required): Unique identifier for each animal
- group (optional): Treatment group assignment
- condition (optional): Experimental condition
- sex (optional): Male or Female
- timepoint (optional): Timepoint name

Usage:
    window = AnimalDistributionWindow(parent, config, callback)
    # User loads Excel file
    # On save, callback is called with updated config
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any

from src.utils.metadata_loader import MetadataLoader, load_metadata


class AnimalDistributionWindow:
    """
    Popup window for loading animal metadata and distributing across experimental factors.

    This window provides an interface for loading animal information from Excel files
    and validating the data against the project configuration.

    Attributes:
        parent: Parent tkinter window
        config (dict): Project configuration
        callback: Function called with updated config when saved
        window: The toplevel window instance
        loaded_data (dict): Data loaded from Excel file
    """

    def __init__(self, parent, config: Dict, callback: Callable[[Dict], None]):
        """
        Initialize the Animal Distribution window.

        Args:
            parent: Parent tkinter window
            config (dict): Project configuration containing groups, conditions, etc.
            callback: Function to call with updated config when user saves
        """
        self.parent = parent
        self.config = config.copy()  # Work on a copy
        self.callback = callback
        self.loaded_data = None  # Store loaded metadata
        self.warnings = []  # Store warnings from loading

        # Create toplevel window
        self.window = tk.Toplevel(parent)
        self.window.title("Load Animal Metadata")
        self.window.transient(parent)
        self.window.grab_set()

        # Set window size
        self.window.geometry("800x700")
        self.window.minsize(700, 600)

        # Center the window
        self._center_window()

        # Build the GUI
        self._create_widgets()

        # Load existing metadata file path if set
        self._load_existing_metadata()

    def _center_window(self):
        """Center the window on the parent."""
        self.window.update_idletasks()
        parent_x = self.parent.winfo_rootx()
        parent_y = self.parent.winfo_rooty()
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()

        window_width = self.window.winfo_width()
        window_height = self.window.winfo_height()

        x = parent_x + (parent_width - window_width) // 2
        y = parent_y + (parent_height - window_height) // 2

        self.window.geometry(f"+{x}+{y}")

    def _create_widgets(self):
        """Create all GUI widgets."""
        # Main container with padding
        main_frame = ttk.Frame(self.window, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(
            main_frame,
            text="Load Animal Metadata",
            font=('Arial', 14, 'bold')
        )
        title_label.pack(pady=(0, 10))

        # Instructions
        instructions = ttk.Label(
            main_frame,
            text="Load animal information from an Excel file containing metadata.\n"
                 "The file should have columns for animal_id, group, condition, sex, and/or timepoint.",
            font=('Arial', 9),
            foreground='gray'
        )
        instructions.pack(pady=(0, 15))

        # Data source directory section
        self._create_data_source_section(main_frame)

        # Excel metadata file section
        self._create_metadata_section(main_frame)

        # Summary section (shows loaded data)
        self._create_summary_section(main_frame)

        # Warnings section
        self._create_warnings_section(main_frame)

        # Button frame at bottom
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(15, 0))

        # Validate button
        validate_btn = ttk.Button(
            button_frame,
            text="Validate Files",
            command=self._validate_files
        )
        validate_btn.pack(side=tk.LEFT, padx=5)

        # Clear button
        clear_btn = ttk.Button(
            button_frame,
            text="Clear",
            command=self._clear_all
        )
        clear_btn.pack(side=tk.LEFT, padx=5)

        # Cancel button
        cancel_btn = ttk.Button(
            button_frame,
            text="Cancel",
            command=self.window.destroy
        )
        cancel_btn.pack(side=tk.RIGHT, padx=5)

        # Save button
        save_btn = ttk.Button(
            button_frame,
            text="Save",
            command=self._save_and_close
        )
        save_btn.pack(side=tk.RIGHT, padx=5)

    def _create_data_source_section(self, parent):
        """Create the data source directory selection section."""
        source_frame = ttk.LabelFrame(parent, text="Data Source Directories", padding="10")
        source_frame.pack(fill=tk.X, pady=(0, 10))

        # LUPE data source
        lupe_frame = ttk.Frame(source_frame)
        lupe_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(lupe_frame, text="LUPE Outputs:", width=15).pack(side=tk.LEFT)

        self.data_source_var = tk.StringVar(value=self.config.get('data_source_dir', ''))
        self.data_source_entry = ttk.Entry(
            lupe_frame,
            textvariable=self.data_source_var,
            width=50
        )
        self.data_source_entry.pack(side=tk.LEFT, padx=(10, 5), fill=tk.X, expand=True)

        browse_lupe_btn = ttk.Button(
            lupe_frame,
            text="Browse...",
            command=self._browse_data_source
        )
        browse_lupe_btn.pack(side=tk.LEFT)

        # AMPS data source (optional)
        amps_frame = ttk.Frame(source_frame)
        amps_frame.pack(fill=tk.X)

        ttk.Label(amps_frame, text="AMPS Outputs:", width=15).pack(side=tk.LEFT)

        self.amps_source_var = tk.StringVar(value=self.config.get('amps_output_dir', ''))
        self.amps_source_entry = ttk.Entry(
            amps_frame,
            textvariable=self.amps_source_var,
            width=50
        )
        self.amps_source_entry.pack(side=tk.LEFT, padx=(10, 5), fill=tk.X, expand=True)

        browse_amps_btn = ttk.Button(
            amps_frame,
            text="Browse...",
            command=self._browse_amps_source
        )
        browse_amps_btn.pack(side=tk.LEFT)

        # Optional label for AMPS
        ttk.Label(
            source_frame,
            text="(AMPS outputs are optional - leave empty if not using AMPS)",
            font=('Arial', 8),
            foreground='gray'
        ).pack(anchor=tk.W, pady=(5, 0))

    def _create_metadata_section(self, parent):
        """Create the Excel metadata file selection section."""
        metadata_frame = ttk.LabelFrame(parent, text="Excel Metadata File", padding="10")
        metadata_frame.pack(fill=tk.X, pady=(0, 10))

        # File selection row
        file_row = ttk.Frame(metadata_frame)
        file_row.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(file_row, text="Metadata File:", width=15).pack(side=tk.LEFT)

        self.metadata_file_var = tk.StringVar(value=self.config.get('metadata_file', ''))
        self.metadata_file_entry = ttk.Entry(
            file_row,
            textvariable=self.metadata_file_var,
            width=50
        )
        self.metadata_file_entry.pack(side=tk.LEFT, padx=(10, 5), fill=tk.X, expand=True)

        browse_metadata_btn = ttk.Button(
            file_row,
            text="Browse...",
            command=self._browse_metadata_file
        )
        browse_metadata_btn.pack(side=tk.LEFT, padx=(0, 5))

        load_btn = ttk.Button(
            file_row,
            text="Load",
            command=self._load_metadata_file
        )
        load_btn.pack(side=tk.LEFT)

        # Expected columns info
        expected_info = ttk.Label(
            metadata_frame,
            text="Expected columns: animal_id (required), group, condition, sex, timepoint (optional)",
            font=('Arial', 8),
            foreground='gray'
        )
        expected_info.pack(anchor=tk.W)

    def _create_summary_section(self, parent):
        """Create the summary section showing loaded data."""
        summary_frame = ttk.LabelFrame(parent, text="Loaded Data Summary", padding="10")
        summary_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Create text widget with scrollbar for summary
        text_frame = ttk.Frame(summary_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)

        self.summary_text = tk.Text(
            text_frame,
            height=12,
            width=60,
            font=('Courier', 9),
            state=tk.DISABLED,
            wrap=tk.WORD
        )
        scrollbar = ttk.Scrollbar(text_frame, command=self.summary_text.yview)
        self.summary_text.configure(yscrollcommand=scrollbar.set)

        self.summary_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Initial message
        self._update_summary_text("No metadata file loaded.\n\nClick 'Browse...' to select an Excel file, then 'Load' to import the data.")

    def _create_warnings_section(self, parent):
        """Create the warnings section."""
        warnings_frame = ttk.LabelFrame(parent, text="Warnings", padding="10")
        warnings_frame.pack(fill=tk.X, pady=(0, 10))

        # Create text widget for warnings
        self.warnings_text = tk.Text(
            warnings_frame,
            height=4,
            width=60,
            font=('Arial', 9),
            state=tk.DISABLED,
            wrap=tk.WORD,
            foreground='orange'
        )
        self.warnings_text.pack(fill=tk.X)

        # Initial message
        self._update_warnings_text("No warnings.")

    def _update_summary_text(self, text: str):
        """Update the summary text widget."""
        self.summary_text.configure(state=tk.NORMAL)
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(tk.END, text)
        self.summary_text.configure(state=tk.DISABLED)

    def _update_warnings_text(self, text: str):
        """Update the warnings text widget."""
        self.warnings_text.configure(state=tk.NORMAL)
        self.warnings_text.delete(1.0, tk.END)
        self.warnings_text.insert(tk.END, text)
        self.warnings_text.configure(state=tk.DISABLED)

    def _browse_data_source(self):
        """Open directory browser for LUPE data source."""
        initial_dir = self.data_source_var.get() or str(Path.home())
        directory = filedialog.askdirectory(
            parent=self.window,
            initialdir=initial_dir,
            title="Select LUPE Outputs Directory"
        )
        if directory:
            self.data_source_var.set(directory)

    def _browse_amps_source(self):
        """Open directory browser for AMPS output source."""
        initial_dir = self.amps_source_var.get() or self.data_source_var.get() or str(Path.home())
        directory = filedialog.askdirectory(
            parent=self.window,
            initialdir=initial_dir,
            title="Select AMPS Outputs Directory"
        )
        if directory:
            self.amps_source_var.set(directory)

    def _browse_metadata_file(self):
        """Open file browser for Excel metadata file."""
        initial_dir = str(Path(self.metadata_file_var.get()).parent) if self.metadata_file_var.get() else str(Path.home())
        file_path = filedialog.askopenfilename(
            parent=self.window,
            initialdir=initial_dir,
            title="Select Metadata Excel File",
            filetypes=[
                ("Excel files", "*.xlsx *.xls"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.metadata_file_var.set(file_path)

    def _load_metadata_file(self):
        """Load and parse the Excel metadata file."""
        file_path = self.metadata_file_var.get().strip()

        if not file_path:
            messagebox.showwarning(
                "No File Selected",
                "Please select an Excel metadata file first."
            )
            return

        # Load using MetadataLoader
        loader = MetadataLoader(self.config)
        success, result = loader.load_excel(file_path)

        if not success:
            # Result contains error messages
            error_msg = "Failed to load metadata file:\n\n" + "\n".join(result)
            messagebox.showerror("Load Error", error_msg)
            self._update_summary_text("Error loading file. See error message for details.")
            self.loaded_data = None
            return

        # Success - store loaded data
        self.loaded_data = result
        self.warnings = result.get('warnings', [])

        # Update summary display
        self._display_loaded_summary(result)

        # Update warnings display
        if self.warnings:
            self._update_warnings_text("\n".join(self.warnings))
        else:
            self._update_warnings_text("No warnings - metadata matches project configuration.")

        messagebox.showinfo(
            "Metadata Loaded",
            f"Successfully loaded metadata for {len(result.get('unique_animals', []))} animals."
        )

    def _display_loaded_summary(self, data: Dict):
        """Display a summary of loaded metadata."""
        lines = []
        lines.append("METADATA LOADED SUCCESSFULLY")
        lines.append("=" * 40)
        lines.append("")

        # Columns found
        columns = data.get('columns_found', {})
        lines.append("Columns detected:")
        for col_type, col_name in columns.items():
            if col_name:
                lines.append(f"  - {col_type}: '{col_name}'")
        lines.append("")

        # Animal count
        unique_animals = data.get('unique_animals', [])
        lines.append(f"Total unique animals: {len(unique_animals)}")
        lines.append("")

        # Preview animal IDs
        if unique_animals:
            preview = unique_animals[:10]
            lines.append("Animal IDs (first 10):")
            for animal in preview:
                lines.append(f"  {animal}")
            if len(unique_animals) > 10:
                lines.append(f"  ... and {len(unique_animals) - 10} more")
        lines.append("")

        # Assignment structure preview
        assignments = data.get('animal_assignments', {})
        lines.append("Assignment structure:")
        self._format_assignments_preview(assignments, lines, indent=2)

        self._update_summary_text("\n".join(lines))

    def _format_assignments_preview(self, obj: Any, lines: List[str], indent: int = 0):
        """Format assignments structure for preview display."""
        prefix = " " * indent

        if isinstance(obj, list):
            # List of animal IDs
            n_animals = len(obj)
            preview = obj[:3]
            preview_str = ", ".join(str(a) for a in preview)
            if n_animals > 3:
                preview_str += f", ... ({n_animals} total)"
            lines.append(f"{prefix}[{preview_str}]")
        elif isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, list):
                    n_animals = len(value)
                    preview = value[:3]
                    preview_str = ", ".join(str(a) for a in preview)
                    if n_animals > 3:
                        preview_str += f", ..."
                    lines.append(f"{prefix}{key}: [{preview_str}] ({n_animals} animals)")
                elif isinstance(value, dict):
                    lines.append(f"{prefix}{key}:")
                    self._format_assignments_preview(value, lines, indent + 2)

    def _load_existing_metadata(self):
        """
        Load existing metadata from config or Excel file.

        Priority:
        1. If animal_assignments exists in config, display it (no Excel needed)
        2. If metadata_file path is set and file exists, offer to reload
        """
        # Check if assignments already exist in config
        existing_assignments = self.config.get('animal_assignments', {})
        metadata_file = self.config.get('metadata_file', '')

        if existing_assignments:
            # Load from config - no Excel file needed
            self._load_assignments_from_config(existing_assignments)

            # Set metadata file path if available (for reference)
            if metadata_file:
                self.metadata_file_var.set(metadata_file)

        elif metadata_file and Path(metadata_file).exists():
            # No saved assignments but Excel file exists - auto-load
            self.metadata_file_var.set(metadata_file)
            self._load_metadata_file()

    def _load_assignments_from_config(self, assignments: Dict):
        """
        Load and display animal assignments that were saved in the project config.

        This allows the project to remember assignments without requiring
        the original Excel file each time.

        Args:
            assignments: The animal_assignments dict from project config
        """
        # Count unique animals
        unique_animals = []

        def extract_animals(obj):
            """Recursively extract animal IDs from nested structure."""
            if isinstance(obj, list):
                for item in obj:
                    if isinstance(item, str) and item.strip():
                        unique_animals.append(item.strip())
            elif isinstance(obj, dict):
                for value in obj.values():
                    extract_animals(value)

        extract_animals(assignments)
        unique_animals = list(set(unique_animals))

        # Create loaded_data structure matching what MetadataLoader returns
        self.loaded_data = {
            'animal_assignments': assignments,
            'unique_animals': unique_animals,
            'columns_found': {},  # Not available from saved config
            'warnings': [],
            'loaded_from_config': True  # Flag to indicate source
        }

        # Display summary
        self._display_config_summary(assignments, unique_animals)
        self._update_warnings_text("Assignments loaded from saved project configuration.")

    def _display_config_summary(self, assignments: Dict, unique_animals: List[str]):
        """
        Display a summary of assignments loaded from project config.

        Args:
            assignments: The animal_assignments dictionary
            unique_animals: List of unique animal IDs
        """
        lines = []
        lines.append("ASSIGNMENTS LOADED FROM PROJECT")
        lines.append("=" * 40)
        lines.append("")
        lines.append("(Loaded from saved project configuration)")
        lines.append("(Use 'Load' button to reload from Excel file)")
        lines.append("")

        # Animal count
        lines.append(f"Total unique animals: {len(unique_animals)}")
        lines.append("")

        # Preview animal IDs
        if unique_animals:
            preview = sorted(unique_animals)[:10]
            lines.append("Animal IDs (first 10):")
            for animal in preview:
                lines.append(f"  {animal}")
            if len(unique_animals) > 10:
                lines.append(f"  ... and {len(unique_animals) - 10} more")
        lines.append("")

        # Assignment structure preview
        lines.append("Assignment structure:")
        self._format_assignments_preview(assignments, lines, indent=2)

        self._update_summary_text("\n".join(lines))

    def _validate_files(self):
        """
        Validate that animal data folders exist in data source.

        Uses flexible folder discovery with substring matching to find folders
        where animal IDs and timepoints appear within folder names.
        """
        data_source = self.data_source_var.get().strip()

        if not data_source:
            messagebox.showwarning(
                "Validation",
                "Please specify the LUPE outputs directory first."
            )
            return

        data_path = Path(data_source)
        if not data_path.exists():
            messagebox.showerror(
                "Validation Error",
                f"Data source directory does not exist:\n{data_source}"
            )
            return

        # Get animal IDs from loaded data
        if not self.loaded_data:
            messagebox.showwarning(
                "Validation",
                "No metadata loaded. Please load an Excel file first."
            )
            return

        all_ids = self.loaded_data.get('unique_animals', [])

        if not all_ids:
            messagebox.showinfo(
                "Validation",
                "No animal IDs found in loaded metadata."
            )
            return

        # Use folder discovery to find LUPE data folders
        from src.utils.folder_discovery import (
            discover_lupe_folders,
            find_folder_for_animal,
            summarize_discovered_folders
        )

        # Discover all folders with LUPE data
        discovered_folders = discover_lupe_folders(data_path)

        if not discovered_folders:
            messagebox.showerror(
                "Validation Error",
                f"No folders with LUPE analysis data found in:\n{data_source}\n\n"
                "Expected to find folders containing CSV files like:\n"
                "- *_bout_counts_summary.csv\n"
                "- *_time_distribution_overall.csv\n"
                "- etc."
            )
            return

        # Get timepoints from config
        has_timepoints = self.config.get('has_timepoints', False)
        timepoints = [t.get('name', '') for t in self.config.get('timepoints', [])]
        timepoints_to_check = timepoints if has_timepoints and timepoints else [None]

        missing = []
        found = []
        partial = []  # Animals with some but not all timepoints
        found_details = []  # Details of what was found

        for animal_id in set(all_ids):
            if has_timepoints and timepoints:
                # Check for each timepoint
                found_tps = []
                missing_tps = []
                for tp in timepoints:
                    folder_info = find_folder_for_animal(discovered_folders, str(animal_id), tp)
                    if folder_info:
                        found_tps.append(tp)
                        found_details.append(f"{animal_id}/{tp} -> {folder_info['folder_name']}")
                    else:
                        missing_tps.append(tp)

                if missing_tps:
                    if found_tps:
                        partial.append(f"{animal_id} (missing: {', '.join(missing_tps)})")
                    else:
                        missing.append(str(animal_id))
                else:
                    found.append(str(animal_id))
            else:
                # No timepoints - just check for animal
                folder_info = find_folder_for_animal(discovered_folders, str(animal_id))
                if folder_info:
                    found.append(str(animal_id))
                    found_details.append(f"{animal_id} -> {folder_info['folder_name']}")
                else:
                    missing.append(str(animal_id))

        # Show results
        result_lines = []
        result_lines.append(f"Discovered {len(discovered_folders)} folders with LUPE data")
        result_lines.append(f"Found: {len(found)} animals with all data")

        if partial:
            result_lines.append(f"\nPartial: {len(partial)} animals with some missing timepoints:")
            for p in partial[:5]:
                result_lines.append(f"  - {p}")
            if len(partial) > 5:
                result_lines.append(f"  ... and {len(partial) - 5} more")

        if missing:
            result_lines.append(f"\nMissing: {len(missing)} animals not found:")
            for m in missing[:10]:
                result_lines.append(f"  - {m}")
            if len(missing) > 10:
                result_lines.append(f"  ... and {len(missing) - 10} more")

        # Show some matching examples
        if found_details:
            result_lines.append(f"\nMatching examples (first 5):")
            for detail in found_details[:5]:
                result_lines.append(f"  - {detail}")
            if len(found_details) > 5:
                result_lines.append(f"  ... and {len(found_details) - 5} more matches")

        if missing or partial:
            messagebox.showwarning("Validation Results", "\n".join(result_lines))
        else:
            messagebox.showinfo(
                "Validation Successful",
                "\n".join(result_lines)
            )

    def _clear_all(self):
        """Clear loaded data and reset display."""
        self.loaded_data = None
        self.warnings = []
        self.metadata_file_var.set("")
        self._update_summary_text("No metadata file loaded.\n\nClick 'Browse...' to select an Excel file, then 'Load' to import the data.")
        self._update_warnings_text("No warnings.")

    def _save_and_close(self):
        """Save assignments and close the window."""
        # Validate data source is specified
        data_source = self.data_source_var.get().strip()
        if not data_source:
            messagebox.showwarning(
                "Missing Information",
                "Please specify the LUPE outputs directory."
            )
            return

        # Check metadata is loaded
        if not self.loaded_data:
            result = messagebox.askyesno(
                "No Metadata",
                "No metadata has been loaded from an Excel file.\n"
                "Save anyway without animal assignments?"
            )
            if not result:
                return

        # Update config
        self.config['data_source_dir'] = data_source
        self.config['amps_output_dir'] = self.amps_source_var.get().strip()
        self.config['metadata_file'] = self.metadata_file_var.get().strip()

        if self.loaded_data:
            self.config['animal_assignments'] = self.loaded_data.get('animal_assignments', {})

        # Show warning if there are discrepancies
        if self.warnings:
            result = messagebox.askyesno(
                "Warnings Present",
                f"There are {len(self.warnings)} warning(s) about discrepancies "
                "between the Excel file and project configuration.\n\n"
                "Do you want to save anyway?"
            )
            if not result:
                return

        # Call the callback with updated config
        self.callback(self.config)

        # Close the window
        self.window.destroy()


def open_animal_distribution(parent, config: Dict, callback: Callable[[Dict], None]):
    """
    Convenience function to open the Animal Distribution window.

    Args:
        parent: Parent tkinter window
        config: Current project configuration
        callback: Function to call with updated config when saved
    """
    window = AnimalDistributionWindow(parent, config, callback)
    return window
