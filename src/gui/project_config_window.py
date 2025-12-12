"""
Analyze Project GUI

This module provides a graphical interface for creating and editing project
configurations and starting analysis. Users can define groups, conditions,
sex variables, timepoints, and output directory settings.

The GUI enforces business rules:
- Conditions cannot be created without at least one group
- Sex can be toggled independently of groups/conditions
- Project names must be unique and valid for filenames

Usage:
    from src.gui.project_config_window import ProjectConfigGUI

    app = ProjectConfigGUI()
    app.run()
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, colorchooser
from pathlib import Path

from src.utils.project_config_manager import ProjectConfigManager
from src.gui.animal_distribution_window import open_animal_distribution
from src.core.project_analysis import ProjectAnalyzer


class AnalysisProgressDialog:
    """
    Simple progress dialog for analysis operations.

    Shows a progress bar and status message during analysis.
    """

    def __init__(self, parent):
        """Initialize the progress dialog."""
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Running Analysis...")
        self.dialog.geometry("400x120")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        self.dialog.resizable(False, False)

        # Center on parent
        self.dialog.update_idletasks()
        parent_x = parent.winfo_rootx()
        parent_y = parent.winfo_rooty()
        parent_width = parent.winfo_width()
        parent_height = parent.winfo_height()
        dialog_width = self.dialog.winfo_width()
        dialog_height = self.dialog.winfo_height()
        x = parent_x + (parent_width - dialog_width) // 2
        y = parent_y + (parent_height - dialog_height) // 2
        self.dialog.geometry(f"+{x}+{y}")

        # Content
        frame = ttk.Frame(self.dialog, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)

        self.status_label = ttk.Label(frame, text="Initializing...", font=('Arial', 10))
        self.status_label.pack(pady=(0, 10))

        self.progress_bar = ttk.Progressbar(frame, length=350, mode='determinate')
        self.progress_bar.pack(pady=(0, 10))

        self.percent_label = ttk.Label(frame, text="0%", font=('Arial', 9))
        self.percent_label.pack()

        # Prevent closing
        self.dialog.protocol("WM_DELETE_WINDOW", lambda: None)

    def update_progress(self, message: str, progress: float):
        """Update the progress bar and message."""
        self.status_label.configure(text=message)
        self.progress_bar['value'] = progress * 100
        self.percent_label.configure(text=f"{progress*100:.0f}%")
        self.dialog.update()

    def close(self):
        """Close the progress dialog."""
        try:
            self.dialog.grab_release()
            self.dialog.destroy()
        except Exception:
            pass


class ProjectConfigGUI:
    """
    GUI for creating and editing project configurations and starting analysis.

    Allows users to define:
    - Project name
    - Groups (e.g., Treatment, Control)
    - Conditions (e.g., Day 0, Day 7) - requires at least one group
    - Sex as a variable (independent toggle)
    - Timepoints settings
    - Output directory for analysis results
    """

    # Fixed height for scrollable sections (in pixels)
    SCROLLABLE_SECTION_HEIGHT = 100

    # Default colors for groups (same as in report_generator.py)
    DEFAULT_GROUP_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    def __init__(self):
        """Initialize the Analyze Project GUI."""
        self.root = tk.Tk()
        self.root.title("Analyze Project")
        self.root.geometry("650x950")
        self.root.resizable(True, True)

        # Initialize manager
        self.manager = ProjectConfigManager()

        # Track dynamic widgets
        self.group_entries = []  # List of (frame, entry) tuples
        self.condition_entries = []  # List of (frame, entry) tuples
        self.timepoint_entries = []  # List of (frame, entry) tuples

        # Track animal assignments (populated via Animal Distribution window)
        self.animal_assignments = {}
        self.data_source_dir = ""
        self.amps_output_dir = ""
        self.metadata_file = ""  # Path to Excel metadata file
        self.folder_structure = "nested"  # "nested" (animal/timepoint) or "flat" (animal_timepoint)

        # Build GUI
        self._create_widgets()

        # Update conditions state based on groups
        self._update_conditions_state()

    def _create_scrollable_frame(self, parent, height: int = None) -> tuple:
        """
        Create a scrollable frame container with fixed height.

        Args:
            parent: Parent widget
            height: Fixed height in pixels (uses SCROLLABLE_SECTION_HEIGHT if None)

        Returns:
            tuple: (outer_frame, inner_frame, canvas) where inner_frame is the
                   scrollable container to add widgets to
        """
        if height is None:
            height = self.SCROLLABLE_SECTION_HEIGHT

        # Outer frame to hold canvas and scrollbar
        outer_frame = ttk.Frame(parent)

        # Canvas for scrolling
        canvas = tk.Canvas(outer_frame, height=height, highlightthickness=0)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Scrollbar
        scrollbar = ttk.Scrollbar(outer_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        canvas.configure(yscrollcommand=scrollbar.set)

        # Inner frame inside canvas
        inner_frame = ttk.Frame(canvas)
        canvas_window = canvas.create_window((0, 0), window=inner_frame, anchor=tk.NW)

        def configure_scroll_region(event):
            """Update scroll region when inner frame size changes."""
            canvas.configure(scrollregion=canvas.bbox("all"))

        def configure_canvas_width(event):
            """Update inner frame width to match canvas width."""
            canvas.itemconfig(canvas_window, width=event.width)

        inner_frame.bind("<Configure>", configure_scroll_region)
        canvas.bind("<Configure>", configure_canvas_width)

        # Enable mouse wheel scrolling
        def on_mousewheel(event):
            """Scroll with mouse wheel."""
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def bind_mousewheel(event):
            """Bind mousewheel when mouse enters."""
            canvas.bind_all("<MouseWheel>", on_mousewheel)

        def unbind_mousewheel(event):
            """Unbind mousewheel when mouse leaves."""
            canvas.unbind_all("<MouseWheel>")

        canvas.bind("<Enter>", bind_mousewheel)
        canvas.bind("<Leave>", unbind_mousewheel)

        return outer_frame, inner_frame, canvas

    def _create_widgets(self):
        """Create all GUI widgets."""
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)

        # ========== Title ==========
        title_label = ttk.Label(
            main_frame,
            text="Analyze Project",
            font=('Arial', 16, 'bold')
        )
        title_label.grid(row=0, column=0, pady=(0, 15), sticky=tk.W)

        # ========== Project Name Section ==========
        name_frame = ttk.LabelFrame(main_frame, text="Project Name", padding="10")
        name_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        name_frame.columnconfigure(1, weight=1)

        ttk.Label(name_frame, text="Name:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.project_name_var = tk.StringVar()
        self.project_name_entry = ttk.Entry(name_frame, textvariable=self.project_name_var, width=30)
        self.project_name_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)

        # Load/New buttons
        btn_frame = ttk.Frame(name_frame)
        btn_frame.grid(row=0, column=2, padx=5)
        ttk.Button(btn_frame, text="Load...", command=self._load_project).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="New", command=self._new_project).pack(side=tk.LEFT, padx=2)

        # ========== Groups and Conditions Row (side by side) ==========
        groups_conditions_frame = ttk.Frame(main_frame)
        groups_conditions_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        groups_conditions_frame.columnconfigure(0, weight=1)
        groups_conditions_frame.columnconfigure(1, weight=1)

        # ========== Groups Section (left column) ==========
        groups_frame = ttk.LabelFrame(groups_conditions_frame, text="Groups", padding="10")
        groups_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        groups_frame.columnconfigure(0, weight=1)

        # Add Group button
        ttk.Button(groups_frame, text="+ Add Group", command=self._add_group).grid(
            row=0, column=0, sticky=tk.W, pady=(0, 5)
        )

        # Scrollable container for groups (fixed height)
        groups_scroll_outer, self.groups_container, self.groups_canvas = \
            self._create_scrollable_frame(groups_frame)
        groups_scroll_outer.grid(row=1, column=0, sticky=(tk.W, tk.E))

        # Info label
        self.groups_info = ttk.Label(
            groups_frame,
            text="No groups defined.",
            font=('Arial', 9, 'italic'),
            foreground='gray'
        )
        self.groups_info.grid(row=2, column=0, sticky=tk.W, pady=(5, 0))

        # ========== Conditions Section (right column) ==========
        self.conditions_frame = ttk.LabelFrame(groups_conditions_frame, text="Conditions", padding="10")
        self.conditions_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        self.conditions_frame.columnconfigure(0, weight=1)

        # Add Condition button
        self.add_condition_btn = ttk.Button(
            self.conditions_frame,
            text="+ Add Condition",
            command=self._add_condition
        )
        self.add_condition_btn.grid(row=0, column=0, sticky=tk.W, pady=(0, 5))

        # Scrollable container for conditions (fixed height)
        conditions_scroll_outer, self.conditions_container, self.conditions_canvas = \
            self._create_scrollable_frame(self.conditions_frame)
        conditions_scroll_outer.grid(row=1, column=0, sticky=(tk.W, tk.E))

        # Info/warning label for conditions
        self.conditions_info = ttk.Label(
            self.conditions_frame,
            text="Requires groups.",
            font=('Arial', 9, 'italic'),
            foreground='gray'
        )
        self.conditions_info.grid(row=2, column=0, sticky=tk.W, pady=(5, 0))

        # ========== Sex Variable Section ==========
        sex_frame = ttk.LabelFrame(main_frame, text="Sex Variable", padding="10")
        sex_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=5)

        self.include_sex_var = tk.BooleanVar(value=False)
        self.sex_checkbox = ttk.Checkbutton(
            sex_frame,
            text="Include sex as a variable (Male / Female)",
            variable=self.include_sex_var
        )
        self.sex_checkbox.grid(row=0, column=0, sticky=tk.W)

        # ========== Timepoints Section ==========
        self.timepoints_frame = ttk.LabelFrame(main_frame, text="Timepoints", padding="10")
        self.timepoints_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=5)
        self.timepoints_frame.columnconfigure(0, weight=1)

        self.has_timepoints_var = tk.BooleanVar(value=False)
        self.has_timepoints_checkbox = ttk.Checkbutton(
            self.timepoints_frame,
            text="Experiment has multiple timepoints (dates)",
            variable=self.has_timepoints_var,
            command=self._update_timepoints_state
        )
        self.has_timepoints_checkbox.grid(row=0, column=0, sticky=tk.W)

        self.separate_timepoints_var = tk.BooleanVar(value=False)
        self.separate_timepoints_checkbox = ttk.Checkbutton(
            self.timepoints_frame,
            text="Keep timepoints separated in analysis",
            variable=self.separate_timepoints_var
        )
        self.separate_timepoints_checkbox.grid(row=1, column=0, sticky=tk.W, padx=(20, 0))
        # Initially disabled until "has timepoints" is checked
        self.separate_timepoints_checkbox.configure(state='disabled')

        # Add Timepoint button (initially hidden)
        self.add_timepoint_btn = ttk.Button(
            self.timepoints_frame,
            text="+ Add Timepoint",
            command=self._add_timepoint
        )
        self.add_timepoint_btn.grid(row=2, column=0, sticky=tk.W, pady=(10, 5))
        self.add_timepoint_btn.grid_remove()  # Hidden initially

        # Scrollable container for timepoints (fixed height, initially hidden)
        self.timepoints_scroll_outer, self.timepoints_container, self.timepoints_canvas = \
            self._create_scrollable_frame(self.timepoints_frame)
        self.timepoints_scroll_outer.grid(row=3, column=0, sticky=(tk.W, tk.E))
        self.timepoints_scroll_outer.grid_remove()  # Hidden initially

        # Info label for timepoints
        self.timepoints_info = ttk.Label(
            self.timepoints_frame,
            text="",
            font=('Arial', 9, 'italic'),
            foreground='gray'
        )
        self.timepoints_info.grid(row=4, column=0, sticky=tk.W, pady=(5, 0))

        # Folder structure selection (initially hidden)
        self.folder_structure_frame = ttk.Frame(self.timepoints_frame)
        self.folder_structure_frame.grid(row=5, column=0, sticky=tk.W, pady=(10, 0))
        self.folder_structure_frame.grid_remove()  # Hidden initially

        ttk.Label(
            self.folder_structure_frame,
            text="Folder structure:",
            font=('Arial', 9)
        ).pack(side=tk.LEFT, padx=(0, 10))

        self.folder_structure_var = tk.StringVar(value="nested")
        ttk.Radiobutton(
            self.folder_structure_frame,
            text="Nested (animal/timepoint/)",
            variable=self.folder_structure_var,
            value="nested"
        ).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(
            self.folder_structure_frame,
            text="Flat (animal_timepoint/)",
            variable=self.folder_structure_var,
            value="flat"
        ).pack(side=tk.LEFT, padx=5)

        # ========== Output Directory Section ==========
        output_frame = ttk.LabelFrame(main_frame, text="Output Directory", padding="10")
        output_frame.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=5)
        output_frame.columnconfigure(1, weight=1)

        ttk.Label(output_frame, text="Directory:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.output_dir_var = tk.StringVar(value="outputs")
        self.output_dir_entry = ttk.Entry(output_frame, textvariable=self.output_dir_var, width=40)
        self.output_dir_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)

        ttk.Button(output_frame, text="Browse...", command=self._browse_output_dir).grid(
            row=0, column=2, padx=5
        )

        # ========== Notes Section ==========
        notes_frame = ttk.LabelFrame(main_frame, text="Notes (Optional)", padding="10")
        notes_frame.grid(row=6, column=0, sticky=(tk.W, tk.E), pady=5)
        notes_frame.columnconfigure(0, weight=1)

        self.notes_text = tk.Text(notes_frame, height=3, width=50, wrap=tk.WORD)
        self.notes_text.grid(row=0, column=0, sticky=(tk.W, tk.E))

        # ========== Action Buttons ==========
        action_frame = ttk.Frame(main_frame)
        action_frame.grid(row=7, column=0, pady=15)

        # Distribute Animals button (orange)
        self.distribute_btn = tk.Button(
            action_frame,
            text="Distribute Animals",
            command=self._open_animal_distribution,
            font=('Arial', 11, 'bold'),
            bg='#FF9800',
            fg='white',
            width=17,
            cursor='hand2'
        )
        self.distribute_btn.pack(side=tk.LEFT, padx=5)

        # Start Analysis button (prominent, green)
        self.start_analysis_btn = tk.Button(
            action_frame,
            text="Start Analysis",
            command=self._start_analysis,
            font=('Arial', 12, 'bold'),
            bg='#4CAF50',
            fg='white',
            width=15,
            cursor='hand2'
        )
        self.start_analysis_btn.pack(side=tk.LEFT, padx=5)

        ttk.Button(
            action_frame,
            text="Save Project",
            command=self._save_project
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            action_frame,
            text="Validate",
            command=self._validate_config
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            action_frame,
            text="Cancel",
            command=self.root.destroy
        ).pack(side=tk.LEFT, padx=5)

        # ========== Status Bar ==========
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(
            main_frame,
            textvariable=self.status_var,
            font=('Arial', 9),
            foreground='gray'
        )
        status_bar.grid(row=8, column=0, sticky=tk.W, pady=(5, 0))

    def _add_group(self, name: str = "", color: str = None):
        """
        Add a new group entry field with color picker.

        Args:
            name: Initial group name
            color: Initial color (hex string). If None, uses default color based on index.
        """
        row_frame = ttk.Frame(self.groups_container)
        row_frame.pack(fill=tk.X, pady=2)

        # Group number label
        group_num = len(self.group_entries) + 1
        ttk.Label(row_frame, text=f"Group {group_num}:").pack(side=tk.LEFT, padx=(0, 5))

        # Entry field
        entry_var = tk.StringVar(value=name)
        entry = ttk.Entry(row_frame, textvariable=entry_var, width=20)
        entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # Color variable - use provided color or default based on group index
        if color is None:
            color = self.DEFAULT_GROUP_COLORS[
                len(self.group_entries) % len(self.DEFAULT_GROUP_COLORS)
            ]
        color_var = tk.StringVar(value=color)

        # Color button - shows current color and opens color picker
        color_btn = tk.Button(
            row_frame,
            text="",
            width=3,
            bg=color,
            relief=tk.RAISED,
            command=lambda cv=color_var: self._pick_group_color(cv)
        )
        color_btn.pack(side=tk.LEFT, padx=5)

        # Store reference to button in color_var for updating background
        color_var.color_btn = color_btn

        # Remove button
        remove_btn = ttk.Button(
            row_frame,
            text="X",
            width=3,
            command=lambda f=row_frame: self._remove_group(f)
        )
        remove_btn.pack(side=tk.RIGHT, padx=5)

        # Track the entry with color
        self.group_entries.append((row_frame, entry_var, color_var))

        # Update UI state
        self._update_groups_info()
        self._update_conditions_state()

        # Focus on new entry
        entry.focus_set()

    def _pick_group_color(self, color_var: tk.StringVar):
        """
        Open color picker dialog for a group.

        Args:
            color_var: StringVar holding the current color
        """
        current_color = color_var.get()

        # Open color chooser dialog
        result = colorchooser.askcolor(
            color=current_color,
            title="Choose Group Color"
        )

        if result[1] is not None:
            # User selected a color (result is ((r,g,b), '#hexcolor'))
            new_color = result[1]
            color_var.set(new_color)

            # Update button background color
            if hasattr(color_var, 'color_btn'):
                color_var.color_btn.configure(bg=new_color)

    def _remove_group(self, frame):
        """Remove a group entry."""
        # Find and remove from tracking list
        for i, (f, name_var, color_var) in enumerate(self.group_entries):
            if f == frame:
                self.group_entries.pop(i)
                break

        # Destroy the frame
        frame.destroy()

        # Renumber remaining groups
        self._renumber_groups()

        # Update UI state
        self._update_groups_info()
        self._update_conditions_state()

    def _renumber_groups(self):
        """Renumber group labels after removal."""
        for i, (frame, name_var, color_var) in enumerate(self.group_entries):
            # Find the label widget and update text
            for child in frame.winfo_children():
                if isinstance(child, ttk.Label):
                    child.configure(text=f"Group {i+1}:")
                    break

    def _update_groups_info(self):
        """Update the groups info label."""
        count = len(self.group_entries)
        if count == 0:
            self.groups_info.configure(
                text="No groups defined.",
                foreground='gray'
            )
        else:
            self.groups_info.configure(
                text=f"{count} group(s) defined",
                foreground='green'
            )

    def _add_condition(self, name: str = ""):
        """Add a new condition entry field."""
        row_frame = ttk.Frame(self.conditions_container)
        row_frame.pack(fill=tk.X, pady=2)

        # Condition number label
        cond_num = len(self.condition_entries) + 1
        ttk.Label(row_frame, text=f"Condition {cond_num}:").pack(side=tk.LEFT, padx=(0, 5))

        # Entry field
        entry_var = tk.StringVar(value=name)
        entry = ttk.Entry(row_frame, textvariable=entry_var, width=25)
        entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # Remove button
        remove_btn = ttk.Button(
            row_frame,
            text="X",
            width=3,
            command=lambda f=row_frame: self._remove_condition(f)
        )
        remove_btn.pack(side=tk.RIGHT, padx=5)

        # Track the entry
        self.condition_entries.append((row_frame, entry_var))

        # Update UI
        self._update_conditions_info()

        # Focus on new entry
        entry.focus_set()

    def _remove_condition(self, frame):
        """Remove a condition entry."""
        # Find and remove from tracking list
        for i, (f, var) in enumerate(self.condition_entries):
            if f == frame:
                self.condition_entries.pop(i)
                break

        # Destroy the frame
        frame.destroy()

        # Renumber remaining conditions
        self._renumber_conditions()

        # Update UI
        self._update_conditions_info()

    def _renumber_conditions(self):
        """Renumber condition labels after removal."""
        for i, (frame, var) in enumerate(self.condition_entries):
            for child in frame.winfo_children():
                if isinstance(child, ttk.Label):
                    child.configure(text=f"Condition {i+1}:")
                    break

    def _update_conditions_state(self):
        """Enable/disable conditions section based on groups count."""
        has_groups = len(self.group_entries) > 0

        if has_groups:
            # Enable conditions section
            self.add_condition_btn.configure(state='normal')
            self.conditions_info.configure(
                text="Apply to all groups.",
                foreground='green' if self.condition_entries else 'gray'
            )
            # Enable existing condition entries
            for frame, var in self.condition_entries:
                for child in frame.winfo_children():
                    if isinstance(child, ttk.Entry):
                        child.configure(state='normal')
                    elif isinstance(child, ttk.Button):
                        child.configure(state='normal')
        else:
            # Disable conditions section
            self.add_condition_btn.configure(state='disabled')
            self.conditions_info.configure(
                text="Requires groups.",
                foreground='orange'
            )
            # Disable existing condition entries (but don't remove them)
            for frame, var in self.condition_entries:
                for child in frame.winfo_children():
                    if isinstance(child, ttk.Entry):
                        child.configure(state='disabled')
                    elif isinstance(child, ttk.Button):
                        child.configure(state='disabled')

    def _update_conditions_info(self):
        """Update the conditions info label."""
        count = len(self.condition_entries)
        if count == 0:
            if len(self.group_entries) > 0:
                self.conditions_info.configure(
                    text="No conditions defined.",
                    foreground='gray'
                )
        else:
            self.conditions_info.configure(
                text=f"{count} condition(s) defined",
                foreground='green'
            )

    def _update_timepoints_state(self):
        """Enable/disable timepoints section based on toggle."""
        if self.has_timepoints_var.get():
            self.separate_timepoints_checkbox.configure(state='normal')
            self.add_timepoint_btn.grid()  # Show add button
            self.timepoints_scroll_outer.grid()  # Show scrollable container
            self.folder_structure_frame.grid()  # Show folder structure options
            self._update_timepoints_info()
        else:
            self.separate_timepoints_var.set(False)
            self.separate_timepoints_checkbox.configure(state='disabled')
            self.add_timepoint_btn.grid_remove()  # Hide add button
            self.timepoints_scroll_outer.grid_remove()  # Hide scrollable container
            self.folder_structure_frame.grid_remove()  # Hide folder structure options
            self.timepoints_info.configure(text="")

    def _add_timepoint(self, name: str = ""):
        """Add a new timepoint entry field."""
        row_frame = ttk.Frame(self.timepoints_container)
        row_frame.pack(fill=tk.X, pady=2)

        # Timepoint number label
        tp_num = len(self.timepoint_entries) + 1
        ttk.Label(row_frame, text=f"Timepoint {tp_num}:").pack(side=tk.LEFT, padx=(0, 5))

        # Entry field
        entry_var = tk.StringVar(value=name)
        entry = ttk.Entry(row_frame, textvariable=entry_var, width=25)
        entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # Remove button
        remove_btn = ttk.Button(
            row_frame,
            text="X",
            width=3,
            command=lambda f=row_frame: self._remove_timepoint(f)
        )
        remove_btn.pack(side=tk.RIGHT, padx=5)

        # Track the entry
        self.timepoint_entries.append((row_frame, entry_var))

        # Update UI state
        self._update_timepoints_info()

        # Focus on new entry
        entry.focus_set()

    def _remove_timepoint(self, frame):
        """Remove a timepoint entry."""
        # Find and remove from tracking list
        for i, (f, var) in enumerate(self.timepoint_entries):
            if f == frame:
                self.timepoint_entries.pop(i)
                break

        # Destroy the frame
        frame.destroy()

        # Renumber remaining timepoints
        self._renumber_timepoints()

        # Update UI state
        self._update_timepoints_info()

    def _renumber_timepoints(self):
        """Renumber timepoint labels after removal."""
        for i, (frame, var) in enumerate(self.timepoint_entries):
            for child in frame.winfo_children():
                if isinstance(child, ttk.Label):
                    child.configure(text=f"Timepoint {i+1}:")
                    break

    def _update_timepoints_info(self):
        """Update the timepoints info label."""
        if not self.has_timepoints_var.get():
            self.timepoints_info.configure(text="")
            return

        count = len(self.timepoint_entries)
        if count == 0:
            self.timepoints_info.configure(
                text="Add timepoint names (e.g., Day0, Day7, Week2)",
                foreground='gray'
            )
        else:
            self.timepoints_info.configure(
                text=f"{count} timepoint(s) defined",
                foreground='green'
            )

    def _browse_output_dir(self):
        """Open directory browser for output directory selection."""
        directory = filedialog.askdirectory(
            title="Select Output Directory",
            initialdir=self.output_dir_var.get() or "."
        )
        if directory:
            self.output_dir_var.set(directory)

    def _open_animal_distribution(self):
        """Open the Animal Distribution window to assign animals to factors."""
        # Build current config to pass to the distribution window
        config = self._get_current_config()

        # Include any existing animal assignments
        config['animal_assignments'] = self.animal_assignments
        config['data_source_dir'] = self.data_source_dir
        config['amps_output_dir'] = self.amps_output_dir

        def on_save(updated_config):
            """Callback when animal distribution is saved."""
            self.animal_assignments = updated_config.get('animal_assignments', {})
            self.data_source_dir = updated_config.get('data_source_dir', '')
            self.amps_output_dir = updated_config.get('amps_output_dir', '')

            # Update status
            animal_count = self._count_animals(self.animal_assignments)
            if animal_count > 0:
                self.status_var.set(f"Animals distributed: {animal_count} animals assigned")
            else:
                self.status_var.set("No animals assigned")

        # Open the window
        open_animal_distribution(self.root, config, on_save)

    def _count_animals(self, assignments: dict) -> int:
        """Count total animals in assignments structure."""
        count = 0

        def count_recursive(obj):
            nonlocal count
            if isinstance(obj, list):
                count += len([a for a in obj if isinstance(a, str) and a.strip()])
            elif isinstance(obj, dict):
                for value in obj.values():
                    count_recursive(value)

        count_recursive(assignments)
        return count

    def _start_analysis(self):
        """Start the analysis with current configuration."""
        # Validate configuration first
        config = self._get_current_config()
        is_valid, errors = self.manager.validate_config(config)

        if not is_valid:
            error_msg = "Cannot start analysis - validation errors:\n\n" + "\n".join(f"- {e}" for e in errors)
            messagebox.showerror("Validation Failed", error_msg)
            return

        # Check output directory is specified
        output_dir = self.output_dir_var.get().strip()
        if not output_dir:
            messagebox.showerror("Error", "Please specify an output directory.")
            return

        # Check animal assignments exist
        if not config.get('animal_assignments'):
            messagebox.showerror(
                "Error",
                "Please distribute animals first using the 'Distribute Animals' button."
            )
            return

        # Check data source directory
        if not config.get('data_source_dir'):
            messagebox.showerror(
                "Error",
                "Please specify the LUPE outputs directory in 'Distribute Animals'."
            )
            return

        # Create and show progress dialog
        progress_dialog = AnalysisProgressDialog(self.root)

        def run_analysis():
            """Run analysis in the main thread with progress updates."""
            try:
                analyzer = ProjectAnalyzer(config)

                def progress_callback(progress_info):
                    """Update progress dialog."""
                    progress_dialog.update_progress(
                        progress_info.message,
                        progress_info.progress
                    )
                    self.root.update()

                success, result = analyzer.run_complete_analysis(progress_callback)

                progress_dialog.close()

                if success:
                    # Show success message with summary
                    outputs = result.get('outputs', {})
                    n_csv = len(outputs.get('csv', []))
                    n_figures = len(outputs.get('figures', []))

                    messagebox.showinfo(
                        "Analysis Complete",
                        f"Statistical analysis completed successfully!\n\n"
                        f"Output files:\n"
                        f"  - {n_csv} CSV files\n"
                        f"  - {n_figures} figure files\n"
                        f"  - 1 statistical report\n\n"
                        f"Time: {result.get('elapsed_time', 'N/A')}\n"
                        f"Animals analyzed: {result.get('n_animals', 0)}\n"
                        f"Statistical tests: {result.get('n_tests', 0)}\n\n"
                        f"Output directory:\n{result.get('output_dir', output_dir)}"
                    )
                    self.status_var.set("Analysis complete!")
                else:
                    messagebox.showerror(
                        "Analysis Failed",
                        f"Analysis failed:\n\n{result}"
                    )
                    self.status_var.set("Analysis failed")

            except Exception as e:
                progress_dialog.close()
                messagebox.showerror(
                    "Error",
                    f"An error occurred during analysis:\n\n{str(e)}"
                )
                self.status_var.set("Analysis error")
                import traceback
                traceback.print_exc()

        # Run analysis
        self.status_var.set("Running analysis...")
        self.root.after(100, run_analysis)

    def _get_current_config(self) -> dict:
        """Build configuration dictionary from current GUI state."""
        config = {
            "project_name": self.project_name_var.get().strip(),
            "groups": [
                {"name": name_var.get().strip(), "color": color_var.get()}
                for _, name_var, color_var in self.group_entries
                if name_var.get().strip()
            ],
            "conditions": [
                {"name": var.get().strip()}
                for _, var in self.condition_entries
                if var.get().strip()
            ],
            "include_sex": self.include_sex_var.get(),
            "has_timepoints": self.has_timepoints_var.get(),
            "separate_timepoints": self.separate_timepoints_var.get(),
            "timepoints": [
                {"name": var.get().strip()}
                for _, var in self.timepoint_entries
                if var.get().strip()
            ],
            "folder_structure": self.folder_structure_var.get(),
            "output_dir": self.output_dir_var.get().strip(),
            "data_source_dir": self.data_source_dir,
            "amps_output_dir": self.amps_output_dir,
            "metadata_file": self.metadata_file,
            "animal_assignments": self.animal_assignments,
            "notes": self.notes_text.get("1.0", tk.END).strip()
        }
        return config

    def _load_config_to_gui(self, config: dict):
        """Load a configuration dictionary into the GUI."""
        # Clear existing entries
        self._clear_all()

        # Set project name
        self.project_name_var.set(config.get("project_name", ""))

        # Add groups with colors
        for group in config.get("groups", []):
            if isinstance(group, dict):
                name = group.get("name", "")
                color = group.get("color", None)
            else:
                name = str(group)
                color = None
            self._add_group(name, color)

        # Add conditions
        for condition in config.get("conditions", []):
            name = condition.get("name", "") if isinstance(condition, dict) else str(condition)
            self._add_condition(name)

        # Set sex variable
        self.include_sex_var.set(config.get("include_sex", False))

        # Set timepoints
        self.has_timepoints_var.set(config.get("has_timepoints", False))
        self.separate_timepoints_var.set(config.get("separate_timepoints", False))
        self._update_timepoints_state()

        # Add timepoint names
        for timepoint in config.get("timepoints", []):
            name = timepoint.get("name", "") if isinstance(timepoint, dict) else str(timepoint)
            self._add_timepoint(name)

        # Set folder structure
        self.folder_structure_var.set(config.get("folder_structure", "nested"))

        # Set output directory
        self.output_dir_var.set(config.get("output_dir", "outputs"))

        # Load animal assignments
        self.animal_assignments = config.get("animal_assignments", {})
        self.data_source_dir = config.get("data_source_dir", "")
        self.amps_output_dir = config.get("amps_output_dir", "")
        self.metadata_file = config.get("metadata_file", "")

        # Set notes
        self.notes_text.delete("1.0", tk.END)
        self.notes_text.insert("1.0", config.get("notes", ""))

        # Update states
        self._update_conditions_state()

        # Update status with animal count
        animal_count = self._count_animals(self.animal_assignments)
        if animal_count > 0:
            self.status_var.set(f"Loaded project with {animal_count} animals assigned")

    def _clear_all(self):
        """Clear all entries and reset to empty state."""
        # Clear groups
        for frame, _ in self.group_entries:
            frame.destroy()
        self.group_entries = []

        # Clear conditions
        for frame, _ in self.condition_entries:
            frame.destroy()
        self.condition_entries = []

        # Clear timepoints
        for frame, _ in self.timepoint_entries:
            frame.destroy()
        self.timepoint_entries = []

        # Reset other fields
        self.project_name_var.set("")
        self.include_sex_var.set(False)
        self.has_timepoints_var.set(False)
        self.separate_timepoints_var.set(False)
        self.folder_structure_var.set("nested")
        self.output_dir_var.set("outputs")
        self.notes_text.delete("1.0", tk.END)

        # Reset animal assignments
        self.animal_assignments = {}
        self.data_source_dir = ""
        self.amps_output_dir = ""
        self.metadata_file = ""

        # Update UI
        self._update_groups_info()
        self._update_conditions_state()
        self._update_timepoints_state()

    def _new_project(self):
        """Start a new empty project."""
        if self.group_entries or self.condition_entries:
            if not messagebox.askyesno(
                "New Project",
                "Clear current configuration and start new project?"
            ):
                return

        self._clear_all()
        self.status_var.set("New project started")
        self.project_name_entry.focus_set()

    def _load_project(self):
        """Load an existing project from file."""
        projects = self.manager.list_projects()

        if not projects:
            messagebox.showinfo("No Projects", "No saved projects found.")
            return

        # Create a simple selection dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Load Project")
        dialog.geometry("300x400")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="Select a project:", font=('Arial', 11)).pack(pady=10)

        # Listbox with projects
        listbox = tk.Listbox(dialog, height=15)
        listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        for project in projects:
            listbox.insert(tk.END, project)

        def on_select():
            selection = listbox.curselection()
            if selection:
                project_name = listbox.get(selection[0])
                dialog.destroy()
                self._do_load_project(project_name)

        def on_delete():
            selection = listbox.curselection()
            if selection:
                project_name = listbox.get(selection[0])
                if messagebox.askyesno("Delete Project", f"Delete '{project_name}'?"):
                    self.manager.delete_project(project_name)
                    listbox.delete(selection[0])

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="Load", command=on_select).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Delete", command=on_delete).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)

        # Double-click to load
        listbox.bind('<Double-1>', lambda e: on_select())

    def _do_load_project(self, name: str):
        """Actually load a project by name."""
        try:
            config = self.manager.load_project(name)
            self._load_config_to_gui(config)
            self.status_var.set(f"Loaded: {name}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load project: {str(e)}")

    def _validate_config(self):
        """Validate current configuration and show results."""
        config = self._get_current_config()
        is_valid, errors = self.manager.validate_config(config)

        if is_valid:
            messagebox.showinfo(
                "Validation Passed",
                "Configuration is valid and ready to save."
            )
            self.status_var.set("Validation passed")
        else:
            error_msg = "Validation errors:\n\n" + "\n".join(f"- {e}" for e in errors)
            messagebox.showwarning("Validation Failed", error_msg)
            self.status_var.set("Validation failed")

    def _save_project(self):
        """Save the current project configuration."""
        config = self._get_current_config()

        # Validate first
        is_valid, errors = self.manager.validate_config(config)
        if not is_valid:
            error_msg = "Cannot save - validation errors:\n\n" + "\n".join(f"- {e}" for e in errors)
            messagebox.showerror("Save Failed", error_msg)
            return

        # Check if overwriting
        project_name = config['project_name']
        if self.manager.project_exists(project_name):
            if not messagebox.askyesno(
                "Overwrite?",
                f"Project '{project_name}' already exists. Overwrite?"
            ):
                return

        # Save
        try:
            file_path = self.manager.save_project(config)
            self.status_var.set(f"Saved: {project_name}")
            messagebox.showinfo(
                "Project Saved",
                f"Project saved successfully!\n\nFile: {file_path}"
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {str(e)}")

    def run(self):
        """Start the GUI application."""
        self.root.mainloop()


if __name__ == '__main__':
    app = ProjectConfigGUI()
    app.run()
