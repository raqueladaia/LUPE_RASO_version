"""
LUPE Analysis Tool - GUI Entry Point

This script launches the graphical user interface for LUPE analysis.

Usage:
    python main_lupe_gui.py

The GUI provides an easy-to-use interface for:
- Loading behavior data
- Running various analyses
- Exporting results
"""

import sys
import os
from pathlib import Path

# FORCE NUMBA TO RECOMPILE (disable caching to ensure new code is used)
os.environ['NUMBA_DISABLE_JIT'] = '0'  # Keep JIT enabled
os.environ['NUMBA_CACHE_DIR'] = ''  # Disable file-based cache
# This forces Numba to compile from source code every time

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Check critical dependencies before importing GUI
def check_dependencies():
    """
    Check if all required dependencies are installed.
    Provides helpful error messages if dependencies are missing.
    """
    missing_deps = []

    # Check psutil (required for memory monitoring in Phase 1 fixes)
    try:
        import psutil
    except ImportError:
        missing_deps.append("psutil")

    # Check matplotlib (required for plotting)
    try:
        import matplotlib
    except ImportError:
        missing_deps.append("matplotlib")

    # Check pandas (required for data processing)
    try:
        import pandas
    except ImportError:
        missing_deps.append("pandas")

    # Check numpy (required for numerical operations)
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")

    # Check numba (required for JIT-compiled feature extraction)
    try:
        import numba
    except ImportError:
        missing_deps.append("numba")

    # Check h5py (required for HDF5 file support)
    try:
        import h5py
    except ImportError:
        missing_deps.append("h5py")

    # Check tables/pytables (required by pandas for HDF5 reading)
    try:
        import tables
    except ImportError:
        missing_deps.append("tables")

    # Check tkinter (usually comes with Python)
    try:
        import tkinter
    except ImportError:
        missing_deps.append("tkinter (comes with Python - may need to reinstall Python)")

    if missing_deps:
        error_msg = (
            "ERROR: Missing required dependencies:\n\n"
            + "\n".join(f"  - {dep}" for dep in missing_deps)
            + "\n\nTo install missing dependencies, run:\n"
            + f"  pip install {' '.join([d.split(' ')[0] for d in missing_deps])}\n"
        )
        print(error_msg)

        # Try to show GUI error if tkinter is available
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Missing Dependencies", error_msg)
            root.destroy()
        except:
            pass

        sys.exit(1)

# Check dependencies before importing GUI
check_dependencies()

from src.gui.main_window import LupeGUI


def main():
    """Launch the LUPE GUI application."""
    print("Starting LUPE Analysis Tool GUI...")
    print("If the GUI doesn't appear, check for error messages below.\n")

    try:
        app = LupeGUI()
        app.run()
    except Exception as e:
        print(f"Error starting GUI: {e}")
        print("\nIf you encounter issues, try using the CLI instead:")
        print("  python main_cli.py --help")
        sys.exit(1)


if __name__ == '__main__':
    main()
