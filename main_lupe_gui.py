"""
LUPE Analysis Tool - GUI Entry Point

This script launches the graphical user interface for LUPE analysis.

Usage:
    python main_gui.py

The GUI provides an easy-to-use interface for:
- Loading behavior data
- Running various analyses
- Exporting results
"""

import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

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
