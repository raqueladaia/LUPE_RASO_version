"""
LUPE Analyze Project GUI

Entry point for the Analyze Project GUI.
This GUI allows users to define project metadata and start analysis:
- Groups (e.g., Treatment, Control)
- Conditions (e.g., Day 0, Day 7, Day 14)
- Whether sex is a variable
- Timepoint settings
- Output directory for analysis results

Project configurations are saved as JSON files in the projects/ directory.

Usage:
    python main_project_config_gui.py

Or launch from the main launcher using the "Analyze Project" button.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.gui.project_config_window import ProjectConfigGUI


def main():
    """Main entry point for the Analyze Project GUI."""
    print("=" * 60)
    print("LUPE Analyze Project")
    print("=" * 60)
    print("\nLaunching Analyze Project GUI...\n")

    try:
        app = ProjectConfigGUI()
        app.run()
    except Exception as e:
        print(f"\nError launching Analyze Project GUI: {str(e)}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")


if __name__ == '__main__':
    main()
