"""
LUPE-AMPS Pain Scale Analysis GUI

Entry point for the LUPE-AMPS graphical user interface.

This application allows researchers to analyze behavior CSV files using
the LUPE-AMPS (Advanced Multivariate Pain Scale) model to quantify
pain-related behaviors on a continuous scale.

Usage:
    python main_lupe_amps_gui.py

Requirements:
    - Pre-classified behavior CSV files (frame, behavior_id columns)
    - LUPE-AMPS PCA model file (model_AMPS.pkl)

The analysis pipeline includes:
    1. Preprocessing: Downsample and calculate behavior metrics
    2. PCA Projection: Project onto pain scale (PC2 = Pain Behavior Scale)
    3. Metrics Visualization: Visualize occupancy, bouts, and durations
    4. Model Fit Analysis: Test feature importance

Output Structure:
    outputs/{project_name}_LUPE-AMPS/
    ├── Section1_preprocessing/
    │   └── metrics_all_files.csv
    ├── Section2_pain_scale/
    │   ├── pain_scale_projection.png/svg/csv
    ├── Section3_behavior_metrics/
    │   ├── fraction_occupancy.png/svg/csv
    │   ├── number_of_bouts.png/svg/csv
    │   └── bout_duration.png/svg/csv
    └── Section4_model_fit/
        └── feature_importance.png/svg/csv
"""

from src.gui.lupe_amps_window import LupeAmpsGUI


def main():
    """Launch the LUPE-AMPS GUI application."""
    print("=" * 60)
    print("LUPE-AMPS Pain Scale Analysis Tool")
    print("=" * 60)
    print("\nLaunching GUI...\n")

    try:
        app = LupeAmpsGUI()
        app.run()
    except Exception as e:
        print(f"\nError launching GUI: {str(e)}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")


if __name__ == '__main__':
    main()
