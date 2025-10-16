"""
Command-Line Interface for LUPE Analysis Tool

This CLI provides access to all LUPE analysis functions from the command line.

Usage:
    python -m src.cli.main classify --model path/to/model.pkl --input data/ --output outputs/
    python -m src.cli.main analyze --behaviors behaviors.pkl --output outputs/ --all
    python -m src.cli.main export --behaviors behaviors.pkl --output outputs/csv/

For help on any command:
    python -m src.cli.main <command> --help
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Import core modules
from src.core.data_loader import load_model, load_behaviors, save_behaviors
from src.core.classification import classify_behaviors
from src.core.dlc_preprocessing import preprocess_dlc_csv, batch_process_dlc_files
from src.core.analysis_csv_export import export_behaviors_to_csv, export_behavior_summary
from src.core.analysis_instance_counts import analyze_instance_counts
from src.core.analysis_total_frames import analyze_total_frames
from src.core.analysis_durations import analyze_bout_durations
from src.core.analysis_binned_timeline import analyze_binned_timeline
from src.core.analysis_transitions import analyze_transitions
from src.utils.config_manager import get_config


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description='LUPE Analysis Tool - Command Line Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preprocess DLC CSV files
  lupe-cli preprocess --input dlc_data/*.csv --output pose_data.pkl

  # Classify behaviors from pose data
  lupe-cli classify --model model.pkl --input pose_data.pkl --output behaviors.pkl

  # Run all analyses
  lupe-cli analyze --behaviors behaviors.pkl --output outputs/ --all

  # Export to CSV
  lupe-cli export --behaviors behaviors.pkl --output csv/

  # Run specific analysis
  lupe-cli analyze --behaviors behaviors.pkl --output outputs/ --instance-counts

For more information, visit: https://github.com/your-repo/LUPE-analysis
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # ============ CLASSIFY Command ============
    classify_parser = subparsers.add_parser(
        'classify',
        help='Classify behaviors from pose data'
    )
    classify_parser.add_argument(
        '--model', '-m',
        required=True,
        help='Path to the trained A-SOiD model (.pkl file)'
    )
    classify_parser.add_argument(
        '--input', '-i',
        required=True,
        nargs='+',
        help='Input pose data files (.npy or .pkl)'
    )
    classify_parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output path for behavior classifications (.pkl file)'
    )
    classify_parser.add_argument(
        '--framerate',
        type=int,
        help='Video framerate (default: from config)'
    )

    # ============ PREPROCESS Command ============
    preprocess_parser = subparsers.add_parser(
        'preprocess',
        help='Preprocess raw DeepLabCut CSV files'
    )
    preprocess_parser.add_argument(
        '--input', '-i',
        required=True,
        nargs='+',
        help='Input DLC CSV files (one or more)'
    )
    preprocess_parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output path for processed pose data (.pkl file)'
    )
    preprocess_parser.add_argument(
        '--likelihood-threshold',
        type=float,
        help='Minimum likelihood threshold for filtering (default: from config)'
    )
    preprocess_parser.add_argument(
        '--skip-rows',
        type=int,
        default=0,
        help='Number of extra rows to skip in CSV (default: 0)'
    )

    # ============ ANALYZE Command ============
    analyze_parser = subparsers.add_parser(
        'analyze',
        help='Run analysis on behavior classifications'
    )
    analyze_parser.add_argument(
        '--behaviors', '-b',
        required=True,
        help='Path to behaviors file (.pkl)'
    )
    analyze_parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output directory for analysis results'
    )
    analyze_parser.add_argument(
        '--all',
        action='store_true',
        help='Run all analyses'
    )
    analyze_parser.add_argument(
        '--instance-counts',
        action='store_true',
        help='Analyze behavior instance counts'
    )
    analyze_parser.add_argument(
        '--total-frames',
        action='store_true',
        help='Analyze total frames per behavior'
    )
    analyze_parser.add_argument(
        '--durations',
        action='store_true',
        help='Analyze behavior bout durations'
    )
    analyze_parser.add_argument(
        '--timeline',
        action='store_true',
        help='Analyze behavior timeline'
    )
    analyze_parser.add_argument(
        '--transitions',
        action='store_true',
        help='Analyze behavior transitions'
    )
    analyze_parser.add_argument(
        '--bin-minutes',
        type=float,
        default=1.0,
        help='Bin size in minutes for timeline analysis'
    )

    # ============ EXPORT Command ============
    export_parser = subparsers.add_parser(
        'export',
        help='Export behaviors to CSV format'
    )
    export_parser.add_argument(
        '--behaviors', '-b',
        required=True,
        help='Path to behaviors file (.pkl)'
    )
    export_parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output directory for CSV files'
    )
    export_parser.add_argument(
        '--with-names',
        action='store_true',
        help='Include behavior names in CSV'
    )
    export_parser.add_argument(
        '--summary',
        action='store_true',
        help='Also create summary CSV'
    )

    # ============ CONFIG Command ============
    config_parser = subparsers.add_parser(
        'config',
        help='View or modify configuration'
    )
    config_parser.add_argument(
        '--show',
        action='store_true',
        help='Show current configuration'
    )
    config_parser.add_argument(
        '--behaviors',
        action='store_true',
        help='Show behavior definitions'
    )

    # Parse arguments
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Execute command
    try:
        if args.command == 'classify':
            cmd_classify(args)
        elif args.command == 'preprocess':
            cmd_preprocess(args)
        elif args.command == 'analyze':
            cmd_analyze(args)
        elif args.command == 'export':
            cmd_export(args)
        elif args.command == 'config':
            cmd_config(args)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


def cmd_classify(args):
    """Execute the classify command."""
    print("=" * 60)
    print("LUPE Behavior Classification")
    print("=" * 60)

    # Load model
    print(f"\nLoading model from: {args.model}")
    model = load_model(args.model)
    print("✓ Model loaded successfully")

    # Load pose data
    print(f"\nLoading {len(args.input)} pose data files...")
    pose_data = []
    file_names = []

    for input_path in args.input:
        path = Path(input_path)
        if path.suffix == '.npy':
            data = np.load(input_path)
        elif path.suffix == '.pkl':
            from src.core.data_loader import load_data
            data = load_data(input_path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        pose_data.append(data)
        file_names.append(path.stem)

    print(f"✓ Loaded {len(pose_data)} files")

    # Classify behaviors
    print("\nClassifying behaviors...")
    predictions = classify_behaviors(model, pose_data, framerate=args.framerate)
    print("✓ Classification complete")

    # Create behaviors dictionary
    behaviors_dict = {name: pred for name, pred in zip(file_names, predictions)}

    # Save results
    print(f"\nSaving results to: {args.output}")
    save_behaviors(behaviors_dict, args.output)
    print("✓ Results saved")

    print("\n" + "=" * 60)
    print("Classification completed successfully!")
    print("=" * 60)


def cmd_preprocess(args):
    """Execute the preprocess command."""
    import pickle

    print("=" * 60)
    print("LUPE DLC CSV Preprocessing")
    print("=" * 60)

    # Handle single or multiple files
    input_files = []
    for input_pattern in args.input:
        input_path = Path(input_pattern)

        # Check if it's a glob pattern or single file
        if '*' in str(input_pattern) or '?' in str(input_pattern):
            # Glob pattern
            parent = input_path.parent if input_path.parent != Path('.') else Path.cwd()
            pattern = input_path.name
            matches = list(parent.glob(pattern))
            input_files.extend(matches)
        else:
            # Single file
            if input_path.exists():
                input_files.append(input_path)
            else:
                raise FileNotFoundError(f"File not found: {input_path}")

    if not input_files:
        raise FileNotFoundError("No CSV files found matching the input pattern")

    print(f"\nFound {len(input_files)} CSV file(s) to process")

    # Process files
    processed_data = {}

    for i, csv_file in enumerate(input_files, 1):
        print(f"\n[{i}/{len(input_files)}] Processing: {csv_file.name}")
        print("-" * 60)

        try:
            # Preprocess single CSV file
            pose_data = preprocess_dlc_csv(
                str(csv_file),
                likelihood_threshold=args.likelihood_threshold,
                save_output=False  # We'll save everything together
            )

            # Use filename stem as key
            file_key = csv_file.stem
            processed_data[file_key] = pose_data

            print(f"✓ Processed {csv_file.name}: shape {pose_data.shape}")

        except Exception as e:
            print(f"✗ Error processing {csv_file.name}: {str(e)}")
            print("  Skipping this file...")
            continue

    if not processed_data:
        raise RuntimeError("No files were successfully processed")

    # Save processed data
    print(f"\nSaving processed data to: {args.output}")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(processed_data, f)

    print(f"✓ Saved {len(processed_data)} file(s) to {args.output}")

    print("\n" + "=" * 60)
    print("Preprocessing completed successfully!")
    print(f"Processed files: {len(processed_data)}")
    print("=" * 60)
    print("\nNext steps:")
    print(f"  1. Classify behaviors:")
    print(f"     python main_cli.py classify --model MODEL.pkl --input {args.output} --output behaviors.pkl")
    print(f"  2. Run analyses:")
    print(f"     python main_cli.py analyze --behaviors behaviors.pkl --output results/ --all")


def cmd_analyze(args):
    """Execute the analyze command."""
    print("=" * 60)
    print("LUPE Behavior Analysis")
    print("=" * 60)

    # Load behaviors
    print(f"\nLoading behaviors from: {args.behaviors}")
    behaviors = load_behaviors(args.behaviors)
    print(f"✓ Loaded behaviors for {len(behaviors)} files")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which analyses to run
    run_all = args.all
    analyses_to_run = []

    if run_all or args.instance_counts:
        analyses_to_run.append(('Instance Counts', 'instance_counts',
                               lambda: analyze_instance_counts(behaviors, str(output_dir / 'instance_counts'))))

    if run_all or args.total_frames:
        analyses_to_run.append(('Total Frames', 'total_frames',
                               lambda: analyze_total_frames(behaviors, str(output_dir / 'total_frames'))))

    if run_all or args.durations:
        analyses_to_run.append(('Bout Durations', 'durations',
                               lambda: analyze_bout_durations(behaviors, str(output_dir / 'durations'))))

    if run_all or args.timeline:
        analyses_to_run.append(('Timeline', 'timeline',
                               lambda: analyze_binned_timeline(behaviors, str(output_dir / 'timeline'),
                                                              bin_size_minutes=args.bin_minutes)))

    if run_all or args.transitions:
        analyses_to_run.append(('Transitions', 'transitions',
                               lambda: analyze_transitions(behaviors, str(output_dir / 'transitions'))))

    if not analyses_to_run:
        print("\nNo analyses selected. Use --all or specify individual analyses.")
        print("Run 'lupe-cli analyze --help' for options.")
        return

    # Run analyses
    print(f"\nRunning {len(analyses_to_run)} analyses...")
    print("-" * 60)

    for name, folder, func in analyses_to_run:
        print(f"\n[{name}]")
        try:
            func()
            print(f"✓ {name} complete")
        except Exception as e:
            print(f"✗ {name} failed: {str(e)}")

    print("\n" + "=" * 60)
    print("Analysis completed!")
    print(f"Results saved to: {args.output}")
    print("=" * 60)


def cmd_export(args):
    """Execute the export command."""
    print("=" * 60)
    print("LUPE Behavior Export")
    print("=" * 60)

    # Load behaviors
    print(f"\nLoading behaviors from: {args.behaviors}")
    behaviors = load_behaviors(args.behaviors)
    print(f"✓ Loaded behaviors for {len(behaviors)} files")

    # Export to CSV
    print(f"\nExporting to: {args.output}")
    export_behaviors_to_csv(behaviors, args.output)
    print("✓ CSV files created")

    # Export summary if requested
    if args.summary:
        summary_path = Path(args.output) / 'behavior_summary.csv'
        export_behavior_summary(behaviors, str(summary_path))
        print(f"✓ Summary saved to: {summary_path}")

    print("\n" + "=" * 60)
    print("Export completed successfully!")
    print("=" * 60)


def cmd_config(args):
    """Execute the config command."""
    config = get_config()

    if args.show:
        print("=" * 60)
        print("LUPE Configuration")
        print("=" * 60)

        print("\nBehaviors:")
        behavior_names = config.get_behavior_names()
        behavior_colors = config.get_behavior_colors()
        for i, (name, color) in enumerate(zip(behavior_names, behavior_colors)):
            print(f"  {i}: {name:20s} (color: {color})")

        print(f"\nFramerate: {config.get_framerate()} fps")
        print(f"Pixel to cm: {config.get_pixel_to_cm()}")
        print(f"Smoothing window: {config.get_smoothing_window()}")

    elif args.behaviors:
        print("Behavior Definitions:")
        behavior_names = config.get_behavior_names()
        for i, name in enumerate(behavior_names):
            print(f"  {i}: {name}")

    else:
        print("Use --show to display configuration")
        print("Use --behaviors to list behavior definitions")


if __name__ == '__main__':
    main()
