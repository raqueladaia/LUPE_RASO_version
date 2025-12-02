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
  # Preprocess DLC files
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
        help='Preprocess raw DeepLabCut files'
    )
    preprocess_parser.add_argument(
        '--input', '-i',
        required=True,
        nargs='+',
        help='Input DLC files (one or more)'
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
    analyze_parser.add_argument(
        '--framerate',
        type=float,
        help='Video framerate in fps (default: from config, typically 60)'
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

    # ============ AMPS Command ============
    amps_parser = subparsers.add_parser(
        'amps',
        help='Run LUPE-AMPS pain scale analysis'
    )
    amps_parser.add_argument(
        '--csv-files', '-c',
        required=True,
        nargs='+',
        help='Behavior CSV files (frame, behavior_id format)'
    )
    amps_parser.add_argument(
        '--model', '-m',
        default='models/model_AMPS.pkl',
        help='Path to LUPE-AMPS PCA model (default: models/model_AMPS.pkl)'
    )
    amps_parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output base directory'
    )
    amps_parser.add_argument(
        '--project-name', '-p',
        default='LUPE-AMPS',
        help='Project name for output folders (default: LUPE-AMPS)'
    )
    amps_parser.add_argument(
        '--target-fps',
        type=int,
        default=20,
        help='Target framerate for downsampling (default: 20 fps)'
    )
    amps_parser.add_argument(
        '--sections',
        type=int,
        nargs='+',
        default=[1, 2, 3, 4],
        choices=[1, 2, 3, 4],
        help='Sections to run: 1=preprocessing, 2=PCA, 3=metrics, 4=model_fit (default: all)'
    )

    # ============ RUN Command (Full Workflow) ============
    run_parser = subparsers.add_parser(
        'run',
        help='Run complete workflow: preprocess -> classify -> analyze (like GUI)'
    )
    run_parser.add_argument(
        '--dlc-csv', '-d',
        required=True,
        nargs='+',
        help='DeepLabCut CSV files to process'
    )
    run_parser.add_argument(
        '--model', '-m',
        default='models/model_LUPE.pkl',
        help='Path to LUPE classification model (default: models/model_LUPE.pkl)'
    )
    run_parser.add_argument(
        '--output', '-o',
        default='outputs/',
        help='Output base directory (default: outputs/)'
    )
    run_parser.add_argument(
        '--likelihood-threshold',
        type=float,
        default=0.1,
        help='Likelihood threshold for DLC filtering (default: 0.1)'
    )
    run_parser.add_argument(
        '--bin-minutes',
        type=float,
        default=1.0,
        help='Timeline bin size in minutes (default: 1.0)'
    )
    run_parser.add_argument(
        '--skip-analyses',
        nargs='*',
        choices=['instance_counts', 'total_frames', 'durations', 'timeline', 'transitions'],
        help='Analyses to skip (default: run all)'
    )
    run_parser.add_argument(
        '--framerate',
        type=float,
        help='Video framerate in fps (default: from config, typically 60)'
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
        elif args.command == 'amps':
            cmd_amps(args)
        elif args.command == 'run':
            cmd_run(args)
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
    print("[OK] Model loaded successfully")

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

    print(f"[OK] Loaded {len(pose_data)} files")

    # Classify behaviors
    print("\nClassifying behaviors...")
    predictions = classify_behaviors(model, pose_data, framerate=args.framerate)
    print("[OK] Classification complete")

    # Create behaviors dictionary
    behaviors_dict = {name: pred for name, pred in zip(file_names, predictions)}

    # Save results
    print(f"\nSaving results to: {args.output}")
    save_behaviors(behaviors_dict, args.output)
    print("[OK] Results saved")

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

            print(f"[OK] Processed {csv_file.name}: shape {pose_data.shape}")

        except Exception as e:
            print(f"[ERROR] Error processing {csv_file.name}: {str(e)}")
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

    print(f"[OK] Saved {len(processed_data)} file(s) to {args.output}")

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
    from src.utils.master_summary import create_analysis_summary
    from src.utils.filename_utils import extract_partial_filename

    print("=" * 60)
    print("LUPE Behavior Analysis")
    print("=" * 60)

    # Load behaviors
    print(f"\nLoading behaviors from: {args.behaviors}")
    behaviors = load_behaviors(args.behaviors)
    print(f"[OK] Loaded behaviors for {len(behaviors)} files")

    # Create base output directory
    base_output_dir = Path(args.output)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which analyses to run
    run_all = args.all
    selected_analyses = []
    if run_all or args.instance_counts:
        selected_analyses.append('instance_counts')
    if run_all or args.total_frames:
        selected_analyses.append('total_frames')
    if run_all or args.durations:
        selected_analyses.append('durations')
    if run_all or args.timeline:
        selected_analyses.append('timeline')
    if run_all or args.transitions:
        selected_analyses.append('transitions')

    if not selected_analyses:
        print("\nNo analyses selected. Use --all or specify individual analyses.")
        print("Run 'python main_cli.py analyze --help' for options.")
        return

    # Get configuration for behavior names and framerate
    config = get_config()
    behavior_names = config.get_behavior_names()

    # Use user-specified framerate or fall back to config default
    framerate = args.framerate if args.framerate else config.get_framerate()
    print(f"Using framerate: {framerate} fps")

    # Process each file individually (matching GUI behavior)
    print(f"\nProcessing {len(behaviors)} file(s)...")
    print("=" * 60)

    for file_idx, (file_name, predictions) in enumerate(behaviors.items(), 1):
        print(f"\n[File {file_idx}/{len(behaviors)}] Processing: {file_name}")
        print("-" * 60)

        # Create per-file output structure matching GUI
        # outputs/{filename}/{filename}_analysis/
        file_output_dir = base_output_dir / file_name
        analysis_dir = file_output_dir / f"{file_name}_analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)

        # Create single-file dictionary for analysis functions
        file_behaviors = {file_name: predictions}

        # Run selected analyses
        analysis_count = 0

        if 'instance_counts' in selected_analyses:
            try:
                print(f"  - Bout counts...")
                analyze_instance_counts(file_behaviors, str(analysis_dir), file_prefix=file_name)
                print(f"    [OK] Complete")
                analysis_count += 1
            except Exception as e:
                print(f"    [ERROR] {str(e)}")

        if 'total_frames' in selected_analyses:
            try:
                print(f"  - Time distribution...")
                analyze_total_frames(file_behaviors, str(analysis_dir), file_prefix=file_name)
                print(f"    [OK] Complete")
                analysis_count += 1
            except Exception as e:
                print(f"    [ERROR] {str(e)}")

        if 'durations' in selected_analyses:
            try:
                print(f"  - Bout durations...")
                analyze_bout_durations(file_behaviors, str(analysis_dir),
                                      framerate=framerate, file_prefix=file_name)
                print(f"    [OK] Complete")
                analysis_count += 1
            except Exception as e:
                print(f"    [ERROR] {str(e)}")

        if 'timeline' in selected_analyses:
            try:
                print(f"  - Timeline...")
                analyze_binned_timeline(file_behaviors, str(analysis_dir),
                                      bin_size_minutes=args.bin_minutes, framerate=framerate,
                                      file_prefix=file_name)
                print(f"    [OK] Complete")
                analysis_count += 1
            except Exception as e:
                print(f"    [ERROR] {str(e)}")

        if 'transitions' in selected_analyses:
            try:
                print(f"  - Behavior transitions...")
                analyze_transitions(file_behaviors, str(analysis_dir), file_prefix=file_name)
                print(f"    [OK] Complete")
                analysis_count += 1
            except Exception as e:
                print(f"    [ERROR] {str(e)}")

        # Create master analysis summary if any analyses were run
        if analysis_count > 0:
            try:
                print(f"  - Creating master summary...")
                create_analysis_summary(
                    file_prefix=file_name,
                    analysis_dir=str(analysis_dir),
                    behavior_names=behavior_names
                )
                print(f"    [OK] Master summary created")
            except Exception as e:
                print(f"    [WARNING] Could not create master summary: {str(e)}")

        print(f"[OK] Completed: {file_name}")

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
    print(f"[OK] Loaded behaviors for {len(behaviors)} files")

    # Export to CSV
    print(f"\nExporting to: {args.output}")
    export_behaviors_to_csv(behaviors, args.output)
    print("[OK] CSV files created")

    # Export summary if requested
    if args.summary:
        summary_path = Path(args.output) / 'behavior_summary.csv'
        export_behavior_summary(behaviors, str(summary_path))
        print(f"[OK] Summary saved to: {summary_path}")

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


def cmd_amps(args):
    """Execute the LUPE-AMPS command."""
    from src.core.analysis_lupe_amps import LupeAmpsAnalysis

    print("=" * 60)
    print("LUPE-AMPS Pain Scale Analysis")
    print("=" * 60)

    # Validate model file
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {args.model}")

    # Validate CSV files
    csv_files = []
    for csv_file in args.csv_files:
        csv_path = Path(csv_file)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        csv_files.append(str(csv_path))

    print(f"\nInput files: {len(csv_files)}")
    print(f"Model: {args.model}")
    print(f"Target FPS: {args.target_fps}")
    print(f"Sections to run: {args.sections}")
    print(f"Output directory: {args.output}")
    print(f"Project name: {args.project_name}")

    # Create analysis object
    analysis = LupeAmpsAnalysis(
        model_path=args.model,
        num_behaviors=6,
        target_fps=args.target_fps
    )

    # Run complete analysis
    print("\n" + "=" * 60)
    print("Starting Analysis...")
    print("=" * 60)

    try:
        results = analysis.run_complete_analysis(
            csv_files=csv_files,
            output_base_dir=args.output,
            project_name=args.project_name,
            sections=args.sections,
            progress_callback=print
        )

        print("\n" + "=" * 60)
        print("LUPE-AMPS Analysis Completed Successfully!")
        print("=" * 60)
        print(f"\nResults saved to: {Path(args.output) / f'{args.project_name}_LUPE-AMPS'}")

        # Print section summaries
        if 'section1_csv' in results:
            print(f"  Section 1: {results['section1_csv']}")
        if 'section2_plot' in results:
            print(f"  Section 2: {results['section2_plot']}.png/svg/csv")
        if 'section3_dir' in results:
            print(f"  Section 3: {results['section3_dir']}/")
        if 'section4_dir' in results:
            print(f"  Section 4: {results['section4_dir']}/")

    except Exception as e:
        print(f"\n[ERROR] Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def cmd_run(args):
    """Execute the full workflow command (preprocess -> classify -> analyze)."""
    import pandas as pd
    import gc
    from src.utils.filename_utils import extract_partial_filename
    from src.utils.master_summary import create_analysis_summary
    from src.core.file_summary import generate_dlc_summary

    print("=" * 60)
    print("LUPE Complete Workflow")
    print("=" * 60)
    print("\nThis command runs the full pipeline:")
    print("  1. Preprocess DLC files")
    print("  2. Classify behaviors with LUPE model")
    print("  3. Run all analyses and generate summaries")

    # Validate model file
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {args.model}")

    # Validate DLC CSV files
    dlc_files = []
    for csv_file in args.dlc_csv:
        csv_path = Path(csv_file)
        if not csv_path.exists():
            raise FileNotFoundError(f"DLC CSV file not found: {csv_file}")
        dlc_files.append(csv_path)

    print(f"\nInput files: {len(dlc_files)}")
    print(f"Model: {args.model}")
    print(f"Likelihood threshold: {args.likelihood_threshold}")
    print(f"Output directory: {args.output}")

    # Load model once for all files
    print("\nLoading LUPE model...")
    model = load_model(str(model_path))
    print("[OK] Model loaded")

    # Get configuration
    config = get_config()
    behavior_names = config.get_behavior_names()

    # Use user-specified framerate or fall back to config default
    framerate = args.framerate if args.framerate else config.get_framerate()
    print(f"Using framerate: {framerate} fps")

    # Get base output directory
    base_output_dir = Path(args.output)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which analyses to run (all except skipped ones)
    all_analyses = ['instance_counts', 'total_frames', 'durations', 'timeline', 'transitions']
    skip_analyses = args.skip_analyses if args.skip_analyses else []
    selected_analyses = [a for a in all_analyses if a not in skip_analyses]

    # Process each DLC CSV file individually
    print(f"\nProcessing {len(dlc_files)} file(s)...")
    print("=" * 60)

    for file_idx, csv_file in enumerate(dlc_files, 1):
        print(f"\n[File {file_idx}/{len(dlc_files)}] Processing: {csv_file.name}")
        print("-" * 60)

        try:
            # Step 1: Preprocess DLC CSV
            print("  [Step 1/4] Preprocessing DLC CSV...")
            pose_data = preprocess_dlc_csv(
                str(csv_file),
                likelihood_threshold=args.likelihood_threshold,
                save_output=False
            )
            print(f"    [OK] Preprocessed shape: {pose_data.shape}")

            # Step 2: Classify behaviors
            print("  [Step 2/4] Classifying behaviors...")
            predictions = classify_behaviors(model, [pose_data])[0]
            print(f"    [OK] Classified {len(predictions):,} frames")
            print(f"    [OK] Found {len(np.unique(predictions))} unique behaviors")

            # Free pose_data to release memory
            del pose_data
            gc.collect()

            # Step 3: Create output folder structure
            partial_name = extract_partial_filename(str(csv_file))
            file_output_dir = base_output_dir / partial_name
            analysis_dir = file_output_dir / f"{partial_name}_analysis"
            file_output_dir.mkdir(parents=True, exist_ok=True)
            analysis_dir.mkdir(parents=True, exist_ok=True)

            print(f"  [Step 3/4] Saving outputs to: {file_output_dir}")

            # Save behaviors CSV (frame, behavior_id)
            behaviors_csv_path = file_output_dir / f"{partial_name}_behaviors.csv"
            df_behaviors = pd.DataFrame({
                'frame': range(1, len(predictions) + 1),
                'behavior_id': predictions
            })
            df_behaviors.to_csv(behaviors_csv_path, index=False)
            print(f"    [OK] Saved: {behaviors_csv_path.name}")

            # Save time vector CSV (frame, time_seconds)
            time_csv_path = file_output_dir / f"{partial_name}_time.csv"
            time_seconds = np.array([i / framerate for i in range(len(predictions))])
            df_time = pd.DataFrame({
                'frame': range(1, len(predictions) + 1),
                'time_seconds': time_seconds
            })
            df_time.to_csv(time_csv_path, index=False)
            print(f"    [OK] Saved: {time_csv_path.name}")

            # Generate file summary
            try:
                dlc_df_headers = pd.read_csv(str(csv_file), header=[0, 1, 2], nrows=0)
                summary_path = generate_dlc_summary(
                    dlc_df=dlc_df_headers,
                    predictions=predictions,
                    framerate=framerate,
                    file_path=str(csv_file),
                    output_dir=str(file_output_dir),
                    behavior_names=behavior_names
                )
                print(f"    [OK] Saved: {Path(summary_path).name}")
            except Exception as e:
                print(f"    [WARNING] Could not generate summary: {str(e)}")

            # Run analyses
            print("  [Step 4/4] Running analyses...")
            file_behaviors = {partial_name: predictions}
            analysis_count = 0

            if 'instance_counts' in selected_analyses:
                try:
                    print("    - Bout counts...")
                    analyze_instance_counts(file_behaviors, str(analysis_dir), file_prefix=partial_name)
                    print("      [OK] Complete")
                    analysis_count += 1
                except Exception as e:
                    print(f"      [ERROR] {str(e)}")

            if 'total_frames' in selected_analyses:
                try:
                    print("    - Time distribution...")
                    analyze_total_frames(file_behaviors, str(analysis_dir), file_prefix=partial_name)
                    print("      [OK] Complete")
                    analysis_count += 1
                except Exception as e:
                    print(f"      [ERROR] {str(e)}")

            if 'durations' in selected_analyses:
                try:
                    print("    - Bout durations...")
                    analyze_bout_durations(file_behaviors, str(analysis_dir),
                                          framerate=framerate, file_prefix=partial_name)
                    print("      [OK] Complete")
                    analysis_count += 1
                except Exception as e:
                    print(f"      [ERROR] {str(e)}")

            if 'timeline' in selected_analyses:
                try:
                    print("    - Timeline...")
                    analyze_binned_timeline(file_behaviors, str(analysis_dir),
                                          bin_size_minutes=args.bin_minutes, framerate=framerate,
                                          file_prefix=partial_name)
                    print("      [OK] Complete")
                    analysis_count += 1
                except Exception as e:
                    print(f"      [ERROR] {str(e)}")

            if 'transitions' in selected_analyses:
                try:
                    print("    - Behavior transitions...")
                    analyze_transitions(file_behaviors, str(analysis_dir), file_prefix=partial_name)
                    print("      [OK] Complete")
                    analysis_count += 1
                except Exception as e:
                    print(f"      [ERROR] {str(e)}")

            # Create master summary
            if analysis_count > 0:
                try:
                    print("    - Creating master summary...")
                    create_analysis_summary(
                        file_prefix=partial_name,
                        analysis_dir=str(analysis_dir),
                        behavior_names=behavior_names
                    )
                    print("      [OK] Master summary created")
                except Exception as e:
                    print(f"      [WARNING] Could not create master summary: {str(e)}")

            print(f"  [OK] Completed processing: {csv_file.name}")

            # Free memory
            gc.collect()

        except Exception as e:
            print(f"  [ERROR] Failed to process {csv_file.name}: {str(e)}")
            print("  Skipping to next file...")
            continue

    print("\n" + "=" * 60)
    print("Complete workflow finished!")
    print(f"Results saved to: {base_output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
