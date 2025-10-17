# LUPE Analysis Tool

A comprehensive, user-friendly analysis tool for LUPE (Light aUtomated Pain Evaluator) behavioral data. This tool converts Jupyter notebook-based analysis workflows into a modular application with both GUI and CLI interfaces.

> **New to LUPE?** Start with [GETTING_STARTED.md](GETTING_STARTED.md) for a step-by-step tutorial.

## Features

- **No Jupyter Notebooks Required**: Runs entirely in VS Code, Cursor, or command line
- **DeepLabCut Integration**: Direct support for raw DLC CSV files
- **Dual Interface**: Choose between GUI for ease-of-use or CLI for automation
- **Two Analysis Modes**:
  - **LUPE Classification**: Behavior classification and standard metrics
  - **LUPE-AMPS**: Pain scale analysis using PCA projection
- **Complete Workflow**:
  - DLC CSV preprocessing (likelihood filtering)
  - Behavior classification from pose data
  - Instance counts (bout frequency)
  - Duration analysis
  - Timeline visualization
  - Transition matrices
  - Pain scale quantification (LUPE-AMPS)
  - CSV exports for external analysis
- **Well-Documented**: Extensive comments and beginner-friendly code
- **Configurable**: JSON-based configuration (no hardcoded values)
- **Simplified Workflow**: Focuses on data analysis without group/condition complexity

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [GUI Mode](#gui-mode)
  - [CLI Mode](#cli-mode)
- [Analysis Types](#analysis-types)
- [Output Files](#output-files)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Reference Repository & Attribution](#reference-repository--attribution)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Installation

### Prerequisites

- Python 3.11 or higher
- pip (Python package manager)

### ⚠️ IMPORTANT: Library Version Compatibility

This tool uses **specific library versions** that are compatible with the pre-trained A-SOiD model.

**Required versions:**
- **scikit-learn 1.2.1** (MUST match the model's training version)
- numpy 1.26.4
- pandas 2.2.2
- numba 0.58.1

**CRITICAL:** The pre-trained A-SOiD model was trained with scikit-learn 1.2.1 and is **INCOMPATIBLE** with newer versions (1.3.0+) due to breaking changes in decision tree internal structures. DO NOT upgrade sklearn or the model will fail to load.

### Steps

1. **Clone or download this repository**
   ```bash
   cd LUPE_analysis_RASO_version
   ```

2. **Create a virtual environment (STRONGLY recommended)**
   ```bash
   python -m venv env

   # On Windows:
   env\Scripts\activate

   # On macOS/Linux:
   source env/bin/activate
   ```

3. **Install dependencies with exact versions**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python -c "import sklearn; print(f'scikit-learn: {sklearn.__version__}')"
   # Should print: scikit-learn: 1.2.1

   python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
   # Should print: NumPy: 1.26.4
   ```

5. **Download the LUPE A-SOiD model**
   - Download from: [LUPE Model Link](https://upenn.box.com/s/9rfslrvcc7m6fji8bmgktnegghyu88b0)
   - Place the model file in a `models/` directory

## Quick Start

### Main Launcher (Recommended)

```bash
python main_launcher.py
```

This opens the LUPE Analysis Suite launcher with two options:
- **Run LUPE**: Opens the behavior classification and metrics analysis GUI
- **Run AMPS**: Opens the LUPE-AMPS pain scale analysis GUI

Both GUIs can run simultaneously, and you can switch between them as needed.

### LUPE Classification GUI

```bash
python main_lupe_gui.py
```

**Option 1: Start from DeepLabCut CSV files**
1. Select "Raw DeepLabCut CSV Files" radio button
2. Browse and select your DLC CSV file(s)
3. Browse and select your LUPE model file (.pkl)
4. Adjust likelihood threshold if needed (default: 0.1)
5. Select output directory and analyses
6. Click "Run Analysis" - preprocessing, classification, and analysis run automatically

**Option 2: Start from pre-classified behaviors**
1. Select "Pre-classified Behaviors (.pkl)" radio button
2. Browse and select your behaviors file (.pkl)
3. Select output directory
4. Choose which analyses to run
5. Click "Run Analysis"
6. Results will be saved to the output directory

### LUPE-AMPS Pain Scale GUI

```bash
python main_lupe_amps_gui.py
```

**Analyze pain-related behaviors on a continuous scale:**
1. Add behavior CSV files (frame, behavior_id format)
2. Select which analysis sections to run (default: all 4)
3. Set output directory and project name
4. Click "Run LUPE-AMPS Analysis"
5. Results include:
   - Pain scale projection (PC1/PC2 coordinates)
   - Behavior metrics visualization
   - Model feature importance analysis

See `docs/LUPE_AMPS_GUIDE.md` for detailed instructions.

### Using the CLI

```bash
# View available commands
python main_cli.py --help

# Option 1: Complete workflow from DLC CSV
python main_cli.py preprocess --input dlc_data/*.csv --output pose_data.pkl
python main_cli.py classify --model model.pkl --input pose_data.pkl --output behaviors.pkl
python main_cli.py analyze --behaviors behaviors.pkl --output results/ --all

# Option 2: Analyze pre-classified behaviors
python main_cli.py export --behaviors behaviors.pkl --output csv/
python main_cli.py analyze --behaviors behaviors.pkl --output results/ --all
```

## Usage

### GUI Mode

The GUI provides a simple, point-and-click interface for running analyses.

**Features:**
- File selection dialogs
- Analysis selection checkboxes
- Real-time progress tracking
- Log window showing analysis status

**Workflow:**
1. Launch GUI: `python main_gui.py`
2. Select behaviors file (previously classified data)
3. Choose output directory
4. Select which analyses to run
5. Click "Run Analysis"

See `docs/GUI_GUIDE.md` for detailed instructions.

### CLI Mode

The CLI provides command-line access for automation and scripting.

**Available Commands:**

```bash
# Classify behaviors from pose data
python main_cli.py classify \
  --model models/model_LUPE-AMPS.pkl \
  --input data/pose_file.npy \
  --output behaviors.pkl

# Run specific analysis
python main_cli.py analyze \
  --behaviors behaviors.pkl \
  --output results/ \
  --instance-counts \
  --total-frames

# Export to CSV
python main_cli.py export \
  --behaviors behaviors.pkl \
  --output csv/ \
  --summary

# View configuration
python main_cli.py config --show
```

See `docs/CLI_GUIDE.md` for detailed command reference.

## Analysis Types

### 1. CSV Export
Exports behavior classifications to CSV format for use in Excel, R, Python, etc.
- Frame-by-frame classifications
- Second-by-second classifications
- Summary statistics

### 2. Instance Counts
Counts how many times each behavior occurs (bout frequency).
- Mean and standard deviation
- Per-file and aggregate statistics
- Bar chart visualization

### 3. Total Frames
Analyzes percentage of time spent in each behavior.
- Overall distribution
- Per-file breakdown
- Pie/donut chart visualization

### 4. Bout Durations
Analyzes how long each behavior bout lasts.
- Duration statistics (mean, median, std, min, max)
- Box-and-whisker plots
- Distribution analysis

### 5. Binned Timeline
Shows how behavior distribution changes over time.
- Configurable time bins
- Line plots with error bands
- Temporal pattern detection

### 6. Behavior Transitions
Analyzes how behaviors transition from one to another.
- Transition probability matrix
- Heatmap visualization
- Sequential pattern analysis

See `docs/ANALYSIS_TYPES.md` for complete descriptions.

### 7. File Summary
Automatically generated for each analyzed file, containing recording metadata and behavior statistics.
- Total frames, duration (seconds/minutes/hours), framerate
- Keypoint information (count and names)
- Behavior statistics (frame counts and percentages per behavior)
- Output: `{filename}_summary.csv`

### 8. Master Analysis Summary
Consolidated summary combining all analysis results into a single file for quick reference.
- Bout counts for all behaviors
- Time distribution (percentages and frame counts)
- Bout duration statistics (mean and median)
- Top 5 behavior transitions with probabilities
- Output: `{filename}_ANALYSIS_SUMMARY.csv`

### 9. LUPE-AMPS Pain Scale Analysis

The LUPE-AMPS (Advanced Multivariate Pain Scale) module provides specialized pain-related behavior analysis:

**What it does:**
- Projects 6-dimensional behavior occupancy onto a 2D pain scale using PCA
- PC1 = Generalized Behavior Scale
- PC2 = **Pain Behavior Scale** (primary measure)
- Higher PC2 values indicate more pain-related behaviors

**Four analysis sections:**
1. **Preprocessing & Metrics**: Downsample, calculate occupancy/bouts/duration
2. **PCA Projection**: Transform to pain scale coordinates
3. **Metrics Visualization**: Bar plots of behavior patterns
4. **Feature Importance**: Model validation and feature contribution

**Inputs:**
- Behavior CSV files with frame-by-frame classifications (output from LUPE classification)
- Summary CSV files (for auto-detection of recording parameters)
- Pre-trained PCA model (`models/model_AMPS.pkl`)

**Auto-Detection Feature:**
- Recording length and original framerate are automatically read from summary CSV files
- No manual parameter entry required
- Only target framerate (default: 20 fps) is user-configurable

**Outputs:**
- Pain scale coordinates (PC1/PC2) for each recording
- Publication-ready scatter plots (PNG/SVG)
- Behavior metrics plots and CSV files
- Feature importance analysis

See `docs/LUPE_AMPS_GUIDE.md` for complete usage instructions.

## Output Files

When you run LUPE analysis on a file (e.g., `mouse01DLC_resnet50.csv`), the following output structure is created:

```
outputs/
└── mouse01/                              # Partial name (extracted before "DLC")
    ├── mouse01_behaviors.csv             # Frame-by-frame behavior classifications
    ├── mouse01_time.csv                  # Time vector (frame, time_seconds)
    ├── mouse01_summary.csv               # Recording metadata & behavior statistics
    └── mouse01_analysis/                 # Analysis results subfolder
        ├── mouse01_ANALYSIS_SUMMARY.csv          # Master summary (all key metrics)
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
```

### Key Output Files:

**Core Files:**
- `{filename}_behaviors.csv`: Frame-by-frame behavior IDs (frame, behavior_id)
- `{filename}_time.csv`: Time vector for each frame (frame, time_seconds)
- `{filename}_summary.csv`: Recording metadata (duration, framerate, keypoints, behavior statistics)

**Master Summary:**
- `{filename}_ANALYSIS_SUMMARY.csv`: Consolidated summary with:
  - Bout counts for all behaviors
  - Time distribution percentages
  - Bout duration statistics (mean, median)
  - Top 5 behavior transitions

**Analysis Files:**
- Files ending in `.csv`: Numerical data (can be opened in Excel, R, Python)
- Files ending in `.svg`: Vector graphics (publication-ready, scalable)
- Files ending in `.png`: Raster graphics (for quick viewing)

**Note:** The `_analysis` folder contains all statistical analyses and visualizations. The `_ANALYSIS_SUMMARY.csv` provides a quick overview of all key metrics in one place.

## Configuration

Configuration files are located in `config/`:

### `metadata.json`
Defines behaviors, colors, keypoints, and physical parameters:
- Behavior names and colors
- Body keypoint definitions
- Pixel-to-cm conversion
- Analysis parameters (framerate, smoothing window)

### `settings.json`
User preferences and default paths:
- Default file paths
- Output preferences (CSV, plots, format)
- Analysis defaults
- GUI preferences

**To modify configuration:**
1. Edit the JSON files directly
2. Restart the application
3. Changes take effect immediately

**Note:** Metadata cannot be saved as .py files (per project rules)

## Project Structure

```
LUPE_analysis_RASO_version/
├── main_launcher.py         # Main launcher (RECOMMENDED entry point)
├── main_lupe_gui.py         # LUPE classification GUI entry point
├── main_lupe_amps_gui.py    # LUPE-AMPS pain scale GUI entry point
├── main_cli.py              # CLI entry point
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── CLAUDE.md                # Project objectives and rules
│
├── config/                  # Configuration files
│   ├── metadata.json        # Behavior definitions, keypoints
│   └── settings.json        # User preferences
│
├── models/                  # Pre-trained models
│   └── model_AMPS.pkl       # LUPE-AMPS PCA model
│
├── src/                     # Source code
│   ├── core/               # Core analysis modules
│   │   ├── data_loader.py
│   │   ├── feature_extraction.py
│   │   ├── classification.py
│   │   ├── dlc_preprocessing.py
│   │   ├── file_summary.py         # Generate summary CSV files
│   │   ├── analysis_*.py           # Standard analysis modules
│   │   ├── analysis_lupe_amps.py   # LUPE-AMPS analysis
│   │   └── ...
│   ├── gui/                # GUI components
│   │   ├── main_window.py         # LUPE classification GUI
│   │   └── lupe_amps_window.py    # LUPE-AMPS GUI
│   ├── cli/                # CLI components
│   │   └── main.py
│   └── utils/              # Utility modules
│       ├── config_manager.py
│       ├── plotting.py
│       ├── amps_summary_reader.py  # Read parameters from summary CSV
│       ├── master_summary.py       # Generate master analysis summaries
│       └── model_utils.py
│
├── outputs/                # Analysis outputs
├── data/                   # Input data directory
├── docs/                   # Documentation
│   ├── LUPE_AMPS_GUIDE.md  # LUPE-AMPS user guide
│   └── ...
└── tests/                  # Test files
```

## Analysis Workflow

1. **Prepare Data**: Obtain pose estimation data from DeepLabCut
2. **Classify (if needed)**: Use the A-SOiD model to classify behaviors
3. **Analyze**: Run desired analyses on the classifications
4. **Export**: Get results as CSV files and visualizations

```
Pose Data → Behavior Classification → Analysis → Results
(DLC .csv)   (A-SOiD model)         (This tool)  (CSV + plots)
```

## Tips and Best Practices

1. **Start with CSV export** to explore your data
2. **Use the GUI** for exploratory analysis
3. **Use the CLI** for batch processing or automation
4. **Adjust bin size** in timeline analysis based on your recording length
5. **Check output files** in the designated output directory after each run

## Troubleshooting

### Model Loading Errors (dtype incompatibility) ⚠️ MOST COMMON ISSUE

**Problem:** You see an error like:
- "incompatible dtype"
- "node array from the pickle has an incompatible dtype"
- "missing_go_to_left" field error

**Root Cause:** The pre-trained model was saved with scikit-learn 1.2.1 and is **INCOMPATIBLE** with sklearn 1.3.0+ due to breaking changes in decision tree structures (added `missing_go_to_left` field).

**Check Your Versions:**
```bash
python -c "import sklearn, numpy; print(f'scikit-learn: {sklearn.__version__}'); print(f'NumPy: {numpy.__version__}')"
```

**Required versions:**
- scikit-learn: **1.2.1** (EXACT version required, NOT 1.3.x or higher)
- NumPy: **1.26.4** (NOT 2.0+)

**SOLUTION (choose one):**

1. **Option 1: Quick fix (if you have sklearn 1.3.x installed)**
   ```bash
   # Uninstall current version
   pip uninstall -y scikit-learn

   # Install correct version
   pip install --prefer-binary scikit-learn==1.2.1

   # Verify
   python -c "import sklearn; print(f'scikit-learn: {sklearn.__version__}')"
   # Should print: scikit-learn: 1.2.1
   ```

2. **Option 2: Manual reinstall (full reset)**
   ```bash
   # Upgrade pip first
   pip install --upgrade pip setuptools wheel

   # Uninstall current versions
   pip uninstall -y scikit-learn numpy pandas scipy

   # Install exact versions
   pip install -r requirements.txt

   # Verify
   python -c "import sklearn; print(f'scikit-learn: {sklearn.__version__}')"
   # Should print: scikit-learn: 1.2.1
   ```

3. **Option 3: If pip installation fails (use conda)**
   ```bash
   # Use conda instead (if you have Anaconda/Miniconda)
   conda create -n lupe python=3.11
   conda activate lupe
   conda install scikit-learn=1.2.1 numpy=1.26.4 pandas=2.2.2 numba=0.58.1
   pip install matplotlib seaborn tqdm
   ```

**Why This Happens:**
- scikit-learn 1.3.0 introduced the `missing_go_to_left` field for missing value support in decision trees
- This broke backward compatibility with models trained in 1.2.x
- The sklearn team does NOT support loading models across major versions
- This is a known issue (sklearn GitHub issue #26798) with no fix planned

**After fixing versions:**
- The model should load without errors
- Do NOT upgrade sklearn to 1.3.x or the error will return

### GUI doesn't start
- Check that all dependencies are installed: `pip install -r requirements.txt`
- Try the CLI instead: `python main_cli.py config --show`
- On some systems, tkinter may need separate installation:
  - Windows: Included with Python
  - macOS: `brew install python-tk`
  - Linux: `sudo apt-get install python3-tk`

### Import errors
- Ensure you're running from the project root directory
- Activate your virtual environment
- Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`
- Check Python version: requires 3.11 or higher

### Analysis takes too long
- Large datasets may take several minutes
- Use CLI for better performance feedback
- Process files in smaller batches
- Consider using a machine with more RAM for very large datasets

### Feature extraction errors
- Ensure your DLC CSV files have the correct format (multi-level headers)
- Check that likelihood threshold is appropriate (default: 0.1)
- Verify the CSV files contain data for all expected keypoints

### Classification produces unexpected results
- Check that you're using the correct A-SOiD model for your data
- Verify your DLC CSV files are from the same camera setup as the training data
- Ensure the framerate setting matches your video framerate (default: 60 fps)

### Memory errors
- For very large videos, process them in smaller batches
- Close other applications to free up RAM
- Consider processing on a machine with more memory

### Getting Help
If you encounter issues not covered here:
1. Check the documentation in the `docs/` folder
2. Verify you're using the latest version of the code
3. Try the model conversion utility: `python -m src.utils.model_utils convert your_model.pkl`
4. Create a detailed issue report including:
   - Error message (full traceback)
   - Python version: `python --version`
   - Library versions: `pip list | grep -E "(numpy|scikit-learn|pandas)"`
   - Operating system

## Reference Repository & Attribution

This tool is a **complete rewrite** of the original LUPE 2.0 Notebook Analysis Package.

**Original Project:** LUPE 2.0 - Light Automated Pain Evaluator
**Original Authors:** Corder Lab (University of Pennsylvania) & Yttri Lab (Carnegie Mellon University)
**Original Repository:** https://github.com/justin05423/LUPE-2.0-NotebookAnalysisPackage

### What Was Changed

This version converts the Jupyter notebook-based workflow into a standalone application while preserving all analytical capabilities:

**Architecture Changes:**
- ❌ Removed: Jupyter notebooks (.ipynb files)
- ✅ Added: GUI application with tkinter
- ✅ Added: CLI for command-line automation
- ✅ Added: Modular Python package structure

**Functional Changes:**
- ❌ Removed: Group and condition comparisons (per project requirements)
- ❌ Removed: Statistical tests between groups
- ✅ Added: LUPE-AMPS standalone GUI with auto-detection
- ✅ Added: Main launcher for easy access
- ✅ Added: Automatic summary CSV generation per file
- ✅ Added: Master analysis summary consolidation
- ✅ Simplified: Focus on individual file analysis with aggregate statistics

**Organization Changes:**
- ✅ Added: Comprehensive documentation (README, guides, tutorials)
- ✅ Added: JSON-based configuration (no hardcoded values)
- ✅ Added: Per-file output directories
- ✅ Added: Dependency management with exact versions

### Why This Rewrite?

The original LUPE 2.0 used Jupyter notebooks, which are excellent for interactive exploration but have limitations:
- Difficult to version control (JSON format)
- Hard to automate and integrate into pipelines
- Requires Jupyter environment setup
- Not suitable for end-user distribution

This rewrite addresses these limitations while preserving all analytical capabilities and properly attributing the original work.

For more details on the relationship to the reference repository, see `reference_repo/README.md`.

## Citation

**If you use this tool in your research, please cite the original LUPE 2.0 project:**

```
LUPE 2.0: Light Automated Pain Evaluator
Corder Lab (University of Pennsylvania) & Yttri Lab (Carnegie Mellon University)
GitHub: https://github.com/justin05423/LUPE-2.0-NotebookAnalysisPackage
```

## License

MIT License - See [LICENSE](LICENSE) file for details.

This project is based on and maintains the same open-source spirit as the original LUPE repository.

## Acknowledgments

**Original LUPE Development:**
- **Corder Lab** - University of Pennsylvania
- **Yttri Lab** - Carnegie Mellon University

**Core Technologies:**
- **A-SOiD**: Active Semi-supervised Clustering algorithm for behavior classification
- **DeepLabCut**: Markerless pose estimation framework
- **LUPE-AMPS**: Advanced Multivariate Pain Scale analysis method

**Special Thanks:**
- Original LUPE 2.0 developers for creating the foundational analysis methods
- The open-source scientific Python community

## Contact

For questions or issues with this analysis tool, please refer to the project documentation or create an issue on GitHub.
