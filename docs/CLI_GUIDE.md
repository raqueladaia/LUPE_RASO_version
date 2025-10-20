# LUPE Analysis Tool - CLI Guide

Complete reference for the Command-Line Interface (CLI).

## Quick Reference

```bash
# Get help
python main_cli.py --help
python main_cli.py <command> --help

# RECOMMENDED: Complete workflow (preprocess -> classify -> analyze)
python main_cli.py run --dlc-csv data/*.csv --model models/model_LUPE.pkl --output outputs/

# Preprocess DeepLabCut CSV files
python main_cli.py preprocess --dlc-csv data/*.csv --output preprocessed/

# Classify behaviors from preprocessed data
python main_cli.py classify --model models/model_LUPE.pkl --input preprocessed/*.npy --output behaviors.pkl

# Run analyses on classified behaviors
python main_cli.py analyze --behaviors behaviors.pkl --output results/ --all

# LUPE-AMPS pain scale analysis
python main_cli.py amps --csv-files outputs/*/summary.csv --model models/model_AMPS.pkl --output amps_results/

# Export to CSV
python main_cli.py export --behaviors behaviors.pkl --output csv/

# View configuration
python main_cli.py config --show
```

## Available Commands

### `run` - Complete Workflow (RECOMMENDED)

Runs the complete LUPE analysis workflow in one command: preprocesses DeepLabCut CSV files, classifies behaviors, and runs all analyses. This mimics the GUI behavior and is the recommended way to use the CLI.

**Usage:**
```bash
python main_cli.py run \
  --dlc-csv PATH_TO_CSV [PATH_TO_CSV ...] \
  --model PATH_TO_MODEL \
  [--output OUTPUT_DIR] \
  [--likelihood-threshold THRESHOLD] \
  [--bin-minutes MINUTES] \
  [--skip-analyses ANALYSIS [ANALYSIS ...]]
```

**Arguments:**
- `--dlc-csv`, `-d`: One or more DeepLabCut CSV files **[Required]**
- `--model`, `-m`: Path to trained LUPE model (default: models/model_LUPE.pkl)
- `--output`, `-o`: Base output directory (default: outputs/)
- `--likelihood-threshold`: Minimum pose confidence (default: 0.1)
- `--bin-minutes`: Bin size for timeline analysis in minutes (default: 1.0)
- `--skip-analyses`: List of analyses to skip (e.g., timeline transitions)

**Output Structure:**
Creates per-file directories with complete analysis results:
```
outputs/
└── recording_name/
    ├── recording_name_behaviors.csv
    ├── recording_name_time.csv
    ├── recording_name_summary.csv
    └── recording_name_analysis/
        ├── recording_name_ANALYSIS_SUMMARY.csv
        ├── instance_counts/
        ├── total_frames/
        ├── durations/
        ├── timeline/
        └── transitions/
```

**Examples:**
```bash
# Process single file with all analyses
python main_cli.py run \
  --dlc-csv data/mouse1DLC_resnet50_*.csv \
  --model models/model_LUPE.pkl

# Process multiple files
python main_cli.py run \
  --dlc-csv data/mouse1*.csv data/mouse2*.csv data/mouse3*.csv \
  --model models/model_LUPE.pkl \
  --output my_results/

# Custom settings and skip some analyses
python main_cli.py run \
  --dlc-csv data/*.csv \
  --model models/model_LUPE.pkl \
  --likelihood-threshold 0.15 \
  --bin-minutes 0.5 \
  --skip-analyses timeline transitions
```

### `preprocess` - Preprocess DeepLabCut CSV Files

Preprocesses DeepLabCut pose estimation CSV files into the format required by the A-SOiD model. This is the first step in the analysis pipeline.

**Usage:**
```bash
python main_cli.py preprocess \
  --dlc-csv PATH_TO_CSV [PATH_TO_CSV ...] \
  --output OUTPUT_DIR \
  [--likelihood-threshold THRESHOLD]
```

**Arguments:**
- `--dlc-csv`, `-d`: One or more DeepLabCut CSV files **[Required]**
- `--output`, `-o`: Output directory for preprocessed files **[Required]**
- `--likelihood-threshold`: Minimum pose confidence (default: 0.1)

**Examples:**
```bash
# Preprocess single file
python main_cli.py preprocess \
  --dlc-csv data/mouse1DLC_resnet50_*.csv \
  --output preprocessed/

# Preprocess multiple files
python main_cli.py preprocess \
  --dlc-csv data/*.csv \
  --output preprocessed/ \
  --likelihood-threshold 0.15

# Then classify the preprocessed data
python main_cli.py classify \
  --model models/model_LUPE.pkl \
  --input preprocessed/*.npy \
  --output behaviors.pkl
```

### `amps` - LUPE-AMPS Pain Scale Analysis

Runs the LUPE-AMPS (Advanced Multivariate Pain Scale) analysis on classified behavior data. This performs PCA projection to generate pain scale scores across 4 sections of the recording.

**Usage:**
```bash
python main_cli.py amps \
  --csv-files PATH_TO_CSV [PATH_TO_CSV ...] \
  --model PATH_TO_MODEL \
  --output OUTPUT_DIR \
  [--project-name NAME] \
  [--target-fps FPS] \
  [--sections SECTION [SECTION ...]]
```

**Arguments:**
- `--csv-files`, `-c`: One or more summary CSV files from LUPE analysis **[Required]**
- `--model`, `-m`: Path to AMPS model (default: models/model_AMPS.pkl)
- `--output`, `-o`: Output directory for AMPS results **[Required]**
- `--project-name`, `-p`: Project name for results (default: LUPE-AMPS)
- `--target-fps`: Target framerate (default: 20)
- `--sections`: Which sections to analyze (default: 1 2 3 4)

**Output:**
Creates comprehensive AMPS analysis results:
```
output_dir/
└── project_name_AMPS_Analysis/
    ├── project_name_AMPS_summary.csv
    ├── project_name_AMPS_section1.csv
    ├── project_name_AMPS_section2.csv
    ├── project_name_AMPS_section3.csv
    ├── project_name_AMPS_section4.csv
    └── visualizations/
```

**Examples:**
```bash
# Run AMPS analysis on all recordings
python main_cli.py amps \
  --csv-files outputs/*/summary.csv \
  --model models/model_AMPS.pkl \
  --output amps_results/

# Analyze specific files with custom settings
python main_cli.py amps \
  --csv-files outputs/mouse1/summary.csv outputs/mouse2/summary.csv \
  --model models/model_AMPS.pkl \
  --output pain_analysis/ \
  --project-name "Baseline_vs_Treatment" \
  --sections 1 2 3
```

**Workflow:**
Typically used after running LUPE analysis:
```bash
# Step 1: Run LUPE analysis
python main_cli.py run --dlc-csv data/*.csv --model models/model_LUPE.pkl

# Step 2: Run AMPS pain scale analysis
python main_cli.py amps \
  --csv-files outputs/*/summary.csv \
  --model models/model_AMPS.pkl \
  --output amps_results/
```

### `classify` - Classify Behaviors from Pose Data

Applies the trained A-SOiD model to pose estimation data to generate behavior classifications.

**Usage:**
```bash
python main_cli.py classify \
  --model PATH_TO_MODEL \
  --input PATH_TO_DATA [PATH_TO_DATA ...] \
  --output OUTPUT_FILE \
  [--framerate FPS]
```

**Arguments:**
- `--model`, `-m`: Path to trained A-SOiD model (.pkl file) **[Required]**
- `--input`, `-i`: One or more pose data files (.npy or .pkl) **[Required]**
- `--output`, `-o`: Output path for behavior classifications (.pkl) **[Required]**
- `--framerate`: Video framerate in fps (default: from config)

**Examples:**
```bash
# Single file
python main_cli.py classify \
  --model models/model_LUPE.pkl \
  --input data/mouse1.npy \
  --output behaviors.pkl

# Multiple files
python main_cli.py classify \
  --model models/model_LUPE.pkl \
  --input data/mouse1.npy data/mouse2.npy data/mouse3.npy \
  --output all_behaviors.pkl

# Custom framerate
python main_cli.py classify \
  --model models/model_LUPE.pkl \
  --input data/*.npy \
  --output behaviors.pkl \
  --framerate 30
```

### `analyze` - Run Behavior Analyses

Performs various analyses on behavior classification data. Creates a per-file directory structure with all analysis results and a master summary CSV.

**Usage:**
```bash
python main_cli.py analyze \
  --behaviors BEHAVIORS_FILE \
  --output OUTPUT_DIR \
  [--all | --instance-counts | --total-frames | --durations | --timeline | --transitions] \
  [--bin-minutes MINUTES] \
  [--framerate FPS]
```

**Arguments:**
- `--behaviors`, `-b`: Path to behaviors file (.pkl) **[Required]**
- `--output`, `-o`: Output directory for results **[Required]**
- `--all`: Run all analyses (recommended)
- `--instance-counts`: Analyze behavior bout counts
- `--total-frames`: Analyze frame percentages
- `--durations`: Analyze bout durations
- `--timeline`: Analyze temporal patterns
- `--transitions`: Analyze behavior transitions
- `--bin-minutes`: Bin size for timeline (default: 1.0)
- `--framerate`: Video framerate in fps (default: from config)

**Output Structure:**
Creates per-file directories with complete analysis results:
```
output_dir/
└── recording_name/
    └── recording_name_analysis/
        ├── recording_name_ANALYSIS_SUMMARY.csv
        ├── instance_counts/
        │   ├── instance_counts_summary.csv
        │   ├── instance_counts_raw.csv
        │   └── instance_counts.svg
        ├── total_frames/
        │   ├── total_frames_summary.csv
        │   └── total_frames_pie.svg
        ├── durations/
        │   ├── bout_durations_statistics.csv
        │   └── bout_durations_boxplot.svg
        ├── timeline/
        │   ├── binned_timeline_1.0min.csv
        │   └── binned_timeline_1.0min.svg
        └── transitions/
            ├── transition_matrix.csv
            └── transition_matrix_heatmap.svg
```

**Examples:**
```bash
# Run all analyses (recommended)
python main_cli.py analyze \
  --behaviors behaviors.pkl \
  --output results/ \
  --all

# Specific analyses only
python main_cli.py analyze \
  --behaviors behaviors.pkl \
  --output results/ \
  --instance-counts \
  --total-frames

# Timeline with custom bin size
python main_cli.py analyze \
  --behaviors behaviors.pkl \
  --output results/ \
  --timeline \
  --bin-minutes 0.5

# All analyses with custom framerate
python main_cli.py analyze \
  --behaviors behaviors.pkl \
  --output results/ \
  --all \
  --framerate 30
```

### `export` - Export to CSV

Exports behavior classifications to CSV format.

**Usage:**
```bash
python main_cli.py export \
  --behaviors BEHAVIORS_FILE \
  --output OUTPUT_DIR \
  [--with-names] \
  [--summary]
```

**Arguments:**
- `--behaviors`, `-b`: Path to behaviors file (.pkl) **[Required]**
- `--output`, `-o`: Output directory **[Required]**
- `--with-names`: Include behavior names (not just IDs)
- `--summary`: Also create summary CSV

**Examples:**
```bash
# Basic export
python main_cli.py export \
  --behaviors behaviors.pkl \
  --output csv/

# With names and summary
python main_cli.py export \
  --behaviors behaviors.pkl \
  --output csv/ \
  --with-names \
  --summary
```

### `config` - View Configuration

Displays current configuration settings.

**Usage:**
```bash
python main_cli.py config [--show | --behaviors]
```

**Arguments:**
- `--show`: Display full configuration
- `--behaviors`: List behavior definitions only

**Examples:**
```bash
# Show all config
python main_cli.py config --show

# Show behaviors only
python main_cli.py config --behaviors
```

## Batch Processing

### Process Multiple Files (Recommended)

Use the `run` command for the simplest workflow:

```bash
# Process all DeepLabCut CSV files in one command
python main_cli.py run \
  --dlc-csv data/*.csv \
  --model models/model_LUPE.pkl \
  --output outputs/

# Then run AMPS analysis on all results
python main_cli.py amps \
  --csv-files outputs/*/summary.csv \
  --model models/model_AMPS.pkl \
  --output amps_results/
```

### Step-by-Step Pipeline (Advanced)

For more control over each step:

```bash
# Step 1: Preprocess DeepLabCut CSV files
python main_cli.py preprocess \
  --dlc-csv data/*.csv \
  --output preprocessed/

# Step 2: Classify behaviors
python main_cli.py classify \
  --model models/model_LUPE.pkl \
  --input preprocessed/*.npy \
  --output behaviors.pkl

# Step 3: Analyze behaviors
python main_cli.py analyze \
  --behaviors behaviors.pkl \
  --output results/ \
  --all

# Step 4: Export to CSV (optional)
python main_cli.py export \
  --behaviors behaviors.pkl \
  --output csv/
```

### Automated Pipeline Script

Create a bash script (`process_all.sh`) for automated analysis:

```bash
#!/bin/bash

# Configuration
MODEL_LUPE="models/model_LUPE.pkl"
MODEL_AMPS="models/model_AMPS.pkl"
DATA_DIR="data/"
OUTPUT_DIR="results_$(date +%Y%m%d)/"

echo "======================================"
echo "LUPE Analysis Pipeline"
echo "======================================"

# Step 1: Run complete LUPE workflow
echo ""
echo "[1/2] Running LUPE analysis..."
python main_cli.py run \
  --dlc-csv $DATA_DIR/*.csv \
  --model $MODEL_LUPE \
  --output $OUTPUT_DIR

# Check if LUPE analysis succeeded
if [ $? -ne 0 ]; then
  echo "ERROR: LUPE analysis failed!"
  exit 1
fi

# Step 2: Run AMPS pain scale analysis
echo ""
echo "[2/2] Running AMPS pain scale analysis..."
python main_cli.py amps \
  --csv-files $OUTPUT_DIR/*/summary.csv \
  --model $MODEL_AMPS \
  --output $OUTPUT_DIR/amps_results/

# Check if AMPS analysis succeeded
if [ $? -ne 0 ]; then
  echo "WARNING: AMPS analysis failed!"
  echo "LUPE results are still available in $OUTPUT_DIR"
  exit 1
fi

echo ""
echo "======================================"
echo "Complete! Results in $OUTPUT_DIR"
echo "======================================"
```

Run with: `bash process_all.sh`

### Windows PowerShell Script

Create a PowerShell script (`process_all.ps1`):

```powershell
# Configuration
$MODEL_LUPE = "models\model_LUPE.pkl"
$MODEL_AMPS = "models\model_AMPS.pkl"
$DATA_DIR = "data\"
$OUTPUT_DIR = "results_$(Get-Date -Format 'yyyyMMdd')\"

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "LUPE Analysis Pipeline" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan

# Step 1: Run LUPE analysis
Write-Host ""
Write-Host "[1/2] Running LUPE analysis..." -ForegroundColor Yellow
python main_cli.py run --dlc-csv $DATA_DIR*.csv --model $MODEL_LUPE --output $OUTPUT_DIR

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: LUPE analysis failed!" -ForegroundColor Red
    exit 1
}

# Step 2: Run AMPS analysis
Write-Host ""
Write-Host "[2/2] Running AMPS pain scale analysis..." -ForegroundColor Yellow
python main_cli.py amps --csv-files $OUTPUT_DIR*\summary.csv --model $MODEL_AMPS --output $OUTPUT_DIR\amps_results\

if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: AMPS analysis failed!" -ForegroundColor Red
    Write-Host "LUPE results are still available in $OUTPUT_DIR" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "======================================" -ForegroundColor Green
Write-Host "Complete! Results in $OUTPUT_DIR" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green
```

Run with: `powershell -File process_all.ps1`

## Output Structure

The CLI creates an organized per-file directory structure matching the GUI output format:

### LUPE Analysis Output (`run` or `analyze` commands)

```
outputs/
└── recording_name/                          # One directory per recording
    ├── recording_name_behaviors.csv         # Behavior classifications
    ├── recording_name_time.csv              # Time points
    ├── recording_name_summary.csv           # Recording metadata (fps, duration, etc.)
    └── recording_name_analysis/             # Analysis results
        ├── recording_name_ANALYSIS_SUMMARY.csv  # Master summary (all metrics)
        ├── instance_counts/
        │   ├── instance_counts_summary.csv
        │   ├── instance_counts_raw.csv
        │   └── instance_counts.svg
        ├── total_frames/
        │   ├── total_frames_summary.csv
        │   └── total_frames_pie.svg
        ├── durations/
        │   ├── bout_durations_statistics.csv
        │   └── bout_durations_boxplot.svg
        ├── timeline/
        │   ├── binned_timeline_1.0min.csv
        │   └── binned_timeline_1.0min.svg
        └── transitions/
            ├── transition_matrix.csv
            └── transition_matrix_heatmap.svg
```

### LUPE-AMPS Analysis Output (`amps` command)

```
output_dir/
└── project_name_AMPS_Analysis/
    ├── project_name_AMPS_summary.csv        # Overall summary
    ├── project_name_AMPS_section1.csv       # Section 1 results
    ├── project_name_AMPS_section2.csv       # Section 2 results
    ├── project_name_AMPS_section3.csv       # Section 3 results
    ├── project_name_AMPS_section4.csv       # Section 4 results
    └── visualizations/
        ├── pain_scale_timeline.svg
        ├── section_comparison.svg
        └── individual_traces.svg
```

### Key Files Explained

**Recording Metadata:**
- `*_summary.csv`: Contains fps, duration, frame count, behavior percentages
- `*_behaviors.csv`: Frame-by-frame behavior classifications (0-5)
- `*_time.csv`: Corresponding timestamps for each frame

**Master Summary:**
- `*_ANALYSIS_SUMMARY.csv`: Consolidated results from all analyses in one file
  - Total frames and percentages per behavior
  - Bout counts and average durations
  - Transition probabilities
  - Timeline binned data

**Individual Analysis Files:**
- `instance_counts_summary.csv`: Number of bouts per behavior
- `total_frames_summary.csv`: Frame counts and percentages
- `bout_durations_statistics.csv`: Mean, median, std of bout durations
- `binned_timeline_*.csv`: Behavior percentages over time bins
- `transition_matrix.csv`: Behavior-to-behavior transition probabilities

## Advanced Usage

### Selective Analysis with Custom Settings

```bash
# Run only specific analyses with custom bin size
python main_cli.py run \
  --dlc-csv data/*.csv \
  --model models/model_LUPE.pkl \
  --bin-minutes 0.25 \
  --skip-analyses transitions timeline

# Timeline analysis only with fine-grained bins (15 seconds)
python main_cli.py analyze \
  --behaviors behaviors.pkl \
  --output temporal_analysis/ \
  --timeline \
  --bin-minutes 0.25

# Run specific subset of analyses
python main_cli.py analyze \
  --behaviors behaviors.pkl \
  --output selected_analyses/ \
  --instance-counts \
  --durations \
  --transitions
```

### Processing Specific Recording Subsets

```bash
# Process only baseline recordings
python main_cli.py run \
  --dlc-csv data/baseline_*.csv \
  --model models/model_LUPE.pkl \
  --output baseline_results/

# Process treatment recordings
python main_cli.py run \
  --dlc-csv data/treatment_*.csv \
  --model models/model_LUPE.pkl \
  --output treatment_results/

# Then compare using AMPS
python main_cli.py amps \
  --csv-files baseline_results/*/summary.csv \
  --model models/model_AMPS.pkl \
  --output baseline_amps/ \
  --project-name "Baseline"

python main_cli.py amps \
  --csv-files treatment_results/*/summary.csv \
  --model models/model_AMPS.pkl \
  --output treatment_amps/ \
  --project-name "Treatment"
```

### Piping and Automation

```bash
# Complete workflow with success checking
python main_cli.py run \
  --dlc-csv data/*.csv \
  --model models/model_LUPE.pkl \
  --output outputs/ && \
echo "Analysis complete! Processed $(ls -d outputs/*/ | wc -l) recordings"

# Conditional AMPS analysis
if python main_cli.py run \
     --dlc-csv data/*.csv \
     --model models/model_LUPE.pkl \
     --output outputs/; then
  echo "LUPE analysis succeeded, running AMPS..."
  python main_cli.py amps \
    --csv-files outputs/*/summary.csv \
    --model models/model_AMPS.pkl \
    --output outputs/amps_results/
else
  echo "LUPE analysis failed" >&2
  exit 1
fi

# Chain multiple operations
python main_cli.py preprocess --dlc-csv data/*.csv --output preprocessed/ && \
python main_cli.py classify --model models/model_LUPE.pkl --input preprocessed/*.npy --output behaviors.pkl && \
python main_cli.py analyze --behaviors behaviors.pkl --output results/ --all && \
echo "Pipeline complete!"
```

## Error Handling

### Common Errors and Solutions

**Error: "FileNotFoundError: Model file not found at models/model_LUPE.pkl"**
```bash
# Check if the model file exists
ls -l models/

# The model must be named exactly 'model_LUPE.pkl'
# If you downloaded it with a different name, rename it:
mv models/model.pkl models/model_LUPE.pkl

# Or use absolute path
python main_cli.py run --dlc-csv data/*.csv --model /full/path/to/model_LUPE.pkl
```

**Error: "FileNotFoundError: Model file not found at models/model_AMPS.pkl"**
```bash
# Check if the AMPS model exists (should be included in repository)
ls -l models/model_AMPS.pkl

# If missing, check the models directory
ls -l models/

# The AMPS model should be included with the repository
# If it's missing, re-clone the repository or download it separately
```

**Error: "No module named 'src'"**
```bash
# Ensure you're in the project root directory
cd LUPE_analysis_RASO_version
python main_cli.py ...

# Or use the full path
cd /path/to/LUPE_analysis_RASO_version
python main_cli.py run ...
```

**Error: "No CSV files found matching pattern"**
```bash
# Check if your CSV files exist
ls -l data/*.csv

# Make sure you're using the correct pattern for your files
# DeepLabCut files typically have this format:
python main_cli.py run --dlc-csv "data/mouseDLC_resnet50_*.csv"

# Use quotes around patterns with wildcards
python main_cli.py preprocess --dlc-csv "data/*.csv"
```

**Error: "Unable to load behaviors from pickle file"**
```bash
# Check if the behaviors file exists and is valid
python -c "import pickle; pickle.load(open('behaviors.pkl', 'rb'))"

# Re-run classification if the file is corrupted
python main_cli.py classify --model models/model_LUPE.pkl --input preprocessed/*.npy --output behaviors.pkl
```

**Error: "Permission denied"**
```bash
# Check output directory permissions
mkdir -p outputs/
chmod 755 outputs/

# Or use a different output location
python main_cli.py run --dlc-csv data/*.csv --output ~/my_results/
```

**Error: "ModuleNotFoundError: No module named 'sklearn'"**
```bash
# Install required dependencies
pip install -r requirements.txt

# Or specifically install scikit-learn
pip install scikit-learn==1.2.1
```

## Performance Tips

1. **Use the `run` Command**: Simplest and most efficient for complete workflows
   ```bash
   python main_cli.py run --dlc-csv data/*.csv --model models/model_LUPE.pkl
   ```

2. **Use Absolute Paths**: Avoid issues with relative paths
   ```bash
   python main_cli.py run --dlc-csv /full/path/to/data/*.csv --output /full/path/to/outputs/
   ```

3. **Process in Batches**: For many files, split into smaller groups
   ```bash
   # Process first batch
   python main_cli.py run --dlc-csv data/batch1_*.csv --output outputs/
   # Process second batch
   python main_cli.py run --dlc-csv data/batch2_*.csv --output outputs/
   ```

4. **Skip Unnecessary Analyses**: Save time by skipping analyses you don't need
   ```bash
   python main_cli.py run --dlc-csv data/*.csv --skip-analyses timeline transitions
   ```

5. **Redirect Output**: Capture logs for debugging and progress tracking
   ```bash
   python main_cli.py run --dlc-csv data/*.csv 2>&1 | tee analysis.log
   ```

6. **Use Custom Bin Sizes**: Smaller bins = more computation time
   ```bash
   # Faster (larger bins)
   python main_cli.py run --dlc-csv data/*.csv --bin-minutes 2.0

   # Slower but more granular
   python main_cli.py run --dlc-csv data/*.csv --bin-minutes 0.25
   ```

## Integration with Other Tools

### Python Scripts

```python
import subprocess
import glob
from pathlib import Path

# Run complete LUPE analysis from Python
def run_lupe_analysis(dlc_files, output_dir='outputs/'):
    """Run LUPE analysis on DeepLabCut CSV files."""
    result = subprocess.run([
        'python', 'main_cli.py', 'run',
        '--dlc-csv', *dlc_files,
        '--model', 'models/model_LUPE.pkl',
        '--output', output_dir
    ], capture_output=True, text=True)

    if result.returncode == 0:
        print("LUPE analysis completed successfully")
        return True
    else:
        print(f"Error: {result.stderr}")
        return False

# Run AMPS analysis from Python
def run_amps_analysis(summary_files, output_dir='amps_results/'):
    """Run LUPE-AMPS pain scale analysis."""
    result = subprocess.run([
        'python', 'main_cli.py', 'amps',
        '--csv-files', *summary_files,
        '--model', 'models/model_AMPS.pkl',
        '--output', output_dir
    ], capture_output=True, text=True)

    if result.returncode == 0:
        print("AMPS analysis completed successfully")
        return True
    else:
        print(f"Error: {result.stderr}")
        return False

# Example usage
if __name__ == '__main__':
    # Find all DLC CSV files
    dlc_files = glob.glob('data/*.csv')

    # Run LUPE analysis
    if run_lupe_analysis(dlc_files):
        # If successful, run AMPS analysis
        summary_files = glob.glob('outputs/*/summary.csv')
        run_amps_analysis(summary_files)
```

### R Integration

```r
# Run LUPE analysis from R
run_lupe <- function(dlc_files, output_dir = "outputs/") {
  cmd <- sprintf(
    'python main_cli.py run --dlc-csv %s --model models/model_LUPE.pkl --output %s',
    paste(dlc_files, collapse = " "),
    output_dir
  )
  system(cmd)
}

# Run AMPS analysis from R
run_amps <- function(summary_files, output_dir = "amps_results/") {
  cmd <- sprintf(
    'python main_cli.py amps --csv-files %s --model models/model_AMPS.pkl --output %s',
    paste(summary_files, collapse = " "),
    output_dir
  )
  system(cmd)
}

# Example: Process all DLC files
dlc_files <- list.files("data", pattern = "*.csv", full.names = TRUE)
run_lupe(dlc_files)

# Read LUPE results
summary_files <- list.files("outputs", pattern = "*_summary.csv",
                           recursive = TRUE, full.names = TRUE)
results <- lapply(summary_files, read.csv)
combined_results <- do.call(rbind, results)

# Run AMPS analysis
amps_input <- list.files("outputs", pattern = "summary.csv",
                        recursive = TRUE, full.names = TRUE)
run_amps(amps_input)

# Read AMPS results
amps_summary <- read.csv("amps_results/LUPE-AMPS_AMPS_Analysis/LUPE-AMPS_AMPS_summary.csv")
```

## Debugging

### Enable Verbose Output

```bash
# Redirect stderr to see all messages
python main_cli.py run --dlc-csv data/*.csv 2>&1

# Save full output for later review
python main_cli.py run --dlc-csv data/*.csv > analysis.log 2>&1

# View output in real-time while saving to file
python main_cli.py run --dlc-csv data/*.csv 2>&1 | tee analysis.log
```

### Test Individual Steps

If the `run` command fails, test each step individually:

```bash
# Test preprocessing
echo "Testing preprocessing..."
python main_cli.py preprocess --dlc-csv data/test_file.csv --output test_preprocess/

# Test classification
echo "Testing classification..."
python main_cli.py classify --model models/model_LUPE.pkl --input test_preprocess/*.npy --output test_behaviors.pkl

# Test analysis
echo "Testing analysis..."
python main_cli.py analyze --behaviors test_behaviors.pkl --output test_results/ --all
```

### Verify File Formats

```bash
# Check DeepLabCut CSV structure
head -20 data/your_file.csv

# Check if preprocessed data is valid
python -c "import numpy as np; data = np.load('preprocessed/file.npy'); print(f'Shape: {data.shape}')"

# Check behaviors pickle file
python -c "import pickle; behaviors = pickle.load(open('behaviors.pkl', 'rb')); print(f'Files: {list(behaviors.keys())}')"
```

### Common Debugging Commands

```bash
# Check Python environment
python --version
pip list | grep -i "scikit-learn\|numpy\|pandas"

# Verify directory structure
ls -R models/
ls -R data/

# Check for file permissions
ls -la outputs/

# Test model loading
python -c "import pickle; model = pickle.load(open('models/model_LUPE.pkl', 'rb')); print('Model loaded successfully')"
```

## See Also

- [GUI Guide](GUI_GUIDE.md) - For graphical interface
- [Analysis Types](ANALYSIS_TYPES.md) - Detailed analysis descriptions
- [README](../README.md) - Overview and installation
