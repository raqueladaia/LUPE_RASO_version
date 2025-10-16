# LUPE Analysis Tool - CLI Guide

Complete reference for the Command-Line Interface (CLI).

## Quick Reference

```bash
# Get help
python main_cli.py --help
python main_cli.py <command> --help

# Classify behaviors
python main_cli.py classify --model model.pkl --input data/*.npy --output behaviors.pkl

# Run analyses
python main_cli.py analyze --behaviors behaviors.pkl --output results/ --all

# Export to CSV
python main_cli.py export --behaviors behaviors.pkl --output csv/

# View configuration
python main_cli.py config --show
```

## Available Commands

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
  --model models/model_LUPE-AMPS.pkl \
  --input data/mouse1.npy \
  --output behaviors.pkl

# Multiple files
python main_cli.py classify \
  --model models/model_LUPE-AMPS.pkl \
  --input data/mouse1.npy data/mouse2.npy data/mouse3.npy \
  --output all_behaviors.pkl

# Custom framerate
python main_cli.py classify \
  --model models/model_LUPE-AMPS.pkl \
  --input data/*.npy \
  --output behaviors.pkl \
  --framerate 30
```

### `analyze` - Run Behavior Analyses

Performs various analyses on behavior classification data.

**Usage:**
```bash
python main_cli.py analyze \
  --behaviors BEHAVIORS_FILE \
  --output OUTPUT_DIR \
  [--all | --instance-counts | --total-frames | --durations | --timeline | --transitions] \
  [--bin-minutes MINUTES]
```

**Arguments:**
- `--behaviors`, `-b`: Path to behaviors file (.pkl) **[Required]**
- `--output`, `-o`: Output directory for results **[Required]**
- `--all`: Run all analyses
- `--instance-counts`: Analyze behavior bout counts
- `--total-frames`: Analyze frame percentages
- `--durations`: Analyze bout durations
- `--timeline`: Analyze temporal patterns
- `--transitions`: Analyze behavior transitions
- `--bin-minutes`: Bin size for timeline (default: 1.0)

**Examples:**
```bash
# Run all analyses
python main_cli.py analyze \
  --behaviors behaviors.pkl \
  --output results/ \
  --all

# Specific analyses
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

### Process Multiple Files

```bash
# Classify all files in a directory
python main_cli.py classify \
  --model models/model.pkl \
  --input data/*.npy \
  --output all_behaviors.pkl

# Then analyze
python main_cli.py analyze \
  --behaviors all_behaviors.pkl \
  --output results/ \
  --all
```

### Automated Pipeline

Create a bash script (`process_all.sh`):

```bash
#!/bin/bash

# Configuration
MODEL="models/model_LUPE-AMPS.pkl"
DATA_DIR="data/"
OUTPUT_DIR="results_$(date +%Y%m%d)/"

# Step 1: Classify
echo "Classifying behaviors..."
python main_cli.py classify \
  --model $MODEL \
  --input $DATA_DIR/*.npy \
  --output $OUTPUT_DIR/behaviors.pkl

# Step 2: Analyze
echo "Running analyses..."
python main_cli.py analyze \
  --behaviors $OUTPUT_DIR/behaviors.pkl \
  --output $OUTPUT_DIR \
  --all

# Step 3: Export
echo "Exporting to CSV..."
python main_cli.py export \
  --behaviors $OUTPUT_DIR/behaviors.pkl \
  --output $OUTPUT_DIR/csv/ \
  --summary

echo "Complete! Results in $OUTPUT_DIR"
```

Run with: `bash process_all.sh`

## Output Structure

After running analyses, your output directory will contain:

```
results/
├── csv/
│   ├── frames/           # Frame-by-frame CSV
│   ├── seconds/          # Second-by-second CSV
│   └── summary.csv       # Overall summary
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

## Advanced Usage

### Selective Analysis with Custom Settings

```bash
# Only timeline with specific bin size
python main_cli.py analyze \
  --behaviors behaviors.pkl \
  --output temporal_analysis/ \
  --timeline \
  --bin-minutes 0.25  # 15-second bins

# Multiple specific analyses
python main_cli.py analyze \
  --behaviors behaviors.pkl \
  --output selected_analyses/ \
  --instance-counts \
  --durations \
  --transitions
```

### Piping and Automation

```bash
# Process and count results
python main_cli.py export \
  --behaviors behaviors.pkl \
  --output csv/ && \
echo "Exported $(ls csv/frames/*.csv | wc -l) files"

# Conditional processing
if python main_cli.py analyze \
     --behaviors behaviors.pkl \
     --output results/ \
     --all; then
  echo "Success!"
else
  echo "Analysis failed" >&2
  exit 1
fi
```

## Error Handling

### Common Errors and Solutions

**Error: "FileNotFoundError: Model file not found"**
```bash
# Check model path
ls -l models/

# Use absolute path
python main_cli.py classify --model /full/path/to/model.pkl ...
```

**Error: "No module named 'src'"**
```bash
# Ensure you're in the project root directory
cd LUPE_analysis_RASO_version
python main_cli.py ...
```

**Error: "Permission denied"**
```bash
# Check output directory permissions
mkdir -p results/
chmod 755 results/
```

## Performance Tips

1. **Use Absolute Paths**: Avoid issues with relative paths
2. **Process in Batches**: For many files, split into smaller groups
3. **Monitor Resources**: Use `--instance-counts` first to gauge performance
4. **Redirect Output**: Capture logs for debugging
   ```bash
   python main_cli.py analyze ... 2>&1 | tee analysis.log
   ```

## Integration with Other Tools

### Python Scripts

```python
import subprocess

# Run classification from Python
subprocess.run([
    'python', 'main_cli.py', 'classify',
    '--model', 'model.pkl',
    '--input', 'data.npy',
    '--output', 'behaviors.pkl'
])
```

### R Integration

```r
# Run from R
system('python main_cli.py export --behaviors behaviors.pkl --output csv/')

# Read results
data <- read.csv('csv/summary.csv')
```

## Debugging

Enable verbose output:

```bash
# Redirect stderr to see all messages
python main_cli.py analyze ... 2>&1

# Save full output
python main_cli.py analyze ... > output.log 2>&1
```

## See Also

- [GUI Guide](GUI_GUIDE.md) - For graphical interface
- [Analysis Types](ANALYSIS_TYPES.md) - Detailed analysis descriptions
- [README](../README.md) - Overview and installation
