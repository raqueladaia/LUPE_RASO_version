# Getting Started with LUPE Analysis Tool

This guide will help you get up and running with the LUPE Analysis Tool in minutes.

> **Looking for an overview?** See [README.md](README.md) for feature list, installation details, and troubleshooting.

## Quick Start Checklist

- [ ] Python 3.11+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] LUPE model downloaded (.pkl format, required for classifying new data)
- [ ] Data file ready:
  - DeepLabCut CSV file (.csv, if processing raw pose data), OR
  - Pre-classified behavior data (.pkl format)

## Step-by-Step Setup

### 1. Verify Python Installation

```bash
python --version
# Should show Python 3.11 or higher
```

If you need to install Python:
- Windows: Download from [python.org](https://www.python.org/downloads/)
- Mac: `brew install python@3.11`
- Linux: `sudo apt-get install python3.11`

### 2. Install Dependencies

```bash
# Navigate to project directory
cd LUPE_analysis_RASO_version

# Install required packages
pip install -r requirements.txt

# Verify installation
python -c "import numpy, pandas, matplotlib; print('All dependencies installed!')"
```

### 3. Test the Installation

**Test GUI:**
```bash
python main_gui.py
```
A window should appear. If so, installation is successful!

**Test CLI:**
```bash
python main_cli.py config --show
```
Should display behavior definitions and settings.

## Your First Analysis

### Option A: Using the GUI (Easiest)

1. **Launch the GUI**
   ```bash
   python main_gui.py
   ```

2. **Load your data**
   - Click "Browse" next to "Behaviors File"
   - Select your `.pkl` file

3. **Choose analyses**
   - Click "Select All" for comprehensive analysis
   - Or select specific analyses

4. **Run**
   - Click "Run Analysis"
   - Watch the progress in the log window
   - Results appear in your output directory

### Option B: Using the CLI (For Automation)

```bash
# Export to CSV (quick check)
python main_cli.py export \
  --behaviors your_behaviors.pkl \
  --output csv/

# Run all analyses
python main_cli.py analyze \
  --behaviors your_behaviors.pkl \
  --output results/ \
  --all
```

## Understanding Your Results

After running analysis, check the output directory:

```
outputs/
├── csv/               # Spreadsheet-compatible data
├── instance_counts/   # How often behaviors occur
├── total_frames/      # Time distribution pie charts
├── durations/         # How long behaviors last
├── timeline/          # Changes over time
└── transitions/       # Behavior sequences
```

**What to look at first:**
1. Open `csv/summary.csv` in Excel
2. View `total_frames/total_frames_pie.svg` in browser
3. Check `instance_counts/instance_counts.svg`

## Working with DeepLabCut CSV Files

If you have raw DeepLabCut CSV output files, you need to preprocess them first before analysis.

### Complete Workflow: DLC CSV → Analysis

**Using GUI:**
1. Launch GUI: `python main_gui.py`
2. Select "Raw DeepLabCut CSV Files" radio button
3. Browse and select your DLC CSV file(s)
4. Browse and select your LUPE model file (.pkl)
5. Adjust likelihood threshold if needed (default: 0.1)
6. Select output directory
7. Choose which analyses to run
8. Click "Run Analysis"

The GUI will automatically:
- Preprocess the CSV files (filter low-confidence poses)
- Classify behaviors using the model
- Run all selected analyses
- Save results to output directory

**Using CLI:**
```bash
# Step 1: Preprocess DLC CSV files
python main_cli.py preprocess \
  --input dlc_data/*.csv \
  --output pose_data.pkl \
  --likelihood-threshold 0.1

# Step 2: Classify behaviors
python main_cli.py classify \
  --model models/model_LUPE-AMPS.pkl \
  --input pose_data.pkl \
  --output behaviors.pkl

# Step 3: Run analyses
python main_cli.py analyze \
  --behaviors behaviors.pkl \
  --output results/ \
  --all
```

### What Does Preprocessing Do?

DeepLabCut CSV files contain:
- x, y coordinates for each tracked keypoint
- Likelihood values (confidence scores 0-1)

Preprocessing:
1. Loads the multi-level header CSV
2. Filters out poses with likelihood below threshold (default: 0.1)
3. Replaces low-confidence points with interpolated values
4. Extracts clean x,y coordinate arrays for feature extraction

## Common Workflows

### Workflow 1: Quick Data Check
```bash
python main_cli.py export --behaviors data.pkl --output csv/
# Then open CSV in Excel to verify data
```

### Workflow 2: Publication Figures
```bash
python main_cli.py analyze \
  --behaviors data.pkl \
  --output figures/ \
  --instance-counts \
  --total-frames \
  --durations
# Generates publication-ready SVG files
```

### Workflow 3: Complete Analysis
```bash
# Using GUI: Click "Select All" → "Run Analysis"
# Using CLI:
python main_cli.py analyze --behaviors data.pkl --output complete/ --all
```

## Next Steps

**For beginners:**
1. Read [GUI Guide](docs/GUI_GUIDE.md) for detailed GUI instructions
2. Start with "Select All" to see all output types
3. Review CSV files to understand data structure

**For advanced users:**
1. Read [CLI Guide](docs/CLI_GUIDE.md) for automation
2. See [ANALYSIS_TYPES.md](docs/ANALYSIS_TYPES.md) for analysis details
3. Customize `config/settings.json` for your needs

**For developers:**
1. Explore `src/` directory for module structure
2. Check `config/metadata.json` for behavior definitions
3. Extend analyses by creating new modules in `src/core/`

## Configuration

### Quick Config Changes

Edit `config/settings.json`:

```json
{
  "paths": {
    "default_output_path": "my_outputs"  // Change default output
  },
  "output_preferences": {
    "plot_format": "png"  // Change from SVG to PNG
  }
}
```

### Behavior Definitions

View current behaviors:
```bash
python main_cli.py config --behaviors
```

Behaviors are defined in `config/metadata.json` (don't modify unless you know what you're doing).

## Troubleshooting

### "Import Error: No module named X"
```bash
pip install -r requirements.txt --upgrade
```

### GUI won't start
Try the CLI first to verify installation:
```bash
python main_cli.py config --show
```

### No output files created
- Check the log window (GUI) or terminal (CLI) for errors
- Verify output directory is writable
- Ensure behaviors file is valid (.pkl format)

### Slow performance
- Normal for large datasets (10+ files)
- Use CLI for better performance feedback
- Close other applications to free memory

## Getting Help

1. **Check documentation**:
   - [README.md](README.md) - Overview
   - [GUI_GUIDE.md](docs/GUI_GUIDE.md) - GUI instructions
   - [CLI_GUIDE.md](docs/CLI_GUIDE.md) - CLI reference
   - [ANALYSIS_TYPES.md](docs/ANALYSIS_TYPES.md) - Analysis details

2. **Run built-in help**:
   ```bash
   python main_cli.py --help
   python main_cli.py analyze --help
   ```

3. **Check your setup**:
   ```bash
   python main_cli.py config --show
   ```

## Tips for Success

1. **Start Small**: Test with one file first
2. **Use CSV Export**: Always start here to verify data
3. **Save Outputs**: Create dated folders for each analysis run
4. **Read the Logs**: GUI log window and CLI output show what's happening
5. **Check Examples**: See `docs/` for example commands and workflows

## Example: Complete Beginner Workflow

```bash
# 1. Verify installation
python main_cli.py config --show

# 2. Quick export to check data
python main_cli.py export \
  --behaviors my_data.pkl \
  --output check/

# 3. Open check/summary.csv in Excel
# - Verify file names are correct
# - Check frame counts make sense
# - Look at behavior percentages

# 4. Run full analysis using GUI
python main_gui.py
# - Load my_data.pkl
# - Select All analyses
# - Run and review results

# 5. Use results
# - Open SVG files in browser
# - Import CSVs into Excel/R/Python
# - Create publication figures
```

## You're Ready!

You now have everything needed to analyze LUPE data. Start with the GUI for exploration, then move to CLI for automation as you become more comfortable.

For detailed information on each analysis type, see [ANALYSIS_TYPES.md](docs/ANALYSIS_TYPES.md).

Happy analyzing!
