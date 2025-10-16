# LUPE Analysis Tool - GUI Guide

This guide explains how to use the graphical user interface (GUI) for LUPE analysis.

## Table of Contents

- [Starting the GUI](#starting-the-gui)
- [Interface Overview](#interface-overview)
- [Step-by-Step Workflow](#step-by-step-workflow)
- [Understanding Output](#understanding-output)
- [Troubleshooting](#troubleshooting)

## Starting the GUI

1. Open a terminal/command prompt
2. Navigate to the project directory
3. Activate your virtual environment (if using one)
4. Run:
   ```bash
   python main_gui.py
   ```

The GUI window will appear with the title "LUPE Analysis Tool".

## Interface Overview

The GUI is divided into several sections:

### 1. Input Type Section
Choose your data source:
- **Pre-classified Behaviors (.pkl)**: If you already have classified behavior data
- **Raw DeepLabCut CSV Files**: If you're starting from DLC pose estimation output

### 2. Data Files Section
Depending on your input type selection:

**For Pre-classified Behaviors:**
- **Behaviors File**: Select your behavior data (.pkl file)
- **Output Directory**: Choose where results will be saved

**For Raw DeepLabCut CSV:**
- **DLC CSV Files**: Select one or more DLC output CSV files
- **Model File**: Select the LUPE A-SOiD model (.pkl file)
- **Likelihood Threshold**: Minimum confidence for pose filtering (0.0-1.0, default: 0.1)
- **Output Directory**: Choose where results will be saved

### 3. Select Analyses Section
Checkboxes for each available analysis:
- ☐ Export to CSV
- ☐ Behavior Instance Counts
- ☐ Total Frames (Pie Charts)
- ☐ Bout Durations
- ☐ Binned Timeline
- ☐ Behavior Transitions

Buttons:
- **Select All**: Check all analyses
- **Deselect All**: Uncheck all analyses

### 3. Options Section
- **Timeline Bin Size**: Adjust the time bin size (in minutes) for timeline analysis
  - Default: 1.0 minute
  - Range: 0.5 - 10.0 minutes

### 4. Action Buttons
- **Run Analysis**: Start the selected analyses
- **Clear Log**: Clear the progress log
- **Exit**: Close the application

### 5. Progress Log Section
- **Progress Bar**: Shows overall completion percentage
- **Log Window**: Displays real-time status messages

## Step-by-Step Workflow

There are two workflows depending on your data type:

### Workflow A: Starting from DeepLabCut CSV Files

#### Step 1: Select Input Type
1. Select the **"Raw DeepLabCut CSV Files"** radio button

#### Step 2: Load Your Data
1. Click **Browse** next to "DLC CSV Files"
2. Select one or more DLC CSV files (hold Ctrl/Cmd to select multiple)
3. The number of selected files will be displayed
4. Click **Browse** next to "Model File"
5. Select your LUPE A-SOiD model file (.pkl)
6. Adjust the **Likelihood Threshold** if needed:
   - Lower values (0.05): Keep more data, risk including bad poses
   - Higher values (0.3): Remove more data, keep only high-confidence poses
   - Default (0.1): Good balance for most cases

#### Step 3: Choose Output Location
1. Click **Browse** next to "Output Directory"
2. Select or create a folder for your results

#### Step 4: Select Analyses and Run
1. Choose which analyses to run (or click "Select All")
2. Click **"Run Analysis"**
3. The GUI will:
   - Preprocess each CSV file (filter low-confidence poses)
   - Classify behaviors using the model
   - Run all selected analyses
   - Save results to output directory

### Workflow B: Starting from Pre-classified Behaviors

#### Step 1: Select Input Type
1. Select the **"Pre-classified Behaviors (.pkl)"** radio button

#### Step 2: Load Your Data
1. Click the **Browse** button next to "Behaviors File"
2. Navigate to your behaviors file (`.pkl` format)
3. Select the file and click "Open"
4. The filename will appear in the interface

**Note:** If you don't have a behaviors file yet, you can:
- Use Workflow A (DLC CSV) instead, OR
- Run classification first using the CLI:
```bash
python main_cli.py preprocess --input dlc_data/*.csv --output pose_data.pkl
python main_cli.py classify --model model.pkl --input pose_data.pkl --output behaviors.pkl
```

#### Step 3: Choose Output Location

1. Click **Browse** next to "Output Directory"
2. Select or create a folder for your results
3. The path will appear in the text field

**Tip:** Create a new folder for each analysis run to keep results organized.

#### Step 4: Select Analyses

Choose which analyses to run by checking the boxes:

**For a quick overview:**
- ☑ Export to CSV
- ☑ Total Frames

**For comprehensive analysis:**
- Click "Select All"

**For specific research questions:**
- Instance Counts: How often does each behavior occur?
- Bout Durations: How long do behaviors last?
- Timeline: How do behaviors change over time?
- Transitions: Which behaviors follow each other?

#### Step 5: Configure Options

Adjust settings as needed:
- **Timeline Bin Size**: Smaller bins (0.5 min) for detailed temporal analysis, larger bins (5-10 min) for overview

#### Step 6: Run the Analysis

1. Click **Run Analysis**
2. Watch the progress bar and log window
3. The "Run Analysis" button will be disabled during processing
4. A popup will notify you when complete

**Processing time depends on:**
- Number of files in your dataset
- Number of selected analyses
- Computer performance
- For DLC CSV workflow: Also includes preprocessing and classification time

#### Step 7: Review Results

1. Navigate to your output directory
2. Each analysis creates its own subfolder:
   ```
   outputs/
   ├── csv/                    # Exported CSV files
   ├── instance_counts/        # Bout frequency analysis
   ├── total_frames/          # Time distribution
   ├── durations/             # Bout duration statistics
   ├── timeline/              # Temporal analysis
   └── transitions/           # Transition matrices
   ```

3. Each folder contains:
   - CSV files with raw data
   - SVG/PNG plots for visualization
   - Summary files

## Understanding the Log Window

The log window shows real-time progress:

```
============================================================
Starting LUPE Analysis
============================================================

Loading behaviors from: behaviors.pkl
✓ Loaded behaviors for 10 files

[1/3] Exporting to CSV...
✓ CSV export complete

[2/3] Analyzing instance counts...
✓ Instance counts complete

[3/3] Analyzing total frames...
✓ Total frames complete

============================================================
All analyses completed successfully!
Results saved to: outputs/
============================================================
```

**Symbols:**
- ✓ = Task completed successfully
- ✗ = Task failed (error message follows)

## Tips for Efficient Use

1. **Start Small**: Run one or two analyses first to verify everything works
2. **Use Select All**: For comprehensive analysis of a new dataset
3. **Clear Log Regularly**: Keep the log window readable
4. **Save Output to Dated Folders**: e.g., `outputs/2024-01-15/` for organization
5. **Check CSV First**: Open CSV files in Excel to verify data before looking at plots

## Common Workflows

### Exploratory Analysis
```
1. Select behaviors file
2. Choose output: outputs/exploratory/
3. Check: Export to CSV, Total Frames, Instance Counts
4. Run Analysis
5. Review CSV files and pie charts
```

### Detailed Temporal Analysis
```
1. Select behaviors file
2. Choose output: outputs/temporal/
3. Check: Binned Timeline, Transitions
4. Set bin size: 0.5 minutes (for 30-min recording)
5. Run Analysis
6. Examine timeline plots
```

### Complete Analysis
```
1. Select behaviors file
2. Choose output: outputs/complete_analysis/
3. Click "Select All"
4. Adjust timeline bin size if needed
5. Run Analysis
6. Review all output folders
```

## Troubleshooting

### GUI Won't Start
- **Check Python version**: Must be 3.11+
- **Verify dependencies**: `pip install -r requirements.txt`
- **Try CLI**: `python main_cli.py config --show`

### "No file selected" Error
- Click Browse button first
- Select a valid `.pkl` file
- Verify file is not corrupted

### Analysis Fails
- **Check log window** for specific error message
- **Verify behaviors file** is properly formatted
- **Check disk space** in output directory
- **Try fewer analyses** at once

### Slow Performance
- **Normal for large datasets**: 10+ files may take several minutes
- **Close other programs**: Free up system resources
- **Use CLI for batching**: Better for very large datasets

### Results Not Appearing
- **Check output directory path**: Ensure it's writable
- **Look in subfolders**: Each analysis has its own folder
- **Check log for errors**: May indicate why files weren't created

## Keyboard Shortcuts

- **Ctrl+C**: Copy selected text from log (when log has focus)
- **Alt+F4** (Windows) / **Cmd+Q** (Mac): Close application

## Advanced Features

### Running Multiple Sessions
You can run multiple instances of the GUI simultaneously:
- Different behaviors files
- Different output directories
- No interference between sessions

### Modifying Configuration
While GUI is running, you can edit `config/settings.json` to change:
- Default output directory
- Plot formats (SVG, PNG, PDF)
- Other preferences

**Note:** Restart GUI for config changes to take effect.

## Next Steps

After running analysis:
1. Open CSV files in Excel/Python/R for further analysis
2. View plots in any SVG viewer or web browser
3. Integrate with statistical software
4. Create custom visualizations from the raw data

For automation or batch processing, consider using the [CLI](CLI_GUIDE.md) instead.
