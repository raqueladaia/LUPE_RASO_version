# LUPE-AMPS Pain Scale Analysis Guide

## Overview

**LUPE-AMPS** (Advanced Multivariate Pain Scale) is a specialized analysis tool that transforms discrete behavioral state classifications into a continuous pain scale using Principal Component Analysis (PCA).

The AMPS model projects 6-dimensional behavior occupancy data onto a 2-dimensional space:
- **PC1**: Generalized Behavior Scale
- **PC2**: Pain Behavior Scale (the primary measure of interest)

This allows researchers to quantify pain-related behaviors on a continuous scale rather than just counting discrete states.

---

## What You Need

### Input Files

**Behavior CSV Files** - Frame-by-frame behavior classifications with this format:

```csv
frame,behavior_id
1,0
2,0
3,1
4,1
...
```

- **Two columns**: `frame` and `behavior_id`
- **Behavior IDs**: Integers from 0 to 5 (representing 6 behavioral states)
- **One file per animal/recording**

These CSV files should be the output from your A-SOiD behavior classification (from the main LUPE GUI).

### Model File

**LUPE-AMPS PCA Model**: `models/model_AMPS.pkl`

This pre-trained model is included in the `models/` directory.

---

## Quick Start

### 1. Launch the GUI

```bash
python main_lupe_amps_gui.py
```

### 2. Add Your Files

Click **"Add Files..."** and select one or more behavior CSV files.

### 3. Configure Parameters (Optional)

The default parameters work for standard 30-minute recordings at 60 fps:
- **Recording Length**: 30 minutes
- **Original Framerate**: 60 fps
- **Target Framerate**: 20 fps (downsampling target)

Adjust these if your recordings differ.

### 4. Select Analysis Sections

All sections are selected by default:
- ☑ **Section 1**: Preprocessing & Metrics Calculation
- ☑ **Section 2**: PCA Pain Scale Projection
- ☑ **Section 3**: Behavior Metrics Visualization
- ☑ **Section 4**: Model Feature Importance

### 5. Set Output Location

- **Output Directory**: Where results will be saved (default: `outputs/`)
- **Project Name**: Name for your analysis (default: `LUPE-AMPS`)

### 6. Run Analysis

Click **"Run LUPE-AMPS Analysis"** and monitor progress in the log window.

---

## Analysis Sections Explained

### Section 1: Preprocessing & Metrics Calculation

**What it does:**
- Loads each CSV file
- Downsamples from 60 fps → 20 fps (reduces computational load)
- Pads or truncates to exactly 30 minutes
- Calculates three key metrics for each behavior state (0-5):
  - **Fraction Occupancy**: % of time in each state
  - **Number of Bouts**: Count of continuous sequences
  - **Bout Duration**: Mean seconds per bout

**Outputs:**
```
Section1_preprocessing/
└── metrics_all_files.csv
```

This CSV contains all metrics for all files.

---

### Section 2: PCA Pain Scale Projection

**What it does:**
- Uses the pre-trained LUPE-AMPS model to project behavior occupancy onto a 2D pain scale
- PC1 = Generalized Behavior Scale
- PC2 = **Pain Behavior Scale** (key measure)
- Higher PC2 values indicate more pain-related behaviors

**Outputs:**
```
Section2_pain_scale/
├── pain_scale_projection.png    # Scatter plot (raster)
├── pain_scale_projection.svg    # Scatter plot (vector, publication-ready)
└── pain_scale_projection.csv    # PC1, PC2 coordinates for each file
```

**Interpreting the plot:**
- Each point represents one recording
- Y-axis (PC2) is the **pain behavior scale**
- Points higher on the Y-axis show more pain-related behavior patterns

---

### Section 3: Behavior Metrics Visualization

**What it does:**
- Creates bar plots showing behavior patterns across all 6 states
- For multiple files: shows mean ± SEM (standard error of mean)
- For single file: shows individual values

**Outputs:**
```
Section3_behavior_metrics/
├── {project}_fraction_occupancy.png/svg/csv
├── {project}_number_of_bouts.png/svg/csv
└── {project}_bout_duration.png/svg/csv
```

**Interpreting the plots:**
- **Fraction Occupancy**: Which states dominate? (higher bars = more time)
- **Number of Bouts**: How fragmented is behavior? (more bouts = more switching)
- **Bout Duration**: How long do behaviors last? (longer = more sustained)

---

### Section 4: Model Feature Importance

**What it does:**
- Tests 9 model variations to understand which behaviors contribute most to classification
- Measures model fit when removing different features
- Helps validate that the model is using meaningful behavioral patterns

**Outputs:**
```
Section4_model_fit/
├── {project}_feature_importance.png
├── {project}_feature_importance.svg
└── {project}_feature_importance.csv
```

**Interpreting the plot:**
- Shows how removing different features affects model performance
- Lower scores when removing a feature = that feature is important
- "Shuffled (Control)" should show worst performance (validation check)

---

## Complete Output Structure

After running all sections, you'll have:

```
outputs/
└── {project_name}_LUPE-AMPS/
    ├── Section1_preprocessing/
    │   └── metrics_all_files.csv
    │
    ├── Section2_pain_scale/
    │   ├── pain_scale_projection.png
    │   ├── pain_scale_projection.svg
    │   └── pain_scale_projection.csv
    │
    ├── Section3_behavior_metrics/
    │   ├── {project}_fraction_occupancy.png
    │   ├── {project}_fraction_occupancy.svg
    │   ├── {project}_fraction_occupancy.csv
    │   ├── {project}_number_of_bouts.png
    │   ├── {project}_number_of_bouts.svg
    │   ├── {project}_number_of_bouts.csv
    │   ├── {project}_bout_duration.png
    │   ├── {project}_bout_duration.svg
    │   └── {project}_bout_duration.csv
    │
    └── Section4_model_fit/
        ├── {project}_feature_importance.png
        ├── {project}_feature_importance.svg
        └── {project}_feature_importance.csv
```

---

## Complete Workflow Example

### Step 1: Classify Behaviors (Main LUPE GUI)

```bash
python main_gui.py
```

1. Load your DLC CSV files
2. Select A-SOiD model
3. Run classification
4. This creates behavior CSVs in: `outputs/{filename}/{filename}_behaviors.csv`

### Step 2: Analyze Pain Scale (LUPE-AMPS GUI)

```bash
python main_lupe_amps_gui.py
```

1. Add the behavior CSV files from Step 1
2. Keep default parameters (or adjust as needed)
3. Set project name (e.g., "Morphine_Study")
4. Run analysis
5. Review results in `outputs/Morphine_Study_LUPE-AMPS/`

### Step 3: Interpret Results

**Key Metrics to Report:**
- **PC2 (Pain Scale)** from Section 2 - primary measure
- **Fraction Occupancy** from Section 3 - which states are most common
- **Bout Duration** from Section 3 - how sustained are behaviors

**Publication-Ready Figures:**
- Use the `.svg` files for high-quality vector graphics
- Pain scale scatter plot (Section 2)
- Behavior metrics bar plots (Section 3)

---

## Technical Details

### Data Processing

1. **Downsampling Method**: Mode (most common value) in 3-frame windows (60→20 fps)
2. **Length Standardization**: All recordings padded/truncated to exactly 30 minutes
3. **Bout Detection**: Uses `scipy.ndimage.label` to identify continuous sequences
4. **Duration Calculation**: Frame count ÷ framerate = seconds

### Model Information

- **Type**: Scikit-learn PCA model
- **Input**: 6-dimensional occupancy vector (fraction of time in each state)
- **Output**: 2 principal components (PC1 and PC2)
- **Training**: Pre-trained on reference pain behavior data

### Performance Considerations

- **Single File**: Processes in seconds
- **Multiple Files (10-20)**: 1-2 minutes total
- **Large Batch (50+)**: Several minutes
- Section 4 (feature importance) is most computationally intensive

---

## Troubleshooting

### CSV Format Errors

**Problem**: "CSV must have columns: ['frame', 'behavior_id']"

**Solution**: Your CSV must have exactly these two columns. Check the output from the main LUPE GUI.

### Model Not Found

**Problem**: "Model file not found: models/model_AMPS.pkl"

**Solution**: Verify the model file exists in the `models/` directory.

### File Length Issues

**Problem**: "File is too short" or unexpected padding

**Solution**: Adjust "Recording Length" parameter to match your actual recording duration.

### Memory Errors

**Problem**: Out of memory with many files

**Solution**: Process files in smaller batches (e.g., 20 files at a time).

### Wrong Framerate

**Problem**: Behavior appears too compressed or expanded

**Solution**: Verify your original recording framerate and adjust the "Original Framerate" parameter.

---

## Best Practices

1. **Batch Similar Recordings**: Analyze recordings from the same experimental session together
2. **Use Descriptive Project Names**: Include date, condition, or cohort (e.g., "Study1_Control_2024")
3. **Keep CSV Files**: Save both original CSVs and LUPE-AMPS outputs for reproducibility
4. **Document Parameters**: Note any non-default parameters used
5. **Save Plots as SVG**: Use `.svg` files for publications (scalable, high-quality)

---

## Comparison with Original Notebook

### Simplifications (No Groups/Conditions)

The GUI version is simplified compared to the original LUPE-AMPS Jupyter notebook:

**Removed Features:**
- ❌ Group comparisons (e.g., Male vs Female)
- ❌ Condition comparisons (e.g., different drug doses)
- ❌ Color-coding by experimental condition
- ❌ Statistical comparisons between groups

**Why?**: Following the project requirement: *"I don't want groups nor conditions. I want this new version only analyzes the data without doing any further analysis to compare groups or conditions."*

**Retained Features:**
- ✅ All 4 analysis sections
- ✅ Downsampling and preprocessing
- ✅ PCA projection to pain scale
- ✅ Behavior metrics calculation
- ✅ Model fit/feature importance analysis
- ✅ Aggregate statistics (mean ± SEM across files)

### What This Means for You

**You can still analyze multiple files**, but the GUI will:
- Show aggregate statistics (mean ± SEM) across all files
- Use single color scheme (not condition-based)
- Report individual file values in CSV exports
- Not perform statistical comparisons

**If you need group/condition comparisons:**
- Export the CSV files from Section 2 (pain scale coordinates)
- Perform statistical analysis in your preferred tool (Excel, R, SPSS, etc.)
- The CSV contains all individual file data needed for comparison

---

## Related Documentation

- **Main LUPE GUI**: See `GETTING_STARTED.md` for behavior classification
- **Reference Repository**: Original LUPE-AMPS notebook in `reference_repo/`
- **CLI Usage**: For command-line batch processing (coming soon)

---

## Citation

If you use LUPE-AMPS analysis in your research, please cite:

```
LUPE 2.0: Light Automated Pain Evaluator
Corder Lab (University of Pennsylvania) & Yttri Lab (Carnegie Mellon)
```

---

## Support

For questions or issues:
1. Check this documentation
2. Review error messages in the Progress Log
3. Verify input file formats
4. Check the LUPE GitHub repository for updates

---

**Last Updated**: Created for LUPE Analysis RASO Version
