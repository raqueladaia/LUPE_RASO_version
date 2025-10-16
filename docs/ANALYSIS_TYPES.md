# LUPE Analysis Types - Complete Reference

This document describes each analysis type available in the LUPE Analysis Tool.

## Table of Contents

1. [CSV Export](#1-csv-export)
2. [Instance Counts](#2-instance-counts)
3. [Total Frames](#3-total-frames)
4. [Bout Durations](#4-bout-durations)
5. [Binned Timeline](#5-binned-timeline)
6. [Behavior Transitions](#6-behavior-transitions)
7. [Advanced Analyses](#7-advanced-analyses)

---

## 1. CSV Export

**Purpose:** Export behavior classifications to CSV format for use in other analysis software (Excel, R, Python, MATLAB, etc.).

**What It Does:**
- Converts pickled behavior data to human-readable CSV files
- Creates both frame-by-frame and time-based exports
- Optionally includes behavior names (not just numeric IDs)
- Generates summary statistics across all files

**Output Files:**
```
csv/
├── frames/
│   ├── file1.csv          # Frame number, behavior ID
│   └── file2.csv
├── seconds/
│   ├── file1.csv          # Time in seconds, behavior ID
│   └── file2.csv
└── summary.csv            # Statistics for all files
```

**CSV Format - Frames:**
```csv
frame,behavior_id
1,0
2,0
3,1
4,1
5,2
```

**CSV Format - Summary:**
```csv
file_name,total_frames,duration_seconds,still_%,walking_%,rearing_%,...
file1,3600,60.0,45.2,30.1,15.3,...
file2,7200,120.0,50.5,25.0,18.2,...
```

**When to Use:**
- Need data in Excel or other spreadsheet software
- Want to perform custom statistical analyses
- Need to integrate with existing analysis pipelines
- Want raw data without visualizations

**Parameters:**
- None (uses configuration defaults)

---

## 2. Instance Counts

**Purpose:** Count how many times each behavior occurs (bout frequency).

**What It Does:**
- Counts the number of bouts for each behavior
- A "bout" is a continuous period of the same behavior
- Calculates mean and standard deviation across files
- Creates bar charts showing results

**Example:**
If a mouse's behavior sequence is:
```
still → walking → still → rearing → still
```
Instance counts would be:
- Still: 3 bouts
- Walking: 1 bout
- Rearing: 1 bout

**Output Files:**
```
instance_counts/
├── instance_counts_summary.csv    # Mean ± std for each behavior
├── instance_counts_raw.csv        # Counts for each file
└── instance_counts.svg            # Bar chart visualization
```

**Metrics:**
- **Mean Count**: Average number of bouts per behavior across all files
- **Std Count**: Standard deviation (variability between files)
- **Per-File Counts**: Raw bout counts for each file

**When to Use:**
- Compare activity levels between datasets
- Measure behavioral switching frequency
- Identify most/least frequent behaviors
- Quality control (unusually low/high counts may indicate issues)

**Interpretation:**
- High counts = frequent switching to/from this behavior
- Low counts = rare behavior or long continuous bouts
- High std = variable between subjects/files
- Compare with duration analysis for complete picture

---

## 3. Total Frames

**Purpose:** Analyze what percentage of total time is spent in each behavior.

**What It Does:**
- Counts total frames for each behavior across entire dataset
- Calculates percentages
- Creates pie/donut charts showing distribution
- Provides per-file breakdown

**Output Files:**
```
total_frames/
├── total_frames_summary.csv       # Overall percentages
├── total_frames_per_file.csv     # Breakdown per file
└── total_frames_pie.svg          # Pie chart
```

**CSV Format:**
```csv
behavior,frames,percentage
still,15000,50.0
walking,9000,30.0
rearing,6000,20.0
```

**When to Use:**
- Get overall behavior distribution
- Identify dominant behaviors
- Compare time budgets between conditions
- Create publication-ready pie charts

**Interpretation:**
- Percentages sum to 100%
- Largest slice = most time-consuming behavior
- Small slices may be rare but important behaviors
- Compare with instance counts to distinguish:
  - Few long bouts vs. many short bouts

**Example Use Case:**
"Mice spent 50% of time still, 30% walking, and 20% in other behaviors."

---

## 4. Bout Durations

**Purpose:** Analyze how long each behavior bout lasts.

**What It Does:**
- Measures duration of every bout
- Calculates statistics (mean, median, std, min, max)
- Creates box-and-whisker plots
- Identifies outliers (unusually long/short bouts)

**Output Files:**
```
durations/
├── bout_durations_statistics.csv  # Summary stats per behavior
├── bout_durations_raw.csv        # All bout durations
└── bout_durations_boxplot.svg    # Box plot
```

**Statistics:**
```csv
behavior,mean_duration_sec,median_duration_sec,std_duration_sec,min_duration_sec,max_duration_sec,total_bouts
still,5.2,3.1,4.8,0.1,45.3,2500
walking,2.3,1.5,2.1,0.1,18.7,1800
rearing,1.8,1.2,1.5,0.1,12.4,1200
```

**When to Use:**
- Understand behavior persistence
- Identify abnormal bout lengths
- Compare behavior "stickiness" between conditions
- Detect sustained vs. transient behaviors

**Interpretation:**
- **High mean/median**: Behavior persists once started
- **High std**: Variable bout lengths
- **Large max**: Some very long episodes
- **Short median but high mean**: Skewed by outliers

**Example Insights:**
- Still bouts average 5 seconds (mice briefly pause)
- Some still bouts last >40 seconds (extended resting)
- Walking bouts are shorter and more uniform

---

## 5. Binned Timeline

**Purpose:** Visualize how behavior distribution changes over time.

**What It Does:**
- Divides recording into equal time bins
- Calculates percentage of each behavior per bin
- Plots timeline with error bands (SEM)
- Reveals temporal patterns

**Parameters:**
- **Bin Size**: Duration of each bin (default: 1 minute)
  - Smaller bins (0.5 min): More detail, more noise
  - Larger bins (5-10 min): Smoother, less detail

**Output Files:**
```
timeline/
├── binned_timeline_1.0min.csv     # Percentages per bin
└── binned_timeline_1.0min.svg     # Line plot
```

**CSV Format:**
```csv
time_bin,still_mean,still_sem,walking_mean,walking_sem,...
0,0.45,0.05,0.35,0.04,...
1,0.50,0.06,0.30,0.03,...
2,0.48,0.05,0.32,0.04,...
```

**When to Use:**
- Detect habituation effects
- Identify circadian/temporal patterns
- Compare early vs. late recording periods
- Assess treatment effects over time

**Interpretation:**
- **Flat lines**: Stable behavior throughout recording
- **Trends**: Increasing/decreasing over time
  - Upward still: Animal calming down
  - Downward walking: Habituation to environment
- **Peaks/Valleys**: Periodic patterns
- **Wide error bands**: High variability between subjects

**Example Patterns:**
- Initial high activity, then settling
- Periodic increases in specific behaviors
- Treatment effects appearing after delay

---

## 6. Behavior Transitions

**Purpose:** Analyze which behaviors follow which other behaviors.

**What It Does:**
- Creates transition probability matrix
- Shows likelihood of switching between behaviors
- Identifies common/rare transition patterns
- Generates heatmap visualization

**Output Files:**
```
transitions/
├── transition_matrix.csv           # Probability matrix
└── transition_matrix_heatmap.svg   # Heatmap
```

**Matrix Format:**
```
          still  walking  rearing  grooming
still     0.00   0.60     0.25     0.15
walking   0.50   0.00     0.30     0.20
rearing   0.70   0.20     0.00     0.10
grooming  0.40   0.40     0.20     0.00
```

Reading: "From walking, 50% chance of transitioning to still, 30% to rearing, 20% to grooming"

**When to Use:**
- Understand behavioral sequences
- Identify stereotyped patterns
- Detect unusual transition patterns
- Build Markov models of behavior

**Interpretation:**
- **Diagonal = 0**: Can't transition to self (by definition)
- **High probability**: Common transition
- **Low probability**: Rare transition
- **Zero**: Transition never observed
- **Row sums = 1.0**: Each row is a probability distribution

**Common Patterns:**
- Still → Walking: High (animals move after resting)
- Walking → Still: High (animals rest after moving)
- Rearing → Grooming: Moderate (sequential behaviors)

**Example Insights:**
- Mice almost never go directly from grooming to rearing (0.10 probability)
- Most transitions from walking lead to still (0.50 probability)
- Rearing usually followed by still (0.70 probability)

---

## 7. Advanced Analyses

### Location Tracking

**Purpose:** Visualize where in the arena the animal spends time.

**Requires:** Pose data (not just behavior classifications)

**Output:**
- Trajectory plots showing movement paths
- Heatmaps of location density
- Per-behavior location preferences

**When to Use:**
- Analyze spatial preferences
- Detect thigmotaxis (wall-hugging)
- Identify preferred zones
- Combine with behavior data

### Kinematics Analysis

**Purpose:** Analyze movement properties (speed, distance, acceleration).

**Requires:** Pose data

**Metrics:**
- Mean speed per behavior
- Total distance traveled
- Acceleration patterns
- Movement efficiency

**When to Use:**
- Distinguish active vs. passive behaviors
- Quantify locomotor activity
- Compare movement quality
- Detect motor impairments

### Distance Traveled Heatmaps

**Purpose:** Show where in the arena the most movement occurs.

**Output:**
- Spatial heatmaps of activity
- Zones of high/low activity
- Per-behavior spatial patterns

---

## Choosing the Right Analyses

### For First-Time Analysis
Start with:
1. **CSV Export** - Verify data integrity
2. **Total Frames** - Get overview
3. **Instance Counts** - Understand frequency

### For Publication
Include:
1. **Total Frames** - For distribution pie charts
2. **Instance Counts** - For activity levels
3. **Bout Durations** - For behavioral persistence
4. **Timeline** - For temporal effects

### For Comparing Conditions
Focus on:
1. **Instance Counts** - Changed activity?
2. **Bout Durations** - Changed persistence?
3. **Transitions** - Changed sequences?
4. **Timeline** - When do differences appear?

### For Detailed Characterization
Run all analyses:
1. Start with basic (CSV, Total Frames)
2. Add frequency (Instance Counts)
3. Add temporal (Durations, Timeline)
4. Add sequential (Transitions)
5. Add spatial (Location, if needed)

---

## Statistical Considerations

### Sample Size
- Minimum 3-5 files per group for meaningful statistics
- More files needed for small effects
- Check variability (std/sem) to assess consistency

### Multiple Comparisons
When comparing multiple behaviors or time bins:
- Consider correction (Bonferroni, FDR)
- Focus on pre-planned comparisons
- Use timeline for exploratory, then confirm

### Outliers
- Check bout durations for extreme values
- Verify with video if possible
- May indicate:
  - Real biological phenomenon
  - Classification errors
  - Technical issues

### Normality
- Behavior data often non-normal
- Consider non-parametric tests
- Transform if needed (log, sqrt)

---

## Best Practices

1. **Always start with CSV export** to verify data
2. **Run Total Frames first** for overview
3. **Combine Instance Counts and Durations** for complete picture
4. **Use Timeline for temporal patterns**
5. **Check Transitions for sequential structure**
6. **Save all outputs** for reproducibility
7. **Document parameters** (bin sizes, etc.)
8. **Compare raw data to plots** for validation

---

## Troubleshooting

### Analysis Produces No Output
- Check that behaviors file loaded correctly
- Verify output directory is writable
- Look for error messages in log

### Strange Results
- Verify behavior classifications make sense (check CSV export)
- Check for very short recordings (may affect statistics)
- Ensure framerate is correct

### Missing Behaviors in Output
- Some behaviors may not occur in dataset
- Check raw data to confirm
- May need more data or longer recordings

---

## Next Steps

After running analyses:
1. Review CSV files to understand raw data
2. Examine plots for patterns
3. Import CSVs into statistical software if needed
4. Combine with other measurements (physiology, etc.)
5. Create publication figures from SVG outputs

For detailed usage instructions, see:
- [GUI Guide](GUI_GUIDE.md)
- [CLI Guide](CLI_GUIDE.md)
- [README](../README.md)
