# Changelog

All notable changes to the LUPE Analysis Tool will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-10-16

### Added
- **Main Launcher GUI**: Unified entry point with buttons to launch LUPE Classification or LUPE-AMPS analysis
- **LUPE-AMPS Pain Scale Analysis**: Complete implementation of Advanced Multivariate Pain Scale
  - PCA-based pain scale projection (PC1/PC2)
  - 4 analysis sections: Preprocessing, Pain Scale Projection, Metrics Visualization, Feature Importance
  - Standalone GUI (`main_lupe_amps_gui.py`)
  - Comprehensive user guide (`docs/LUPE_AMPS_GUIDE.md`)
- **LUPE Classification GUI**: Behavior classification and standard metrics analysis
  - Renamed from `main_gui.py` to `main_lupe_gui.py` for clarity
  - Support for raw DeepLabCut CSV files
  - Pre-classified behavior analysis
  - Per-file output organization
- **CLI Interface**: Command-line access for automation
  - Preprocessing, classification, and analysis commands
  - Export functionality for CSV generation
- **Comprehensive Documentation**:
  - `README.md` - Project overview and installation
  - `GETTING_STARTED.md` - Step-by-step tutorial
  - `docs/GUI_GUIDE.md` - GUI usage instructions
  - `docs/CLI_GUIDE.md` - CLI reference
  - `docs/ANALYSIS_TYPES.md` - Detailed analysis descriptions
  - `docs/LUPE_AMPS_GUIDE.md` - LUPE-AMPS specific guide
- **Configuration System**:
  - `config/metadata.json` - Behavior definitions and parameters
  - `config/settings.json` - User preferences
  - No hardcoded values throughout codebase
- **Dependency Management**:
  - Pinned library versions for model compatibility
  - scikit-learn 1.2.1 (required for A-SOiD model)
  - Helper scripts for dependency reinstallation

### Changed
- **Project Structure**: Converted from Jupyter notebook-based to modular Python application
- **Analysis Philosophy**: Removed group/condition comparisons (per project requirements)
  - Focus on individual file analysis
  - Aggregate statistics (mean ± SEM) for multiple files
  - No statistical comparisons between groups
- **Output Organization**: Per-file directories with consistent naming
  - Format: `{filename}_LUPE_output/`
  - All outputs in dedicated subdirectories

### Fixed
- **Model Compatibility**: Resolved scikit-learn 1.3.x incompatibility issues
  - Documented exact version requirements
  - Created reinstall scripts for clean environment setup
- **Feature Extraction**: Removed inefficient vectorization attempts
  - Maintained original loop-based extraction
  - Better memory management
- **GUI Layout**: Fixed LUPE-AMPS GUI scrolling issues
  - Added scrollable canvas for left column
  - All controls now accessible

### Technical Details
- Python 3.11+ required
- scikit-learn 1.2.1 (CRITICAL - do not upgrade)
- numpy 1.26.4, pandas 2.2.2, scipy 1.13.0
- Pre-trained models: model_LUPE.pkl (308MB), model_AMPS.pkl (1KB)
- No Jupyter notebooks required

### Known Issues
- Large model files (308MB) require Git LFS for efficient version control
- Windows-specific path handling in some modules

### Migration from Reference Repository
This version is a complete rewrite of the LUPE 2.0 Notebook Analysis Package with:
- GUI and CLI interfaces replacing Jupyter notebooks
- Modular architecture for better maintainability
- Simplified workflow (no group/condition complexity)
- Enhanced documentation and user guidance

### Reference
Original repository: https://github.com/justin05423/LUPE-2.0-NotebookAnalysisPackage

---

## [Unreleased]

### Planned Features
- Batch processing mode for large datasets
- Additional export formats (JSON, HDF5)
- Integration tests and CI/CD pipeline
- Performance optimizations for large files

---

**Note**: This is the initial public release. Previous development history is not included as this represents a significant architectural change from the reference repository.
