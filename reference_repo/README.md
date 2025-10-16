# Reference Repository Information

This directory contains information and documentation about the original LUPE 2.0 reference repository that this project is based on.

## Purpose

This folder serves as a reference point during development to:
1. **Preserve original documentation** - Installation instructions and usage guides from the reference repository
2. **Ensure feature parity** - Verify that all functionality from the original notebooks has been implemented
3. **Maintain attribution** - Keep track of the source repository for proper citation and licensing

## Contents

- `github_repository.txt` - URL to the original LUPE 2.0 repository on GitHub
- `location.txt` - Local filesystem location of the cloned reference repository (if available)
- `Instructions for using reference repo in Windows.docx` - Setup and usage documentation from the original project

## Original Repository

**GitHub URL**: https://github.com/justin05423/LUPE-2.0-NotebookAnalysisPackage

**Project**: LUPE 2.0 - Light Automated Pain Evaluator
**Authors**: Corder Lab (University of Pennsylvania) & Yttri Lab (Carnegie Mellon University)

## What Was Changed

This LUPE Analysis Tool is a **complete rewrite** of the reference repository with the following major changes:

### Architecture Changes
- ❌ **Removed**: Jupyter notebooks (.ipynb files)
- ✅ **Added**: GUI application with tkinter
- ✅ **Added**: CLI for command-line automation
- ✅ **Added**: Modular Python package structure

### Functional Changes
- ❌ **Removed**: Group and condition comparisons
- ❌ **Removed**: Statistical tests between groups
- ✅ **Added**: LUPE-AMPS standalone GUI
- ✅ **Added**: Main launcher for easy access
- ✅ **Simplified**: Focus on individual file analysis with aggregate statistics

### Organization Changes
- ✅ **Added**: Comprehensive documentation (README, guides, tutorials)
- ✅ **Added**: JSON-based configuration (no hardcoded values)
- ✅ **Added**: Per-file output directories
- ✅ **Added**: Dependency management with exact versions

## Using the Reference Repository

If you need to compare functionality with the original notebooks:

1. Clone the reference repository:
   ```bash
   git clone https://github.com/justin05423/LUPE-2.0-NotebookAnalysisPackage
   ```

2. Follow the instructions in `Instructions for using reference repo in Windows.docx`

3. The original notebooks can be found in the repository root:
   - Main behavior classification notebooks
   - LUPE-AMPS pain scale analysis notebook
   - Example data and models

## Why This Rewrite?

The original LUPE 2.0 used Jupyter notebooks, which are excellent for interactive exploration but have limitations:
- Difficult to version control (JSON format)
- Hard to automate and integrate into pipelines
- Requires Jupyter environment setup

This rewrite addresses these limitations while preserving all analytical capabilities.

## Attribution

If you use this tool in your research, please cite the original LUPE 2.0 project:

```
LUPE 2.0: Light Automated Pain Evaluator
Corder Lab (University of Pennsylvania) & Yttri Lab (Carnegie Mellon University)
GitHub: https://github.com/justin05423/LUPE-2.0-NotebookAnalysisPackage
```

## Developer Note

**DO NOT MODIFY** files in this directory. This is reference material only.
All development should occur in the main project directories (`src/`, `docs/`, etc.).

---

**Last Updated**: 2024-10-16
