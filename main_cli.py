"""
LUPE Analysis Tool - CLI Entry Point

This script provides command-line access to LUPE analysis functions.

Usage:
    python main_cli.py --help
    python main_cli.py classify --model model.pkl --input data/ --output behaviors.pkl
    python main_cli.py analyze --behaviors behaviors.pkl --output results/ --all

For detailed help on any command:
    python main_cli.py <command> --help
"""

# CRITICAL: Set matplotlib backend BEFORE any other imports
# This prevents threading issues when running analysis in parallel
import matplotlib
matplotlib.use('Agg')

import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.cli.main import main


if __name__ == '__main__':
    main()
