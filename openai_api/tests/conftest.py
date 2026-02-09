"""Pytest configuration and shared fixtures."""

import sys
from pathlib import Path

# Add the parent directory to the path so tests can import the openai_api module
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
