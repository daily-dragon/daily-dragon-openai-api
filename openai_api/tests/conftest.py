"""Pytest configuration and shared fixtures."""

import sys
import os
from pathlib import Path

# Add the parent directory to the path so tests can import the openai_api module
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Prevent OpenAI() from failing at module import time due to missing API key in CI
os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy-key-for-testing")
