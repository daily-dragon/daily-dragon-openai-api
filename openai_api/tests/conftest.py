"""Pytest configuration and shared fixtures."""

import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so tests can import the openai_api module
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Prevent OpenAI() from being instantiated with a real API key at module import time
os.environnsetdefault("OPENAI_API_KEY", "sk-test-dummy-key-for-testing")
