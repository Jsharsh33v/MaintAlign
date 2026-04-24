"""
pytest configuration — adds the project root to sys.path so tests can
import from core/, utils/, analysis/ without needing an installed package.
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
