#!/usr/bin/env python3
"""
Root-level wrapper — delegates to src/mybci.py.

Allows running from the project root:
    uv run python mybci.py <subject> <run> <mode> [options]
"""
import os
import sys

_SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.insert(0, _SRC_DIR)
os.chdir(_SRC_DIR)

from mybci import main  # noqa: E402

if __name__ == '__main__':
    sys.exit(main())
