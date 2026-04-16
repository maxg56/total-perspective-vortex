#!/usr/bin/env python3
"""
Root-level wrapper — delegates to src/mybci.py.

Allows running from the project root:
    uv run python mybci.py <subject> <run> <mode> [options]

When imported as a module (e.g. during tests), transparently replaces
itself in sys.modules with src/mybci.py so all symbols are available.
"""
import os
import sys
import importlib.util

_SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')

if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# Load src/mybci.py and register it as 'mybci' in sys.modules so that
# `from mybci import parse_args` (used in tests) resolves to src/mybci.py.
_spec = importlib.util.spec_from_file_location(
    'mybci',
    os.path.join(_SRC_DIR, 'mybci.py'),
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules['mybci'] = _mod  # pre-register before exec to prevent circular import
_spec.loader.exec_module(_mod)

if __name__ == '__main__':
    os.chdir(_SRC_DIR)
    sys.exit(_mod.main())
