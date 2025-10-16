#!/usr/bin/env python3

# Copyright 2025 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

"""
Run all Python tests in this folder.

- Uses pytest if available (preferred for verbose, clear, colorized output)
- Falls back to Python's unittest discovery if pytest is not installed
- Adds ../src to PYTHONPATH so tests import the local package
- Skips tests marked with @pytest.mark.slow by default; include them with --include-long
- Can show the slowest N tests via --duration when using pytest
- Forwards any extra arguments to the underlying test runner
- Default pytest flags can be overridden via the environment variable NERVA_PYTEST_FLAGS
"""

import os
import sys
from pathlib import Path
import subprocess
import argparse

def ensure_src_importable(repo_root: Path):
    """Make the src folder importable in this Python process and subprocesses."""
    src_dir = repo_root / "src"
    src_str = str(src_dir.resolve())

    if src_str not in sys.path:
        sys.path.insert(0, src_str)

    sep = os.pathsep
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    paths = current_pythonpath.split(sep) if current_pythonpath else []
    if src_str not in paths:
        new_pythonpath = sep.join([src_str] + paths) if paths else src_str
        os.environ["PYTHONPATH"] = new_pythonpath

def main():
    SCRIPT_DIR = Path(__file__).resolve().parent
    REPO_ROOT = SCRIPT_DIR.parent
    ensure_src_importable(REPO_ROOT)
    TEST_DIR = SCRIPT_DIR

    # --- Argument parsing ---
    parser = argparse.ArgumentParser(description="Run all Python tests in this folder")
    parser.add_argument("--include-long", action="store_true",
                        help="Include tests marked @pytest.mark.slow")
    parser.add_argument("--duration", type=int, default=None,
                        help="Show the slowest N tests (pytest only)")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER,
                        help="Additional arguments to pass to the test runner (pytest or unittest)")
    args = parser.parse_args()

    include_long = args.include_long
    duration_flag = args.duration
    extra_args = args.extra_args

    # --- Determine marker/filter ---
    marker_args = []
    if not include_long:
        marker_args = ["-m", "not slow"]

    # --- Determine pytest flags ---
    PYTEST_FLAGS_DEFAULT = ["-v", "-ra", "--disable-warnings"]
    pytest_flags_env = os.environ.get("NERVA_PYTEST_FLAGS")
    PYTEST_FLAGS = pytest_flags_env.split() if pytest_flags_env else PYTEST_FLAGS_DEFAULT

    # --- Optional durations ---
    duration_args = []
    if duration_flag:
        duration_args = [f"--durations={duration_flag}"]

    # --- Build test command ---
    try:
        import pytest
        cmd = [sys.executable, "-m", "pytest", *PYTEST_FLAGS, *marker_args, str(TEST_DIR), *duration_args, *extra_args]
        print("Detected pytest. Running tests with pytest...")
    except ImportError:
        print("pytest not found. Falling back to unittest discovery...")
        cmd = [sys.executable, "-m", "unittest", "discover", "-b", "-s", str(TEST_DIR), "-p", "test_*.py", "-v", *extra_args]

    # --- Run the command ---
    ret = subprocess.run(cmd, check=False)
    sys.exit(ret.returncode)

if __name__ == "__main__":
    main()

