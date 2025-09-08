#!/usr/bin/env python3

# Copyright 2025 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

# Run all Python tests in this folder.
# - Uses pytest if available (preferred for verbose, clear output)
# - Falls back to Python's unittest discovery if pytest is not installed
# - Adds ../src to PYTHONPATH so tests import the local package
# - Forwards any extra arguments to the underlying test runner
# - Default pytest flags can be overridden via the environment variable NERVA_PYTEST_FLAGS

import os
import sys
from pathlib import Path
import subprocess


def ensure_src_importable(repo_root: Path):
    """Make the src folder importable in this Python process and subprocesses."""
    src_dir = repo_root / "src"
    src_str = str(src_dir.resolve())

    # Add to sys.path for this process
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

    # Prepend to PYTHONPATH for subprocesses
    sep = os.pathsep  # ':' on Unix, ';' on Windows
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    paths = current_pythonpath.split(sep) if current_pythonpath else []
    if src_str not in paths:
        new_pythonpath = sep.join([src_str] + paths) if paths else src_str
        os.environ["PYTHONPATH"] = new_pythonpath


def main():
    # Resolve repo root
    SCRIPT_DIR = Path(__file__).resolve().parent
    REPO_ROOT = SCRIPT_DIR.parent

    # Make src importable
    ensure_src_importable(REPO_ROOT)

    # Determine test directory
    TEST_DIR = SCRIPT_DIR

    # Use NERVA_PYTEST_FLAGS if set, else default flags
    PYTEST_FLAGS_DEFAULT = ["-v", "-ra", "--disable-warnings"]
    pytest_flags_env = os.environ.get("NERVA_PYTEST_FLAGS")
    PYTEST_FLAGS = pytest_flags_env.split() if pytest_flags_env else PYTEST_FLAGS_DEFAULT

    # Use pytest if available
    try:
        import pytest  # check if pytest is installed
        print("Detected pytest. Running tests with pytest...")
        cmd = [sys.executable, "-m", "pytest", *PYTEST_FLAGS, str(TEST_DIR)] + sys.argv[1:]
    except ImportError:
        print("pytest not found. Falling back to unittest discovery...")
        cmd = [
            sys.executable, "-m", "unittest", "discover",
            "-b", "-s", str(TEST_DIR), "-p", "test_*.py", "-v"
        ] + sys.argv[1:]

    # Run the chosen test command
    ret = subprocess.run(cmd, check=False)
    sys.exit(ret.returncode)


if __name__ == "__main__":
    main()

