#!/usr/bin/env bash

# Copyright 2025 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

# Run all tests in this folder.
# - Uses pytest if available (preferred for clearer, verbose output)
# - Falls back to Python's unittest discovery otherwise
# - Sets PYTHONPATH so tests import the local package from ../src
# - Forwards any extra arguments to the underlying test runner
# - Default pytest flags can be overridden with NERVA_PYTEST_FLAGS

set -euo pipefail

# Resolve repository root (parent of this tests directory)
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Ensure src/ is on PYTHONPATH so tests import the local package
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# The directory containing the tests is this directory
TEST_DIR="${SCRIPT_DIR}"

# Default pytest flags: verbose test names, summary of fails, suppress warnings
PYTEST_FLAGS_DEFAULT="-v -ra --disable-warnings"
PYTEST_FLAGS="${NERVA_PYTEST_FLAGS:-$PYTEST_FLAGS_DEFAULT}"

if command -v pytest >/dev/null 2>&1; then
  echo "Detected pytest. Running tests with pytest..." >&2
  exec pytest ${PYTEST_FLAGS} "${TEST_DIR}" "$@"
else
  echo "pytest not found. Falling back to unittest discovery..." >&2
  # -b buffers stdout/stderr from tests (only shown on failure), -v for verbose test names
  exec python3 -m unittest discover -b -s "${TEST_DIR}" -p "test_*.py" -v "$@"
fi
