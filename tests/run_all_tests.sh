#!/usr/bin/env bash

# Copyright 2025 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

# Run all tests in this directory.
# - Sets PYTHONPATH so examples can import the src package without installation

set -u

# Resolve paths
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# Prefer our enhanced Python runner for clearer, colored output and timings
python3 "${SCRIPT_DIR}/run_tests.py" "$@"
