#!/bin/bash
set -euo pipefail

# Resolve repository root and script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Output directory at repo root
OUTPUT_DIRECTORY="${REPO_ROOT}/output_directory"
mkdir -p "${OUTPUT_DIRECTORY}"

# Ensure Sphinx is installed (local builds only)
python3 -m pip install --quiet --upgrade sphinx sphinx-rtd-theme

# Build into subfolder /sphinx (keeps site root clean)
# Extra options from the environment, e.g. SPHINX_EXTRA_OPTS="-W --keep-going" (used in CI)
sphinx-build ${SPHINX_EXTRA_OPTS:-} -b html "${REPO_ROOT}/doc/sphinx" "${OUTPUT_DIRECTORY}/sphinx"

echo "Generated: ${OUTPUT_DIRECTORY}/sphinx/index.html"

# Copy landing page into output_directory
cp "${REPO_ROOT}/doc/index.html" "${OUTPUT_DIRECTORY}/"
