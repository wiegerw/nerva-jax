#!/bin/bash
set -euo pipefail

# Resolve repository root and script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Output directory at repo root
OUTPUT_DIRECTORY="${REPO_ROOT}/output_directory"
mkdir -p "${OUTPUT_DIRECTORY}"

# Pre-clean only our outputs
rm -f "${OUTPUT_DIRECTORY}/nerva-jax.html"

# Common Asciidoctor options
COMMON_OPTS=(-r asciidoctor-bibtex -a "source-highlighter=rouge" -D "${OUTPUT_DIRECTORY}")

# Build Python manual (AsciiDoc → single HTML file in root)
pushd "${SCRIPT_DIR}/asciidoc" >/dev/null
asciidoctor "${COMMON_OPTS[@]}" -a "bibtex-file=../latex/nerva.bib" nerva-jax.adoc
echo "Generated: ${OUTPUT_DIRECTORY}/nerva-jax.html"
popd >/dev/null

# Copy landing page into output_directory
cp "${REPO_ROOT}/doc/index.html" "${OUTPUT_DIRECTORY}/"
