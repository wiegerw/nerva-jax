#!/usr/bin/env bash

# Copyright 2025 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

# Run all examples in this directory and report a summary of results.
# - Sets PYTHONPATH so examples can import the src package without installation
# - Skips examples with missing prerequisites (e.g., datasets or optional packages)
# - Reports successes, failures, and skipped items with exit codes

set -u

# Resolve paths
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

MNIST_DATA="${REPO_ROOT}/data/mnist-flattened.npz"
CIFAR_DATA="${REPO_ROOT}/data/cifar10-flattened.npz"

# Collect results
SUCCESS_NAMES=()
FAIL_NAMES=()
FAIL_CODES=()
SKIP_NAMES=()
SKIP_REASONS=()

run_example() {
  local name="$1"
  shift
  local cmd=("$@")

  echo "============================================================"
  echo "Running: ${name}"
  echo "Command: ${cmd[*]}"
  echo "------------------------------------------------------------"

  # Run the command; capture output and status
  # Run inside examples dir to preserve relative paths inside examples
  pushd "${SCRIPT_DIR}" > /dev/null || { echo "Failed to enter examples dir"; exit 2; }
  "${cmd[@]}"
  local status=$?
  popd > /dev/null || true

  if [[ $status -eq 0 ]]; then
    echo "[OK] ${name}"
    SUCCESS_NAMES+=("${name}")
  else
    echo "[FAIL] ${name} (exit ${status})"
    FAIL_NAMES+=("${name}")
    FAIL_CODES+=("${status}")
  fi
}

skip_example() {
  local name="$1"; shift
  local reason="$1"; shift || true
  echo "[SKIP] ${name}: ${reason}"
  SKIP_NAMES+=("${name}")
  SKIP_REASONS+=("${reason}")
}

# Check optional dependency for SymPy validation example
has_nerva_sympy=0
python3 - <<'PY'
try:
    import nerva_sympy  # type: ignore
    import sys
    sys.exit(0)
except Exception:
    import sys
    sys.exit(1)
PY
if [[ $? -eq 0 ]]; then
  has_nerva_sympy=1
fi

# Examples list
# 1) Shell-driven examples
if [[ -f "${CIFAR_DATA}" ]]; then
  run_example "train_cifar10_cli.sh" bash ./train_cifar10_cli.sh
else
  skip_example "train_cifar10_cli.sh" "Dataset not found at ${CIFAR_DATA}"
fi

if [[ -f "${MNIST_DATA}" ]]; then
  run_example "train_mnist_cli.sh" bash ./train_mnist_cli.sh
else
  skip_example "train_mnist_cli.sh" "Dataset not found at ${MNIST_DATA}"
fi

# 2) Python examples that require MNIST data
if [[ -f "${MNIST_DATA}" ]]; then
  run_example "custom_components.py" python3 ./custom_components.py
  run_example "mlp_construction.py" python3 ./mlp_construction.py
  run_example "train_mnist.py" python3 ./train_mnist.py
else
  skip_example "custom_components.py" "Dataset not found at ${MNIST_DATA}"
  skip_example "mlp_construction.py" "Dataset not found at ${MNIST_DATA}"
  skip_example "train_mnist.py" "Dataset not found at ${MNIST_DATA}"
fi

# 3) Python example with synthetic data
run_example "train_synthetic.py" python3 ./train_synthetic.py

# 4) SymPy-based validation (optional dependency)
if [[ ${has_nerva_sympy} -eq 1 ]]; then
  run_example "custom_components_validation.py" python3 ./custom_components_validation.py
else
  skip_example "custom_components_validation.py" "Optional dependency 'nerva-sympy' not installed"
fi

# Summary
echo "============================================================"
echo "Summary:"
printf '  Success: %d\n' "${#SUCCESS_NAMES[@]}"
for n in "${SUCCESS_NAMES[@]}"; do
  printf '    - %s\n' "$n"
done
printf '  Failed:  %d\n' "${#FAIL_NAMES[@]}"
for i in "${!FAIL_NAMES[@]}"; do
  printf '    - %s (exit %s)\n' "${FAIL_NAMES[$i]}" "${FAIL_CODES[$i]}"
done
printf '  Skipped: %d\n' "${#SKIP_NAMES[@]}"
for i in "${!SKIP_NAMES[@]}"; do
  printf '    - %s: %s\n' "${SKIP_NAMES[$i]}" "${SKIP_REASONS[$i]}"
done

# Exit with non-zero if any failures occurred
if [[ ${#FAIL_NAMES[@]} -gt 0 ]]; then
  exit 1
fi
exit 0
