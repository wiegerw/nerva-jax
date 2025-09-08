#!/usr/bin/env python3

# Copyright 2025 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

# Run all example scripts in this folder.
# - Executes all .py and .sh files except this runner script itself
# - Adds ../src to PYTHONPATH so Python examples can import the local package
# - Checks that required datasets (MNIST and CIFAR10) exist before running
# - Captures output and prints warnings if final loss did not decrease or test accuracy did not increase significantly
# - Provides a summary of successful, failed, and skipped examples

import os
import sys
import re
from pathlib import Path
import subprocess

# --------------------------
# Paths
# --------------------------
SCRIPT_PATH = Path(__file__).resolve()
EXAMPLES_DIR = SCRIPT_PATH.parent
REPO_ROOT = SCRIPT_PATH.parent.parent
SRC_DIR = REPO_ROOT / "src"

# Add src to PYTHONPATH for current process and subprocesses
os.environ["PYTHONPATH"] = f"{SRC_DIR}:{os.environ.get('PYTHONPATH','')}"
sys.path.insert(0, str(SRC_DIR.resolve()))

# Dataset files
MNIST_DATA = REPO_ROOT / "data/mnist-flattened.npz"
CIFAR_DATA = REPO_ROOT / "data/cifar10-flattened.npz"

# --------------------------
# Output checker
# --------------------------
def check_training_output(output, name):
    runs = output.split("Total training time for the")
    for i, run in enumerate(runs[:-1]):  # last split may be empty
        loss_pattern = re.compile(r"loss:\s*([\d\.]+)")
        acc_pattern = re.compile(r"test accuracy:\s*([\d\.]+)")
        losses = [float(m) for m in loss_pattern.findall(run)]
        accs = [float(m) for m in acc_pattern.findall(run)]
        if losses and accs:
            if losses[-1] > losses[0]:
                print(f"[WARN] {name} (run {i+1}): final loss did not decrease ({losses[0]} -> {losses[-1]})")
            if accs[-1] <= accs[0] + 0.05:
                print(f"[WARN] {name} (run {i+1}): test accuracy did not increase significantly ({accs[0]} -> {accs[-1]})")

# --------------------------
# Simple runner
# --------------------------
SUCCESS = []
FAIL = []
SKIP = []

data_required = {MNIST_DATA, CIFAR_DATA}

if not all(f.exists() for f in data_required):
    for f in data_required:
        if not f.exists():
            print(f"[SKIP] Dataset not found: {f}")
    sys.exit(1)

for f in EXAMPLES_DIR.iterdir():
    if f == SCRIPT_PATH:
        continue
    if f.suffix == ".py":
        cmd = [sys.executable, str(f)]
    elif f.suffix == ".sh":
        cmd = ["bash", str(f)]
    else:
        continue

    print(f"Running {f.name}...")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    output_lines = []
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            print(line, end="")
            output_lines.append(line)
    retcode = process.wait()
    full_output = "".join(output_lines)
    check_training_output(full_output, f.name)

    if retcode == 0:
        SUCCESS.append(f.name)
    else:
        FAIL.append(f.name)

# --------------------------
# Summary
# --------------------------
print("="*60)
print("Summary:")
print(f"  Success: {len(SUCCESS)}")
for s in SUCCESS:
    print(f"    - {s}")
print(f"  Failed: {len(FAIL)}")
for f_name in FAIL:
    print(f"    - {f_name}")

if FAIL:
    sys.exit(1)

