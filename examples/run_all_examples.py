#!/usr/bin/env python3

import subprocess
import sys
from pathlib import Path
import re

# === Determine paths ===
# Assume this script lives somewhere in the repo
SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent  # repo root = parent of "examples" folder
EXAMPLES_DIR = SCRIPT_PATH.parent  # examples folder

MNIST_DATA = REPO_ROOT / "data/mnist-flattened.npz"
CIFAR_DATA = REPO_ROOT / "data/cifar10-flattened.npz"

SUCCESS = []
FAIL = []
SKIP = []

def run_example(name, cmd):
    def is_tf_warning(line: str) -> bool:
        # Exclude known TensorFlow warnings
        return ("cuda_" in line or "computation_placer.cc" in line) and line.strip().startswith(("W", "E"))

    print("="*60)
    print(f"Running: {name}")
    print(f"Command: {' '.join(cmd)}")
    print("-"*60)
    try:
        process = subprocess.Popen(cmd, cwd=EXAMPLES_DIR,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,
                                   text=True)
        output_lines = []
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                if not is_tf_warning(line):
                    print(line, end="")  # only print non-warning lines
                output_lines.append(line)
        retcode = process.wait()
        full_output = "".join(output_lines)
        check_training_output(full_output, name)
        if retcode == 0:
            print(f"[OK] {name}")
            SUCCESS.append(name)
        else:
            print(f"[FAIL] {name} (exit {retcode})")
            FAIL.append((name, retcode))
    except Exception as e:
        print(f"[FAIL] {name}: {e}")
        FAIL.append((name, -1))

def skip_example(name, reason):
    print(f"[SKIP] {name}: {reason}")
    SKIP.append((name, reason))

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

def main():
    # Optional dataset checks
    data_files = {"train_cifar10_cli.sh": CIFAR_DATA,
                  "train_mnist_cli.sh": MNIST_DATA,
                  "custom_components.py": MNIST_DATA,
                  "mlp_construction.py": MNIST_DATA,
                  "train_mnist.py": MNIST_DATA}

    for f in EXAMPLES_DIR.iterdir():
        if f.name == SCRIPT_PATH.name:
            continue  # skip itself
        if f.suffix == ".sh":
            if f.name in data_files and not data_files[f.name].exists():
                skip_example(f.name, f"Dataset not found at {data_files[f.name]}")
            else:
                run_example(f.name, ["bash", str(f)])
        elif f.suffix == ".py":
            if f.name in data_files and not data_files[f.name].exists():
                skip_example(f.name, f"Dataset not found at {data_files[f.name]}")
            else:
                run_example(f.name, ["python3", str(f)])

    # Summary
    print("="*60)
    print("Summary:")
    print(f"  Success: {len(SUCCESS)}")
    for s in SUCCESS:
        print(f"    - {s}")
    print(f"  Failed: {len(FAIL)}")
    for f_name, code in FAIL:
        print(f"    - {f_name} (exit {code})")
    print(f"  Skipped: {len(SKIP)}")
    for s_name, reason in SKIP:
        print(f"    - {s_name}: {reason}")

    if FAIL:
        sys.exit(1)

if __name__ == "__main__":
    main()

