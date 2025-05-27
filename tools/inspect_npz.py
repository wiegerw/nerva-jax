#!/usr/bin/env python3

# Copyright 2023 - 2025 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

"""
inspect_npz.py

A utility script to inspect the contents of a .npz file storing a dictionary of NumPy arrays.
Displays the name, shape, infinity norm, and optionally the array contents.

This is useful for checking datasets or model weights saved by mlptool or similar tools.

Usage:
    python inspect_npz.py file.npz [--shapes-only]
"""

import argparse
import numpy as np


def inf_norm(x: np.ndarray) -> float:
    """
    Compute the infinity norm (maximum absolute value) of a NumPy array.
    """
    return np.abs(x).max()


def format_shape(shape: tuple[int]) -> str:
    """
    Format a shape tuple like (50000, 3072) into a readable string "50000x3072".
    """
    return "x".join(str(dim) for dim in shape)


def print_array_summary(name: str, x: np.ndarray, shapes_only: bool = False):
    """
    Print a summary of a NumPy array, including name, shape, and inf-norm.
    If shapes_only is False, print the full array contents as well.

    Args:
        name: Name of the array.
        x: NumPy array.
        shapes_only: If True, only show metadata.
    """
    shape_str = format_shape(x.shape)
    norm = inf_norm(x)
    label = f"{name:<8} ({shape_str:<12})  inf-norm = {norm:.8f}"
    print(label)
    if not shapes_only:
        print(x)
        print()


def load_npz_dict(filename: str) -> dict[str, np.ndarray]:
    """
    Load a dictionary of arrays from a .npz file.

    Args:
        filename: Path to a .npz file.

    Returns:
        A dictionary mapping keys to NumPy arrays.
    """
    return dict(np.load(filename, allow_pickle=False))


def inspect_npz_file(filename: str, shapes_only: bool):
    """
    Load and display the contents of a .npz file.

    Args:
        filename: Path to the .npz file to inspect.
        shapes_only: Whether to suppress array contents.
    """
    data = load_npz_dict(filename)
    for key, array in data.items():
        print_array_summary(key, array, shapes_only=shapes_only)


def main():
    parser = argparse.ArgumentParser(description="Inspect the contents of a .npz file storing NumPy arrays.")
    parser.add_argument("filename", metavar="FILE", type=str, help="Path to the .npz file")
    parser.add_argument("--shapes-only", action="store_true", help="Print only array shapes and norms, suppress values")
    args = parser.parse_args()

    inspect_npz_file(args.filename, shapes_only=args.shapes_only)


if __name__ == "__main__":
    main()
