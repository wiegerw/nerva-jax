# Copyright 2024 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import math
from typing import Sequence, Union

import numpy as np
import jax.numpy as jnp


# ------------------------
# Tensor conversion
# ------------------------

def to_tensor(array: Union[Sequence, np.ndarray]) -> jnp.ndarray:
    """
    Convert a Python list or NumPy array to a JAX array.
    - Float arrays become float32.
    - Integer arrays become int64.
    """
    if isinstance(array, np.ndarray):
        if np.issubdtype(array.dtype, np.integer):
            return jnp.array(array, dtype=jnp.int64)
        else:
            return jnp.array(array, dtype=jnp.float32)
    # Assume Python sequence
    arr = np.array(array)
    if np.issubdtype(arr.dtype, np.integer):
        return jnp.array(arr, dtype=jnp.int64)
    return jnp.array(arr, dtype=jnp.float32)


def to_long(array: Union[Sequence, np.ndarray]) -> jnp.ndarray:
    """Convert a Python list or NumPy array to int64 JAX array."""
    return jnp.array(array, dtype=jnp.int64)


# ------------------------
# Tensor comparison
# ------------------------

def equal_tensors(x: jnp.ndarray, y: jnp.ndarray) -> bool:
    """Check if two arrays are exactly equal."""
    return bool(jnp.array_equal(x, y))


def almost_equal(a: Union[float, int, jnp.ndarray],
                 b: Union[float, int, jnp.ndarray],
                 rel_tol: float = 1e-5,
                 abs_tol: float = 1e-8) -> bool:
    """
    Compare two numeric scalars approximately.
    Returns True if close within given relative and absolute tolerances.
    """
    if isinstance(a, jnp.ndarray):
        a = float(a)
    if isinstance(b, jnp.ndarray):
        b = float(b)
    return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)


def all_close(X1: jnp.ndarray, X2: jnp.ndarray,
              rtol: float = 1e-6, atol: float = 1e-6) -> bool:
    """Compare two arrays approximately. Returns True if all elements are close."""
    return bool(jnp.allclose(X1, X2, rtol=rtol, atol=atol))


def all_true(mask: jnp.ndarray) -> bool:
    """Return True if all elements of a boolean array are True."""
    return bool(jnp.all(mask))


def all_finite(x: jnp.ndarray) -> bool:
    """Return True if all elements of an array are finite."""
    return bool(jnp.isfinite(x).all())


def all_positive(X: jnp.ndarray) -> bool:
    """Return True if all entries of X are strictly positive."""
    return bool((X > 0).all())


# ------------------------
# Random tensors (NumPy fallback)
# ------------------------

def randn(*shape: int) -> np.ndarray:
    """Return a random normal array of given shape (NumPy, dtype float32)."""
    return np.random.randn(*shape).astype(np.float32)


def rand(*shape: int) -> np.ndarray:
    """Return a uniform random array in [0,1) of given shape (NumPy, dtype float32)."""
    return np.random.rand(*shape).astype(np.float32)


# ------------------------
# Test helpers
# ------------------------

def assert_tensors_are_close(name1: str, X1: jnp.ndarray,
                             name2: str, X2: jnp.ndarray,
                             rtol: float = 1e-6, atol: float = 1e-6):
    """
    Assert that two arrays are close, with helpful diagnostics.
    Raises AssertionError if not.
    """
    if not all_close(X1, X2, rtol=rtol, atol=atol):
        diff = jnp.abs(X1 - X2)
        max_diff = float(jnp.max(diff))
        raise AssertionError(
            f"Arrays {name1} and {name2} are not close. Max diff: {max_diff:.8f}"
        )

# ------------------------
# Test generation
# ------------------------

def random_float_matrix(shape, a, b):
    """
    Generates a random numpy array with the given shape and float values in the range [a, b].

    Parameters:
    shape (tuple): The shape of the numpy array to generate.
    a (float): The minimum value in the range.
    b (float): The maximum value in the range.

    Returns:
    np.ndarray: A numpy array of the specified shape with random float values in the range [a, b].
    """
    # Generate a random array with values in the range [0, 1)
    rand_array = np.random.rand(*shape)

    # Scale and shift the array to the range [a, b]
    scaled_array = a + (b - a) * rand_array

    return scaled_array


def make_target(Y: np.ndarray) -> np.ndarray:
    """
    Creates a boolean matrix T with the same shape as Y,
    where each row of T has exactly one value set to 1.

    Parameters:
    Y (np.ndarray): The input numpy array.

    Returns:
    np.ndarray: A boolean matrix with the same shape as Y,
                with exactly one True value per row.
    """
    if Y.ndim != 2:
        raise ValueError("The input array must be two-dimensional")

    # Get the shape of the input array
    rows, cols = Y.shape

    # Initialize an array of zeros with the same shape as Y
    T = np.zeros((rows, cols), dtype=bool)

    # Set one random element in each row to True
    for i in range(rows):
        random_index = np.random.randint(0, cols)
        T[i, random_index] = True

    return T
