# Copyright 2022 - 2025 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import json
import os
import unittest
from pathlib import Path
from typing import Dict, Any

from nerva_jax.utilities import load_dict_from_npz

import jax.numpy as jnp

from nerva_jax.layers import (
    LinearLayer,
    ActivationLayer,
    SoftmaxLayer,
    LogSoftmaxLayer,
    BatchNormalizationLayer,
    SReLULayer,
)
from nerva_jax.activation_functions import (
    SReLUActivation,
    parse_activation,
)

ESSENTIAL_ATOL = 1e-6
ESSENTIAL_RTOL = 1e-6

# Optional debug output controlled via ONE_LAYER_DEBUG environment variable.
# Values: "0" or unset = silent (default), "1" = basic, "2" = verbose tensors.
DEBUG_LEVEL = int(os.environ.get("ONE_LAYER_DEBUG", "0") or 0)

def _print_debug(msg: str):
    if DEBUG_LEVEL > 0:
        print(msg)


def assert_close(name: str, a: jnp.ndarray, b: jnp.ndarray, atol=ESSENTIAL_ATOL, rtol=ESSENTIAL_RTOL):
    if a.shape != b.shape:
        _print_debug(f"ASSERT {name}: shape mismatch: {a.shape} vs {b.shape}")
        raise AssertionError(f"Shape mismatch for {name}: {a.shape} vs {b.shape}")
    close = bool(jnp.allclose(a, b, atol=atol, rtol=rtol))
    if not close or DEBUG_LEVEL > 0:
        max_diff = float(jnp.max(jnp.abs(a - b)))
        _print_debug(f"ASSERT {name}: shape={tuple(a.shape)}, atol={atol}, rtol={rtol}, max|diff|={max_diff}")
        if DEBUG_LEVEL >= 2:
            # Print tensors if they are reasonably small
            def tensor_str(t: jnp.ndarray):
                if t.size <= 200:
                    return str(jnp.array(t))
                return f"tensor(shape={tuple(t.shape)}, numel={t.size})"
            _print_debug(f"  A: {tensor_str(a)}")
            _print_debug(f"  B: {tensor_str(b)}")
    if not close:
        raise AssertionError(f"Mismatch in {name}: max|diff|={max_diff}")



def run_case(manifest_dir: Path, meta: Dict[str, Any]):
    tensors = load_dict_from_npz(str(manifest_dir / meta["file"]))
    # Convert numpy arrays to JAX arrays for computations
    tensors = {k: jnp.array(v) for k, v in tensors.items()}
    X = tensors["X"]

    _print_debug(f"CASE: type={meta.get('type')} file={meta.get('file')} D={meta.get('D')} K={meta.get('K')} activation={meta.get('activation')}")

    if meta["type"] == "Linear":
        D = meta["D"]; K = meta["K"]
        layer = LinearLayer(D, K)
        layer.W = tensors["W"]
        layer.b = tensors["b"]
        Y = layer.feedforward(X)
        assert_close("Y", Y, tensors["Y"])
        DY = tensors["DY"]
        layer.backpropagate(Y, DY)
        assert_close("DX", layer.DX, tensors["DX"])
        assert_close("DW", layer.DW, tensors["DW"])
        assert_close("Db", layer.Db, tensors["Db"])
        # optimize verification
        if "optimizer_spec" in meta:
            layer.set_optimizer(meta["optimizer_spec"])
            layer.optimize(meta.get("lr", 0.1))
            assert_close("W_opt", layer.W, tensors["W_opt"])
            assert_close("b_opt", layer.b, tensors["b_opt"])
        return

    if meta["type"] == "Activation":
        D = meta["D"]; K = meta["K"]
        act = parse_activation(meta["activation_spec"])
        layer = ActivationLayer(D, K, act)
        layer.W = tensors["W"]
        layer.b = tensors["b"]
        Y = layer.feedforward(X)
        assert_close("Y", Y, tensors["Y"])
        DY = tensors["DY"]
        layer.backpropagate(Y, DY)
        assert_close("DX", layer.DX, tensors["DX"])
        assert_close("DW", layer.DW, tensors["DW"])
        assert_close("Db", layer.Db, tensors["Db"])
        # optimize verification
        if "optimizer_spec" in meta:
            layer.set_optimizer(meta["optimizer_spec"])
            layer.optimize(meta.get("lr", 0.1))
            assert_close("W_opt", layer.W, tensors["W_opt"])
            assert_close("b_opt", layer.b, tensors["b_opt"])
        return

    if meta["type"] == "SReLU":
        D = meta["D"]; K = meta["K"]
        act = parse_activation(meta["activation_spec"])
        layer = SReLULayer(D, K, act)
        layer.W = tensors["W"]
        layer.b = tensors["b"]
        act.x = tensors["act_x"]
        Y = layer.feedforward(X)
        assert_close("Y", Y, tensors["Y"])
        DY = tensors["DY"]
        layer.backpropagate(Y, DY)
        assert_close("DX", layer.DX, tensors["DX"])
        assert_close("DW", layer.DW, tensors["DW"])
        assert_close("Db", layer.Db, tensors["Db"])
        assert_close("act.Dx", act.Dx, tensors["act_Dx"])
        # optimize verification
        if "optimizer_spec" in meta:
            layer.set_optimizer(meta["optimizer_spec"])
            layer.optimize(meta.get("lr", 0.1))
            assert_close("W_opt", layer.W, tensors["W_opt"])
            assert_close("b_opt", layer.b, tensors["b_opt"])
            assert_close("act_x_opt", act.x, tensors["act_x_opt"])
        return

    if meta["type"] == "Softmax":
        D = meta["D"]; K = meta["K"]
        layer = SoftmaxLayer(D, K)
        layer.W = tensors["W"]
        layer.b = tensors["b"]
        Y = layer.feedforward(X)
        assert_close("Y", Y, tensors["Y"])
        DY = tensors["DY"]
        layer.backpropagate(Y, DY)
        assert_close("DX", layer.DX, tensors["DX"])
        assert_close("DW", layer.DW, tensors["DW"])
        assert_close("Db", layer.Db, tensors["Db"])
        if "optimizer_spec" in meta:
            layer.set_optimizer(meta["optimizer_spec"])
            layer.optimize(meta.get("lr", 0.1))
            assert_close("W_opt", layer.W, tensors["W_opt"])
            assert_close("b_opt", layer.b, tensors["b_opt"])
        return

    if meta["type"] == "LogSoftmax":
        D = meta["D"]; K = meta["K"]
        layer = LogSoftmaxLayer(D, K)
        layer.W = tensors["W"]
        layer.b = tensors["b"]
        Y = layer.feedforward(X)
        assert_close("Y", Y, tensors["Y"])
        DY = tensors["DY"]
        layer.backpropagate(Y, DY)
        assert_close("DX", layer.DX, tensors["DX"])
        assert_close("DW", layer.DW, tensors["DW"])
        assert_close("Db", layer.Db, tensors["Db"])
        if "optimizer_spec" in meta:
            layer.set_optimizer(meta["optimizer_spec"])
            layer.optimize(meta.get("lr", 0.1))
            assert_close("W_opt", layer.W, tensors["W_opt"])
            assert_close("b_opt", layer.b, tensors["b_opt"])
        return

    if meta["type"] == "BatchNormalization":
        D = meta["D"]
        layer = BatchNormalizationLayer(D)
        layer.gamma = tensors["gamma"]
        layer.beta = tensors["beta"]
        Y = layer.feedforward(X)
        assert_close("Y", Y, tensors["Y"])  # BN determinism within our math
        DY = tensors["DY"]
        layer.backpropagate(Y, DY)
        assert_close("DX", layer.DX, tensors["DX"]) 
        assert_close("Dgamma", layer.Dgamma, tensors["Dgamma"]) 
        assert_close("Dbeta", layer.Dbeta, tensors["Dbeta"]) 
        if "optimizer_spec" in meta:
            layer.set_optimizer(meta["optimizer_spec"])
            layer.optimize(meta.get("lr", 0.1))
            assert_close("gamma_opt", layer.gamma, tensors["gamma_opt"]) 
            assert_close("beta_opt", layer.beta, tensors["beta_opt"]) 
        return

    raise ValueError(f"Unknown case type: {meta['type']}")


class TestOneLayer(unittest.TestCase):
    def test_cases(self):
        # We set the constant for numerical stability to 0, in order to be consistent with nerva-sympy
        import nerva_jax
        nerva_jax.matrix_operations.epsilon = 0

        # Read cases from the default directory under tests/one_layer_cases
        out_dir = Path(__file__).parent / "one_layer_cases"
        manifest_path = out_dir / "manifest.json"
        if not manifest_path.exists():
            self.skipTest(f"No one_layer_cases manifest found in '{manifest_path}'")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(manifest), 1, "No cases found in manifest")

        for meta in manifest:
            with self.subTest(name=meta.get("name", meta.get("file", "unknown"))):
                run_case(out_dir, meta)


if __name__ == '__main__':
    unittest.main()
