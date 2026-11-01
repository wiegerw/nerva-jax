# One-Layer Forward/Backward Test Cases

This directory contains reference test cases for single layers (feedforward, backpropagation and an
optimizer step), used by `tests/test_one_layer.py`. The same cases are used by all Nerva Python
backends, so passing them shows that the backends produce the same results.

The cases are generated in the [nerva-sympy](https://github.com/wiegerw/nerva-sympy) repository with

```bash
python scripts/generate_one_layer_tests.py
```

which also copies them to the `tests/one_layer_cases` directories of sibling checkouts of the backends.
Do not edit the files here; regenerate them in nerva-sympy instead.

To run the test:

```bash
python -m pytest tests/test_one_layer.py
```

Set `ONE_LAYER_DEBUG=1` or `ONE_LAYER_DEBUG=2` for per-case debug output.
