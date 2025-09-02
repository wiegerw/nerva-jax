nerva_jax documentation
=========================

A tiny, educational set of neural network components built on JAX.

Install and build
-----------------

.. code-block:: bash

    # from repository root
    python -m pip install -U sphinx sphinx-rtd-theme
    # build HTML docs into docs_sphinx/_build/html
    sphinx-build -b html docs_sphinx docs_sphinx/_build/html

API reference
-------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   nerva_jax
   nerva_jax.activation_functions
   nerva_jax.datasets
   nerva_jax.layers
   nerva_jax.learning_rate
   nerva_jax.loss_functions
   nerva_jax.matrix_operations
   nerva_jax.multilayer_perceptron
   nerva_jax.optimizers
   nerva_jax.softmax_functions
   nerva_jax.training
   nerva_jax.utilities
   nerva_jax.weight_initializers
