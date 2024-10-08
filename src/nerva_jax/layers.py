# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import jax.numpy as jnp

from nerva_jax.activation_functions import ActivationFunction, SReLUActivation, parse_activation
from nerva_jax.matrix_operations import column_repeat, columns_mean, columns_sum, diag, elements_sum, hadamard, \
    identity, ones, inv_sqrt, row_repeat, rows_sum, vector_size, zeros
from nerva_jax.optimizers import CompositeOptimizer, parse_optimizer
from nerva_jax.softmax_functions import log_softmax, softmax
from nerva_jax.weight_initializers import set_layer_weights

Matrix = jnp.ndarray


class Layer(object):
    """
    Base class for layers of a neural network with data in column layout
    """
    def __init__(self):
        self.X = None
        self.DX = None
        self.optimizer = None

    def feedforward(self, X: Matrix) -> Matrix:
        raise NotImplementedError

    def backpropagate(self, Y: Matrix, DY: Matrix) -> None:
        raise NotImplementedError

    def optimize(self, eta):
        if self.optimizer:
            self.optimizer.update(eta)


class LinearLayer(Layer):
    """
    Linear layer of a neural network
    """
    def __init__(self, D: int, K: int):
        super().__init__()
        self.W = zeros(K, D)
        self.DW = zeros(K, D)
        self.b = zeros(K)
        self.Db = zeros(K)
        self.optimizer = None

    def feedforward(self, X: Matrix) -> Matrix:
        self.X = X
        N, D = X.shape
        W = self.W
        b = self.b

        Y = X @ W.T + row_repeat(b, N)

        return Y

    def backpropagate(self, Y: Matrix, DY: Matrix) -> None:
        X = self.X
        W = self.W

        DW = DY.T @ X
        Db = columns_sum(DY)
        DX = DY @ W

        self.DW = DW
        self.Db = Db
        self.DX = DX

    def input_size(self) -> int:
        return self.W.shape[1]

    def output_size(self) -> int:
        return self.W.shape[0]

    def set_optimizer(self, optimizer: str):
        make_optimizer = parse_optimizer(optimizer)
        self.optimizer = CompositeOptimizer([make_optimizer(self, 'W', 'DW'), make_optimizer(self, 'b', 'Db')])

    def set_weights(self, weight_initializer):
        set_layer_weights(self, weight_initializer)


class ActivationLayer(LinearLayer):
    """
    Linear layer with an activation function
    """
    def __init__(self, D: int, K: int, act: ActivationFunction):
        super().__init__(D, K)
        self.Z = None
        self.DZ = None
        self.act = act

    def feedforward(self, X: Matrix) -> Matrix:
        self.X = X
        N, D = X.shape
        W = self.W
        b = self.b
        act = self.act

        Z = X @ W.T + row_repeat(b, N)
        Y = act(Z)

        self.Z = Z
        return Y

    def backpropagate(self, Y: Matrix, DY: Matrix) -> None:
        X = self.X
        W = self.W
        Z = self.Z
        act = self.act

        DZ = hadamard(DY, act.gradient(Z))
        DW = DZ.T @ X
        Db = columns_sum(DZ)
        DX = DZ @ W

        self.DZ = DZ
        self.DW = DW
        self.Db = Db
        self.DX = DX


class SReLULayer(ActivationLayer):
    """
    Linear layer with an SReLU activation function. It adds learning of the parameters
    al, tl, ar and tr.
    """
    def __init__(self, D: int, K: int, act: SReLUActivation):
        super().__init__(D, K, act)
        self.Dal = 0
        self.Dtl = 0
        self.Dar = 0
        self.Dtr = 0

    def backpropagate(self, Y: Matrix, DY: Matrix) -> None:
        super().backpropagate(Y, DY)
        Z = self.Z
        al, tl, ar, tr = self.act.x

        Al = lambda Z: jnp.where(Z <= tl, Z - tl, 0)
        Tl = lambda Z: jnp.where(Z <= tl, 1 - al, 0)
        Ar = lambda Z: jnp.where((Z <= tl) | (Z < tr), 0, Z - tr)
        Tr = lambda Z: jnp.where((Z <= tl) | (Z < tr), 0, 1 - ar)

        Dal = elements_sum(hadamard(DY, Al(Z)))
        Dtl = elements_sum(hadamard(DY, Tl(Z)))
        Dar = elements_sum(hadamard(DY, Ar(Z)))
        Dtr = elements_sum(hadamard(DY, Tr(Z)))

        self.act.Dx = jnp.array([Dal, Dtl, Dar, Dtr])

    def set_optimizer(self, optimizer: str):
        make_optimizer = parse_optimizer(optimizer)
        self.optimizer = CompositeOptimizer([make_optimizer(self, 'W', 'DW'),
                                             make_optimizer(self, 'b', 'Db'),
                                             make_optimizer(self.act, 'x', 'Dx')
                                            ])


class SoftmaxLayer(LinearLayer):
    """
    Linear layer with a softmax activation function
    """
    def __init__(self, D: int, K: int):
        super().__init__(D, K)
        self.Z = None
        self.DZ = None

    def feedforward(self, X: Matrix) -> Matrix:
        self.X = X
        N, D = X.shape
        W = self.W
        b = self.b

        Z = X @ W.T + row_repeat(b, N)
        Y = softmax(Z)

        self.Z = Z
        return Y

    def backpropagate(self, Y: Matrix, DY: Matrix) -> None:
        K, N = self.Z.shape
        X = self.X
        W = self.W

        DZ = hadamard(Y, DY - column_repeat(diag(DY @ Y.T), N))
        DW = DZ.T @ X
        Db = columns_sum(DZ)
        DX = DZ @ W

        self.DZ = DZ
        self.DW = DW
        self.Db = Db
        self.DX = DX


class LogSoftmaxLayer(LinearLayer):
    """
    Linear layer with a log_softmax activation function
    """
    def __init__(self, D: int, K: int):
        super().__init__(D, K)
        self.Z = None
        self.DZ = None

    def feedforward(self, X: Matrix) -> Matrix:
        self.X = X
        N, D = X.shape
        W = self.W
        b = self.b

        Z = X @ W.T + row_repeat(b, N)
        Y = log_softmax(Z)

        self.Z = Z
        return Y

    def backpropagate(self, Y: Matrix, DY: Matrix) -> None:
        K, N = self.Z.shape
        X = self.X
        W = self.W
        Z = self.Z

        DZ = DY - hadamard(softmax(Z), column_repeat(rows_sum(DY), N))
        DW = DZ.T @ X
        Db = columns_sum(DZ)
        DX = DZ @ W

        self.DZ = DZ
        self.DW = DW
        self.Db = Db
        self.DX = DX


class BatchNormalizationLayer(Layer):
    """
    A batch normalization layer
    """
    def __init__(self, D: int):
        super().__init__()
        self.Z = None
        self.DZ = None
        self.gamma = ones(D)
        self.Dgamma = zeros(D)
        self.beta = zeros(D)
        self.Dbeta = zeros(D)
        self.inv_sqrt_Sigma = zeros(D)
        self.optimizer = None

    def feedforward(self, X: Matrix) -> Matrix:
        self.X = X
        N, D = X.shape
        gamma = self.gamma
        beta = self.beta

        R = X - row_repeat(columns_mean(X), N)
        Sigma = diag(R.T @ R) / N
        inv_sqrt_Sigma = inv_sqrt(Sigma)
        Z = hadamard(row_repeat(inv_sqrt_Sigma, N), R)
        Y = hadamard(row_repeat(gamma, N), Z) + row_repeat(beta, N)

        self.inv_sqrt_Sigma = inv_sqrt_Sigma
        self.Z = Z
        return Y

    def backpropagate(self, Y: Matrix, DY: Matrix) -> None:
        N, D = self.X.shape
        Z = self.Z
        gamma = self.gamma
        inv_sqrt_Sigma = self.inv_sqrt_Sigma

        DZ = hadamard(row_repeat(gamma, N), DY)
        Dbeta = columns_sum(DY)
        Dgamma = columns_sum(hadamard(DY, Z))
        DX = hadamard(row_repeat(inv_sqrt_Sigma / N, N), (N * identity(N) - ones(N, N)) @ DZ - hadamard(Z, row_repeat(diag(Z.T @ DZ), N)))

        self.DZ = DZ
        self.Dbeta = Dbeta
        self.Dgamma = Dgamma
        self.DX = DX

    def input_size(self) -> int:
        return vector_size(self.gamma)

    def output_size(self) -> int:
        return vector_size(self.gamma)

    def set_optimizer(self, optimizer: str):
        make_optimizer = parse_optimizer(optimizer)
        self.optimizer = CompositeOptimizer([make_optimizer(self, 'beta', 'Dbeta'), make_optimizer(self, 'gamma', 'Dgamma')])


def parse_linear_layer(text: str,
                       D: int,
                       K: int,
                       optimizer: str,
                       weight_initializer: str
                      ) -> Layer:
    if text == 'Linear':
        layer = LinearLayer(D, K)
    elif text == 'Softmax':
        layer = SoftmaxLayer(D, K)
    elif text == 'LogSoftmax':
        layer = LogSoftmaxLayer(D, K)
    elif text.startswith('SReLU'):
        act = parse_activation(text)
        layer = SReLULayer(D, K, act)
    else:
        act = parse_activation(text)
        layer = ActivationLayer(D, K, act)
    layer.set_optimizer(optimizer)
    layer.set_weights(weight_initializer)
    return layer
