# Copyright 2023-2025 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

"""Test cases for MLPs."""

import unittest
from dataclasses import dataclass
from typing import List

from nerva_jax.learning_rate import ConstantScheduler
from nerva_jax.datasets import DataLoader, from_one_hot
from nerva_jax.loss_functions import SoftmaxCrossEntropyLossFunction
from nerva_jax.matrix_operations import Matrix
from nerva_jax.multilayer_perceptron import parse_multilayer_perceptron, MultilayerPerceptron
from nerva_jax.training import stochastic_gradient_descent_plain, stochastic_gradient_descent


# ----------------------------
# Package-agnostic helpers (imported from tests.utilities)
# ----------------------------
from utilities import to_tensor, assert_tensors_are_close

# ----------------------------
# MLPSpec dataclass
# ----------------------------


@dataclass
class MLPSpec:
    X: Matrix
    T: Matrix
    W: List[Matrix]
    b: List[Matrix]
    Y1: Matrix
    DY1: Matrix
    Y2: Matrix
    DY2: Matrix
    lr: float
    sizes: List[int]
    batch_size: int

# ----------------------------
# MLP construction helper
# ----------------------------


def construct_mlp(linear_layer_sizes: List[int]):
    layer_specifications = ["ReLU", "ReLU", "Linear"]
    linear_layer_weights = ["XavierNormal", "XavierNormal", "XavierNormal"]
    layer_optimizers = ["GradientDescent", "GradientDescent", "GradientDescent"]
    return parse_multilayer_perceptron(layer_specifications,
                                       linear_layer_sizes,
                                       layer_optimizers,
                                       linear_layer_weights)

# ----------------------------
# Test case
# ----------------------------

class TestMLP(unittest.TestCase):

    def _initialize_mlp(self, spec: MLPSpec) -> MultilayerPerceptron:
        """Helper to construct and initialize the MLP with fixed weights/biases."""
        M = construct_mlp(spec.sizes)
        for i, (W, b) in enumerate(zip(spec.W, spec.b)):
            M.layers[i].W = W
            M.layers[i].b = b
        return M

    def _test_mlp(self, spec: MLPSpec):
        """Test MultilayerPerceptron against precomputed values."""
        M = self._initialize_mlp(spec)
        loss = SoftmaxCrossEntropyLossFunction()

        # First forward pass before training
        Y = M.feedforward(spec.X)
        DY = loss.gradient(Y, spec.T) / spec.batch_size
        assert_tensors_are_close("Y", Y, "Y1", spec.Y1)
        assert_tensors_are_close("DY", DY, "DY1", spec.DY1)

        M.backpropagate(Y, DY)
        M.optimize(spec.lr)
        Y = M.feedforward(spec.X)
        M.backpropagate(Y, DY)

        assert_tensors_are_close("Y", Y, "Y2", spec.Y2)
        # Allow a slightly looser tolerance for DY due to accumulation of small numeric differences
        assert_tensors_are_close("DY", DY, "DY2", spec.DY2, atol=1e-3, rtol=1e-3)

    def _test_sgd_plain(self, spec: MLPSpec):
        """Test stochastic_gradient_descent_plain against precomputed values."""
        M = self._initialize_mlp(spec)
        loss = SoftmaxCrossEntropyLossFunction()
        lr_sched = ConstantScheduler(spec.lr)

        # First forward pass before training
        Y = M.feedforward(spec.X)
        DY = loss.gradient(Y, spec.T) / spec.batch_size
        assert_tensors_are_close("Y (sgd_plain before)", Y, "Y1", spec.Y1)
        assert_tensors_are_close("DY (sgd_plain before)", DY, "DY1", spec.DY1)

        # Run one epoch (one batch) of SGD with plain tensors
        stochastic_gradient_descent_plain(M, spec.X, spec.T, loss,
                                          lr_sched, epochs=1,
                                          batch_size=spec.batch_size,
                                          shuffle=False)

        # After one update step
        Y = M.feedforward(spec.X)
        DY = loss.gradient(Y, spec.T) / spec.batch_size
        assert_tensors_are_close("Y (sgd_plain after)", Y, "Y2", spec.Y2)
        assert_tensors_are_close("DY (sgd_plain after)", DY, "DY2", spec.DY2, atol=1e-3, rtol=1e-3)

    def _test_sgd_loader(self, spec: MLPSpec):
        """Test stochastic_gradient_descent with DataLoader against precomputed values."""
        M = self._initialize_mlp(spec)
        loss = SoftmaxCrossEntropyLossFunction()
        lr_sched = ConstantScheduler(spec.lr)

        train_loader = DataLoader(spec.X, from_one_hot(spec.T), batch_size=spec.batch_size, num_classes=spec.sizes[-1])
        test_loader = DataLoader(spec.X, from_one_hot(spec.T), batch_size=spec.batch_size, num_classes=spec.sizes[-1])

        # First forward pass before training
        Y = M.feedforward(spec.X)
        DY = loss.gradient(Y, spec.T) / spec.batch_size
        assert_tensors_are_close("Y (sgd_loader before)", Y, "Y1", spec.Y1)
        assert_tensors_are_close("DY (sgd_loader before)", DY, "DY1", spec.DY1)

        # Run one epoch (one batch) of SGD with loader
        stochastic_gradient_descent(M, epochs=1, loss=loss,
                                    learning_rate=lr_sched,
                                    train_loader=train_loader,
                                    test_loader=test_loader)

        # After one update step
        Y = M.feedforward(spec.X)
        DY = loss.gradient(Y, spec.T) / spec.batch_size
        assert_tensors_are_close("Y (sgd_loader after)", Y, "Y2", spec.Y2)
        assert_tensors_are_close("DY (sgd_loader after)", DY, "DY2", spec.DY2, atol=1e-3, rtol=1e-3)

    def test_mlp0(self):
        """
        Test MLP with the following architecture:

        Layer 1: Linear(2 → 6) + ReLU
        Layer 2: Linear(6 → 4) + ReLU
        Layer 3: Linear(4 → 3)

        W[i] and b[i] correspond to the weights and biases of layer i+1.
        """
        spec = MLPSpec(
            X=to_tensor([
                [0.37454012, 0.95071429],
                [0.73199391, 0.59865850],
                [0.15601864, 0.15599452],
                [0.05808361, 0.86617613],
                [0.60111499, 0.70807260],
            ]),
            T=to_tensor([
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]),
            W=[
                to_tensor([
                    [0.56110168, 0.66345596],
                    [-0.12117522, -0.37362885],
                    [0.58880562, -0.04527364],
                    [-0.47821110, 0.44252452],
                    [0.70591444, -0.14112198],
                    [0.67916799, -0.60361999],
                ]),
                to_tensor([
                    [0.08186211, 0.10790163, 0.10656221, 0.05315667, -0.18149444, 0.35163686],
                    [-0.19592771, 0.12398510, 0.31654948, -0.07420450, 0.16522692, -0.34778017],
                    [0.31287605, -0.02583930, 0.23740244, 0.36834371, -0.06145998, -0.03954251],
                    [-0.10137446, -0.21237525, 0.25827262, -0.38412368, -0.11736847, -0.02998422],
                ]),
                to_tensor([
                    [-0.30466300, 0.16379684, 0.14415038, 0.46986455],
                    [0.11790103, -0.04250759, -0.06847537, -0.43189043],
                    [-0.36789089, -0.46546835, -0.06826532, -0.03290677],
                ])
            ],
            b=[
                to_tensor([[0.20102479, 0.59402120, 0.21540157, 0.35205203, 0.34316611, 0.39400283]]),
                to_tensor([[0.17123953, -0.11965782, 0.35985306, 0.39208883]]),
                to_tensor([[0.07861263, 0.34228545, 0.07193470]])
            ],
            Y1=to_tensor([
                [0.15708828, 0.28446376, -0.10275675],
                [0.16033420, 0.25479689, -0.14251150],
                [0.11323270, 0.28988251, -0.11241873],
                [0.12644258, 0.31084442, -0.09212098],
                [0.15833774, 0.26541382, -0.12861101],
            ]),
            DY1=to_tensor([
                [0.06879912, -0.12185498, 0.05305589],
                [-0.12952240, 0.07745969, 0.05206273],
                [0.06686258, -0.12021869, 0.05335608],
                [0.06652980, -0.11999798, 0.05346817],
                [0.06984292, -0.12226351, 0.05242061],
            ]),
            Y2=to_tensor([
                [0.14954381, 0.29581159, -0.10952257],
                [0.15290099, 0.26589465, -0.14921829],
                [0.10718645, 0.29935911, -0.11815142],
                [0.11962160, 0.32149324, -0.09845620],
                [0.15089402, 0.27656406, -0.13531879],
            ]),
            DY2=to_tensor([
                [0.06827622, -0.12096987, 0.05269366],
                [-0.13004242, 0.07832626, 0.05171615],
                [0.06644239, -0.11947980, 0.05303741],
                [0.06605557, -0.11916840, 0.05311283],
                [0.06932384, -0.12139316, 0.05206933],
            ]),
            lr=0.01,
            sizes=[2, 6, 4, 3],
            batch_size=5
        )

        self._test_mlp(spec)
        self._test_sgd_plain(spec)
        self._test_sgd_loader(spec)

    def test_mlp1(self):
        """
        Test MLP with the following architecture:

        Layer 1: Linear(3 → 5) + ReLU
        Layer 2: Linear(5 → 2) + ReLU
        Layer 3: Linear(2 → 4)

        W[i] and b[i] correspond to the weights and biases of layer i+1.
        """
        spec = MLPSpec(
            X=to_tensor([
                [0.00077877, 0.99221158, 0.61748153],
                [0.61165315, 0.00706631, 0.02306242],
                [0.52477467, 0.39986098, 0.04666566],
                [0.97375554, 0.23277134, 0.09060644],
            ]),
            T = to_tensor([
                [0.00000000, 1.00000000, 0.00000000, 0.00000000],
                [0.00000000, 0.00000000, 1.00000000, 0.00000000],
                [0.00000000, 0.00000000, 0.00000000, 1.00000000],
                [0.00000000, 0.00000000, 1.00000000, 0.00000000],
            ]),
            W=[
                to_tensor([
                    [-0.39624736, 0.52474105, 0.21831584],
                    [-0.38542053, 0.22781183, 0.24430284],
                    [-0.48814151, -0.43212008, 0.15193383],
                    [-0.42474371, -0.12206060, 0.32209969],
                    [0.42924264, 0.54005784, -0.27541694],
                ]),
                to_tensor([
                    [-0.01198715, 0.07376724, -0.05684799, -0.18777776, 0.29133716],
                    [0.15762907, -0.22121911, 0.35209563, -0.42210799, -0.38685778],
                ]),
                to_tensor([
                    [-0.09037616, 0.70386237],
                    [-0.26921728, 0.64686316],
                    [0.34774554, -0.56019980],
                    [0.66119587, -0.23288755],
                ]),
            ],
            b=[
                to_tensor([
                    [0.07732374, 0.30383354, 0.02999540, -0.43342215, -0.32955059],
                ]),
                to_tensor([
                    [0.43432558, -0.14905037],
                ]),
                to_tensor([
                    [-0.40916497, -0.64210689, -0.35259017, 0.22382185],
                ]),
            ],
            Y1 = to_tensor([
                [-0.45312327, -0.77305222, -0.18344933, 0.54542261],
                [-0.44891989, -0.76053095, -0.19962291, 0.51467049],
                [-0.45228270, -0.77054822, -0.18668362, 0.53927302],
                [-0.45342341, -0.77394629, -0.18229441, 0.54761851],
            ]),
            DY1 = to_tensor([
                [0.04347773, -0.21842644, 0.05693571, 0.11801299],
                [0.04435392, 0.03247889, -0.19308847, 0.11625564],
                [0.04365253, 0.03175326, 0.05693214, -0.13233793],
                [0.04341538, 0.03150955, -0.19306317, 0.11813824],
            ]),
            Y2 = to_tensor([
                [-0.45506847, -0.77091932, -0.18090963, 0.54122663],
                [-0.45096460, -0.75883305, -0.19661310, 0.51152146],
                [-0.45433164, -0.76874924, -0.18372914, 0.53589314],
                [-0.45551044, -0.77222097, -0.17921844, 0.54442573],
            ]),
            DY2 = to_tensor([
                [0.04345694, -0.21831259, 0.05716429, 0.11769136],
                [0.04430398, 0.03256395, -0.19286448, 0.11599655],
                [0.04360865, 0.03184365, 0.05716021, -0.13261250],
                [0.04336601, 0.03159395, -0.19283350, 0.11787353],
            ]),
            lr = 0.01,
            sizes = [3, 5, 2, 4],
            batch_size = 4
        )
        self._test_mlp(spec)
        self._test_sgd_plain(spec)
        self._test_sgd_loader(spec)

    def test_mlp2(self):
        """
        Test MLP with the following architecture:

        Layer 1: Linear(6 → 2) + ReLU
        Layer 2: Linear(2 → 2) + ReLU
        Layer 3: Linear(2 → 3)

        W[i] and b[i] correspond to the weights and biases of layer i+1.
        """
        spec = MLPSpec(
            X=to_tensor([
                [0.98323089, 0.46676290, 0.85994041, 0.68030757, 0.45049927, 0.01326496],
                [0.94220173, 0.56328821, 0.38541651, 0.01596625, 0.23089382, 0.24102546],
                [0.68326354, 0.60999668, 0.83319491, 0.17336465, 0.39106062, 0.18223609],
                [0.75536144, 0.42515588, 0.20794167, 0.56770033, 0.03131329, 0.84228480],
                [0.44975412, 0.39515024, 0.92665887, 0.72727197, 0.32654077, 0.57044399],
                [0.52083427, 0.96117204, 0.84453386, 0.74732012, 0.53969210, 0.58675116],
                [0.96525532, 0.60703427, 0.27599919, 0.29627350, 0.16526695, 0.01563641],
                [0.42340147, 0.39488152, 0.29348817, 0.01407982, 0.19884241, 0.71134198],
            ]),
            T=to_tensor([
                [0.00000000, 0.00000000, 1.00000000],
                [1.00000000, 0.00000000, 0.00000000],
                [0.00000000, 0.00000000, 1.00000000],
                [1.00000000, 0.00000000, 0.00000000],
                [0.00000000, 1.00000000, 0.00000000],
                [0.00000000, 0.00000000, 1.00000000],
                [0.00000000, 1.00000000, 0.00000000],
                [1.00000000, 0.00000000, 0.00000000],
            ]),
            W=[
                to_tensor([
                    [-0.32607165, 0.14416048, -0.18457964, 0.37571555, 0.31530383, -0.23843847],
                    [0.26372054, 0.36656415, -0.09508932, -0.06380705, -0.21398570, -0.25577575],
                ]),
                to_tensor([
                    [-0.23665118, -0.08175746],
                    [0.23666702, -0.65622914],
                ]),
                to_tensor([
                    [-0.12040216, 0.11809724],
                    [0.67121518, 0.29201975],
                    [0.11411048, -0.23803940],
                ])
            ],
            b=[
                to_tensor([
                    [-0.01039284, -0.27663937],
                ]),
                to_tensor([
                    [0.18114421, -0.16169840],
                ]),
                to_tensor([
                    [-0.00158843, -0.40333885, -0.31654432],
                ]),
            ],
            Y1 = to_tensor([
                [-0.02339859, -0.28175211, -0.29587388],
                [-0.02310725, -0.28337622, -0.29614997],
                [-0.02339859, -0.28175211, -0.29587388],
                [-0.02339859, -0.28175211, -0.29587388],
                [-0.02339859, -0.28175211, -0.29587388],
                [-0.02016460, -0.29978088, -0.29893887],
                [-0.02225747, -0.28811356, -0.29695535],
                [-0.02339859, -0.28175211, -0.29587388],
            ]),
            DY1 = to_tensor([
                [0.04933274, 0.03810076, -0.08743350],
                [-0.07563005, 0.03805654, 0.03757351],
                [0.04933274, 0.03810076, -0.08743350],
                [-0.07566726, 0.03810076, 0.03756650],
                [0.04933274, -0.08689924, 0.03756650],
                [0.04974561, 0.03761135, -0.08735696],
                [0.04947848, -0.08707230, 0.03759383],
                [-0.07566726, 0.03810076, 0.03756650],
            ]),
            Y2 = to_tensor([
                [-0.02357430, -0.28248075, -0.29513836],
                [-0.02328351, -0.28410131, -0.29541418],
                [-0.02357430, -0.28248075, -0.29513836],
                [-0.02357430, -0.28248075, -0.29513836],
                [-0.02357430, -0.28248075, -0.29513836],
                [-0.02033705, -0.30052209, -0.29820901],
                [-0.02243426, -0.28883424, -0.29621974],
                [-0.02357430, -0.28248075, -0.29513836],
            ]),
            DY2 = to_tensor([
                [0.04932753, 0.03807569, -0.08740322],
                [-0.07563534, 0.03803158, 0.03760376],
                [0.04932753, 0.03807569, -0.08740322],
                [-0.07567246, 0.03807569, 0.03759678],
                [0.04932753, -0.08692431, 0.03759678],
                [0.04974060, 0.03758618, -0.08732678],
                [0.04947305, -0.08709708, 0.03762402],
                [-0.07567246, 0.03807569, 0.03759678],
            ]),
            lr = 0.01,
            sizes = [6, 2, 2, 3],
            batch_size = 8
        )
        self._test_mlp(spec)
        self._test_sgd_plain(spec)
        self._test_sgd_loader(spec)
