import numpy as np
import tensorflow as tf

from ecodse_funtime_alpha.models import TestMLP, SimpleCNN


class TestTestMLP(object):

    def test_outshape(self):
        testmodel = TestMLP(5, 9)
        testinput = tf.zeros([2, 256 * 256 * 3])
        testout = testmodel(testinput)
        assert np.array_equal(tf.shape(testout).numpy(), [2, 9])

    def test_numparam(self):
        testmodel = TestMLP(5, 9)
        testinput = tf.zeros([2, 256 * 256 * 3])
        _ = testmodel(testinput)
        nparam = np.sum([np.product([xi for xi in x.get_shape()]) for x in testmodel.trainable_variables])
        assert nparam == 256 * 256 * 3 * 5 + 5 + 5 * 9 + 9


class TestSimpleCNN(object):

    def test_outshape(self):
        testmodel = SimpleCNN(5, 4, 9)
        testinput = tf.zeros([2, 256, 256, 3])
        testout = testmodel(testinput)
        assert np.array_equal(tf.shape(testout).numpy(), [2, 9])

    def test_numparam(self):
        testmodel = SimpleCNN(5, 4, 9)
        testinput = tf.zeros([2, 256, 256, 3])
        _ = testmodel(testinput)
        nparam = np.sum([np.product([xi for xi in x.get_shape()]) for x in testmodel.trainable_variables])
        assert nparam == (4 * 4 * 3 + 1) * 5 + (5 * ((256 - (4 - 1)) // 2) ** 2 + 1) * 9
