import numpy as np
import pytest
import tensorflow as tf

from ecodse_funtime_alpha.models import TestMLP, SimpleCNN


class TestTestMLP(object):
    @pytest.fixture(autouse=True)
    def create_model(self):
        self.batchsize = 2
        self.insize = 256 * 256 * 3
        self.hiddensize = 5
        self.outsize = 9
        self.testmodel = TestMLP(self.hiddensize, self.outsize)
        self.testinput = tf.zeros([self.batchsize, self.insize])

    def test_outshape(self):
        testout = self.testmodel(self.testinput)
        assert np.array_equal(tf.shape(testout).numpy(), [self.batchsize, self.outsize])

    def test_numparam(self):
        _ = self.testmodel(self.testinput)
        nparam = np.sum([np.product([xi for xi in x.get_shape()]) for x in self.testmodel.trainable_variables])
        assert nparam == (self.insize + 1) * self.hiddensize + (self.hiddensize + 1) * self.outsize


class TestSimpleCNN(object):
    @pytest.fixture(autouse=True)
    def create_model(self):
        self.batchsize = 2
        self.img_size = 256
        self.img_channel = 3
        self.kernels = 5
        self.kernel_size = 4
        self.outsize = 9
        self.testmodel = SimpleCNN(self.kernels, self.kernel_size, self.outsize)
        self.testinput = tf.zeros([self.batchsize, self.img_size, self.img_size, self.img_channel])

    def test_outshape(self):
        testout = self.testmodel(self.testinput)
        assert np.array_equal(tf.shape(testout).numpy(), [self.batchsize, self.outsize])

    def test_numparam(self):
        _ = self.testmodel(self.testinput)
        nparam = np.sum([np.product([xi for xi in x.get_shape()]) for x in self.testmodel.trainable_variables])
        assert nparam == (self.kernel_size * self.kernel_size * self.img_channel + 1) * self.kernels + \
                         (self.kernels * ((self.img_size - (self.kernel_size - 1)) // 2) ** 2 + 1) * self.outsize
