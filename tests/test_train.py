import math
import os

from copy import deepcopy

import pytest
import tensorflow as tf

from ecodse_funtime_alpha.train import batch_dataset
from ecodse_funtime_alpha.train import fit_loop
from ecodse_funtime_alpha.train import get_args
from ecodse_funtime_alpha.train import train_loop


class TestArgparse(object):

    def test_argparsenormal(self):
        fakearg = ['--imagepath=./', '--labelpath=fakedir/name.csv',
                   '--seed=1', '--kernels=10', '--ksize=1',
                   '--lr=0.01', '--nepoch=2', '--batchsize=4'
                   ]
        args = get_args(fakearg)
        assert args.imagepath == './'
        assert args.labelpath == 'fakedir/name.csv'
        assert args.seed == 1
        assert args.kernels == 10
        assert args.ksize == 1
        assert args.lr == 0.01
        assert args.nepoch == 2
        assert args.batchsize == 4

    @pytest.mark.xfail(raises=SystemExit)
    def test_argparse_lr(self):
        fakearg = ['--lr=a']
        _ = get_args(fakearg)

    @pytest.mark.xfail(raises=SystemExit)
    def test_argparse_seed(self):
        fakearg = ['--seed=a']
        _ = get_args(fakearg)

    def test_argparse_imagepath(self):
        fakearg = ['--imagepath=notavalidpath']
        args = get_args(fakearg)
        assert not os.path.isdir(args.imagepath)

    def test_argparse_labelpath(self):
        fakearg = ['--labelpath=invalid.csv']
        args = get_args(fakearg)
        assert not os.path.exists(args.labelpath)


class TestBatchDataset(object):
    @pytest.fixture(autouse=True)
    def mock_file(self):
        self.nimage = 10
        self.nlabel = 8
        img_ds = tf.data.Dataset.from_tensor_slices(tf.zeros([self.nimage, 28 * 28 * 3]))
        label_ds = tf.data.Dataset.from_tensor_slices(tf.zeros([self.nimage, self.nlabel]))
        self.dataset = tf.data.Dataset.zip((img_ds, label_ds))

    def test_dataset(self):
        batchsize = 4
        nepoch = 3
        dataset = batch_dataset(self.dataset, nepoch, batchsize)
        # get sizes of mini-batches in one epoch
        size_of_batch = [batchsize] * (self.nimage // batchsize)
        # add remainder if number of examples is not a multiple of batchsize
        size_of_batch += [self.nimage % batchsize] if self.nimage % batchsize != 0 else []
        # multiply by number of epochs
        size_of_batch = [*size_of_batch] * nepoch
        assert [x[0].shape[0].value for x in dataset] == size_of_batch

    def test_datasetsize(self):
        batchsize = 4
        nepoch = 3
        dataset = batch_dataset(self.dataset, nepoch, batchsize)
        assert tf.data.experimental.cardinality(dataset).numpy() == math.ceil(self.nimage / batchsize) * nepoch


class TestFitLoop(object):
    @pytest.fixture(autouse=True)
    def mock_file(self):
        self.nimage = 1
        self.nlabel = 8
        img_ds = tf.data.Dataset.from_tensor_slices(tf.random.uniform([self.nimage, 28 * 28 * 3]))
        label_ds = tf.data.Dataset.from_tensor_slices(tf.random.uniform([self.nimage, self.nlabel]))
        self.dataset = tf.data.Dataset.zip((img_ds, label_ds))
        self.model = tf.keras.Sequential([tf.keras.layers.Dense(self.nlabel, input_shape=(28 * 28 * 3,))])

    def test_fitvarchanged(self):
        before = deepcopy(self.model.trainable_variables)
        model = fit_loop(self.dataset, self.model, tf.keras.optimizers.Adam(lr=0.1), 1, 1)
        after = model.trainable_variables
        for b, a in zip(before, after):
            # make sure something changed
            assert (b.numpy() != a.numpy()).any()

    def test_trainvarchanged(self):
        before = deepcopy(self.model.trainable_variables)
        model = train_loop(self.dataset, self.model, tf.train.AdamOptimizer(learning_rate=0.1), 1, 1)
        after = model.trainable_variables
        for b, a in zip(before, after):
            # make sure something changed
            assert (b.numpy() != a.numpy()).any()
