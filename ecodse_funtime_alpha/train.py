import argparse
import math
import sys

import tensorflow as tf

import ecodse_funtime_alpha.data as data
import ecodse_funtime_alpha.models as models

tf.enable_eager_execution()


def batch_dataset(dataset, nepoch, batchsize):
    # shuffling the dataset before mini-batches to shuffle elements and not mini-batches
    dataset = dataset.shuffle(buffer_size=10 * batchsize)
    # split into mini-batches
    dataset = dataset.batch(batchsize)
    # repeat for multiple epochs; the earlier shuffle is different for each epoch
    dataset = dataset.repeat(nepoch)
    return dataset


def train_loop(dataset, model, optimizer, nepoch, batchsize):
    dataset = batch_dataset(dataset, nepoch, batchsize)
    for x, y in dataset:
        with tf.GradientTape() as tape:
            predictions = model(x)
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(y, tf.float32), logits=predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return model


def fit_loop(dataset, model, optimizer, nepoch, batchsize):
    # number of steps in an epoch is len(dataset) / batchsize (math.ceil for the remainder)
    nstep = math.ceil(tf.data.experimental.cardinality(dataset).numpy() / batchsize)
    dataset = batch_dataset(dataset, nepoch, batchsize)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=["accuracy"])
    model.fit(dataset, epochs=nepoch, steps_per_epoch=nstep)
    return model


def get_args(args):
    argparser = argparse.ArgumentParser()
    def_impath = '../../rainforest/fixed-train-jpg/'
    argparser.add_argument('--imagepath',
                           default=def_impath,
                           help=f'path to image directory (default {def_impath})')
    def_labelpath = '../../rainforest/train_v3.csv'
    argparser.add_argument('--labelpath',
                           default=def_labelpath,
                           help=f'path to csv file for labels (defautlt {def_labelpath})')
    def_seed = -1
    argparser.add_argument('-s',
                           '--seed',
                           default=def_seed,
                           type=int,
                           help=f'Set random seed to this number (default {def_seed})')
    def_kernels = 4
    argparser.add_argument('-k',
                           '--kernels',
                           default=def_kernels,
                           type=int,
                           help=f'Number of kernels in the CNN (default {def_kernels})')
    def_ksize = 2
    argparser.add_argument('-ks',
                           '--ksize',
                           default=def_ksize,
                           type=int,
                           help=f'Size of kernels in CNN (default {def_ksize})')
    def_lr = 0.1
    argparser.add_argument('-l',
                           '--lr',
                           default=def_lr,
                           type=float,
                           help=f'Learning rate (default {def_lr})')
    def_nepoch = 1
    argparser.add_argument('-n',
                           '--nepoch',
                           default=def_nepoch,
                           type=int,
                           help=f'Number of epoch for training (default {def_nepoch})')
    def_batch = 4
    argparser.add_argument('-b',
                           '--batchsize',
                           default=def_batch,
                           type=int,
                           help=f'batch size (default {def_batch})')
    args = argparser.parse_args(args)
    return args


if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    tf.random.set_random_seed(args.seed)
    dataset = data.get_dataset(args.imagepath, args.labelpath)
    # model = models.TestMLP(10, 9)
    model = models.SimpleCNN(args.kernels, args.ksize, 9)
    optimizer = tf.keras.optimizers.Adam(lr=args.lr)
    model = fit_loop(dataset, model, optimizer, args.nepoch, args.batchsize)
