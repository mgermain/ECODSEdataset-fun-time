import argparse
import sys

import tensorflow as tf

import ecodse_funtime_alpha.data as data
import ecodse_funtime_alpha.models as models


def train_loop(dataset, model, optimizer):
    for x, y in dataset.batch(5):
        with tf.GradientTape() as tape:
            predictions = model(tf.squeeze(x))
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(y, tf.float32), logits=predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        print(loss)


def fit_loop(dataset, lendataset, model, optimizer, nepoch, batchsize):
    nstep = lendataset // batchsize
    dataset = dataset.shuffle(12)  # 12 = buffer size
    dataset = dataset.repeat(nepoch)
    dataset = dataset.batch(batchsize)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=["accuracy"])
    model.fit(dataset, epochs=nepoch, steps_per_epoch=nstep)


def get_args(args):
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--imagepath', default='../../rainforest/fixed-train-jpg/',
                           help='path to image directory')
    argparser.add_argument('--labelpath', default='../../rainforest/train_v3.csv',
                           help='path to csv file for labels')
    argparser.add_argument('--seed', default=-1, type=int,
                           help='Set random seed to this number (if >0)')
    argparser.add_argument('--kernels', default=1, type=int,
                           help='Number of kernels in the CNN')
    argparser.add_argument('--ksize', default=1, type=int,
                           help='Size of kernels in CNN')
    argparser.add_argument('--lr', default=0.1, type=float,
                           help='Learning rate (default=0.1')
    argparser.add_argument('--nepoch', default=1, type=int,
                           help='Number of epoch for training')
    argparser.add_argument('--batchsize', default=32, type=int,
                           help='batch size (default=32)')
    args = argparser.parse_args(args)
    return args


if __name__ == "__main__":
    tf.enable_eager_execution()
    args = get_args(sys.argv[1:])
    if args.seed > 0:
        tf.random.set_seed(args.seed)
    dataset, lendataset = data.get_dataset(args.imagepath, args.labelpath)
    # model = models.TestMLP(10, 9)
    model = models.SimpleCNN(args.kernels, args.ksize, 9)
    optimizer = tf.keras.optimizers.Adam(lr=args.lr)
    fit_loop(dataset, lendataset, model, optimizer, args.nepoch, args.batchsize)
    # nepoch = 2
    # for _ in range(nepoch):
    #     train_loop(dataset, model, optimizer)
