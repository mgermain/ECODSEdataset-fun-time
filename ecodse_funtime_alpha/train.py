import tensorflow as tf
import models
import data
import argparse
import os


def train_loop(dataset, model, optimizer):
    for x, y in dataset.batch(5):
        with tf.GradientTape() as tape:
            predictions = model(tf.squeeze(x))
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(y, tf.float32), logits=predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        print(loss)


def fit_loop(dataset, model, optimizer, nepoch, batchsize):
    nstep = len(list(dataset)) // batchsize
    dataset = dataset.shuffle(12)
    dataset = dataset.repeat(nepoch)
    dataset = dataset.batch(batchsize)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=["accuracy"])
    model.fit(dataset, epochs=nepoch, steps_per_epoch=nstep)


def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--imagepath', type=dir_path,
                           help='path to image directory')
    argparser.add_argument('--labelpath',
                           help='path to csv file for labels')
    argparser.add_argument('--set-seed', default=-1, type=int,
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
    args = argparser.parse_args()

    dataset = data.get_dataset(args.imagepath, args.labelpath)
    # model = models.SimpleCNN(args.kernels, args.ksize, 9)
    model = models.RetrainResnet()
    optimizer = tf.keras.optimizers.Adam(lr=args.lr)
    fit_loop(dataset, model, optimizer, args.nepoch, args.batchsize)
    # for _ in range(nepoch):
    #     train_loop(dataset, model, optimizer)
