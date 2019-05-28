import csv
from os.path import join, exists

import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np


def preprocess_image(image):
    image = tf.image.decode_image(image)
    print(image.shape)
    print(image.dtype)
    print(np.unique(image.numpy()))
    image = tf.cast(image, tf.float32)
    image /= 255.0  # normalize to [0,1] range

    return image


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    print(path)
    img = plt.imread(path)
    plt.imshow(img)
    plt.show()
    return preprocess_image(image)


class AmazonDataset:
    def __init__(self, image_dir, labels_csv):
        self.samples = []
        with open(labels_csv) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for i, row in enumerate(reader):
                if i < 1:  # Skip first row as it is column names
                    continue
                self.samples.append((row[0], join(image_dir, '{}.jpg'.format(row[0])), row[1].split(' ')))

        # Check all images exists
        for sample in self.samples:
            if not exists(sample[1]):
                print("WARNING: {} does not exist".format(sample[1]))

    def __len__(self):
        return len(self.samples)

    def get_tf_version(self):
        img_ds = tf.data.Dataset.from_tensor_slices([s[1] for s in self.samples])
        image = load_and_preprocess_image([s[1] for s in self.samples][0])
        plt.imshow(image)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.show()
        plt.close()
        img_ds = img_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        for n, image in enumerate(img_ds.take(4)):
            image = image.numpy()
            print(image.shape)
            print(np.unique(image))
            plt.imshow(image)
            plt.grid(False)
            plt.xticks([])
            plt.yticks([])
            plt.show()
            plt.close()


if __name__ == "__main__":
    image_dir = '/home/hadrien/Downloads/rainforest/train-jpg/'
    labels_csv = '/home/hadrien/Downloads/rainforest/train_v2.csv'
    dataset = AmazonDataset(image_dir, labels_csv)

    print(len(dataset))
    dataset.get_tf_version()
