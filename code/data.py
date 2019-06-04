import csv
from os.path import join, exists

import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer


def preprocess_image(image):
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    image /= 255.0  # normalize to [0,1] range
    return image


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
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

        # Check all images exists and create labels list
        self.labels = []
        for sample in self.samples:
            if not exists(sample[1]):
                print("WARNING: {} does not exist".format(sample[1]))

            for label in sample[2]:
                if label not in self.labels:
                    self.labels.append(label)

        self.mb = MultiLabelBinarizer(classes=self.labels)
        self.mb.fit(self.labels)

        # Transform all labels
        self.transformed_labels = []
        for sample in self.samples:
            self.transformed_labels.append(self.mb.transform([sample[2]]).squeeze())

    def __len__(self):
        return len(self.samples)

    def get_tf_version(self):
        img_ds = tf.data.Dataset.from_tensor_slices([s[1] for s in self.samples])
        img_ds = img_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(self.transformed_labels, tf.int64))

        img_label_ds = tf.data.Dataset.zip((img_ds, label_ds))

        return img_label_ds


if __name__ == "__main__":
    image_dir = '/home/hadrien/Downloads/rainforest/fixed-train-jpg/'
    labels_csv = '/home/hadrien/Downloads/rainforest/train_v2.csv'
    dataset = AmazonDataset(image_dir, labels_csv)

    print("Dataset len: {}".format(len(dataset)))
    ds = dataset.get_tf_version()

    for n, sample in enumerate(ds.take(4)):
        image, label = sample[0], sample[1]
        image = image.numpy()
        print(image.shape)
        print(np.unique(image))
        plt.imshow(image)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.show()
        plt.close()
