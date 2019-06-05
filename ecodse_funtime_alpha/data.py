import csv
from os.path import join, exists

import tensorflow as tf

import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer


def preprocess_image(image):
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    image /= 255.0  # normalize to [0,1] range
    return image


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


def get_dataset(image_dir, labels_csv):
    samples = []
    with open(labels_csv) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(reader):
            if i < 1:  # Skip first row as it is column names
                continue
            samples.append((row[0], join(image_dir, '{}.jpg'.format(row[0])), row[1].split(' ')))

    # Check all images exists and create labels list
    labels = []
    for sample in samples:
        if not exists(sample[1]):
            print("WARNING: {} does not exist".format(sample[1]))

        for label in sample[2]:
            if label not in labels:
                labels.append(label)

    mb = MultiLabelBinarizer(classes=labels)
    mb.fit(labels)

    # Transform all labels
    transformed_labels = []
    for sample in samples:
        transformed_labels.append(mb.transform([sample[2]]).squeeze())

    # Create tensorflow dataset
    img_ds = tf.data.Dataset.from_tensor_slices([s[1] for s in samples])
    img_ds = img_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(transformed_labels, tf.int64))

    img_label_ds = tf.data.Dataset.zip((img_ds, label_ds))

    return img_label_ds


if __name__ == "__main__":
    image_dir = '/home/hadrien/Downloads/rainforest/fixed-train-jpg/'
    labels_csv = '/home/hadrien/Downloads/rainforest/train_v2.csv'
    dataset = get_dataset(image_dir, labels_csv)

    for n, sample in enumerate(dataset.take(4)):
        image, label = sample[0], sample[1]
        image = image.numpy()
        plt.imshow(image)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.title(label)
        plt.show()
        plt.close()
