import csv
from os.path import join, exists

import tensorflow as tf

from sklearn.preprocessing import MultiLabelBinarizer


def preprocess_image(image):
    """
    Decode a jpeg and normalize it to [0, 1] range.

    Parameters
    ----------
    image : tf tensor
        Raw image as a tf tensor

    Returns
    -------
        tf tensor in float32
    """
    image = tf.io.decode_jpeg(image, dct_method="INTEGER_ACCURATE")
    image = tf.cast(image, tf.float32)
    image /= 255.0  # normalize to [0,1] range
    return image


def load_and_preprocess_image(path):
    """
    Read a jpeg and normalize it to [0, 1] range.

    Parameters
    ----------
    path : str
        Path to the jpeg to read

    Returns
    -------
        tf tensor in float32
    """
    image = tf.io.read_file(path)
    return preprocess_image(image)


def get_dataset(image_dir, labels_csv):
    """
    Prepare the dataset for training. Images are normalized to [0, 1] range. Labels are n-hot encoded.

    Parameters
    ----------
    image_dir : str
        Path to the directory containing the images
    labels_csv : str
        Path to the csv file

    Returns
    -------
        tf.data.ZipDataset
            Elements of the dataset are structured (image, labels)
    """
    samples = []
    with open(labels_csv) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        reader.next()  # Skip first row as it is column names
        for i, row in enumerate(reader):
            samples.append((row[0], join(image_dir, f'{row[0]}.jpg'), row[1].split(' ')))

    # Check all images exists and create labels list
    labels = []
    for sample in samples:
        if not exists(sample[1]):
            print(f"WARNING: {sample[1]} does not exist")

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


def get_labels_distribution(labels_csv):
    """
    Returns a dict where the keys are the labels and the values are the number of occurences.

    Parameters
    ----------
    labels_csv : str
        Path to the csv file

    Returns
    -------
        dict
    """
    # Read the csv
    sample_labels = []
    with open(labels_csv) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(reader):
            if i < 1:  # Skip first row as it is column names
                continue
            sample_labels.append(row[1].split(' '))

    # Count labels occurences
    labels = {}
    for sample in sample_labels:
        for label in sample:
            if label not in labels:
                labels[label] = 1
            else:
                labels[label] += 1

    return labels
