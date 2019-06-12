from collections import Counter
import csv
from os.path import join, exists

import tensorflow as tf

from sklearn.preprocessing import MultiLabelBinarizer

tf.enable_eager_execution()


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
        next(reader)  # Skip first row as it is column names
        for row in reader:
            samples.append((row[0], join(image_dir, f'{row[0]}.jpg'), row[1].split(' ')))
            if not exists(samples[-1][1]):
                print(f"WARNING: {samples[-1][1]} does not exist")

    # Create labels list
    labels = sorted(list(set(sum(list(zip(*samples))[2], []))))
    binarizer = MultiLabelBinarizer(classes=labels)
    binarizer.fit(labels)

    # Transform all labels
    transformed_labels = list(map(lambda x: binarizer.transform([x[2]]).squeeze(), samples))

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
        next(reader)  # Skip first row as it is column names
        for row in reader:
            sample_labels.append(row[1].split(' '))

    # Count labels occurences
    labels = Counter(x for xs in sample_labels for x in set(xs))

    return labels
