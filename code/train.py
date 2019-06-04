import tensorflow as tf
import numpy as np
import models
import data


def train_loop(dataset, model, optimizer):
    for x, y in dataset.batch(5):
        with tf.GradientTape() as tape:
            predictions = model(x)
            loss = tf.keras.metrics.MSE(y, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        print(loss)


def fit_loop(dataset, model):
    dataset.batch(4)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss="MSE",
                  metrics=["accuracy"])
    print("Dataset", dataset)
    model.fit(dataset, epochs=20, steps_per_epoch=2)


if __name__ == "__main__":
    image_dir = '../../rainforest/fixed-train-jpg/'
    labels_csv = '../../rainforest/train_v3.csv'
    dataset = data.get_dataset(image_dir, labels_csv)
    model = models.testMLP(10, 9)
    optimizer = tf.keras.optimizers.Adam(lr=0.001)
    fit_loop(dataset, model)
    nepoch = 2
    # for _ in range(nepoch):
    #    train_loop(dataset, model, optimizer)
