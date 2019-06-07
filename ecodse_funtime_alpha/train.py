import tensorflow as tf

import models
import data

tf.enable_eager_execution()


def train_loop(dataset, model, optimizer):
    for x, y in dataset.batch(5):
        with tf.GradientTape() as tape:
            predictions = model(tf.squeeze(x))
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(y, tf.float32), logits=predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        print(loss)


def fit_loop(dataset, model, optimizer):
    dataset = dataset.shuffle(12)
    dataset = dataset.repeat(20)
    dataset = dataset.batch(4)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=["accuracy"])
    model.fit(dataset, epochs=20, steps_per_epoch=2)


if __name__ == "__main__":
    image_dir = '../../rainforest/fixed-train-jpg/'
    labels_csv = '../../rainforest/train_v3.csv'
    dataset = data.get_dataset(image_dir, labels_csv)
    # model = models.TestMLP(10, 9)
    model = models.SimpleCNN(10, 3, 9)
    optimizer = tf.keras.optimizers.Adam(lr=0.001)
    fit_loop(dataset, model, optimizer)
    # nepoch = 2
    # for _ in range(nepoch):
    #     train_loop(dataset, model, optimizer)
