import tensorflow as tf
import numpy as np
import models
from data import AmazonDataset


def train_loop(dataset, model, optimizer):
    for x in dataset.img_ds.batch(4):
        gt = np.zeros((x.shape[0], 3))
        gt[:,0] = 1
        gt = tf.convert_to_tensor(gt)  
        with tf.GradientTape() as tape:
            predictions = model(tf.reshape(x, [x.shape[0], -1]))
            loss = tf.keras.metrics.categorical_crossentropy(gt, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)        
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

if __name__=="__main__":
    image_dir = 'rainforest/fixed-train-jpg/'
    labels_csv = 'rainforest/train_v3.csv'
    dataset = AmazonDataset(image_dir, labels_csv)
    model = models.testMLP(256*256*3, 10, 3)
    optimizer = tf.keras.optimizers.Adam(lr=0.001)
    nepoch = 2 
    for _ in range(nepoch):
        train_loop(dataset, model, optimizer)
    