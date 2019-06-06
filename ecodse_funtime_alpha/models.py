import tensorflow as tf


class TestMLP(tf.keras.Model):
    def __init__(self, hiddensize, outsize):
        super(TestMLP, self).__init__(self)
        self.model = tf.keras.Sequential([
            # tf.keras.layers.Reshape((256*256*3,)),
            tf.keras.layers.Dense(hiddensize, input_shape=(256 * 256 * 3,), activation=tf.nn.relu),
            tf.keras.layers.Dense(outsize, activation=tf.nn.sigmoid)
        ])

    def call(self, inputs):
        return self.model(inputs)


class SimpleCNN(tf.keras.Model):
    def __init__(self, nkernel, kernelsize, outsize):
        super(SimpleCNN, self).__init__(self)
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(nkernel, kernelsize, input_shape=(256, 256, 3,), activation=tf.nn.relu),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(outsize, activation=tf.nn.sigmoid)
        ])

    def call(self, inputs):
        return self.model(inputs)


class RetrainResnet(tf.keras.Model):
    def __init__(self):
        super(RetrainResnet, self).__init__(self)
        base_model = tf.keras.applications.ResNet50(input_shape=(256, 256, 3),
                                                    include_top=False,
                                                    weights='imagenet')
        base_model.trainable = False
        global_avg_layer = tf.keras.layers.GlobalAveragePooling2D()
        pred_layer = tf.keras.layers.Dense(9, activation=tf.nn.sigmoid)
        self.model = tf.keras.Sequential([
            base_model,
            global_avg_layer,
            pred_layer
        ])

    def call(self, inputs):
        return self.model(inputs)


if __name__ == "__main__":
    testmodel = TestMLP(5, 2)
    testin = tf.random.uniform([7, 256 * 256 * 3])
    print(testmodel(testin))
    testin3 = tf.random.uniform([2, 256, 256, 3])
    testres = RetrainResnet()
    print(testres(testin3))
