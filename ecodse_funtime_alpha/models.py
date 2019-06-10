import tensorflow as tf

tf.enable_eager_execution()


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
            tf.keras.layers.Dense(outsize)
        ])

    def call(self, inputs):
        return self.model(inputs)


if __name__ == "__main__":
    testmodel = TestMLP(5, 2)
    testin = tf.random.uniform([7, 256 * 256 * 3])
    print(testmodel(testin))
    testmodel2 = SimpleCNN(2, 2, 4)
    testin2 = tf.random.uniform([10, 32, 32, 1])
    print(testmodel2(testin2))
