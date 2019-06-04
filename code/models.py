import tensorflow as tf


class testMLP(tf.keras.Model):
    def __init__(self, hiddensize, outsize):
        super(testMLP, self).__init__(self)
        self.model = tf.keras.Sequential([
            #tf.keras.layers.Reshape((256*256*3,)),
            tf.keras.layers.Dense(hiddensize, input_shape=(256*256*3,), activation=tf.nn.relu),
            tf.keras.layers.Dense(outsize, activation=tf.nn.softmax)
        ])
    
    def call(self, inputs):
        print(inputs)
        return self.model(inputs)


if __name__=="__main__":
    testmodel = testMLP(5, 10, 2)
    testin = tf.random.uniform([7,5])
    print(testmodel(testin))