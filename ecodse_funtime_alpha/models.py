import tensorflow as tf

tf.enable_eager_execution()


class TestMLP(tf.keras.Model):
    """
    A fully connected neural network with 1 hidden layer.

    Attributes
    ----------
    model : tf.keras.Model
        sequential model made of 1 hidden layer with ReLU activation and 1 output layer.

    Methods
    -------
    call:
        inherit from keras
        Usage example: model(x)
    """

    def __init__(self, hiddensize, outsize):
        """
        Class construction

        Parameters
        ----------
        hiddensize : int
            size of the hidden layer
        outsize : int
            size of the output layer
        """
        super(TestMLP, self).__init__(self)
        self.model = tf.keras.Sequential([
            # tf.keras.layers.Reshape((256*256*3,)),
            tf.keras.layers.Dense(hiddensize, input_shape=(256 * 256 * 3,), activation=tf.nn.relu),
            tf.keras.layers.Dense(outsize, activation=tf.nn.sigmoid)
        ])

    def call(self, inputs):
        """
        forward propagation function of the model

        Parameters
        ----------
        inputs : tf.tensor shape = (batchsize, 256 * 256 * 3)
            mini-batch of batchsize flatten images

        Returns
        -------
        tf.tensor shape = (batchsize, outsize)
            logits prediction of the model for the outsize classes
        """
        return self.model(inputs)


class SimpleCNN(tf.keras.Model):
    """
    A convolutional neural network with 1 conv layer, 1 maxpool, and 1 fully-connected output layer.

    Attributes
    ----------
    model : tf.keras.Model
        sequential model made of 1 conv layer, 1 maxpool layer and 1 dense layer

    Methods
    -------
    call:
        inherit from keras
        Usage example: model(x)
    """

    def __init__(self, nkernel, kernelsize, outsize):
        """
        Class constructor

        Parameters
        ----------
        nkernel : int
            number of kernels in the convolutional layer
        kernelsize : int
            size of the kernels in the convolutional layer
        outsize : int
            number of dimensions of the output
        """
        super(SimpleCNN, self).__init__(self)
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(nkernel, kernelsize, input_shape=(256, 256, 3,), activation=tf.nn.relu),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(outsize)
        ])

    def call(self, inputs):
        """
        Forward function for the model

        Parameters
        ----------
        inputs : tf.tensor shape = (batchsize, 256, 256, 3)
            mini-batch of batchsize images of size 256 x 256 x 3

        Returns
        -------
        tf.tensor shape = (batchsize, outsize)
            logits prediction of the model for the outsize classes
        """
        return self.model(inputs)
