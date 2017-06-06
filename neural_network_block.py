from nio.block.base import Block
from nio.properties import *
from enum import Enum
import tensorflow as tf


# class ActivationFunctions(Enum):
    # softmax = 'softmax'

# class Layers(PropertyHolder):

    # count = IntProperty(default=1, title='Number of Neurons')
    # activation = SelectProperty(ActivationFunctions,
                                # title='Activation Function',
                                # default=ActivationFunctions.softmax)

class NeuralNetwork(Block):

    version = VersionProperty('0.1.0')
    # layers = ListProperty(Layers, title='Network Layers', default=[])

    # set random seed for repeatable computations
    tf.set_random_seed(0)
    # input images [minibatch size, width, height, color channels]
    X = tf.placeholder(tf.float32, [None, 28, 28, 1])
    # desired output
    Y_ = tf.placeholder(tf.float32, [None, 10])
    # weights, 784 inputs to 10 neurons
    W = tf.Variable(tf.zeros([784, 10]))
    # biases, one per neuron
    b = tf.Variable(tf.zeros([10]))
    # flatten images
    XX = tf.reshape(X, [-1, 784])

    def start(self):
        # build the model, Y = computed output
        Y = tf.nn.softmax(tf.matmul(self.XX, self.W) + self.b)
        # define loss function
        cross_entropy = -tf.reduce_mean(self.Y_ * tf.log(Y)) * 1000.0
        # initialize model
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        super().start()

    def process_signals(self, signals):
        for signal in signals:
            pass
        self.notify_signals(signals)
