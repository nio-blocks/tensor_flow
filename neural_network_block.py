from nio.block.base import Block
from nio.block.terminals import input
from nio.properties import *
from enum import Enum
import tensorflow as tf


# not used just yet, hardocoding mnist project values for now...
# class ActivationFunctions(Enum):
    # softmax = 'softmax'

# class Layers(PropertyHolder):

    # count = IntProperty(title='Number of Neurons', default=10)
    # activation = SelectProperty(ActivationFunctions,
                                # title='Activation Function',
                                # default=ActivationFunctions.softmax)

@input('predict')
@input('train')
class NeuralNetwork(Block):

    version = VersionProperty('0.1.0')
    # layers = ListProperty(Layers, title='Network Layers', default=[])

    # set random seed for repeatable computations
    tf.set_random_seed(0)
    # input images [minibatch size, height, width, color channels]
    # todo: verify order of height/width args
    X = tf.placeholder(tf.float32, [None, 28, 28, 1])
    # desired output
    Y_ = tf.placeholder(tf.float32, [None, 10])
    # weights, 784 inputs to 10 neurons
    W = tf.Variable(tf.zeros([784, 10]))
    # biases, one per neuron
    b = tf.Variable(tf.zeros([10]))
    # flatten images
    XX = tf.reshape(X, [-1, 784])

    def __init__(self):
        self.loss_function = None
        self.train_Step = None
        self.sess = None
        self.correct_prediction = None
        self.accuracy = None
        super().__init__()

    def start(self):
        # build the model, Y = computed output
        Y = tf.nn.softmax(tf.matmul(self.XX, self.W) + self.b)
        # define loss function
        self.loss_function = -tf.reduce_mean(self.Y_ * tf.log(Y)) * 1000.0
        # define training step
        self.train_step = tf.train.GradientDescentOptimizer(0.005).minimize(self.loss_function)
        # define accuracy functions
        self.correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(self.Y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        # initialize model
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        super().start()

    def process_signals(self, signals, input_id=None):
        for signal in signals:
            if input_id == 'train':
                self._training(signal)
        self.notify_signals(signals)

    def _training(self, signal):
        batch_X, batch_Y = signal.batch
        self.sess.run(self.train_step,
                      feed_dict={self.X: batch_X, self.Y_: batch_Y})
