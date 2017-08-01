import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # supress TF build warnings
from enum import Enum
from nio.block.base import Block
from nio.block.terminals import input
from nio.properties import VersionProperty, Property, FloatProperty, \
                           PropertyHolder, IntProperty, SelectProperty, \
                           ListProperty, BoolProperty
from nio.signal.base import Signal
import tensorflow as tf


# not used just yet, hardocoding mnist project values for now...
class ActivationFunctions(Enum):
    softmax = 'softmax'

class Layers(PropertyHolder):

    count = IntProperty(title='Number of Neurons', default=10)
    activation = SelectProperty(ActivationFunctions,
                                title='Activation Function',
                                default=ActivationFunctions.softmax)
    bias = BoolProperty(title='Add Bias Unit', default=True)

@input('predict')
@input('test')
@input('train')
class NeuralNetwork(Block):

    version = VersionProperty('0.1.0')
    input_dims = Property(title='Input Tensor Dimensions',
                          default='{{ [None, 28, 28, 1] }}',
                          visible=False)
    learning_rate = FloatProperty(title='Learning Rate', default=0.005)
    layers = ListProperty(Layers, title='Network Layers', default=[])

    def __init__(self):
        super().__init__()
        self.X = None
        self.Y_ = None

    def configure(self, context):
        super().configure(context)
        width, height= self.input_dims()[1:-1]
        pixels = width * height
        tf.set_random_seed(0)
        # input images [minibatch size, width, height, color channels]
        self.X = tf.placeholder(tf.float32, self.input_dims())
        # desired output
        self.Y_ = tf.placeholder(tf.float32, [None, 10])
        #################
        prev_layer = tf.reshape(self.X, [-1, pixels])
        for i, layer in enumerate(self.layers()):
            name = 'layer{}'.format(i)
            globals()[name] = getattr(tf.nn, layer.activation)(inputs=prev_layer,
                                                               units=layer['count'],
                                                               activation=layer['activation'],
                                                               use_bias=layer['bias'],
                                                               name=name)
            prev_layer = layer
        #################
        # weights, 784 inputs to 10 neurons
        W = tf.Variable(tf.zeros([pixels, 10]))
        # biases, one per neuron
        b = tf.Variable(tf.zeros([10]))
        # flatten images
        XX = tf.reshape(self.X, [-1, pixels])
        # build the model, Y = computed output
        Y = tf.nn.softmax(tf.matmul(XX, W) + b)
        # define loss function, cross-entropy
        self.loss_function = -tf.reduce_mean(self.Y_ * tf.log(Y)) * 1000.0
        # define training step
        self.train_step = tf.train.GradientDescentOptimizer(
            self.learning_rate()).minimize(self.loss_function)
        # define accuracy functions
        self.correct_prediction = tf.equal(tf.argmax(Y, 1),
                                           tf.argmax(self.Y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction,
                                       tf.float32))
        self.prediction = (tf.argmax(Y, 1), Y)
        # initialize model
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def process_signals(self, signals, input_id=None):
        for signal in signals:
            if input_id == 'train':
                acc, loss = self._train(signal)[1:]
                self.notify_signals([Signal({'input_id': input_id,
                                             'accuracy': acc, 
                                             'loss': loss})])
            elif input_id == 'test':
                acc, loss = self._test(signal)
                self.notify_signals([Signal({'input_id': input_id,
                                             'accuracy': acc,
                                             'loss': loss})])
            else:
                predict = self._predict(signal)
                self.notify_signals([Signal({'input_id': input_id,
                                             'prediction': predict})])

    def stop(self):
        # todo: use context manager and remove this
        self.sess.close()
        super().stop()

    def _train(self, signal):
        batch_X = signal.images
        batch_Y = signal.labels
        return self.sess.run(
            [self.train_step, self.accuracy, self.loss_function],
            feed_dict={self.X: batch_X, self.Y_: batch_Y})

    def _test(self, signal):
        batch_X = signal.images
        batch_Y = signal.labels
        return self.sess.run(
            [self.accuracy, self.loss_function],
            feed_dict={self.X: batch_X, self.Y_: batch_Y})

    def _predict(self, signal):
        batch_X = signal.images
        return self.sess.run(self.prediction, feed_dict={self.X: batch_X})
