import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # supress TF build warnings
from enum import Enum
from nio.block.base import Block
from nio.block.terminals import input
from nio.properties import VersionProperty, Property, FloatProperty, \
                           PropertyHolder, IntProperty, SelectProperty, \
                           ListProperty, BoolProperty, StringProperty
from nio.signal.base import Signal
import tensorflow as tf


class LossFunctions(Enum):

    cross_entropy = 'cross_entropy'
    softmax_cross_entropy_with_logits = 'softmax_cross_entropy_with_logits'

class ActivationFunctions(Enum):

    softmax = 'softmax'
    sigmoid = 'sigmoid'
    tanh = 'tanh'
    relu = 'relu'
    dropout = 'drouput'

class InitialValues(Enum):

    random = 'truncated_normal'
    zeros = 'zeros'
    ones = 'ones'

class Layers(PropertyHolder):

    count = FloatProperty(title='Number of Neurons (alt: Dropout Percent)',
                          default=10)
    activation = SelectProperty(ActivationFunctions,
                                title='Activation Function',
                                default=ActivationFunctions.softmax)
    initial_weights = SelectProperty(InitialValues,
                                     title='Initial Weight Values',
                                     default=InitialValues.random)
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
    loss = SelectProperty(LossFunctions,
                                   title='Loss Function',
                                   default=LossFunctions.cross_entropy)
    optimizer = StringProperty(title='Optimizer',
                               default='GradientDescentOptimizer')

    def __init__(self):
        super().__init__()
        self.X = None
        self.Y_ = None

    def configure(self, context):
        super().configure(context)
        width, height= self.input_dims()[1:-1]
        tf.set_random_seed(0)
        # input tensors [batch size, width, height, color channels]
        self.X = tf.placeholder(tf.float32, self.input_dims())
        # desired output, labels
        self.Y_ = tf.placeholder(tf.float32, [None, int(self.layers()[-1].count())])
        prev_layer = tf.reshape(self.X, [-1, width * height])
        for i, layer in enumerate(self.layers()):
            name = 'layer{}'.format(i)
            W = tf.Variable(getattr(tf, layer.initial_weights().value)([int(prev_layer.shape[-1]), int(layer.count())]))
            b = tf.Variable(getattr(tf, layer.initial_weights().value)([int(layer.count())]))
            # logits may be used by loss function so we create a variable for
            # each layer before and after activation
            # todo: only for last layer!!!
            if layer.activation().value != 'dropout':
                if layer.bias.value:
                    globals()[name + '_logits'] = tf.matmul(prev_layer, W) + b
                else:
                    globals()[name + '_logits'] = tf.matmul(prev_layer, W)
                globals()[name] = getattr(tf.nn, layer.activation().value)(globals()[name + '_logits'])
            else:
                name = name + '_d'
                globals()[name] = tf.nn.dropout(prev_layer, 1 - layer.count())
            prev_layer = globals()[name]
        Y = globals()['layer{}'.format(len(self.layers()) - 1)]
        Y_logits = globals()['layer{}_logits'.format(len(self.layers()) - 1)]
        # define loss functions
        if self.loss().value == 'cross_entropy':
            self.loss_function = -tf.reduce_mean(self.Y_ * tf.log(Y)) # * 1000.0
        if self.loss().value == 'softmax_cross_entropy_with_logits':
            self.loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y_logits, labels=self.Y_)) # * 100
        self.train_step = getattr(tf.train, self.optimizer())(self.learning_rate()).minimize(self.loss_function)
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
