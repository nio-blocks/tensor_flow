import os
from enum import Enum

from nio.block.base import Block
from nio.block.terminals import input
from nio.block.mixins.enrich.enrich_signals import EnrichSignals
from nio.properties import VersionProperty, Property, FloatProperty, \
                           PropertyHolder, IntProperty, SelectProperty, \
                           ListProperty, BoolProperty, ObjectProperty
from nio.signal.base import Signal

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # supress TF build warnings


class LossFunctions(Enum):
    cross_entropy = 'cross_entropy'
    softmax_cross_entropy_with_logits = 'softmax_cross_entropy_with_logits'
    mean_absolute_error = 'mean_absolute_error'


class Optimizers(Enum):
    GradientDescentOptimizer = 'GradientDescentOptimizer'
    ProximalGradientDescentOptimizer = 'ProximalGradientDescentOptimizer'
    AdadeltaOptimizer = 'AdadeltaOptimizer'
    AdagradOptimizer = 'AdagradOptimizer'
    ProximalAdagradOptimizer = 'ProximalAdagradOptimizer'
    AdagradDAOptimizer = 'AdagradDAOptimizer'
    MomentumOptimizer = 'MomentumOptimizer'
    AdamOptimizer = 'AdamOptimizer'
    FtrlOptimizer = 'FtrlOptimizer'
    RMSPropOptimizer = 'RMSPropOptimizer'


class ActivationFunctions(Enum):
    softmax = 'softmax'
    softplus = 'softplus'
    softsign = 'softsign'
    sigmoid = 'sigmoid'
    tanh = 'tanh'
    elu = 'elu'
    relu = 'relu'
    relu6 = 'relu6'
    crelu = 'crelu'
    dropout = 'dropout'
    bias_add = 'bias_add'


class InitialValues(Enum):
    random = 'truncated_normal'
    zeros = 'zeros'
    ones = 'ones'


class Layers(PropertyHolder):
    count = IntProperty(title='Number of Neurons',
                        default=10)
    activation = SelectProperty(ActivationFunctions,
                                title='Activation Function',
                                default=ActivationFunctions.softmax)
    initial_weights = SelectProperty(InitialValues,
                                     title='Initial Weight Values',
                                     default=InitialValues.random)
    bias = BoolProperty(title='Add Bias Unit', default=True)


class NetworkConfig(PropertyHolder):
    input_dim = IntProperty(title='Number of Inputs',
                            default=784)
    learning_rate = FloatProperty(title='Learning Rate', default=0.005)
    loss = SelectProperty(LossFunctions,
                          title='Loss Function',
                          default=LossFunctions.cross_entropy)
    optimizer = SelectProperty(Optimizers,
                               title="Optimizer",
                               default=Optimizers.GradientDescentOptimizer)
    dropout = FloatProperty(title='Dropout Percentage During Training',
                            default=0)
    random_seed = IntProperty(title="Random seed", default=0, visible=False)


@input('predict')
@input('test')
@input('train')
class NeuralNetwork(EnrichSignals, Block):

    layers = ListProperty(Layers,
                          title='Network Layers',
                          default=[{'count': 10,
                                    'activation': 'softmax',
                                    'initial_weights': 'random',
                                    'bias': True}])
    network_config = ObjectProperty(NetworkConfig,
                                    title='ANN Configuration',
                                    defaul=NetworkConfig())
    version = VersionProperty('0.3.0')

    def __init__(self):
        super().__init__()
        self.X = None
        self.Y_ = None
        self.prob_keep = None
        self.train_step = None
        self.correct_prediction = None
        self.accuracy = None
        self.prediction = None
        self.sess = None
        self.loss_function = None

    def configure(self, context):
        super().configure(context)
        tf.set_random_seed(self.network_config().random_seed())
        # input tensors shape
        self.X = tf.placeholder(tf.float32,
                                shape=[None,
                                       self.network_config().input_dim()])
        # specify desired output (labels)
        self.Y_ = tf.placeholder(tf.float32,
                                 shape=[None, self.layers()[-1].count()])
        self.prob_keep = tf.placeholder(tf.float32)

        layers_logits = {}
        prev_layer = self.X
        for i, layer in enumerate(self.layers()):
            name = 'layer{}'.format(i)
            with tf.name_scope(name):
                W = tf.Variable(
                    getattr(tf, layer.initial_weights().value)
                    ([int(prev_layer.shape[-1]), layer.count()]))
                b = tf.Variable(
                    getattr(tf, layer.initial_weights().value)
                    ([layer.count()]))

                if layer.activation().value != 'dropout':
                    if i == (len(self.layers()) - 1):
                        # calculate logits seperately for use by loss function
                        if layer.bias.value:
                            layers_logits[name + '_logits'] = \
                                tf.matmul(prev_layer, W) + b
                        else:
                            layers_logits[name + '_logits'] = \
                                tf.matmul(prev_layer, W)
                        layers_logits[name] = \
                            getattr(tf.nn, layer.activation().value) \
                                (layers_logits[name + '_logits'])
                    else:
                        if layer.bias.value:
                            logits = tf.matmul(prev_layer, W) + b
                        else:
                            logits = tf.matmul(prev_layer, W)
                        layers_logits[name] = \
                            getattr(tf.nn, layer.activation().value)(logits)
                else:
                    name = 'layer{}_d'.format(i)
                    layers_logits[name] = tf.nn.dropout(prev_layer,
                                                        self.prob_keep)
                prev_layer = layers_logits[name]

        output_layer_num = len(self.layers()) - 1
        Y = layers_logits['layer{}'.format(output_layer_num)]
        Y_logits = layers_logits['layer{}_logits'.format(output_layer_num)]
        self.accuracy = 1 - tf.reduce_mean(abs(self.Y_ - Y))
        if self.network_config().loss().value == 'cross_entropy':
            self.loss_function = -tf.reduce_mean(self.Y_ * tf.log(Y))
        if self.network_config().loss().value == \
                'softmax_cross_entropy_with_logits':
            self.loss_function = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=Y_logits,
                                                        labels=self.Y_))
        if self.network_config().loss().value == 'mean_absolute_error':
            self.loss_function = tf.reduce_mean(abs(self.Y_ - Y))
        self.train_step = \
            getattr(tf.train, self.network_config().optimizer().value) \
            (self.network_config().learning_rate()).minimize(
                self.loss_function)
        self.prediction = Y
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def process_signals(self, signals, input_id=None):
        new_signals = []
        for signal in signals:
            if input_id == 'train':
                acc, loss = self._train(signal)[1:]
                new_signals.append(self.get_output_signal(
                    {'input_id': input_id,
                     'accuracy': acc,
                     'loss': loss,
                     'prediction': None},
                     signal))
            elif input_id == 'test':
                acc, loss = self._test(signal)
                new_signals.append(self.get_output_signal(
                    {'input_id': input_id,
                     'accuracy': acc,
                     'loss': loss,
                     'prediction': None},
                     signal))
            else:
                predict = self._predict(signal)
                new_signals.append(self.get_output_signal(
                    {'input_id': input_id,
                     'accuracy': None,
                     'loss': None,
                     'prediction': predict},
                     signal))
        self.notify_signals(new_signals)

    def stop(self):
        # todo: use context manager and remove this
        self.sess.close()
        super().stop()

    def _train(self, signal):
        batch_X = signal.batch
        batch_Y_ = signal.labels
        return self.sess.run(
            [self.train_step, self.accuracy, self.loss_function],
            feed_dict={self.X: batch_X,
                       self.Y_: batch_Y_,
                       self.prob_keep: 1 - self.network_config().dropout()})

    def _test(self, signal):
        batch_X = signal.batch
        batch_Y_ = signal.labels
        return self.sess.run(
            [self.accuracy, self.loss_function],
            feed_dict={self.X: batch_X,
                       self.Y_: batch_Y_,
                       self.prob_keep: 1})

    def _predict(self, signal):
        batch_X = signal.batch
        return self.sess.run(
            self.prediction,
            feed_dict={self.X: batch_X,
                       self.prob_keep: 1})
