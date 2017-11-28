import os
from enum import Enum
from nio.block.base import Block
from nio.block.terminals import input
from nio.block.mixins.enrich.enrich_signals import EnrichSignals
from nio.properties import VersionProperty, FloatProperty, StringProperty, \
                           PropertyHolder, IntProperty, SelectProperty, \
                           ListProperty, BoolProperty, ObjectProperty
import tensorflow as tf


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # supress TF build warnings

class LossFunctions(Enum):
    # todo: need a better way for user to define loss
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
    input_dim = IntProperty(title='Number of Inputs', default=784)
    learning_rate = FloatProperty(title='Learning Rate', default=0.01)
    loss = SelectProperty(LossFunctions,
                          title='Loss Function',
                          default=LossFunctions.cross_entropy)
    optimizer = SelectProperty(Optimizers,
                               title="Optimizer",
                               default=Optimizers.GradientDescentOptimizer)
    dropout = FloatProperty(title='Dropout Percentage During Training',
                            default=0)
    random_seed = IntProperty(title="Random Seed", default=0, visible=False)


class ModelManagement(PropertyHolder):

    default_tag = '{{ datetime.datetime.now().strftime(\'%Y%m%d%H%M%S\') }}'
    save_file = StringProperty(title='Save Weights to File',
                               default='',
                               allow_none=True)
    load_file = StringProperty(title='Load Weights From File',
                               default='',
                               allow_none=True)
    tensorboard_int = IntProperty(title='Train Steps per TensorBoard Record',
                                  default=0)
    tensorboard_tag = StringProperty(title='TensorBoard Run Label',
                                     default=default_tag,
                                     allow_none=True,
                                     visible=False)
    tensorboard_dir = StringProperty(title='TensorBoard File Directory',
                                     default='tensorboard',
                                     visible=False)

@input('predict')
@input('test')
@input('train')
class TensorFlow(EnrichSignals, Block):

    layers = ListProperty(Layers,
                          title='Network Layers',
                          default=[{'count': 10,
                                    'activation': 'softmax',
                                    'initial_weights': 'random',
                                    'bias': True}])
    network_config = ObjectProperty(NetworkConfig,
                                    title='ANN Configuration',
                                    defaul=NetworkConfig())
    models = ObjectProperty(ModelManagement,
                            title='Model Management',
                            default=ModelManagement())
    version = VersionProperty('0.4.0')

    def __init__(self):
        super().__init__()
        self.X = None
        self.Y_ = None
        self.prob_keep = None
        self.train_step = None
        self.correct_prediction = None
        self.prediction = None
        self.sess = None
        self.loss_function = None
        self.saver = None
        self.iter = 0
        self.summaries = None
        self.summary_writer = None

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
                if layer.activation().value != 'dropout':
                    W = tf.Variable(
                        getattr(tf, layer.initial_weights().value)
                        ([int(prev_layer.shape[-1]), layer.count()]))
                    b = tf.Variable(
                        getattr(tf, layer.initial_weights().value)
                        ([layer.count()]))
                    if self.models().tensorboard_int():
                        with tf.name_scope('weights'):
                            tf.summary.histogram('weights', W)
                        with tf.name_scope('biases'):
                            tf.summary.histogram('biases', b)
                    if i == (len(self.layers()) - 1):
                        # calculate logits seperately for use by loss function
                        if layer.bias.value:
                            layers_logits[name + '_logits'] = \
                                tf.matmul(prev_layer, W) + b
                        else:
                            layers_logits[name + '_logits'] = \
                                tf.matmul(prev_layer, W)
                        layers_logits[name] = getattr(
                            tf.nn,
                            layer.activation().value
                        )(layers_logits[name + '_logits'])
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
        if self.network_config().loss().value == 'cross_entropy':
            self.loss_function = tf.reduce_mean(abs(self.Y_ * tf.log(Y)))
        if self.network_config().loss().value == \
                'softmax_cross_entropy_with_logits':
            self.loss_function = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=Y_logits,
                                                        labels=self.Y_))
        if self.network_config().loss().value == 'mean_absolute_error':
            self.loss_function = tf.reduce_mean(abs(self.Y_ - Y))
        if self.models().tensorboard_int():
            with tf.name_scope('loss'):
                tf.summary.scalar(self.network_config().loss().value,
                                  self.loss_function)
        self.train_step = getattr(
            tf.train,
            self.network_config().optimizer().value)(
                self.network_config().learning_rate()
            ).minimize(self.loss_function)
        self.prediction = Y
        if self.models().load_file() or self.models().save_file():
            self.saver = tf.train.Saver(max_to_keep=None)
        self.sess = tf.Session()
        if self.models().tensorboard_int():
            label = self.models().tensorboard_tag()
            self.summaries = tf.summary.merge_all()
            self.summary_writer = tf.summary.FileWriter(
                '{}/{}'.format(self.models().tensorboard_dir(), label),
                self.sess.graph)
            self.logger.debug('TensorBoard summary label: {}'.format(label))
        if self.models().load_file():
            self.saver.restore(self.sess, self.models().load_file())
        else:
            self.sess.run(tf.global_variables_initializer())

    def process_signals(self, signals, input_id=None):
        new_signals = []
        for signal in signals:
            if input_id == 'train':
                if self.models().tensorboard_int():
                    if self.iter % self.models().tensorboard_int() == 0:
                        summary, _, loss, predict = self._train(signal)
                        self.summary_writer.add_summary(summary, self.iter)
                    else:
                        _, loss, predict = self._train(signal)
                    self.iter += 1
                else:
                    _, loss, predict = self._train(signal)
                output = {'input_id': input_id,
                          'loss': loss,
                          'prediction': predict}
                new_signals.append(self.get_output_signal(output, signal))
            elif input_id == 'test':
                loss, predict = self._test(signal)
                output = {'input_id': input_id,
                          'loss': loss,
                          'prediction': predict}
                new_signals.append(self.get_output_signal(output, signal))
            else:
                predict = self._predict(signal)
                output = {'input_id': input_id,
                          'loss': None,
                          'prediction': predict}
                new_signals.append(self.get_output_signal(output, signal))
        self.notify_signals(new_signals)

    def stop(self):
        if self.models().save_file():
            self.logger.debug('saving model to {}'.format(
                self.models().save_file()))
            self.saver.save(self.sess, self.models().save_file())
        if self.models().tensorboard_int():
            self.summary_writer.close()
        self.sess.close()
        super().stop()

    def _train(self, signal):
        batch_X = signal.batch
        batch_Y_ = signal.labels
        fetches = [self.train_step, self.loss_function, self.prediction]
        dropout_rate = 1 - self.network_config().dropout()
        if self.models().tensorboard_int():
            if self.iter % self.models().tensorboard_int() == 0:
                fetches = [self.summaries] + fetches
        return self.sess.run(fetches,
                             feed_dict={self.X: batch_X,
                                        self.Y_: batch_Y_,
                                        self.prob_keep: dropout_rate})

    def _test(self, signal):
        batch_X = signal.batch
        batch_Y_ = signal.labels
        fetches = [self.loss_function, self.prediction]
        return self.sess.run(fetches,
                             feed_dict={self.X: batch_X,
                                        self.Y_: batch_Y_,
                                        self.prob_keep: 1})

    def _predict(self, signal):
        batch_X = signal.batch
        fetches = self.prediction
        return self.sess.run(fetches,
                             feed_dict={self.X: batch_X, self.prob_keep: 1})
