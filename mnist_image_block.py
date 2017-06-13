from nio.block.base import Block
from nio.block.terminals import input
from nio.properties import *
from nio.signal.base import Signal
from tensorflow.examples.tutorials.mnist import input_data as mnist_data


@input('test')
@input('train')
class MNISTImageLoader(Block):

    """Creates 4-Dimensional numpy arrays from MNIST handwriting dataset using
    data and code examples included with TensorFlow 1.1.0. Image data is 
    stored in `data/` and will be downloaded if it does not exist.

    Each signal processed by `train` or `test` input loads the next 
    `batch_size` images from the corresponding dataset. The output signal is 
    ready to be used by a NeuralNetwork block.

    The data contains 60,000 images with labels for training, and another
    10,000 for testing. Additional information on this dataset: 
    http://yann.lecun.com/exdb/mnist/

    Properties:
        `batch_size` (int): How many images and labels to load per signal
        `shuffle`: If True the contents of each batch will be in random order

    """

    version = VersionProperty('0.1.0')
    batch_size = Property(title='Images per Batch', default=100)
    shuffle = BoolProperty(title='Shuffle Images', default=True, visible=False)
    # todo: validation_size prop

    def start(self):
        self.mnist = mnist_data.read_data_sets('data',
                                               one_hot=True,
                                               reshape=False,
                                               validation_size=0)
        super().start()

    def process_signals(self, signals, input_id=None):
        # todo: notify a list of new signals instead of new list for each 
        # signal
        # todo: signal attr for epoch count
        for signal in signals:
            kwargs = {'batch_size': self.batch_size(signal),
                      'shuffle': self.shuffle(signal)}
            if input_id == 'train':
                self.notify_signals(
                    [Signal({'batch': self.mnist.train.next_batch(**kwargs)})])
            else:
                self.notify_signals(
                    [Signal({'batch': self.mnist.test.next_batch(**kwargs)})])
