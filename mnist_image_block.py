from nio.block.base import Block
from nio.block.terminals import input
from nio.properties import *
from nio.signal.base import Signal
from tensorflow.examples.tutorials.mnist import input_data as mnist_data


@input('test')
@input('train')
class MNISTImageLoader(Block):

    version = VersionProperty('0.1.0')
    batch_size = Property(title='Images per Batch', default=100)
    shuffle = BoolProperty(title='Shuffle Images', default=True, visible=False)

    def start(self):
        self.mnist = mnist_data.read_data_sets('data',
                                               one_hot=True,
                                               reshape=False,
                                               validation_size=0)
        super().start()

    def process_signals(self, signals, input_id=None):
        for signal in signals:
            kwargs = {'batch_size': self.batch_size(signal),
                      'shuffle': self.shuffle(signal)}
            if input_id == 'train':
                self.notify_signals(
                    [Signal({'batch': self.mnist.train.next_batch(**kwargs)})])
            else:
                self.notify_signals(
                    [Signal({'batch': self.mnist.test.next_batch(**kwargs)})])
            self.notify_signals([Signal(temp)])
