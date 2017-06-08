from nio.block.base import Block
from nio.block.terminals import input
from nio.signal.base import Signal
from nio.properties import *
from tensorflow.examples.tutorials.mnist import input_data as mnist_data


@input('test')
@input('train')
class MNISTImageLoader(Block):

    version = VersionProperty('0.1.0')
    batch_size = Property(title='Images per Batch', default=100)

    def start(self):
        self.mnist = mnist_data.read_data_sets('data',
                                               one_hot=True,
                                               reshape=False,
                                               validation_size=0)
        super().start()

    def process_signals(self, signals, input_id=None):
        for signal in signals:
            count = self.batch_size(signal)
            temp = {}
            if input_id == 'train':
                temp['batch'] = self.mnist.train.next_batch(count)
            else:
                temp['batch'] = self.mnist.test.next_batch(count)
            self.notify_signals([Signal(temp)])
