from nio.block.base import Block
from nio.block.terminals import input
from nio.properties import *
from nio.signal.base import Signal
from tensorflow.examples.tutorials.mnist import input_data as mnist_data


@input('test')
@input('train')
class MNISTImageLoader(Block):

    """Generates pixel data and labels from MNIST handwriting dataset. 
    If not already present in `data/` the source data will be downloaded 
    automatically. The output signal is ready to use by a NeuralNetwork 
    block.

    Each signal processed loads the next `batch_size` images from the 
    dataset corresponding to `input_id`.
    """

    version = VersionProperty('0.1.0')
    batch_size = Property(title='Images per Batch', default=100)
    shuffle = BoolProperty(title='Shuffle Batch', default=True, visible=False)
    # todo: validation_size prop

    def start(self):
        super().start()
        self.mnist = mnist_data.read_data_sets('data',
                                               one_hot=True,
                                               reshape=False,
                                               validation_size=0)

    def process_signals(self, signals, input_id=None):
        output_signals = []
        for signal in signals:
            kwargs = {'batch_size': self.batch_size(signal),
                      'shuffle': self.shuffle(signal)}
            output_signals.append(Signal(
                {'batch': getattr(self.mnist, input_id).next_batch(**kwargs)}))
        self.notify_signals(output_signals)
