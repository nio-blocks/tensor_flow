from nio.block.base import Block
from nio.block.terminals import input
from nio.properties import IntProperty, BoolProperty, VersionProperty
from nio.signal.base import Signal
from nio.block.mixins.enrich.enrich_signals import EnrichSignals
from tensorflow.examples.tutorials.mnist import input_data as mnist_data


@input('test')
@input('train')
class MNISTImageLoader(EnrichSignals, Block):

    """Generates pixel data and labels from MNIST handwriting dataset.
    If not already present in `data/` the source data will be downloaded
    automatically. The output signal is ready to use by a NeuralNetwork
    block.

    Each signal processed loads the next `batch_size` images from the
    dataset corresponding to `input_id`.
    """

    version = VersionProperty('0.2.0')
    batch_size = IntProperty(title='Images per Batch', default=100)
    shuffle = BoolProperty(title='Shuffle Batch', default=True, visible=False)

    def __init__(self):
        super().__init__()
        self.mnist = None

    def configure(self, context):
        super().configure(context)
        self.mnist = mnist_data.read_data_sets(
            'data',
            one_hot=True,
            reshape=True,
            validation_size=0)

    def process_signals(self, signals, input_id=None):
        output_signals = []
        for signal in signals:
            kwargs = {'batch_size': self.batch_size(signal),
                      'shuffle': self.shuffle(signal)}
            batch = getattr(self.mnist, input_id).next_batch(**kwargs)
            new_signal = self.get_output_signal(
                {'batch': batch[0], 'labels': batch[1], 'input_id': input_id},
                signal)
            output_signals.append(new_signal)
        self.notify_signals(output_signals)
