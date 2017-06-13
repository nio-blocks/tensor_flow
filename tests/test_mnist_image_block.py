from unittest.mock import patch, MagicMock, ANY
from nio.block.terminals import DEFAULT_TERMINAL
from nio.signal.base import Signal
from nio.testing.block_test_case import NIOBlockTestCase
from ..mnist_image_block import MNISTImageLoader


class TestMNISTImageLoader(NIOBlockTestCase):

    @patch('tensorflow.examples.tutorials.mnist.input_data.read_data_sets')
    def test_process_signals(self, mock_dataset):
        """For each input signal call next_batch(batch_size)"""
        blk = MNISTImageLoader()
        self.configure_block(blk, {'batch_size': '{{ $foo }}' })
        blk.start()
        blk.process_signals([Signal({'foo': 10})], input_id='train')
        blk.process_signals([Signal({'foo': 1})], input_id='test')
        blk.stop()
        # todo: assert mock_dataset args
        self.assert_num_signals_notified(2)
        self.assertDictEqual({'batch': ANY},
                             self.last_notified[DEFAULT_TERMINAL][0].to_dict())
        mock_dataset.return_value.train.next_batch.assert_called_once_with(
            batch_size=10,
            shuffle=True)
        mock_dataset.return_value.test.next_batch.assert_called_once_with(
            batch_size=1,
            shuffle=True)

    @patch('tensorflow.examples.tutorials.mnist.input_data.read_data_sets')
    def test_shuffle_images(self, mock_dataset):
        """Shuffle can be disabled for repeatable output."""
        blk = MNISTImageLoader()
        self.configure_block(blk, {'shuffle': False})
        blk.start()
        blk.process_signals([Signal()], input_id='train')
        blk.stop()
        mock_dataset.return_value.train.next_batch.assert_called_once_with(
            batch_size=100,
            shuffle=False)

class TestSignalLists(NIOBlockTestCase):

    def signals_notified(self, block, signals, output_id):
        """Override so that last_notified is list of signal lists instead of
        list of signals"""
        self.last_notified[output_id].append(signals)

    @patch('tensorflow.examples.tutorials.mnist.input_data.read_data_sets')
    def test_signal_lists(self, mock_dataset):
        """Notify a list of signals of equal length to signals received"""
        input_signals = [Signal()] * 3
        blk = MNISTImageLoader()
        self.configure_block(blk, {})
        blk.start()
        blk.process_signals(input_signals, input_id='train')
        blk.stop()
        # assert that one list has been notified ...
        self.assertEqual(len(self.last_notified[DEFAULT_TERMINAL]), 1)
        # ... and it contains 3 signals
        self.assertEqual(len(self.last_notified[DEFAULT_TERMINAL][-1]),
                         len(input_signals))
