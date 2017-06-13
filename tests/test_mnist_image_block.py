from unittest.mock import patch, MagicMock, ANY
from nio.block.terminals import DEFAULT_TERMINAL
from nio.signal.base import Signal
from nio.testing.block_test_case import NIOBlockTestCase
from ..mnist_image_block import MNISTImageLoader


class TestMNISTImageLoader(NIOBlockTestCase):

    @patch('tensorflow.examples.tutorials.mnist.input_data.read_data_sets')
    def test_process_signals(self, mock_dataset):
        """For each input signal call next_batch(batch_size) on the 
        corresponding attribute of returned DataSet object.
        """
        blk = MNISTImageLoader()
        self.configure_block(blk, {'batch_size': '{{ $foo }}' })
        blk.start()
        blk.process_signals([Signal({'foo': 10})], input_id='train')
        blk.process_signals([Signal({'foo': 1})], input_id='test')
        blk.stop()
        # todo: assert mock_dataset args
        print(mock_dataset.call_args_list)
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
