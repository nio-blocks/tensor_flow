from nio.block.terminals import DEFAULT_TERMINAL
from nio.signal.base import Signal
from nio.testing.block_test_case import NIOBlockTestCase
from ..mnist_image_block import MNISTImageLoader
from unittest.mock import patch, MagicMock


class TestMNISTImageLoader(NIOBlockTestCase):

    @patch('tensorflow.examples.tutorials.mnist.input_data.read_data_sets')
    # @patch('tensorflow.contrib.learn.datasets.mnist.DataSet')
    def test_process_signals(self, mock_read):
        mock_arrays = (MagicMock(), MagicMock())
        mock_read.return_value.train.next_batch.return_value = mock_arrays
        mock_read.return_value.test.next_batch.return_value = mock_arrays
        blk = MNISTImageLoader()
        self.configure_block(blk, {})
        blk.start()
        blk.process_signals([Signal({'foo': 'bar'})])
        blk.stop()
        self.assert_num_signals_notified(1)
        self.assertDictEqual(
            self.last_notified[DEFAULT_TERMINAL][0].to_dict(),
            {'batch': mock_arrays})
        print(mock_read.call_args_list)
