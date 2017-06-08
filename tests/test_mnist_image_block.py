from nio.block.terminals import DEFAULT_TERMINAL
from nio.signal.base import Signal
from nio.testing.block_test_case import NIOBlockTestCase
from ..mnist_image_block import MNISTImageLoader
from unittest.mock import patch, MagicMock, ANY


class TestMNISTImageLoader(NIOBlockTestCase):

    @patch('tensorflow.examples.tutorials.mnist.input_data.read_data_sets')
    def test_process_signals(self, mock_read):
        blk = MNISTImageLoader()
        self.configure_block(blk, {'batch_size': '{{ $foo }}' })
        blk.start()
        blk.process_signals([Signal({'foo': 10})], input_id='train')
        blk.process_signals([Signal({'foo': 1})], input_id='test')
        blk.stop()
        self.assert_num_signals_notified(2)
        self.assertDictEqual({'batch': ANY},
                             self.last_notified[DEFAULT_TERMINAL][0].to_dict())
        mock_read.return_value.train.next_batch.assert_called_once_with(10)
        mock_read.return_value.test.next_batch.assert_called_once_with(1)
