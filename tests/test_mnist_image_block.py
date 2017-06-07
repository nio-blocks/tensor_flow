from nio.block.terminals import DEFAULT_TERMINAL
from nio.signal.base import Signal
from nio.testing.block_test_case import NIOBlockTestCase
from ..mnist_image_block import MNISTImageLoader
from unittest.mock import patch, MagicMock


class TestMNISTImageLoader(NIOBlockTestCase):

    @patch('tensorflow.contrib.learn.datasets.mnist')
    def test_process_signals(self, mock_images):
        # patch is not working
        blk = MNISTImageLoader()
        self.configure_block(blk, {})
        blk.start()
        blk.process_signals([Signal({"hello": "n.io"})])
        blk.stop()
        self.assert_num_signals_notified(1)
        self.assertDictEqual(
            self.last_notified[DEFAULT_TERMINAL][0].to_dict(),
            {"hello": "n.io"})
