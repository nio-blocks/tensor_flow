from unittest.mock import patch, MagicMock, ANY
from nio.block.terminals import DEFAULT_TERMINAL
from nio.signal.base import Signal
from nio.testing.block_test_case import NIOBlockTestCase
from ..neural_network_block import NeuralNetwork


class TestNeuralNetworkBlock(NIOBlockTestCase):

    @patch('tensorflow.Session')
    def test_process_signals(self, mock_sess):
        """Signals processed by 'train' input execute one training iteration"""
        mock_sess.return_value.run.return_value = [MagicMock()] * 3
        input_signal = {'batch': (MagicMock(), MagicMock())}
        blk = NeuralNetwork()
        self.configure_block(blk, {})
        blk.start()
        blk.process_signals([Signal(input_signal)], input_id='train')
        blk.stop()
        self.assert_num_signals_notified(1)
        self.assertDictEqual(
            {'loss': ANY, 'accuracy': ANY},
            self.last_notified[DEFAULT_TERMINAL][0].to_dict())
