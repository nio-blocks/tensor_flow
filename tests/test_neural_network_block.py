from nio.block.terminals import DEFAULT_TERMINAL
from nio.signal.base import Signal
from nio.testing.block_test_case import NIOBlockTestCase
from ..neural_network_block import NeuralNetwork
from unittest.mock import patch, MagicMock, ANY


class TestNeuralNetworkBlock(NIOBlockTestCase):

    @patch('tensorflow.Session')
    def test_process_signals(self, mock_tf):
        """numbers get computered like a motherfucker"""
        input_signal = {'batch': (MagicMock(), MagicMock())}
        blk = NeuralNetwork()
        self.configure_block(blk, {})
        blk.start()
        blk.process_signals([Signal(input_signal)], input_id='train')
        blk.stop()
        self.assert_num_signals_notified(1)
        self.assertDictEqual(
            {'loss': ANY, 'accuracy': ANY, 'iteration': 1},
            self.last_notified[DEFAULT_TERMINAL][0].to_dict())
