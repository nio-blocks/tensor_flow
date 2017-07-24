from unittest.mock import patch, MagicMock, ANY
from nio.block.terminals import DEFAULT_TERMINAL
from nio.signal.base import Signal
from nio.testing.block_test_case import NIOBlockTestCase
from ..neural_network_block import NeuralNetwork


class TestNeuralNetworkBlock(NIOBlockTestCase):

    @patch('tensorflow.Session')
    def test_process_train_signals(self, mock_sess):
        """Signals processed by 'train' input execute one training iteration"""
        mock_sess.return_value.run.return_value = [MagicMock()] * 3
        input_signal = {'batch': (MagicMock(), MagicMock())}
        blk = NeuralNetwork()
        self.configure_block(blk, {})
        blk.start()
        blk.process_signals([Signal(input_signal)], input_id='train')
        blk.stop()
        self.assertEqual(mock_sess.call_count, 1)
        # sess.run() is called in start() and process_signals()
        self.assertEqual(mock_sess.return_value.run.call_count, 2)
        self.assertEqual(mock_sess.return_value.close.call_count, 1)
        self.assert_num_signals_notified(1)
        self.assertDictEqual(
            {'loss': ANY, 'accuracy': ANY, 'input_id': 'train'},
            self.last_notified[DEFAULT_TERMINAL][0].to_dict())

    @patch('tensorflow.Session')
    def test_process_test_signals(self, mock_sess):
        """Signals processed by 'test' return accuracy and loss"""
        mock_sess.return_value.run.return_value = [MagicMock()] * 2
        input_signal = {'batch': (MagicMock(), MagicMock())}
        blk = NeuralNetwork()
        self.configure_block(blk, {})
        blk.start()
        blk.process_signals([Signal(input_signal)], input_id='test')
        blk.stop()
        self.assertEqual(mock_sess.call_count, 1)
        # sess.run() is called in start() and process_signals()
        self.assertEqual(mock_sess.return_value.run.call_count, 2)
        self.assertEqual(mock_sess.return_value.close.call_count, 1)
        self.assert_num_signals_notified(1)
        self.assertDictEqual(
            {'loss': ANY, 'accuracy': ANY, 'input_id': 'test'},
            self.last_notified[DEFAULT_TERMINAL][0].to_dict())

    @patch('tensorflow.Session')
    def test_process_predict_signals(self, mock_sess):
        """Signals processed by 'predict' return classification"""
        mock_sess.return_value.run.return_value = [MagicMock()] * 2
        input_signal = {'batch': (MagicMock(), MagicMock())}
        blk = NeuralNetwork()
        self.configure_block(blk, {})
        blk.start()
        blk.process_signals([Signal(input_signal)], input_id='predict')
        blk.stop()
        self.assertEqual(mock_sess.call_count, 1)
        # sess.run() is called in start() and process_signals()
        self.assertEqual(mock_sess.return_value.run.call_count, 2)
        self.assertEqual(mock_sess.return_value.close.call_count, 1)
        self.assert_num_signals_notified(1)
        self.assertDictEqual(
            {'prediction': ANY, 'accuracy': ANY, 'input_id': 'predict'},
            self.last_notified[DEFAULT_TERMINAL][0].to_dict())
