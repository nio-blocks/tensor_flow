from unittest.mock import patch, MagicMock, ANY
from nio.block.terminals import DEFAULT_TERMINAL
from nio.signal.base import Signal
from nio.testing.block_test_case import NIOBlockTestCase
from ..neural_network_block import NeuralNetwork
# https://www.tensorflow.org/api_guides/python/test


class TestNeuralNetworkBlock(NIOBlockTestCase):

    block_config = {'layers': [{}]}

    @patch('tensorflow.Session')
    def test_process_train_signals(self, mock_sess):
        """Signals processed by 'train' input execute one training iteration"""
        mock_sess.return_value.run.return_value = [MagicMock()] * 3
        input_signal = {'batch': MagicMock(), 'labels': MagicMock()}
        blk = NeuralNetwork()
        self.configure_block(blk, self.block_config)
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
        input_signal = {'batch': MagicMock(), 'labels': MagicMock()}
        blk = NeuralNetwork()
        self.configure_block(blk, self.block_config)
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
        input_signal = {'batch': MagicMock(), 'labels': MagicMock()}
        blk = NeuralNetwork()
        self.configure_block(blk, self.block_config)
        blk.start()
        blk.process_signals([Signal(input_signal)], input_id='predict')
        blk.stop()
        self.assertEqual(mock_sess.call_count, 1)
        # sess.run() is called in start() and process_signals()
        self.assertEqual(mock_sess.return_value.run.call_count, 2)
        self.assertEqual(mock_sess.return_value.close.call_count, 1)
        self.assert_num_signals_notified(1)
        self.assertDictEqual(
            {'prediction': ANY,'input_id': 'predict'},
            self.last_notified[DEFAULT_TERMINAL][0].to_dict())

    @patch('tensorflow.Session')
    def test_multi_input(self, mock_sess):
        # these mocks are the return values from sess.run() for _train, _test,
        # and _predict methods in the block. These are expected to return
        # acc, loss except _predict, which returns a single prediction
        train_mock = MagicMock()
        train_acc = 0
        train_loss = 0
        # mock list slicing to correctly return acc, loss
        train_mock.__getitem__.return_value = [train_acc, train_loss]

        test_mock = MagicMock()
        test_acc = 0
        test_loss = 0
        test_mock.return_value = [test_acc, test_loss]

        predict_mock = MagicMock()
        prediction = 0
        predict_mock.return_value = prediction

        # global variable initialization, train, test, predict
        mock_sess.return_value.run.side_effect = [MagicMock(),
                                                  train_mock,
                                                  test_mock(),
                                                  predict_mock()]
        # this should be data that
        train_input_signal = {'batch': [], 'labels': []}
        # this should be data that
        test_input_signal = {'batch': [], 'labels': []}
        # this should be data that
        predict_input_signal = {'batch': [], 'labels': []}

        blk = NeuralNetwork()
        self.configure_block(blk, self.block_config)
        blk.start()
        blk.process_signals([Signal(train_input_signal)], input_id='train')
        blk.process_signals([Signal(test_input_signal)], input_id='test')
        blk.process_signals([Signal(predict_input_signal)], input_id='predict')
        blk.stop()

        # first assert all signals were at least processed
        self.assertEqual(mock_sess.call_count, 1)
        # one for global variable initialization in config, 3 for processed
        # signals
        self.assertEqual(mock_sess.return_value.run.call_count, 4)
        self.assertEqual(mock_sess.return_value.close.call_count, 1)
        self.assert_num_signals_notified(3)

        self.assertDictEqual(
            {'loss': train_loss, 'accuracy': train_acc, 'input_id': 'train'},
            self.last_notified[DEFAULT_TERMINAL][0].to_dict())
        self.assertDictEqual(
            {'loss': test_loss, 'accuracy': test_acc, 'input_id': 'test'},
            self.last_notified[DEFAULT_TERMINAL][1].to_dict())
        self.assertDictEqual(
            {'prediction': prediction, 'input_id': 'predict'},
            self.last_notified[DEFAULT_TERMINAL][2].to_dict())



class TestNeuralNetworkBlockMultiLayer(TestNeuralNetworkBlock):

    # test two layers
    block_config = {
        'layers': [{
            "count": 10,
            "activation": "softmax",
            "initial_weights": "truncated_normal",
            "bias": True
        }]
    }
