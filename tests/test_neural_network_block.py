from unittest.mock import patch, MagicMock, ANY
from nio.block.terminals import DEFAULT_TERMINAL
from nio.signal.base import Signal
from nio.testing.block_test_case import NIOBlockTestCase
from ..neural_network_block import NeuralNetwork
import tensorflow as tf
# https://www.tensorflow.org/api_guides/python/test


class TestNeuralNetworkBlock(NIOBlockTestCase, tf.test.TestCase):

    block_config = {}
    input_signals = [Signal({'batch': MagicMock(), 'labels': MagicMock()})]

    @patch('tensorflow.Session')
    def test_process_train_signals(self, mock_sess):
        """Signals processed by 'train' input execute one training iteration"""
        # run() returns 3 values
        mock_sess.return_value.run.return_value = [MagicMock()] * 3
        blk = NeuralNetwork()
        self.configure_block(blk, self.block_config)
        blk.start()
        blk.process_signals(self.input_signals, input_id='train')
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
        # run() returns 2 values
        mock_sess.return_value.run.return_value = [MagicMock()] * 2
        blk = NeuralNetwork()
        self.configure_block(blk, self.block_config)
        blk.start()
        blk.process_signals(self.input_signals, input_id='test')
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
        blk = NeuralNetwork()
        self.configure_block(blk, self.block_config)
        blk.start()
        blk.process_signals(self.input_signals, input_id='predict')
        blk.stop()
        self.assertEqual(mock_sess.call_count, 1)
        # sess.run() is called in start() and process_signals()
        self.assertEqual(mock_sess.return_value.run.call_count, 2)
        self.assertEqual(mock_sess.return_value.close.call_count, 1)
        self.assert_num_signals_notified(1)
        self.assertDictEqual(
            {'prediction': ANY, 'input_id': 'predict'},
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
        # "batch" here should be a batch of input data to train with. This data
        # should not appear in test or predict. "Labels" is the corresponding
        # correct results for the given input
        train_input_signal = {'batch': [], 'labels': []}
        # "batch" here should be input data that didn't exist in train.
        # "Labels" again being correct output.
        test_input_signal = {'batch': [], 'labels': []}
        # "batch" here is input data for the network to do prediction on
        predict_input_signal = {'batch': []}

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

    @staticmethod
    def _get_number_of_layers(sess):
        """returns the number of created layers in a session"""
        unique_layers = [name.split('/')[0] for name in
                         sess.graph._nodes_by_name.keys()
                         if "layer" in name.split('/')[0]]
        return len(set(unique_layers))

    def test_layers_created(self):
        # create two layers and assert that the number of layers were
        # actually created
        block_config = {
            'layers': [
                {
                    "count": 2,
                    "activation": "softmax",
                    "initial_weights": "truncated_normal",
                    "bias": True
                },
                {
                    "count": 2,
                    "activation": "softmax",
                    "initial_weights": "truncated_normal",
                    "bias": True
                },
            ]
        }

        blk = NeuralNetwork()
        self.configure_block(blk, block_config)
        blk.start()
        blk.stop()

        self.assertEqual(len(blk.layers()),
                         self._get_number_of_layers(blk.sess))


class TestSignalLists(NIOBlockTestCase):

    block_config = {}
    input_signals = [Signal({'batch': MagicMock(), 'labels': MagicMock()})] * 2

    def signals_notified(self, block, signals, output_id):
        """Override so that last_notified is list of signal lists"""
        self.last_notified[output_id].append(signals)

    @patch('tensorflow.Session')
    def test_process_signals(self, mock_sess):
        """Notified signal list is equal length to input"""
        mock_sess.return_value.run.return_value = [MagicMock()] * 3
        blk = NeuralNetwork()
        self.configure_block(blk, self.block_config)
        blk.start()
        blk.process_signals(self.input_signals, input_id='train')
        blk.stop()
        self.assertEqual(len(self.last_notified[DEFAULT_TERMINAL]), 1)
