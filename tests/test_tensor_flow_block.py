from unittest.mock import patch, MagicMock, ANY
import tensorflow as tf
# https://www.tensorflow.org/api_guides/python/test

from nio.block.terminals import DEFAULT_TERMINAL
from nio.signal.base import Signal
from nio.testing.block_test_case import NIOBlockTestCase

from ..tensor_flow_block import TensorFlow


class TestTensorFlowBlock(NIOBlockTestCase, tf.test.TestCase):

    block_config = {}
    input_signals = [Signal({'batch': MagicMock(), 'labels': MagicMock()})]

    def output_dict(self, input_id):
        sig = {'loss': ANY if input_id != 'predict' else None,
               'prediction': ANY,
               'input_id': input_id}
        return sig

    @patch('tensorflow.Session')
    @patch('tensorflow.summary')
    def test_process_train_signals(self, mock_summary, mock_sess):
        """Signals processed by 'train' input execute one training iteration"""
        input_id='train'
        # run() returns 3 values
        mock_sess.return_value.run.return_value = [MagicMock()] * 3
        blk = TensorFlow()
        self.configure_block(blk, self.block_config)
        blk.start()
        blk.process_signals(self.input_signals, input_id)
        blk.stop()
        self.assertEqual(mock_sess.call_count, 1)
        # sess.run() is called in start() and process_signals()
        self.assertEqual(mock_sess.return_value.run.call_count, 2)
        self.assertEqual(mock_sess.return_value.close.call_count, 1)
        self.assert_num_signals_notified(1)
        self.assertDictEqual(
            self.output_dict(input_id),
            self.last_notified[DEFAULT_TERMINAL][0].to_dict())

    @patch('tensorflow.Session')
    @patch('tensorflow.summary')
    def test_process_test_signals(self, mock_summary, mock_sess):
        """Signals processed by 'test' return loss"""
        input_id='test'
        # run() returns 2 values
        mock_sess.return_value.run.return_value = [MagicMock()] * 2
        blk = TensorFlow()
        self.configure_block(blk, self.block_config)
        blk.start()
        blk.process_signals(self.input_signals, input_id)
        blk.stop()
        self.assertEqual(mock_sess.call_count, 1)
        # sess.run() is called in start() and process_signals()
        self.assertEqual(mock_sess.return_value.run.call_count, 2)
        self.assertEqual(mock_sess.return_value.close.call_count, 1)
        self.assert_num_signals_notified(1)
        self.assertDictEqual(
            self.output_dict(input_id),
            self.last_notified[DEFAULT_TERMINAL][0].to_dict())

    @patch('tensorflow.Session')
    @patch('tensorflow.summary')
    def test_process_predict_signals(self, mock_summary, mock_sess):
        """Signals processed by 'predict' return classification"""
        input_id = 'predict'
        blk = TensorFlow()
        self.configure_block(blk, self.block_config)
        blk.start()
        blk.process_signals(self.input_signals, input_id)
        blk.stop()
        self.assertEqual(mock_sess.call_count, 1)
        # sess.run() is called in start() and process_signals()
        self.assertEqual(mock_sess.return_value.run.call_count, 2)
        self.assertEqual(mock_sess.return_value.close.call_count, 1)
        self.assert_num_signals_notified(1)
        self.assertDictEqual(
            self.output_dict(input_id),
            self.last_notified[DEFAULT_TERMINAL][0].to_dict())

    @patch('tensorflow.Session')
    @patch('tensorflow.summary')
    def test_multi_input(self, mock_summary, mock_sess):
        # these mocks are the return values from sess.run() for _train, _test,
        # and _predict methods in the block. These are expected to return loss
        loss = 0
        prediction = 1
        train_mock = [MagicMock(), loss, prediction]
        test_mock = [loss, prediction]
        predict_mock = prediction

        # variable initialization, train, test, predict = 4 calls
        mock_sess.return_value.run.side_effect = [MagicMock(),
                                                  train_mock,
                                                  test_mock,
                                                  predict_mock]
        # "batch" here should be a batch of input data to train with. This data
        # should not appear in test or predict. "Labels" is the corresponding
        # correct results for the given input
        train_input_signal = {'batch': [], 'labels': []}
        # "batch" here should be input data that didn't exist in train.
        # "Labels" again being correct output.
        test_input_signal = {'batch': [], 'labels': []}
        # "batch" here is input data for the network to do prediction on
        predict_input_signal = {'batch': []}

        blk = TensorFlow()
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
            self.output_dict('train'),
            self.last_notified[DEFAULT_TERMINAL][0].to_dict())
        self.assertDictEqual(
            self.output_dict('test'),
            self.last_notified[DEFAULT_TERMINAL][1].to_dict())
        self.assertDictEqual(
            self.output_dict('predict'),
            self.last_notified[DEFAULT_TERMINAL][2].to_dict())

    @patch('tensorflow.Session')
    @patch('tensorflow.summary')
    def test_tensorboard(self, mock_summary, mock_sess):
        mock_graph = mock_sess.return_value.graph = MagicMock()
        mock_FileWriter = mock_summary.FileWriter.return_value = MagicMock()
        mock_merge_all = mock_summary.merge_all.return_value = MagicMock()
        # run returns 4 values in train when tensorboard is enabled
        train_return = [mock_merge_all] + [MagicMock()] * 3
        # 4 calls to run() from global var init and three train steps
        # second train step does not create TB record, returns 3 values
        mock_sess.return_value.run.side_effect = [MagicMock(),
                                                  train_return,
                                                  [MagicMock()] * 3,
                                                  train_return]
        train_input_signal = {'batch': [], 'labels': []}
        tb_dir = 'foo'
        tb_tag = 'bar'
        blk = TensorFlow()
        self.configure_block(blk, {'models': {'tensorboard_int': 2,
                                               'tensorboard_dir': tb_dir,
                                               'tensorboard_tag': tb_tag},
                                   'layers': [{}, {}]})
        blk.start()
        blk.process_signals([Signal(train_input_signal)], input_id='train')
        blk.process_signals([Signal(train_input_signal)], input_id='train')
        blk.process_signals([Signal(train_input_signal)], input_id='train')
        blk.stop()

        # two histograms per layer
        self.assertEqual(mock_summary.histogram.call_count, 4)
        self.assertEqual(mock_summary.scalar.call_count, 1)
        mock_summary.FileWriter.assert_called_once_with(
            '{}/{}'.format(tb_dir, tb_tag), mock_graph)
        self.assertEqual(mock_FileWriter.add_summary.call_count, 2)
        self.assertEqual(
            mock_FileWriter.add_summary.call_args_list[0][0],
            (mock_merge_all, 0))
        self.assertEqual(
            mock_FileWriter.add_summary.call_args_list[1][0],
            (mock_merge_all, 2))
        mock_FileWriter.close.assert_called_once_with()

    def test_layers_created(self):
        # create two layers and assert that the number of layers were
        # actually created
        layer_config = {'layers': [{}, {},]}

        blk = TensorFlow()
        self.configure_block(blk, layer_config)
        blk.start()
        blk.stop()

        self.assertEqual(len(blk.layers()),
                         self._get_number_of_layers(blk.sess))

    @staticmethod
    def _get_number_of_layers(sess):
        """returns the number of created layers in a session"""
        unique_layers = [name.split('/')[0] for name in
                         sess.graph._nodes_by_name.keys()
                         if "layer" in name.split('/')[0]]
        return len(set(unique_layers))


class TestSignalLists(NIOBlockTestCase):

    block_config = {}
    input_signals = [Signal({'batch': MagicMock(), 'labels': MagicMock()})] * 2

    @patch('tensorflow.Session')
    @patch('tensorflow.summary')
    def test_process_signals(self, mock_summary, mock_sess):
        """Notified signal list is equal length to input"""
        mock_sess.return_value.run.return_value = [MagicMock()] * 3
        blk = TensorFlow()
        self.configure_block(blk, self.block_config)
        blk.start()
        blk.process_signals(self.input_signals, input_id='train')
        blk.stop()
        # input and output are both one list of two signals
        self.assertEqual(len(self.notified_signals[DEFAULT_TERMINAL]), 1)
        self.assertEqual(len(self.notified_signals[DEFAULT_TERMINAL][-1]),
                         len(self.input_signals))


class TestSignalEnrichment(NIOBlockTestCase):

    block_config = {'enrich': {'exclude_existing': False}}
    input_signals = [Signal({'batch': MagicMock(), 'labels': MagicMock()})]
    output_dict = {'batch': input_signals[0].batch,
                   'labels': input_signals[0].labels,
                   'loss': ANY,
                   'prediction': ANY,
                   'input_id': 'train'}

    @patch('tensorflow.Session')
    @patch('tensorflow.summary')
    def test_enrich_mixin(self, mock_summary, mock_sess):
        mock_sess.return_value.run.return_value = [MagicMock()] * 3
        blk = TensorFlow()
        self.configure_block(blk, self.block_config)
        blk.start()
        blk.process_signals(self.input_signals, input_id='train')
        blk.stop()
        self.assert_num_signals_notified(1)
        self.assertDictEqual(
            self.output_dict,
            self.last_notified[DEFAULT_TERMINAL][0].to_dict())

class TestVariableSaveAndLoad(NIOBlockTestCase):

    save_path = 'path/to/save.ext'
    load_path = 'path/to/load.ext'

    @patch('tensorflow.Session')
    @patch('tensorflow.train')
    @patch('tensorflow.summary')
    def test_save_and_load(self, mock_summary, mock_train, mock_sess):
        """A path is specified, variables are saved to and loaded from file"""
        session_obj = mock_sess.return_value = MagicMock()
        blk = TensorFlow()
        self.configure_block(blk, {'models': {'save_file': self.save_path,
                                              'load_file': self.load_path}})
        blk.start()
        mock_train.Saver.assert_called_once_with(max_to_keep=None)
        blk.stop()
        mock_train.Saver.return_value.save.assert_called_once_with(
            session_obj,
            self.save_path)
        mock_train.Saver.return_value.restore.assert_called_once_with(
            session_obj,
            self.load_path)

    @patch('tensorflow.Session')
    @patch('tensorflow.train')
    @patch('tensorflow.summary')
    def test_no_save_or_load(self, mock_summary, mock_train, mock_sess):
        """No path is specified, variables are not saved nor loaded"""
        session_obj = mock_sess.return_value = MagicMock()
        blk = TensorFlow()
        self.configure_block(blk, {'models': {'save_file': '',
                                              'load_file': ''}})
        blk.start()
        blk.stop()
        mock_train.Saver.return_value.save.assert_not_called()
        mock_train.Saver.return_value.restore.assert_not_called()
