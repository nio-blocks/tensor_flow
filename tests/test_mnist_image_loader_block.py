from unittest.mock import patch, ANY

from nio.block.terminals import DEFAULT_TERMINAL
from nio.signal.base import Signal
from nio.testing.block_test_case import NIOBlockTestCase

from ..mnist_image_loader_block import MNISTImageLoader


class TestMNISTImageLoader(NIOBlockTestCase):

    @patch('tensorflow.examples.tutorials.mnist.input_data.read_data_sets')
    def test_process_signals(self, mock_dataset):
        """For each input signal call next_batch(batch_size)"""
        train_signals = [Signal({'batch_size': 10, 'shuffle': True})] * 2
        test_signals = [Signal({'batch_size': 1, 'shuffle': False})]
        blk = MNISTImageLoader()
        self.configure_block(blk, {
            'batch_size': '{{ $batch_size }}',
            'shuffle': '{{ $shuffle }}',
            'enrich': {'exclude_existing': False}})
        blk.start()
        blk.process_signals(train_signals, input_id='train')
        blk.process_signals(test_signals, input_id='test')
        blk.stop()
        mock_dataset.assert_called_once_with(
            'data',
            one_hot=True,
            reshape=False,
            validation_size=0)
        self.assert_num_signals_notified(len(train_signals + test_signals))
        self.assertDictEqual(
            {
                'batch': ANY,
                'labels': ANY,
                'input_id': ANY,
                **test_signals[0].to_dict()},  # 'exclude_existing': False
            self.last_notified[DEFAULT_TERMINAL][-1].to_dict())
        for i, arg in enumerate(
                mock_dataset.return_value.train.next_batch.call_args_list):
            self.assertEqual((train_signals[i].to_dict()), arg[1])
        mock_dataset.return_value.test.next_batch.assert_called_once_with(
            batch_size=test_signals[0].batch_size,
            shuffle=False)

    @patch('tensorflow.examples.tutorials.mnist.input_data.read_data_sets')
    def test_reshape_images(self, mock_dataset):
        """optionally flatten images"""
        blk = MNISTImageLoader()
        self.configure_block(blk, {'reshape': True})
        blk.start()
        blk.stop()
        mock_dataset.assert_called_once_with(
            'data',
            one_hot=True,
            reshape=True,
            validation_size=0)