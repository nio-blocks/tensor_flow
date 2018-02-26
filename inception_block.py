import os
import sys
import re
import tarfile
import tensorflow as tf
import base64
import numpy as np
from six.moves import urllib
from nio.util.threading import spawn
from nio.block.base import Block
from nio.block.mixins.enrich.enrich_signals import EnrichSignals
from nio.properties import VersionProperty, IntProperty
from nio.signal.base import Signal


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress TF build warnings

class Inception(EnrichSignals, Block):

    version = VersionProperty('0.1.0')
    num_top_predictions = IntProperty(
        title='Return Top k Predictions',
        default=10)

    def __init__(self):
        super().__init__()
        self.node_lookup = None
        self.sess = None

    def configure(self, context):
        super().configure(context)
        self.maybe_download_and_extract()
        self.node_lookup = NodeLookup()
        self.sess = tf.Session(graph=self.create_graph())

    def process_signals(self, signals):
        for signal in signals:
            self.logger.debug('decoding image')
            image = base64.decodestring(signal.base64Image.encode('utf-8'))
            self.logger.debug('running inference')
            predictions = self.run_inference_on_image(image)
            self.logger.debug('building output signals')
            output_signals = []
            for prediction in predictions:
                output_signal = self.get_output_signal(prediction, signal)
                output_signals.append(output_signal)
            self.logger.debug(
                'notifying {} signals'.format(len(output_signals)))
            self.notify_signals(output_signals)

    def maybe_download_and_extract(self):
        """Download and extract model tar file."""
        DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
        dest_directory = 'tf_models'
        if not os.path.exists(dest_directory):
            self.logger.debug('creating directory: {}'.format(dest_directory))
            os.makedirs(dest_directory)
        filename = DATA_URL.split('/')[-1]
        filepath = os.path.join(dest_directory, filename)
        if not os.path.exists(filepath):
            self.logger.debug('downloading model files')
            filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath)
        self.logger.debug('extracting model files')
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)

    def run_inference_on_image(self, image):
        """Runs inference on an image."""
        softmax_tensor = self.sess.graph.get_tensor_by_name('softmax:0')
        predictions = self.sess.run(
            softmax_tensor,
            {'DecodeJpeg/contents:0': image})
        predictions = np.squeeze(predictions)
        top_k = predictions.argsort()[-self.num_top_predictions():][::-1]
        inference = []
        self.logger.debug('mapping predictions to labels')
        for node_id in top_k:
            human_string = self.node_lookup.id_to_string(node_id)
            score = predictions[node_id]
            inference.append(
                {'label': human_string.split(',')[0], 'confidence': score })
        self.logger.debug('%s (score = %.5f)' % (
            self.node_lookup.id_to_string(top_k[0]), predictions[top_k[0]]))
        return inference

    def create_graph(self):
        """Creates a graph from saved GraphDef file and returns a saver."""
        self.logger.debug('defining graph')
        with tf.gfile.FastGFile('tf_models/classify_image_graph_def.pb', 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            return tf.import_graph_def(graph_def, name='')

class NodeLookup(object):
    """Converts integer node ID's to human readable labels."""

    def __init__(self, label_lookup_path=None, uid_lookup_path=None):
        if not label_lookup_path:
            label_lookup_path = 'tf_models/imagenet_2012_challenge_label_map_proto.pbtxt'
        if not uid_lookup_path:
            uid_lookup_path = 'tf_models/imagenet_synset_to_human_label_map.txt'
        self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

    def load(self, label_lookup_path, uid_lookup_path):
        """Loads a human readable English name for each softmax node.

        Args:
          label_lookup_path: string UID to integer node ID.
          uid_lookup_path: string UID to human-readable string.

        Returns:
          dict from integer node ID to human-readable string.
        """
        if not tf.gfile.Exists(uid_lookup_path):
            tf.logging.fatal('File does not exist %s', uid_lookup_path)
        if not tf.gfile.Exists(label_lookup_path):
            tf.logging.fatal('File does not exist %s', label_lookup_path)

        # Loads mapping from string UID to human-readable string
        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human = {}
        p = re.compile(r'[n\d]*[ \S,]*')
        for line in proto_as_ascii_lines:
            parsed_items = p.findall(line)
            uid = parsed_items[0]
            human_string = parsed_items[2]
            uid_to_human[uid] = human_string

        # Loads mapping from string UID to integer node ID.
        node_id_to_uid = {}
        proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
        for line in proto_as_ascii:
            if line.startswith('  target_class:'):
                target_class = int(line.split(': ')[1])
            if line.startswith('  target_class_string:'):
                target_class_string = line.split(': ')[1]
                node_id_to_uid[target_class] = target_class_string[1:-2]

        # Loads the final mapping of integer node ID to human-readable string
        node_id_to_name = {}
        for key, val in node_id_to_uid.items():
            if val not in uid_to_human:
                tf.logging.fatal('Failed to locate: %s', val)
            name = uid_to_human[val]
            node_id_to_name[key] = name

        return node_id_to_name

    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]