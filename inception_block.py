import os
import sys
import re
import tarfile
import tensorflow as tf
import cv2
import PIL
import numpy as np
from six.moves import urllib
from nio.util.threading import spawn
from nio.block.base import Block
from nio.properties import VersionProperty, IntProperty
from nio.signal.base import Signal


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress TF build warnings

class Inception(Block):

    version = VersionProperty('0.0.1')
    num_top_predictions = IntProperty(title='Return Top k Predictions', default=20)

    def __init__(self):
        super().__init__()
        self._kill = False
        self.video_thread = None
        self.camera = None
        self.DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

    def start(self):
        super().start()
        self.video_thread = spawn(self.classify_frames)

    def stop(self):
        self._kill = True
        self.video_thread.join()
        super().stop()

    def classify_frames(self):
        self.maybe_download_and_extract()
        self.camera=cv2.VideoCapture(0)
        with tf.Graph().as_default():
            self.create_graph()
            while not self._kill:
                self.logger.debug('capturing frame')
                _, frame = self.camera.read()
                img = PIL.Image.fromarray(frame)
                img.save('inception/webcam.jpg')
                self.logger.debug('classifying image')
                cloud = self.run_inference_on_image('inception/webcam.jpg')
                if not self._kill:
                    self.notify_signals([Signal({'wordcloud': cloud})])
            self.camera.release()

    def maybe_download_and_extract(self):
        """Download and extract model tar file."""
        dest_directory = 'inception'
        if not os.path.exists(dest_directory):
            os.makedirs(dest_directory)
        filename = self.DATA_URL.split('/')[-1]
        filepath = os.path.join(dest_directory, filename)
        if not os.path.exists(filepath):
            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                    filename, float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()
            filepath, _ = urllib.request.urlretrieve(self.DATA_URL, filepath, _progress)
            print()
            statinfo = os.stat(filepath)
            print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)

    def run_inference_on_image(self, image):
        """Runs inference on an image."""
        image_data = tf.gfile.FastGFile(image, 'rb').read()
        # self.create_graph()
        with tf.Session() as sess:
            softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
            predictions = sess.run(softmax_tensor,
                                   {'DecodeJpeg/contents:0': image_data})
            predictions = np.squeeze(predictions)

            # Creates node ID --> English string lookup.
            node_lookup = NodeLookup()

            top_k = predictions.argsort()[-self.num_top_predictions():][::-1]
            cloud = []
            for node_id in top_k:
                human_string = node_lookup.id_to_string(node_id)
                score = predictions[node_id]
                cloud.append(
                    {'text': human_string.split(',')[0], 'value': score * 10000})
            self.logger.debug('%s (score = %.5f)' % (
                node_lookup.id_to_string(top_k[0]), predictions[top_k[0]]))
            return cloud

    def create_graph(self):
        """Creates a graph from saved GraphDef file and returns a saver."""
        with tf.gfile.FastGFile('inception/classify_image_graph_def.pb', 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

class NodeLookup(object):
    """Converts integer node ID's to human readable labels."""

    def __init__(self, label_lookup_path=None, uid_lookup_path=None):
        if not label_lookup_path:
            label_lookup_path = 'inception/imagenet_2012_challenge_label_map_proto.pbtxt'
        if not uid_lookup_path:
            uid_lookup_path = 'inception/imagenet_synset_to_human_label_map.txt'
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