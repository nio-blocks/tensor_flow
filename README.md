TensorFlow
===========
Collection of blocks to implement TensorFlow in nio services.

Dependencies:
----------------
* [tensorflow](https://github.com/tensorflow/tensorflow)
* [numpy](https://github.com/numpy/numpy)

NeuralNetwork
===========
This is a general purpose user-configurable neural network, assuming flattened inputs. Executes one training step per input signal.

Properties:
--------------
None

Commands:
----------------
None

Input:
-------
`{'batch': (ndarray(<images>), ndarray(<labels>))}`
* `input_id` (kwarg): Possible values: `train`, `test`, `predict`

Output:
---------
A list of signals of equal length to input signals.
* `accuracy` (attr): Percentage of samples correctly labeled
* `loss` (attr): hard-coded loss function is cross-entropy

`{'loss': <float>, 'accuracy': <float>}`


MNISTImageLoader
===========
Generates pixel data and labels from MNIST handwriting dataset. 
If not already present in `data/` the source data will be downloaded 
automatically. The output signal is ready to use by a NeuralNetwork 
block.

Each signal processed loads the next `batch_size` images from the 
dataset corresponding to `input_id`.

Information on this dataset and using it: 
https://www.tensorflow.org/get_started/mnist/beginners

Properties:
-----------
* `batch_size` (int): Number of images and labels to load per signal
processed
* `shuffle` (bool): Randomize order of each batch

Commands:
---------
None

Input:
------
Any list of signals.
* `input_id` (kwarg): Possible values `train, test`

Output:
-------
A list of signals of equal length to input signals.
* `batch` (attr): `batch_size` images and labels in a tuple of arrays.

`{'batch': (ndarray(<images>), ndarray(<labels>)}`
