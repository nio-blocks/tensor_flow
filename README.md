TensorFlow
===========
Collection of blocks to implement TensorFlow in n.io services.

Dependencies
----------------
* [tensorflow](https://github.com/tensorflow/tensorflow)
* [numpy](https://github.com/numpy/numpy)

NeuralNetwork
===========
Hard-coded 784-10 softmax network, executes one training step per input signal.

Properties
--------------
None

Commands
----------------
None

Input
-------
* `train`: `{'batch': (ndarray(<images>), ndarray(<labels>))}`
* `predict`: Unused

Output
---------
`{'loss': <float>, 'accuracy': <float>}`


MNISTImageLoader
===========
Creates 4-Dimensional numpy arrays from MNIST handwriting dataset using
data and examples included with TensorFlow 1.1.0

Each signal processed by `train` or `test` inputs loads the next 
`batch_size` images from the corresponding dataset. Output is ready to use by 
NeuralNetwork block.

The data contains 60,000 images with labels for training, and another
10,000 for testing. Additional information on this dataset: 
http://yann.lecun.com/exdb/mnist/

Properties:
--------------
* batch_size (int): How many images and labels to load per signal
* shuffle: If True the contents of each batch will be in random order

Commands
----------------
None

Input
-------
* `train`: Any list of signals, returns data from training dataset
* `test`: Any list of signals, returns data from testing dataset

Output
---------
`{'batch': (ndarray(<images>), ndarray(<labels>))}`
