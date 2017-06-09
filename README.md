MNISTImageLoader
===========
Generate 4-Dimensional numpy arrays from MNIST handwriting dataset using data 
and examples included with TensorFlow 1.1.0

The data contains 60,000 images with labels for training, and another
10,000 for testing. This block loads the next `batch_size` images and labels
from either test or train data and notifies a signal containing the resulting
arrays.

Additional information on this dataset: http://yann.lecun.com/exdb/mnist/

Properties
--------------
* `batch_size`: How many images and labels (array depth) to load at once.

Dependencies
----------------
* [tensorflow](https://github.com/tensorflow/tensorflow)
* [numpy](https://github.com/numpy/numpy)

Commands
----------------
None

Input
-------
* Train: Any list of signals, returns data from Training dataset
* Test: Any list of signals, returns data from Testing dataset

Output
---------
A list of one signal for each input signal. The new signal has the attribute 
`batch` which is a tuple of two arrays from the dataset specified by the input 
node used, each with depth `batch_size`. The first array is image pixel data 
and the second is image labels (one-hot encoded).
