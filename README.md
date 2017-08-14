TensorFlow
===========
Collection of blocks to implement TensorFlow in nio services.

Dependencies:
----------------
* [tensorflow](https://github.com/tensorflow/tensorflow)
* [numpy](https://github.com/numpy/numpy)

NeuralNetwork
===========
A general purpose, user-configurable neural network. See the full documentation here: [Python API Guides](https://www.tensorflow.org/api_guides/python/)

Properties:
--------------
* `network_config`
  * `input_dim` (int) Number of inputs to first layer
  * `learning_rate` (float) Constant multiplier for weight updates
  * `loss` (select) Objective function to be minimized, [Losses](https://www.tensorflow.org/api_guides/python/nn#Losses)
  * `optimizer` (select) Optimizer to use in minimizing loss, [Optimizers](https://www.tensorflow.org/api_guides/python/train#Optimizers)
  * `dropout` (float, 0-1) Dropout rate to be applied to all dropout layers, if present
  * `random_seed` (int) Non-zero seeds can be set to produce repeatable calculations
* `layers`
  * `count` (int) Number of neurons in the layer
  * `activation` (select) Activation function of the layer, [Activation Functions](https://www.tensorflow.org/api_guides/python/nn#Activation_Functions)
  * initial_weights` (select) All weights (including bias) of this layer will be initialized with this value
  * `bias` (bool) Add bias unit to layer

Commands:
----------------
None

Input:
-------
* `input_id` (kwarg) Possible values `train, test, predict`

Signal Attributes:

* `batch` (array) Data with shape (*n*, `network_config.input_dim`), where *n* is `batch_size`
* `labels` (array) Target value for each datum with shape (*n*, `layers[-1].count`). Not required if `input_id` is `predict`

Output:
---------
A list of signals of equal length to input signals.  

* `accuracy` (float) Percentage of batch correctly identified
* `loss` (float) Result of loss function `network_config.loss`
* `input_id` (str)

If `input_id` is `predict`, `loss` and `accuracy` will be replaced by:

* `prediction` (array) Network output with shape (*n*, `layers[-1].count`), where *n* is `batch_size`

MNISTImageLoader
===========
Generates pixel data and labels from MNIST handwriting dataset. If not already present in `data/` the source data will be downloaded automatically. The output signal is ready to use by a NeuralNetwork block.

Each signal processed loads the next `batch_size` images from the dataset corresponding to `input_id`.

Information on this dataset and using it: 
https://www.tensorflow.org/get_started/mnist/beginners

Properties:
-----------
* `batch_size` (int) Number of images and labels to load per signal processed
* `shuffle` (bool) Randomize order of each batch

Commands:
---------
None

Input:
------
Any list of signals.

* `input_id` (kwarg) Possible values `train, test`

Output:
-------
A list of signals of equal length to input signals.

* `batch` (array) Flattened image data with shape (*n*, 784), where *n* is `batch_size`
* `labels` (array) Image  labels with shape (*n*, 10)