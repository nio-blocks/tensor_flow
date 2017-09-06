MNISTImageLoader
================
Each signal processed loads the next `batch_size` images from the dataset corresponding to `input_id`. Output is a rank-two tensor with shape (`batch_size`, 784), ready to be used by a TensorFlow block.

Properties
----------
- **batch_size**: Number of images and labels per batch.
- **shuffle**: Randomize the order of each batch.
- **validation_size**: Number of images and labels to reserve from training data for cross-validation.

Inputs
------
- **test**: Load `batch_size` images from testing dataset.
- **train**: Load `batch_size` images from training dataset.

Outputs
-------
- **default**: A list of signals of equal length to input signals.

Commands
--------
None

Dependencies
------------
* [tensorflow](https://github.com/tensorflow/tensorflow)
* [numpy](https://github.com/numpy/numpy)

Output Details
--------------
* `batch` (array) Flattened image data with shape (*n*, 784), where *n* is `batch_size`
* `labels` (array) Image  labels with shape (*n*, 10)

TensorFlow
==========
Accepts rank-two input tensors, each is fed-forward through a configured aritifial neural network, which predicts values for each of its outputs. Training and testing data will be compared to their empirical labels and evaluated for accuracy and loss, as defined by the user. During training weights are updated through back-propogration, according to the optimizer selected.

Properties
----------
- **enrich**: *enrich_field:* The attribute on the signal to store the results from this block. If this is empty, the results will be merged onto the incoming signal. This is the default operation. Having this field allows a block to 'save' the results of an operation to a single field on an incoming signal and notify the enriched signal.  *results field:* The attribute on the signal to store the results from this block. If this is empty, the results will be merged onto the incoming signal. This is the default operation. Having this field allows a block to save the results of an operation to a single field on an incoming signal and notify the enriched signal.
- **layers**: Define one or more network layers with the specified number of neurons, a bias unit if selected, an activation function, and initial weight values. Each layer's input is the layer before it (or input data, in the case of the first layer).
- **network_config**: Parameters of the artifical neural network. Dropout is only used if one or more dropout layers are configured in the network.

Inputs
------
- **predict**: Create new predictions for un-labeled input tensor.
- **test**: Compare predictions for input tensor to labels, return accuracy and loss.
- **train**: Compare predictions for input tensor to labels, return accuracy and loss, and optimze weights.

Outputs
-------
- **default**: A list of signals of equal length to input signals. Attribute values will be `None` if not applicable to the operation named by `input_id`.

Commands
--------
None

Dependencies
------------
* [tensorflow](https://github.com/tensorflow/tensorflow)
* [numpy](https://github.com/numpy/numpy)

