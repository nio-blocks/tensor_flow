NOTICE:
==========
**The TensorFlow library is updated frequently, these blocks have been developed for and tested with v1.4.0**

MNISTImageLoader
================
Each signal processed loads the next `batch_size` images from the dataset corresponding to `input_id`. Output is a batch of flattened images in a rank-two tensor, with labels, ready to be used by a TensorFlow block.

Properties
----------
- **batch_size**: Number of images and labels per batch.
- **enrich**: Signal Enrichment
  - *exclude_existing*: If checked (true), the attributes of the incoming signal will be excluded from the outgoing signal. If unchecked (false), the attributes of the incoming signal will be included in the outgoing signal.
  - *enrich_field*: (hidden) The attribute on the signal to store the results from this block. If this is empty, the results will be merged onto the incoming signal. This is the default operation. Having this field allows a block to 'save' the results of an operation to a single field on an incoming signal and notify the enriched signal.
- **shuffle**: (hidden) Randomize the order of each batch.

Inputs
------
- **test**: Load `batch_size` images from testing dataset.
- **train**: Load `batch_size` images from training dataset.

Outputs
-------
- **default**: A list of signals of equal length to input signals.
  - *batch* (array) Flattened image data with shape (`batch_size`, 784).
  - *labels* (array) Image labels with shape (`batch_size`, 10).

Commands
--------
None

Dependencies
------------
* [tensorflow](https://github.com/tensorflow/tensorflow)
* [numpy](https://github.com/numpy/numpy)

***

TensorFlow
==========
Accepts rank-two input tensors, each is fed-forward through a configured artificial neural network, which predicts values for each of its outputs. Training and testing data will be compared to their empirical labels and evaluated for loss as defined by the user. During training weights are updated through back-propogration, according to the optimizer selected. The default configuration is ready to take input from an MNISTImageLoader block, and will surpass 90% accuracy in about 2,000 training steps.

Properties
----------
- **enrich**: Signal Enrichment
  - *exclude_existing*: If checked (true), the attributes of the incoming signal will be excluded from the outgoing signal. If unchecked (false), the attributes of the incoming signal will be included in the outgoing signal.
  - *enrich_field*: (hidden) The attribute on the signal to store the results from this block. If this is empty, the results will be merged onto the incoming signal. This is the default operation. Having this field allows a block to 'save' the results of an operation to a single field on an incoming signal and notify the enriched signal.
- **layers**: Define one or more network layers. Each layer's input is the layer before it (or input data, in the case of the first layer).
  - *count*: Number of neurons in this layer
  - *activation*: Activation function, use *bias_add* to use no activation function.
  - *initial_weights*: Initialize newly created model weights with random or fixed values.
  - *bias*: Add a bias unit to this layer's input.
- **models**: Visualize models using [TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard#launching_tensorboard). Save and Load training progress in [Checkpoint Files](https://www.tensorflow.org/versions/master/get_started/checkpoints).
  - *save_file*: If not blank, when stopped the block will save its internal weights and bias values to this path.
  - *load_file*: If not blank, when configured the block will load weight values from this path instead of applying the `initial_weights` specified for each layer.
  - *tensorboard_int*: Number of training steps between each update of TensorBoard event files, set to 0 to disable.
  - *tensorboard_tag*: (hidden) Run label, records will be saved to a subdirectory with this name inside `tensorboard_dir`. Default is a string of the current local time, as `YYYYMMDDHHMMSS`.
  - *tensorboard_dir*: (hidden) Path to event files, defaults to `<project>/tensorboard`.
- **network_config**: Hyperparameters of the artifical neural network.
  - *input_dim*: Number of input values to the first layer.
  - *learning_rate*: Multiplier for updates to weight values.
  - *loss*: Loss function to quantify prediction accuracy.
  - *optimizer*: Optimizer algorithm to compute gradients and apply weight updates.
  - *dropout*: Percentage of this neurons to disable during training, applied to each layer where `activation == 'dropout'`.
  - *random_seed*: (hidden) Set to non-zero for repeatable random values.

Inputs
------
- **predict**: Create new predictions for un-labeled input tensor.
  - *batch*: (array) Input data, rank-two tensor.
- **test**: Compare predictions for input tensor to labels, return prediction and loss.
  - *batch*: (array) Input data, rank-two tensor.
  - *labels*: (array) Input labels, rank-two tensor.
- **train**: Compare predictions for input tensor to labels, return prediction and loss, and optimze network weights.
  - *batch*: (array) Input data, rank-two tensor.
  - *labels*: (array) Input labels, rank-two tensor.

Outputs
-------
- **default**: A list of signals of equal length to input signals.
  - *input_id*: (string) The input which processed this list of signals.
  - *loss*: (float) The measured loss for this batch, will be `None` if `input_id == 'predict'`.
  - *prediction*: (array) Tensor containing network output as predictions.

Commands
--------
None

Dependencies
------------
* [tensorflow](https://github.com/tensorflow/tensorflow)
* [numpy](https://github.com/numpy/numpy)

