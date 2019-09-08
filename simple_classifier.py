import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import errno

tf.logging.set_verbosity(tf.logging.INFO)

# def _pre_model_check():
#     check_pass = False
#     if not os.path.exists(LOG_PATH):
#         os.mkdir(LOG_PATH)
#     if not os.path.exists(TRAIN_DATA):
#         raise(FileNotFoundError(TRAIN_DATA))
#     if not os.path.exists(TEST_DATA):
#         raise(FileNotFoundError(errno., TEST_DATA))
#     check_pass = True
#     return check_pass

IMG_SIZE = 32
RGB_CHANNELS = 3
FLAT_IMG_SIZE = IMG_SIZE * IMG_SIZE * RGB_CHANNELS

LOG_PATH = "./logs"
TRAIN_DATA = "./data/train"
TEST_DATA = "./data/test"

NUM_CLASSES = 2

# TODO: Edit hyperparameters

lr = 0.001
epochs = 1000
batch_size = 100
result_disp_fqy = 10

# Layer Configuration
# Convolutional Layer 1
filter_size_1 = 5
num_filters_1 = 16
stride_1 = 2

# Convolutional Layer 2
filter_size_2 = 5
num_filters_2 = 32
stride_2 = 1

# Fully Connected Layer
h_1 = 128

def _weight_variable(shape):
    """
    Create and Initialize weight variables
    Arguments:
        shape: Shape of the weight tensor
        name: Name of weight tensor
    Returns:
        Initialized weight variable
    """
    wt_init = tf.truncated_normal_initializer(stddev=0.075)
    return tf.get_variable('W', dtype=tf.float32, shape=shape, initializer=wt_init)

def _bias_variable(shape):
    """
    Create and Initialize bias variables
    Arguments:
        shape: Shape of bias tensor
        name: Name of bias tensor
    Returns:
        Initialized bias variable
    """
    b_init = tf.constan(0., shape=shape, dtype=tf.float32)
    return tf.get_variable('b', dtype=tf.float32, initializer=b_init)

# Defining helper functions for the convolutional, pooling, flattening and fully connected layers
def conv_layer(x, filter_size, num_filters, stride, name):
    """
    Create a 2D convolutional layer based on parameters from previous layers
    Arguments:
        x: Input from previous layer
        filter_size: size of the convolutional filter
        num_filters: Numbers of convolutional filters
        stride: Filter stride
        name: Layer name
    Returns:
        Output array
    """
    with tf.variable_scope(name):
        num_input_channels = x.get_shape().as_list()[-1]
        shape = [filter_size, filter_size, num_input_channels, num_filters]
        W = _weight_variable(shape=shape)
        tf.summary.histogram('weight', W)
        b = _bias_variable(shape=[num_filters])
        tf.summary.histogram('bias', b)
        layer = tf.layers.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")
        layer += b
    return(tf.nn.relu(layer))

def max_pool(x, kernel_size, stride, name):
    """
    Create a Maximum-Pooling layer
    Arguments:
        x: Input from previous layer
        kernel_size: Size of the max-pooling filter
        stride: Stride of max-pool filter
        name: Layer name
    Returns:
        Output array
    """
    return tf.nn.max_pool(x, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride, stride, 1], padding="SAME", name=name)

def flatten(layer):
    """
    Flattens output of convolutional layer to be fed to fully connected layer
    Arguments:
        layer: Input array
    Returns:
        Flattened array for fully connected layer
    """
    with tf.variable_scope("Flatten"):
        layer_shape = layer.get_shape()
        num_features = layer_shape[1:4].num_elements()
        layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat

def fc_layer(x, hidden_units, name, activation):
    """
    Create a fully connected layer
    Arguments:
        x: Input from previous layer
        hidden_units: Number of hidden units
        name: Layer name
        activation: Choice of activation function for fully connected layer (ReLU, tanh, Sigmoid). Activation is relu by default.
    Returns:
        Output array
    """
    with tf.variable_scope(name):
        input_dim = x.get_shape()[1]
        W = _weight_variable(shape=[input_dim, hidden_units])
        tf.summary.histogram('weight', W)
        b = _bias_variable(shape=[hidden_units])
        tf.summary.histogram('bias', b)
        layer = tf.matmul(x, W)
        layer += b
        if activation == "tanh":
            layer = tf.nn.tanh(layer)
        elif activation == "sigmoid":
            layer = tf.nn.sigmoid(layer)
        else: 
            layer = tf.nn.relu(layer)
    return layer

# TODO: Add graph methods and comments