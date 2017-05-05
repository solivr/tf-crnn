#!/usr/bin/env python
__author__ = 'solivr'

from typing import Callable
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell


class Model:

    def __init__(self, name: str, function: Callable, pretrained_file=None, trainable=False):
        self.name = name
        self.function = function
        self.pretrained_file = pretrained_file
        self.trainable = trainable

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)


def deep_bidirectional_lstm(input_tensor : tf.Tensor, list_n_hidden=[256, 256], seq_len=None,
                            name_scope='deep_bidirectional_lstm') -> tf.Tensor:
    # Prepare data shape to match `bidirectional_rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input) "(batch, time, height)"

    with tf.name_scope(name_scope):
        # Forward direction cells
        fw_cell_list = [BasicLSTMCell(nh, forget_bias=1.0) for nh in list_n_hidden]
        # Backward direction cells
        bw_cell_list = [BasicLSTMCell(nh, forget_bias=1.0) for nh in list_n_hidden]

        outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(fw_cell_list,
                                                                       bw_cell_list,
                                                                       input_tensor,
                                                                       dtype=tf.float32,
                                                                       # sequence_length=sequence_length=batch_size*[n_steps]
                                                                       # sequence_length=seq_len
                                                                       )

        return outputs


def deep_cnn(input_tensor: tf.Tensor, name_scope='deep_cnn'):

    with tf.name_scope(name_scope):
        net = tf.layers.conv2d(input_tensor, 64, (3, 3), strides=(1, 1), padding='same', name='conv1')
        net = tf.layers.max_pooling2d(net, (2, 2), strides=(2, 2), name='pool1')

        net = tf.layers.conv2d(net, 128, (3, 3), strides=(1, 1), padding='same', name='conv2')
        net = tf.layers.max_pooling2d(net, (2, 2), strides=(2, 2), name='pool2')

        net = tf.layers.conv2d(net, 256, (3, 3), strides=(1, 1), padding='same', name='conv3')
        net = tf.layers.conv2d(net, 256, (3, 3), strides=(1, 1), padding='same', name='conv4')
        net = tf.layers.max_pooling2d(net, (1, 2), strides=(2, 2), name='pool4')

        net = tf.layers.conv2d(net, 512, (3, 3), strides=(1, 1), padding='same', name='conv5')
        net = tf.layers.batch_normalization(net, axis=-1, name='bn5')

        net = tf.layers.conv2d(net, 512, (3, 3), strides=(1,1), padding='same', name='conv6')
        net = tf.layers.batch_normalization(net, axis=-1, name='bn6')
        net = tf.layers.max_pooling2d(net, (1,2), strides=(2,2), name='pool6')

        net = tf.layers.conv2d(net, 512, (2, 2), strides=(1, 1), padding='valid', name='conv7')

    return net