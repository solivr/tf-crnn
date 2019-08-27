#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.backend import ctc_batch_cost
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, MaxPool2D, Input, Permute, \
    Reshape, Bidirectional, LSTM, Dense, Softmax, Lambda
from typing import List, Tuple
from .config import Params


class ConvBlock(Layer):
    def __init__(self,
                 features: int,
                 kernel_size: int,
                 stride: Tuple[int, int],
                 cnn_padding: str,
                 pool_size: Tuple[int, int],
                 pool_strides: Tuple[int, int],
                 batchnorm: bool):
        super(ConvBlock, self).__init__()
        self.conv = Conv2D(features,
                           kernel_size,
                           strides=stride,
                           padding=cnn_padding)
        self.bn = BatchNormalization(renorm=True,
                                     renorm_clipping={'rmax': 1e2, 'rmin': 1e-1, 'dmax': 1e1},
                                     trainable=True) if batchnorm else None
        self.pool = MaxPool2D(pool_size=pool_size,
                              strides=pool_strides,
                              padding='same')

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        if self.bn is not None:
            x = self.bn(x, training=training)
        if self.pool is not None:
            x = self.pool(x)
        x = tf.nn.relu(x)
        return x


def get_crnn_output(input_images, parameters: Params=None):

    # params of the architecture
    cnn_features_list = parameters.cnn_features_list
    cnn_kernel_size = parameters.cnn_kernel_size
    cnn_pool_size = parameters.cnn_pool_size
    cnn_pool_strides = parameters.cnn_pool_strides
    cnn_stride_size = parameters.cnn_stride_size
    cnn_batch_norm = parameters.cnn_batch_norm
    rnn_units = parameters.rnn_units

    # CNN layers
    cnn_params = zip(cnn_features_list, cnn_kernel_size, cnn_stride_size, cnn_pool_size,
                     cnn_pool_strides, cnn_batch_norm)
    conv_layers = [ConvBlock(ft, ks, ss, 'same', psz, pst, bn) for ft, ks, ss, psz, pst, bn in cnn_params]

    x = conv_layers[0](input_images)
    for conv in conv_layers[1:]:
        x = conv(x)

    # Permutation and reshape
    x = Permute((2, 1, 3))(x)
    shape = x.get_shape().as_list()
    x = Reshape((shape[1], shape[2] * shape[3]))(x)  # [B, W, H*C]

    # RNN layers
    rnn_layers = [Bidirectional(LSTM(ru, dropout=0.5, return_sequences=True, time_major=False)) for ru in
                  rnn_units]
    for rnn in rnn_layers:
        x = rnn(x)

    # Dense and softmax
    x = Dense(parameters.alphabet.n_classes)(x)
    net_output = Softmax(name='sorftmax_output')(x)

    return net_output


def get_model_train(params_dict: dict=None):
    parameters = params_dict['parameters']
    training_parameters = params_dict['training_parameters']

    h, w = parameters.input_shape
    c = parameters.input_channels

    input_images = Input(shape=(h, w, c), name='input_images')
    input_seq_len = Input(shape=[1], dtype=tf.int32, name='input_seq_length')

    label_codes = Input(shape=[parameters.max_chars_per_string], dtype='int32', name='label_codes')
    label_seq_length = Input(shape=[1], dtype='int64', name='label_seq_length')

    net_output = get_crnn_output(input_images, parameters)

    # Loss
    def _ctc_loss_fn(args):
        preds, label_codes, input_length, label_length = args
        return ctc_batch_cost(label_codes, preds, input_length, label_length)
    loss_ctc = Lambda(_ctc_loss_fn, output_shape=(1,), name='ctc_loss')(
        [net_output, label_codes, input_seq_len, label_seq_length])

    # tf.summary.scalar('loss', tf.reduce_mean(loss_ctc))

    # Define model and compile it
    model = Model(inputs=[input_images, label_codes, input_seq_len, label_seq_length], outputs=loss_ctc)
    optimizer = tf.keras.optimizers.Adam(learning_rate=training_parameters.learning_rate)
    model.compile(loss={'ctc_loss': lambda i, j: j}, optimizer=optimizer)  # loss has already been added to the model

    return model
