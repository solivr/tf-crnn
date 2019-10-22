#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.backend import ctc_batch_cost, ctc_decode
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, MaxPool2D, Input, Permute, \
    Reshape, Bidirectional, LSTM, Dense, Softmax, Lambda
from typing import List, Tuple
from .config import Params


class ConvBlock(Layer):
    """
    Convolutional block class.
    It is composed of a `Conv2D` layer, a `BatchNormaization` layer (optional),
    a `MaxPool2D` layer (optional) and a `ReLu` activation.

    :ivar features: number of features of the convolutional layer
    :vartype features: int
    :ivar kernel_size: size of the convolutional kernel
    :vartype kernel_size: int
    :ivar stride: stride of the convolutional layer
    :vartype stride: int, int
    :ivar cnn_padding: padding of the convolution ('same' or 'valid')
    :vartype cnn_padding:
    :ivar pool_size: size of the maxpooling
    :vartype pool_size: int, int
    :ivar batchnorm: use batch norm or not
    :vartype batchnorm: bool
    """
    def __init__(self,
                 features: int,
                 kernel_size: int,
                 stride: Tuple[int, int],
                 cnn_padding: str,
                 pool_size: Tuple[int, int],
                 batchnorm: bool,
                 **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        self.conv = Conv2D(features,
                           kernel_size,
                           strides=stride,
                           padding=cnn_padding)
        self.bn = BatchNormalization(renorm=True,
                                     renorm_clipping={'rmax': 1e2, 'rmin': 1e-1, 'dmax': 1e1},
                                     trainable=True) if batchnorm else None
        self.pool = MaxPool2D(pool_size=pool_size,
                              padding='same') if list(pool_size) > [1, 1] else None

        # for config purposes
        self._features = features
        self._kernel_size = kernel_size
        self._stride = stride
        self._cnn_padding = cnn_padding
        self._pool_size = pool_size
        self._batchnorm = batchnorm

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        if self.bn is not None:
            x = self.bn(x, training=training)
        if self.pool is not None:
            x = self.pool(x)
        x = tf.nn.relu(x)
        return x

    def get_config(self) -> dict:
        """
        Get a dictionary with all the necessary properties to recreate the same layer.

        :return: dictionary containing the properties of the layer
        """
        super_config = super(ConvBlock, self).get_config()
        config = {
            'features': self._features,
            'kernel_size': self._kernel_size,
            'stride': self._stride,
            'cnn_padding': self._cnn_padding,
            'pool_size': self._pool_size,
            'batchnorm': self._batchnorm
        }
        return dict(list(super_config.items()) + list(config.items()))


def get_crnn_output(input_images,
                    parameters: Params=None) -> tf.Tensor:
    """
    Creates the CRNN network and returns it's output.
    Passes the `input_images` through the network and returns its output

    :param input_images: images to process (B, H, W, C)
    :param parameters: parameters of the model (``Params``)
    :return: the output of the CRNN model
    """

    # params of the architecture
    cnn_features_list = parameters.cnn_features_list
    cnn_kernel_size = parameters.cnn_kernel_size
    cnn_pool_size = parameters.cnn_pool_size
    cnn_stride_size = parameters.cnn_stride_size
    cnn_batch_norm = parameters.cnn_batch_norm
    rnn_units = parameters.rnn_units

    # CNN layers
    cnn_params = zip(cnn_features_list, cnn_kernel_size, cnn_stride_size, cnn_pool_size, cnn_batch_norm)
    conv_layers = [ConvBlock(ft, ks, ss, 'same', psz, bn) for ft, ks, ss, psz, bn in cnn_params]

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
    net_output = Softmax()(x)

    return net_output


def get_model_train(parameters: Params):
    """
    Constructs the full model for training.
    Defines inputs and outputs, loss function and metric (CER).

    :param parameters: parameters of the model (``Params``)
    :return: the model (``tf.Keras.Model``)
    """

    h, w = parameters.input_shape
    c = parameters.input_channels

    input_images = Input(shape=(h, w, c), name='input_images')
    input_seq_len = Input(shape=[1], dtype=tf.int32, name='input_seq_length')

    label_codes = Input(shape=(parameters.max_chars_per_string), dtype=tf.int32, name='label_codes')
    label_seq_length = Input(shape=[1], dtype=tf.int32, name='label_seq_length')

    net_output = get_crnn_output(input_images, parameters)

    # Loss function
    def warp_ctc_loss(y_true, y_pred):
        return ctc_batch_cost(label_codes, y_pred, input_seq_len, label_seq_length)

    # Metric function
    def warp_cer_metric(y_true, y_pred):
        pred_sequence_length, true_sequence_length = input_seq_len, label_seq_length

        # y_pred needs to be decoded (its the logits)
        pred_codes_dense = ctc_decode(y_pred, tf.squeeze(pred_sequence_length, axis=-1), greedy=True)
        pred_codes_dense = tf.squeeze(tf.cast(pred_codes_dense[0], tf.int64), axis=0)  # only [0] if greedy=true

        # create sparse tensor
        idx = tf.where(tf.not_equal(pred_codes_dense, -1))
        pred_codes_sparse = tf.SparseTensor(tf.cast(idx, tf.int64),
                                            tf.gather_nd(pred_codes_dense, idx),
                                            tf.cast(tf.shape(pred_codes_dense), tf.int64))

        idx = tf.where(tf.not_equal(label_codes, 0))
        label_sparse = tf.SparseTensor(tf.cast(idx, tf.int64),
                                       tf.gather_nd(label_codes, idx),
                                       tf.cast(tf.shape(label_codes), tf.int64))
        label_sparse = tf.cast(label_sparse, tf.int64)

        # Compute edit distance and total chars count
        distance = tf.reduce_sum(tf.edit_distance(pred_codes_sparse, label_sparse, normalize=False))
        count_chars = tf.reduce_sum(true_sequence_length)

        return tf.divide(distance, tf.cast(count_chars, tf.float32), name='CER')

    # Define model and compile it
    model = Model(inputs=[input_images, label_codes, input_seq_len, label_seq_length], outputs=net_output, name='CRNN')
    optimizer = tf.keras.optimizers.Adam(learning_rate=parameters.learning_rate)
    model.compile(loss=[warp_ctc_loss],
                  optimizer=optimizer,
                  metrics=[warp_cer_metric],
                  experimental_run_tf_function=False) # TODO this is set to true by default but does not seem to work...

    return model


def get_model_inference(parameters: Params,
                        weights_path: str=None):
    """
    Constructs the full model for prediction.
    Defines inputs and outputs, and loads the weights.


    :param parameters: parameters of the model (``Params``)
    :param weights_path: path to the weights (.h5 file)
    :return: the model (``tf.Keras.Model``)
    """
    h, w = parameters.input_shape
    c = parameters.input_channels

    input_images = Input(shape=(h, w, c), name='input_images')
    input_seq_len = Input(shape=[1], dtype=tf.int32, name='input_seq_length')
    filename_images = Input(shape=[1], dtype=tf.string, name='filename_images')

    net_output = get_crnn_output(input_images, parameters)
    output_seq_len = tf.identity(input_seq_len)  # need this op to pass it to output
    filenames = tf.identity(filename_images)

    model = Model(inputs=[input_images, input_seq_len, filename_images], outputs=[net_output, output_seq_len, filenames])

    if weights_path:
        model.load_weights(weights_path)

    return model
