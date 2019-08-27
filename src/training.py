#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

from .model import CRNNModel
from .config import Params
import tensorflow as tf
from tensorflow.keras.layers import Input


def training(parameters: Params):

    h, w = parameters.input_shape
    c = parameters.input_channels
    data_imgs = Input(shape=(h, w, c), name='input_images')
    data_labels = Input(shape=[], dtype=tf.string, name='labels')
    seq_len_inputs = Input(shape=[], dtype=tf.int32, name='input_widths')

    crnn_model = CRNNModel(dense_output_dims=parameters.alphabet.n_classes)
    net_output = crnn_model(data_imgs, training=True)

    # Alphabet and codes
    keys_alphabet_units = parameters.alphabet.alphabet_units
    values_alphabet_codes = parameters.alphabet.codes
    table_str2int = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys_alphabet_units, values_alphabet_codes), -1)

    # Get labels formatted
    labels_splited = tf.string_split(data_labels, delimiter=parameters.string_split_delimiter)
    codes = table_str2int.lookup(labels_splited.values)
    sparse_code_target = tf.SparseTensor(labels_splited.indices, codes, labels_splited.dense_shape)

    seq_len_labels = tf.math.bincount(tf.cast(sparse_code_target.indices[:, 0], tf.int32),
                                      minlength=tf.shape(net_output)[1])

    # Add CTC loss to model
    ctc_loss = tf.nn.ctc_loss_v2(logits=net_output,
                                 labels=tf.sparse.to_dense(sparse_code_target, default_value=-1),
                                 label_length=seq_len_labels,
                                 logit_length=seq_len_inputs,
                                 logits_time_major=False)

    crnn_model.add_loss(ctc_loss)

    crnn_model.compile(optimizer='adam')

