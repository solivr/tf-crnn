#!/usr/bin/env python
__author__ = 'solivr'

from typing import Callable
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell
import os
import warpctc_tensorflow
import numpy as np

class Model:

    def __init__(self, name: str, function: Callable, pretrained_file=None, trainable=False):
        self.name = name
        self.function = function
        self.pretrained_file = pretrained_file
        self.trainable = trainable

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)
# ----------------------------------------------------------


# def deep_bidirectional_lstm(input_tensor: tf.Tensor, list_n_hidden=[256, 256], seq_len=None,
#                             name_scope='deep_bidirectional_lstm') -> tf.Tensor:
#     # Prepare data shape to match `bidirectional_rnn` function requirements
#     # Current data input shape: (batch_size, n_steps, n_input) "(batch, time, height)"
#
#     with tf.name_scope(name_scope):
#         # Forward direction cells
#         fw_cell_list = [BasicLSTMCell(nh, forget_bias=1.0) for nh in list_n_hidden]
#         # Backward direction cells
#         bw_cell_list = [BasicLSTMCell(nh, forget_bias=1.0) for nh in list_n_hidden]
#
#         outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(fw_cell_list,
#                                                                        bw_cell_list,
#                                                                        input_tensor,
#                                                                        dtype=tf.float32,
#                                                                        # sequence_length=sequence_length=batch_size*[n_steps]
#                                                                        sequence_length=seq_len
#                                                                        )
#
#         return outputs
# ----------------------------------------------------------


# def deep_cnn(input_tensor: tf.Tensor, resize_shape=[32, 32], name_scope='deep_cnn'):
#
#     if resize_shape:
#         # resize image to have h x w
#         input_tensor = tf.image.resize_images(input_tensor, resize_shape)
#
#     # Following source code, not paper
#
#     with tf.variable_scope(name_scope):
#         # conv1 - maxPool2x2
#         net = tf.layers.conv2d(input_tensor, 64, (3, 3),
#                                strides=(1, 1), padding='same',
#                                activation=tf.nn.relu, name='conv1')
#         net = tf.layers.max_pooling2d(net, (2, 2), strides=(2, 2), name='pool1')
#
#         # conv2 - maxPool 2x2
#         net = tf.layers.conv2d(net, 128, (3, 3),
#                                strides=(1, 1), padding='same',
#                                activation=tf.nn.relu, name='conv2')
#         net = tf.layers.max_pooling2d(net, (2, 2), strides=(2, 2), name='pool2')
#
#         # conv3 - w/batch-norm (as source code, not paper)
#         with tf.variable_scope('conv3'):
#             net = tf.layers.conv2d(net, 256, (3, 3),
#                                    strides=(1, 1), padding='same')
#             net = tf.layers.batch_normalization(net, axis=-1, name='batch-norm')
#             net = tf.nn.relu(net, name='ReLU')
#
#         # conv4 - maxPool 2x1
#         net = tf.layers.conv2d(net, 256, (3, 3), strides=(1, 1), padding='same',
#                                activation=tf.nn.relu, name='conv4')
#         net = tf.layers.max_pooling2d(net, (2, 2), strides=(2, 1), name='pool4')
#
#         # conv5 - w/batch-norm
#         with tf.variable_scope('conv5'):
#             net = tf.layers.conv2d(net, 512, (3, 3), strides=(1, 1), padding='same')
#             net = tf.layers.batch_normalization(net, axis=-1, name='batch-norm')
#             net = tf.nn.relu(net, name='ReLU')
#
#         # conv6 - maxPool 2x1 (as source code, not paper)
#         net = tf.layers.conv2d(net, 512, (3, 3), strides=(1,1), padding='same',
#                                activation=tf.nn.relu, name='conv6')
#         net = tf.layers.max_pooling2d(net, (2,2), strides=(2,1), name='pool6')
#
#         # conv 7 - w/batch-norm (as source code, not paper)
#         with tf.variable_scope('conv7'):
#             net = tf.layers.conv2d(net, 512, (2, 2), strides=(1, 1), padding='valid')
#             net = tf.layers.batch_normalization(net, axis=-1, name='batch-norm')
#             net = tf.nn.relu(net, name='ReLU')
#
#     return net
# ---------------------------------------------------------


# def crnn(input: tf.Tensor, cnn_input_shape=[32, 100]):
#     # Convolutional NN
#     conv = deep_cnn(input, resize_shape=cnn_input_shape)
#
#     with tf.variable_scope('Reshaping'):
#         shape = conv.get_shape().as_list()  # [batch, height, width, features]
#         transposed = tf.transpose(conv, perm=[0, 2, 3, 1], name='transposed')  # [batch, width, features, height]
#         conv_reshaped = tf.reshape(transposed, [-1, shape[2], shape[1] * shape[3]],
#                                    name='reshaped')  # [batch, width, height x features]
#
#     # Recurrent NN (BiLSTM)
#     output_rnn = deep_bidirectional_lstm(conv_reshaped, list_n_hidden=[256, 256])
#
#     return output_rnn
# ----------------------------------------------------------


class CRNN():
    def __init__(self, inputImgs, conf, rnnSeqLengths: list, isTraining: bool, keep_prob: float, session=None):
        self.inputImgs = inputImgs
        self.sess = session
        self.config = conf
        self.isTraining = isTraining
        self.keep_prob = keep_prob
        self.rnnSeqLengths = rnnSeqLengths
        self.conv = self.deep_cnn()
        self.prob = self.deep_bidirectional_lstm()

    def deep_cnn(self) -> tf.Tensor:
        if self.config.inputShape:
            # resize image to have h x w
            input_tensor = tf.image.resize_images(self.inputImgs, self.config.inputShape)
            tf.summary.image('input_image', input_tensor, 1)

        # Following source code, not paper

        with tf.variable_scope('deep_cnn'):
            # - conv1 - maxPool2x2
            net = tf.layers.conv2d(input_tensor, 64, (3, 3),
                                   kernel_initializer=tf.constant_initializer(np.random.normal(loc=0.0, scale=0.1,
                                                                                               size=(3, 3, 1, 64))),
                                   strides=(1, 1), padding='same',
                                   activation=tf.nn.relu, name='conv1')
            net = tf.layers.max_pooling2d(net, (2, 2), strides=(2, 2), name='pool1')

            tf.summary.image('conv1_1st_sample', net[:, :, :, :1], 1)
            weights = [var for var in tf.global_variables() if var.name == 'deep_cnn/conv1/kernel:0'][0]
            tf.summary.histogram('conv1_weights', weights)

            # - conv2 - maxPool 2x2
            net = tf.layers.conv2d(net, 128, (3, 3),
                                   strides=(1, 1), padding='same',
                                   kernel_initializer=tf.constant_initializer(np.random.normal(loc=0.0, scale=0.1,
                                                                                               size=(3, 3, 64, 128))),
                                   activation=tf.nn.relu, name='conv2')
            net = tf.layers.max_pooling2d(net, (2, 2), strides=(2, 2), padding='same', name='pool2')

            weights = [var for var in tf.global_variables() if var.name == 'deep_cnn/conv2/kernel:0'][0]
            tf.summary.histogram('conv2_weights', weights)

            # - conv3 - w/batch-norm (as source code, not paper)
            with tf.variable_scope('conv3'):
                net = tf.layers.conv2d(net, 256, (3, 3),
                                       kernel_initializer=tf.constant_initializer(np.random.normal(loc=0.0, scale=0.1,
                                                                                                   size=(3, 3, 1, 256))),
                                       strides=(1, 1), padding='same')
                net = tf.layers.batch_normalization(net, axis=-1, name='batch-norm')
                net = tf.nn.relu(net, name='ReLU')

            weights = [var for var in tf.global_variables() if var.name == 'deep_cnn/conv3/conv2d/kernel:0'][0]
            tf.summary.histogram('conv3_weights', weights)

            # - conv4 - maxPool 2x1
            net = tf.layers.conv2d(net, 256, (3, 3), strides=(1, 1), padding='same',
                                   kernel_initializer=tf.constant_initializer(np.random.normal(loc=0.0, scale=0.1,
                                                                                               size=(3, 3, 1, 256))),
                                   activation=tf.nn.relu, name='conv4')
            net = tf.layers.max_pooling2d(net, (2, 2), strides=(2, 1), padding='same', name='pool4')

            weights = [var for var in tf.global_variables() if var.name == 'deep_cnn/conv4/kernel:0'][0]
            tf.summary.histogram('conv4_weights', weights)

            # - conv5 - w/batch-norm
            with tf.variable_scope('conv5'):
                net = tf.layers.conv2d(net, 512, (3, 3), strides=(1, 1),
                                       kernel_initializer=tf.constant_initializer(np.random.normal(loc=0.0, scale=0.1,
                                                                                                   size=(3, 3, 1, 512))),
                                       padding='same')
                net = tf.layers.batch_normalization(net, axis=-1, name='batch-norm')
                net = tf.nn.relu(net, name='ReLU')

            weights = [var for var in tf.global_variables() if var.name == 'deep_cnn/conv5/conv2d/kernel:0'][0]
            tf.summary.histogram('conv5_weights', weights)

            # - conv6 - maxPool 2x1 (as source code, not paper)
            net = tf.layers.conv2d(net, 512, (3, 3), strides=(1, 1), padding='same',
                                   kernel_initializer=tf.constant_initializer(np.random.normal(loc=0.0, scale=0.1,
                                                                                               size=(3, 3, 1, 512))),
                                   activation=tf.nn.relu, name='conv6')
            net = tf.layers.max_pooling2d(net, (2, 2), strides=(2, 1), padding='same', name='pool6')

            weights = [var for var in tf.global_variables() if var.name == 'deep_cnn/conv6/kernel:0'][0]
            tf.summary.histogram('conv6_weights', weights)

            # conv 7 - w/batch-norm (as source code, not paper)
            with tf.variable_scope('conv7'):
                net = tf.layers.conv2d(net, 512, (2, 2), strides=(1, 1),
                                       kernel_initializer=tf.constant_initializer(np.random.normal(loc=0.0, scale=0.1,
                                                                                                   size=(2, 2, 1, 512))),
                                       padding='valid')
                net = tf.layers.batch_normalization(net, axis=-1, name='batch-norm')
                net = tf.nn.relu(net, name='ReLU')

            weights = [var for var in tf.global_variables() if var.name == 'deep_cnn/conv7/conv2d/kernel:0'][0]
            tf.summary.histogram('conv7_weights', weights)

            self.cnn_net = net

            with tf.variable_scope('Reshaping_cnn'):
                shape = self.cnn_net.get_shape().as_list()  # [batch, height, width, features]
                transposed = tf.transpose(self.cnn_net, perm=[0, 2, 1, 3],
                                          name='transposed')  # [batch, width, height, features]
                conv_reshaped = tf.reshape(transposed, [-1, shape[2], shape[1] * shape[3]],
                                           name='reshaped')  # [batch, width, height x features]

        return conv_reshaped

    def deep_bidirectional_lstm(self) -> tf.Tensor:
        # Prepare data shape to match `bidirectional_rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input) "(batch, time, height)"

        with tf.name_scope('deep_bidirectional_lstm'):
            # Forward direction cells
            fw_cell_list = [BasicLSTMCell(nh, forget_bias=1.0) for nh in self.config.listNHidden]
            # Backward direction cells
            bw_cell_list = [BasicLSTMCell(nh, forget_bias=1.0) for nh in self.config.listNHidden]

            self.lstm_net, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(fw_cell_list,
                                                                                 bw_cell_list,
                                                                                 self.conv,
                                                                                 dtype=tf.float32,
                                                                                 sequence_length=self.rnnSeqLengths
                                                                                 )

            # Dropout layer
            self.lstm_net = tf.nn.dropout(self.lstm_net, keep_prob=self.keep_prob)

            with tf.variable_scope('Reshaping_rnn'):
                shape = self.lstm_net.get_shape().as_list()  # [batch, width, 2*n_hidden]
                rnn_reshaped = tf.reshape(self.lstm_net, [-1, shape[-1]])  # [batch x width, 2*n_hidden]

            fc_out = tf.contrib.layers.fully_connected(inputs=rnn_reshaped,
                                                       num_outputs=self.config.nClasses,
                                                       activation_fn=None,
                                                       trainable=True
                                                       # weights_initializer=tf.Variable(tf.truncated_normal([2*list_n_hidden[-1], n_classes])),
                                                       # biases_initializer=tf.Variable(tf.truncated_normal([n_classes])),
                                                       )  # [batch x width, n_classes]

            lstm_out = tf.reshape(fc_out, [-1, shape[1], self.config.nClasses], name='reshape_out')  # [batch, width, n_classes]

            self.rawPred = tf.argmax(tf.nn.softmax(lstm_out), axis=2, name='raw_prediction')
            tf.summary.tensor_summary('raw_preds', tf.nn.softmax(lstm_out))

            # Swap batch and time axis
            lstm_out = tf.transpose(lstm_out, [1, 0, 2], name='transpose_time_major')  # [width(time), batch, n_classes]

            return lstm_out

    def saveModel(self, model_dir, step):
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)

        save_path = os.path.join(model_dir, 'chkpt-{}'.format(step))
        saver = tf.train.Saver()

        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        p = saver.save(self.sess, os.path.join(save_path, 'ckpt-{}'.format(step)))
        print('Model saved at: {}'.format(p))
        return p

    def loadModel(self, file):
        saver = tf.train.Saver()
        saver.restore(self.sess, file)
        print('Model restored')



class CTC:
    # def __init__(self, result, inputSeqLengths, lossTarget, targetSeqLengths, pred_labels, true_labels):
    def __init__(self, result: tf.Tensor, target, target_warp, targetSeqLengths: list, inputSeqLength=None, pred_labels=None, true_labels=None):
        self.result = result
        self.target = target
        self.target_warp = target_warp
        self.targetSeqLengths = targetSeqLengths
        self.inputSeqLength = inputSeqLength
        self.pred_labels = pred_labels
        self.true_labels = true_labels
        self.createCtcCriterion()

    def createCtcCriterion(self):
        # using built-in ctc loss calculator
        self.loss = tf.nn.ctc_loss(self.target, self.result, self.targetSeqLengths, time_major=True)
        self.cost = tf.reduce_mean(self.loss)

        # using baidu's warp ctc loss calculator
        self.loss_warp = warpctc_tensorflow.ctc(activations=self.result,
                                                flat_labels=self.target_warp,
                                                label_lengths=self.targetSeqLengths,
                                                input_lengths=self.inputSeqLength,
                                                blank_label=36)
        self.cost_warp = tf.reduce_mean(self.loss_warp)
