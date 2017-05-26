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


def weightVar(shape, mean=0.0, stddev=0.1, name='weights'):
    initW = tf.truncated_normal(shape=shape, mean=mean, stddev=stddev)
    return tf.Variable(initW, name=name)


def biasVar(shape, value=0.0, name='bias'):
    initb = tf.constant(value=value, shape=shape)
    return tf.Variable(initb, name=name)


def conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME', name=None):
    return tf.nn.conv2d(input, filter, strides=strides, padding=padding, name=name)


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
            with tf.variable_scope('layer1'):
                W = weightVar([3, 3, 1, 64])
                b = biasVar([64])
                conv = conv2d(input_tensor, W, name='conv')
                out = tf.nn.bias_add(conv, b)
                conv1 = tf.nn.relu(out)
                pool1 = tf.nn.max_pool(conv1, [1, 2, 2, 1], strides=[1, 2, 2, 1],
                                        padding='SAME', name='pool')

                tf.summary.image('conv1_1st_sample', pool1[:, :, :, :1], 1)
                weights = [var for var in tf.global_variables() if var.name == 'deep_cnn/layer1/weights:0'][0]
                tf.summary.histogram('weights', weights)
                bias = [var for var in tf.global_variables() if var.name == 'deep_cnn/layer1/bias:0'][0]
                tf.summary.histogram('bias', bias)

            # - conv2 - maxPool 2x2
            with tf.variable_scope('layer2'):
                W = weightVar([3, 3, 64, 128])
                b = biasVar([128])
                conv = conv2d(pool1, W)
                out = tf.nn.bias_add(conv, b)
                conv2 = tf.nn.relu(out)
                pool2 = tf.nn.max_pool(conv2, [1, 2, 2, 1], strides=[1, 2, 2, 1],
                                       padding='SAME', name='pool1')

                weights = [var for var in tf.global_variables() if var.name == 'deep_cnn/layer2/weights:0'][0]
                tf.summary.histogram('weights', weights)
                bias = [var for var in tf.global_variables() if var.name == 'deep_cnn/layer2/bias:0'][0]
                tf.summary.histogram('bias', bias)

            # - conv3 - w/batch-norm (as source code, not paper)
            with tf.variable_scope('layer3'):
                W = weightVar([3, 3, 128, 256])
                b = biasVar([256])
                conv = conv2d(pool2, W)
                out = tf.nn.bias_add(conv, b)
                b_norm = tf.layers.batch_normalization(out, axis=-1,
                                                       training=self.isTraining, name='batch-norm')
                conv3 = tf.nn.relu(b_norm, name='ReLU')

                weights = [var for var in tf.global_variables() if var.name == 'deep_cnn/layer3/weights:0'][0]
                tf.summary.histogram('weights', weights)
                bias = [var for var in tf.global_variables() if var.name == 'deep_cnn/layer3/bias:0'][0]
                tf.summary.histogram('bias', bias)

            # - conv4 - maxPool 2x1
            with tf.variable_scope('layer4'):
                W = weightVar([3, 3, 256, 256])
                b = biasVar([256])
                conv = conv2d(conv3, W)
                out = tf.nn.bias_add(conv, b)
                conv4 = tf.nn.relu(out)
                pool4 = tf.nn.max_pool(conv4, [1, 2, 2, 1], strides=[1, 2, 1, 1],
                                       padding='SAME', name='pool4')

                weights = [var for var in tf.global_variables() if var.name == 'deep_cnn/layer4/weights:0'][0]
                tf.summary.histogram('weights', weights)
                bias = [var for var in tf.global_variables() if var.name == 'deep_cnn/layer4/bias:0'][0]
                tf.summary.histogram('bias', bias)

            # - conv5 - w/batch-norm
            with tf.variable_scope('layer5'):
                W = weightVar([3, 3, 256, 512])
                b = biasVar([512])
                conv = conv2d(pool4, W)
                out = tf.nn.bias_add(conv, b)
                b_norm = tf.layers.batch_normalization(out, axis=-1,
                                                       training=self.isTraining, name='batch-norm')
                conv5 = tf.nn.relu(b_norm)

                weights = [var for var in tf.global_variables() if var.name == 'deep_cnn/layer5/weights:0'][0]
                tf.summary.histogram('weights', weights)
                bias = [var for var in tf.global_variables() if var.name == 'deep_cnn/layer5/bias:0'][0]
                tf.summary.histogram('bias', bias)

            # - conv6 - maxPool 2x1 (as source code, not paper)
            with tf.variable_scope('layer6'):
                W = weightVar([3, 3, 512, 512])
                b = biasVar([512])
                conv = conv2d(conv5, W)
                out = tf.nn.bias_add(conv, b)
                conv6 = tf.nn.relu(out)
                pool6 = tf.nn.max_pool(conv6, [1, 2, 2, 1], strides=[1, 2, 1, 1],
                                       padding='SAME', name='pool6')

                weights = [var for var in tf.global_variables() if var.name == 'deep_cnn/layer6/weights:0'][0]
                tf.summary.histogram('weights', weights)
                bias = [var for var in tf.global_variables() if var.name == 'deep_cnn/layer6/bias:0'][0]
                tf.summary.histogram('bias', bias)

            # conv 7 - w/batch-norm (as source code, not paper)
            with tf.variable_scope('layer7'):
                W = weightVar([2, 2, 512, 512])
                b = biasVar([512])
                conv = conv2d(pool6, W, padding='VALID')
                out = tf.nn.bias_add(conv, b)
                b_norm = tf.layers.batch_normalization(out, axis=-1,
                                                       training=self.isTraining, name='batch-norm')
                conv7 = tf.nn.relu(b_norm)

                weights = [var for var in tf.global_variables() if var.name == 'deep_cnn/layer7/weights:0'][0]
                tf.summary.histogram('weights', weights)
                bias = [var for var in tf.global_variables() if var.name == 'deep_cnn/layer7/bias:0'][0]
                tf.summary.histogram('bias', bias)

            self.cnn_net = conv7

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

            with tf.variable_scope('fully_connected'):
                W = weightVar([self.config.listNHidden[-1]*2, self.config.nClasses])
                b = biasVar([self.config.nClasses])
                fc_out = tf.nn.bias_add(tf.matmul(rnn_reshaped, W), b)

                weights = [var for var in tf.global_variables()
                           if var.name == 'deep_bidirectional_lstm/fully_connected/weights:0'][0]
                tf.summary.histogram('weights', weights)
                bias = [var for var in tf.global_variables()
                        if var.name == 'deep_bidirectional_lstm/fully_connected/bias:0'][0]
                tf.summary.histogram('bias', bias)

            # fc_out = tf.contrib.layers.fully_connected(inputs=rnn_reshaped,
            #                                            num_outputs=self.config.nClasses,
            #                                            activation_fn=None,
            #                                            trainable=True
            #                                            # weights_initializer=tf.Variable(tf.truncated_normal([2*list_n_hidden[-1], n_classes])),
            #                                            # biases_initializer=tf.Variable(tf.truncated_normal([n_classes])),
            #                                            )  # [batch x width, n_classes]

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
