#!/usr/bin/env python
__author__ = 'solivr'

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from tensorflow.contrib.rnn import BasicLSTMCell
import warpctc_tensorflow
import cv2
import numpy as np
import os
from crnn.decoding import get_words_from_chars


def weightVar(shape, mean=0.0, stddev=0.02, name='weights'):
    initW = tf.truncated_normal(shape=shape, mean=mean, stddev=stddev)
    return tf.Variable(initW, name=name)


def biasVar(shape, value=0.0, name='bias'):
    initb = tf.constant(value=value, shape=shape)
    return tf.Variable(initb, name=name)


def conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME', name=None):
    return tf.nn.conv2d(input, filter, strides=strides, padding=padding, name=name)


def deep_cnn(inputImgs: tf.Tensor, isTraining: bool) -> tf.Tensor:
    input_tensor = inputImgs

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

            # tf.summary.image('1st_sample', pool1[:, :, :, :1], 1)
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

            # tf.summary.image('1st_sample', pool2[:, :, :, :1], 1)
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
                                                   training=isTraining, name='batch-norm')
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

            # tf.summary.image('1st_sample', pool4[:, :, :, :1], 1)
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
                                                   training=isTraining, name='batch-norm')
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
                                                   training=isTraining, name='batch-norm')
            conv7 = tf.nn.relu(b_norm)

            weights = [var for var in tf.global_variables() if var.name == 'deep_cnn/layer7/weights:0'][0]
            tf.summary.histogram('weights', weights)
            bias = [var for var in tf.global_variables() if var.name == 'deep_cnn/layer7/bias:0'][0]
            tf.summary.histogram('bias', bias)

        cnn_net = conv7

        with tf.variable_scope('Reshaping_cnn'):
            shape = cnn_net.get_shape().as_list()  # [batch, height, width, features]
            transposed = tf.transpose(cnn_net, perm=[0, 2, 1, 3],
                                      name='transposed')  # [batch, width, height, features]
            conv_reshaped = tf.reshape(transposed, [-1, shape[2], shape[1] * shape[3]],
                                       name='reshaped')  # [batch, width, height x features]

    return conv_reshaped


def deep_bidirectional_lstm(inputs: tf.Tensor, params: dict) -> tf.Tensor:
    # Prepare data shape to match `bidirectional_rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input) "(batch, time, height)"

    listNHidden = [256, 256]
    nClasses = 37

    with tf.name_scope('deep_bidirectional_lstm'):
        # Forward direction cells
        fw_cell_list = [BasicLSTMCell(nh, forget_bias=1.0) for nh in listNHidden]
        # Backward direction cells
        bw_cell_list = [BasicLSTMCell(nh, forget_bias=1.0) for nh in listNHidden]

        lstm_net, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(fw_cell_list,
                                                                        bw_cell_list,
                                                                        inputs,
                                                                        dtype=tf.float32
                                                                        )

        # Dropout layer
        lstm_net = tf.nn.dropout(lstm_net, keep_prob=params['keep_prob'])

        with tf.variable_scope('Reshaping_rnn'):
            shape = lstm_net.get_shape().as_list()  # [batch, width, 2*n_hidden]
            rnn_reshaped = tf.reshape(lstm_net, [-1, shape[-1]])  # [batch x width, 2*n_hidden]

        with tf.variable_scope('fully_connected'):
            W = weightVar([listNHidden[-1]*2, nClasses])
            b = biasVar([nClasses])
            fc_out = tf.nn.bias_add(tf.matmul(rnn_reshaped, W), b)

            weights = [var for var in tf.global_variables()
                       if var.name == 'deep_bidirectional_lstm/fully_connected/weights:0'][0]
            tf.summary.histogram('weights', weights)
            bias = [var for var in tf.global_variables()
                    if var.name == 'deep_bidirectional_lstm/fully_connected/bias:0'][0]
            tf.summary.histogram('bias', bias)

        lstm_out = tf.reshape(fc_out, [-1, shape[1], nClasses], name='reshape_out')  # [batch, width, n_classes]

        rawPred = tf.argmax(tf.nn.softmax(lstm_out), axis=2, name='raw_prediction')

        # Swap batch and time axis
        lstm_out = tf.transpose(lstm_out, [1, 0, 2], name='transpose_time_major')  # [width(time), batch, n_classes]

        return lstm_out, rawPred


def data_loader(csv_filename, batch_size: int, input_shape=[32, 100]):

    def input_fn():
        # Choose case one csv file or list of csv files
        if not isinstance(csv_filename, list):
            dirname = os.path.dirname(csv_filename)
            filename_queue = tf.train.string_input_producer([csv_filename], num_epochs=50)
        elif isinstance(csv_filename, list):
            dirname = os.path.dirname(csv_filename[0])
            filename_queue = tf.train.string_input_producer(csv_filename, num_epochs=50)
        else:
            raise TypeError

        reader = tf.TextLineReader(name='CSV_Reader')
        key, value = reader.read(filename_queue, name='file_reading_op')

        default_line = [['None'], ['None']]
        path, label = tf.decode_csv(value, record_defaults=default_line, field_delim=' ', name='csv_reading_op')

        # Shuffle queue -> batch of 1 (allowing to use batch with dynamic pad after)

        # Get full path
        # full_dir = tf.constant([os.path.abspath(os.path.join(dirname, '..'))], tf.string)
        full_dir = dirname
        full_path = tf.string_join([full_dir, path], separator=os.path.sep)

        # Read image
        image_content = tf.read_file(full_path, name='image_reader')
        image = tf.image.decode_jpeg(image_content, channels=1, try_recover_truncated=True)
        # Reshape
        image = tf.image.resize_images(image, size=input_shape, method=tf.image.ResizeMethod.BICUBIC)

        # Data augmentation
        # TODO

        # Batch
        img_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=15,
                                                capacity=3000, dynamic_pad=False)

        return {'images': img_batch}, label_batch  # features = {img_batch, image width (rnn_seq_length)}

    return input_fn


def crnn_fn(features, labels, mode, params):
    """
    :param features: dict {
                            'images'
                            'rnn_seq_length'
                            'target_seq_length' }
    :param labels: labels. flattend (1D) array with encoded label (one code per character)
    :param mode:
    :param params: dict {
                            'input_shape'
                            'keep_prob'
                            'starting_learning_rate'
                            'optimizer'
                            'decay_steps'
                            'decay_rate'
                            'max_length'}
    :return:
    """
    if mode == 'train':
        isTraining = True
        params['keep_prob'] = 0.7
    else:
        isTraining = False
        params['keep_prob'] = 1.0

    # Alphabet and codes
    alphabet = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-'
    keys = [c for c in alphabet]
    values = list(range(36)) + list(range(10, 37))

    # Convert string to code
    table_str2int = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(keys, values), -1)
    splited = tf.string_split(labels, delimiter='')
    codes = table_str2int.lookup(splited.values)
    sparse_code_target = tf.SparseTensor(splited.indices, codes, splited.dense_shape)

    sequence_lengths = tf.segment_max(sparse_code_target.indices[:, 1], sparse_code_target.indices[:, 0]) + 1

    conv = deep_cnn(features['images'], isTraining)
    prob, raw_pred = deep_bidirectional_lstm(conv, params=params)  # params: rnn_seq_length, keep_prob
    predictions_dict = {'prob': prob, 'raw_predictions': raw_pred}

    # Loss
    loss_ctc = warpctc_tensorflow.ctc(activations=prob,
                                      flat_labels=sparse_code_target.values,
                                      label_lengths=tf.cast(sequence_lengths, tf.int32),
                                      input_lengths=tf.ones([tf.shape(labels)[0]], dtype=tf.int32)*params['max_length'],
                                      blank_label=36)
    loss_ctc = tf.reduce_mean(loss_ctc)

    # Train op
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(params['starting_learning_rate'], global_step, params['decay_steps'],
                                               params['decay_rate'], staircase=True)

    tf.summary.scalar('learning_rate', learning_rate)

    if params['optimizer'] == 'ada':
        optimizer = tf.train.AdadeltaOptimizer(learning_rate)
    elif params['optimizer'] == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5)
    elif params['optimizer'] == 'rms':
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
    else:
        print('Error, no optimizer. RMS by default.')
        optimizer = tf.train.RMSPropOptimizer(learning_rate)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss_ctc, global_step=global_step)

    # Evaluation ops

    # Convert code labels to string labels
    keys = np.arange(37, dtype=np.int64)
    alphabet_short = '0123456789abcdefghijklmnopqrstuvwxyz-'
    values = [c for c in alphabet_short]
    table_int2str = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(keys, values), '?')

    (sparse_code_pred,), neg_sum_logits = tf.nn.ctc_greedy_decoder(predictions_dict['prob'],
                                                                   tf.ones([tf.shape(labels)[0]],
                                                                           dtype=tf.int32) * params['max_length'],
                                                                   merge_repeated=True)
    # sparse_code_pred = tf.cast(sparse_code_pred, dtype=tf.int64)

    sequence_lengths = tf.segment_max(sparse_code_pred.indices[:, 1], sparse_code_pred.indices[:, 0]) + 1

    pred_chars = table_int2str.lookup(sparse_code_pred)
    predictions_dict['words'] = get_words_from_chars(pred_chars.values, sequence_lengths=sequence_lengths)

    CER = tf.metrics.mean(tf.edit_distance(sparse_code_pred, tf.cast(sparse_code_target, dtype=tf.int64)))
    WER = tf.metrics.accuracy(labels, predictions_dict['words'])

    eval_metric_ops = {'WER': WER,
                       'accuracy': 1 - WER[0],
                       'CER': CER,
                       'loss': loss_ctc}

    # Output
    return model_fn_lib.ModelFnOps(
        mode=mode,
        predictions=predictions_dict,
        loss=loss_ctc,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops
    )
