#!/usr/bin/env python
__author__ = 'solivr'

import tensorflow as tf
import numpy as np
import time
from model import CRNN, CTC
from config import Conf
from dataset import Dataset
from decoding import convertSparseArrayToStr, simpleDecoder, eval_accuracy, simpleDecoderWithBlank


# CONFIG PARAMETERS
# -----------------

# Limit the usage of GPU memory to 30%
config_sess = tf.ConfigProto()
config_sess.gpu_options.per_process_gpu_memory_fraction = 0.3

config = Conf(n_classes=37,
              train_batch_size=128,
              test_batch_size=32,
              learning_rate=0.001,  # 0.001 for adadelta
              max_iteration=3000000,
              max_epochs=100,
              display_interval=200,
              test_interval=200,
              save_interval=2500,
              file_writer='../rms01',
              data_set='/home/soliveir/NAS-DHProcessing/mnt/ramdisk/max/90kDICT32px/',
              model_dir='../model-crnn-rms01/',
              input_shape=[32, 100],
              list_n_hidden=[256, 256],
              max_len=24)

session = tf.Session(config=config_sess)


def crnn_train(conf=config, sess=session):

    # PLACEHOLDERS
    # ------------

    # Sequence length, parameter of stack_bidirectional_dynamic_rnn,
    rnn_seq_len = tf.placeholder(tf.int32, [None], name='sequence_length')
    target_seq_len = tf.placeholder(tf.int32, [None], name='target_seq_len')
    input_ctc_seq_len = tf.placeholder(tf.int32, [None], name='input_ctc_seq_len')
    is_training = tf.placeholder(tf.bool, name='trainable')

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, conf.inputShape[0], conf.inputShape[1], 1], name='input')
    keep_prob = tf.placeholder(tf.float32, name='dropout_prob')
    labels = tf.placeholder(tf.int32, [None], name='labels')

    # Sequence length
    train_seq_len = [conf.maxLength for _ in range(conf.trainBatchSize)]
    test_seq_len = [conf.maxLength for _ in range(conf.testBatchSize)]


    # NETWORK
    # -------

    # Network and ctc definition
    crnn = CRNN(x, conf, rnn_seq_len, is_training, keep_prob, session=sess)
    ctc = CTC(crnn.prob, labels, target_seq_len, inputSeqLength=input_ctc_seq_len)

    # Optimizer defintion
    global_step = tf.Variable(0)
    #optimizer = tf.train.AdadeltaOptimizer(conf.learning_rate).minimize(ctc.loss, global_step=global_step)
    optimizer = tf.train.RMSPropOptimizer(conf.learning_rate).minimize(ctc.loss, global_step=global_step)


    # SUMMARIES
    # ---------

    # Cost
    #tf.summary.scalar('cost', ctc.cost)
    tf.summary.scalar('cost_warp', ctc.cost)
    # Time spent per batch
    time_batch = tf.placeholder(tf.float32, None, name='time_var')
    tf.summary.scalar('time', time_batch)
    # Accuracy
    accuracy = tf.placeholder(tf.float32, None, name='accuracy_var')
    tf.summary.scalar('accuracy', accuracy)

    # Summary Writer
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(conf.fileWriter,
                                         graph=tf.get_default_graph(),
                                         flush_secs=10)

    # tf.summary.FileWriter("./graph", graph=tf.get_default_graph(), flush_secs=10)

    # GRAPH
    # -----

    # Initialize graph
    init = tf.global_variables_initializer()
    sess.run(init)
    step = 0

    # Data
    data_train = Dataset(conf,
                         path=conf.dataSet,
                         mode='train')
    data_test = Dataset(conf,
                        path=conf.dataSet,
                        mode='test')

    # RUN SESSION
    # -----------

    start_time = time.time()
    t = start_time
    accuracy_train = 0
    while step < conf.maxIteration:
        # Prepare batch and add channel dimension
        images_batch, label_set, seq_len = data_train.nextBatch(conf.trainBatchSize)
        images_batch = np.expand_dims(images_batch, axis=-1)

        cost, _, step = sess.run([ctc.cost, optimizer, global_step],
                                 feed_dict={
                                               x: images_batch,
                                               keep_prob: 0.7,
                                               rnn_seq_len: train_seq_len,
                                               input_ctc_seq_len: train_seq_len,
                                               target_seq_len: seq_len,
                                               labels: label_set[1],
                                               is_training: True,
                                             })

        # Eval accuarcy
        if step != 0 and step % conf.testInterval == 0:
            images_batch_test, label_set_test, seq_len_test = data_test.nextBatch(conf.testBatchSize)
            images_batch_test = np.expand_dims(images_batch_test, axis=-1)

            raw_pred = sess.run([crnn.rawPred],
                                feed_dict={
                                            x: images_batch_test,
                                            keep_prob: 1.0,
                                            is_training: False,
                                            rnn_seq_len: test_seq_len,
                                            input_ctc_seq_len: test_seq_len,
                                            target_seq_len: seq_len_test,
                                            labels: label_set_test[1]
                                           })

            # convert coding to strings
            str_pred_orginal = label_set_test[0]
            str_pred_blank = simpleDecoderWithBlank(raw_pred[0])
            str_pred = simpleDecoder(raw_pred[0])

            # evaluate accuracy
            accuracy_train = eval_accuracy(str_pred, str_pred_orginal)
            print('step: {}, training accuracy: {}'.format(step, accuracy_train))

            for i in range(5):
                print('original: {}, predicted(no decode): {}, predicted: {}'.format(str_pred_orginal[i], str_pred_blank[i],
                                                                                     str_pred[i]))

        # Display
        if step % conf.displayInterval == 0:
            time_elapse = time.time() - t
            t = time.time()
            total_time = time.time() - start_time
            print('* step: {}, cost: {}, step time: {:.2}s, total time: {:.2}s'.format(step, cost, time_elapse,
                                                                                       total_time))

            summary, step = sess.run([merged, global_step],
                                     feed_dict={
                                         x: images_batch,
                                         keep_prob: 0.7,
                                         rnn_seq_len: train_seq_len,
                                         input_ctc_seq_len: train_seq_len,
                                         target_seq_len: seq_len,
                                         labels: label_set[1],
                                         is_training: False,
                                         time_batch: time_elapse,
                                         accuracy: accuracy_train
                                     })

            train_writer.add_summary(summary, step)

        if step != 0 and step % conf.saveInterval == 0:
            crnn.saveModel(conf.modelDir, step)

        if step >= conf.maxIteration:
            print('{} training has completed'.format(conf.maxIteration))
            crnn.saveModel(conf.modelDir, step)


if __name__ == '__main__':
    crnn_train()
