#!/usr/bin/env python
__author__ = 'solivr'

import tensorflow as tf
import numpy as np
import time
from model import CRNN, CTC
from config import Conf
from dataset import Dataset
from decoding import simpleDecoder, evaluation_metrics
import argparse


# CONFIG PARAMETERS
# -----------------

# Limit the usage of GPU memory to 30%
config_sess = tf.ConfigProto()
config_sess.gpu_options.per_process_gpu_memory_fraction = 0.3

# config = Conf(n_classes=37,
#               train_batch_size=128,
#               eval_batch_size=256,
#               learning_rate=0.01,  # 0.001 for adadelta
#               decay_rate=0.9,
#               max_iteration=3000000,
#               max_epochs=100,
#               eval_interval=100,
#               save_interval=2500,
#               file_writer='../ada_d09-01',
#               data_set='/home/soliveir/NAS-DHProcessing/mnt/ramdisk/max/90kDICT32px/',
#               model_dir='../model-crnn-ada_d09-01/',
#               input_shape=[32, 100],
#               list_n_hidden=[256, 256],
#               max_len=24)

session = tf.Session(config=config_sess)


def crnn_train(conf, sess=session):

    # PLACEHOLDERS
    # ------------

    # Sequence length, parameter of stack_bidirectional_dynamic_rnn,
    # rnn_seq_len = tf.placeholder(tf.int32, [None], name='sequence_length')
    target_seq_len = tf.placeholder(tf.int32, [None], name='target_seq_len')
    input_ctc_seq_len = tf.placeholder(tf.int32, [None], name='input_ctc_seq_len')
    is_training = tf.placeholder(tf.bool, name='trainable')

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, conf.inputShape[0], conf.inputShape[1], 1], name='input')
    keep_prob = tf.placeholder(tf.float32, name='dropout_prob')
    labels = tf.placeholder(tf.int32, [None], name='labels')

    # Sequence length
    train_seq_len = [conf.maxLength for _ in range(conf.trainBatchSize)]
    test_seq_len = [conf.maxLength for _ in range(conf.evalBatchSize)]

    # NETWORK
    # -------

    # Network and ctc definition
    crnn = CRNN(x, conf, is_training, keep_prob, session=sess)
    ctc = CTC(crnn.prob, labels, target_seq_len, inputSeqLength=input_ctc_seq_len)

    # Optimizer defintion
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(conf.learning_rate, global_step, conf.decay_steps,
                                               conf.decay_rate, staircase=True)
    if conf.optimizer == 'ada':
        optimizer = tf.train.AdadeltaOptimizer(learning_rate).minimize(ctc.loss, global_step=global_step)
    elif conf.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ctc.loss, global_step=global_step)
    elif conf.optimizer == 'rms':
        optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(ctc.loss, global_step=global_step)
    else:
        print('Error, no optimizer. RMS by default.')
        optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(ctc.loss, global_step=global_step)

    # ERRORS EVALUATION
    # -----------------
    # accuracy, WER, CER = evaluation_metrics(predicted_string, true_string)


    # SUMMARIES
    # ---------

    # Cost
    tf.summary.scalar('cost', ctc.cost)
    # Learning rate
    tf.summary.scalar('learning_rate', learning_rate)
    # Accuracy
    accuracy = tf.placeholder(tf.float32, None, name='accuracy_var')
    tf.summary.scalar('accuracy', accuracy)
    # WER
    WER = tf.placeholder(tf.float32, None, name='WER')
    tf.summary.scalar('WordER', WER)
    # CER
    CER = tf.placeholder(tf.float32, None, name='CER')
    tf.summary.scalar('CharER', CER)

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
                        mode='val')

    # RUN SESSION
    # -----------

    start_time = time.time()
    t = start_time
    while step < conf.maxIteration:
        # Prepare batch and add channel dimension
        images_batch, label_set, seq_len = data_train.nextBatch(conf.trainBatchSize)
        if images_batch is None:
            continue

        images_batch = np.expand_dims(images_batch, axis=-1)

        cost, _, step = sess.run([ctc.cost, optimizer, global_step],
                                 feed_dict={
                                           x: images_batch,
                                           keep_prob: 0.7,
                                           # rnn_seq_len: train_seq_len,
                                           input_ctc_seq_len: train_seq_len,
                                           target_seq_len: seq_len,
                                           labels: label_set[1],
                                           is_training: True,
                                        })

        # Eval accuarcy
        if step != 0 and step % conf.evalInterval == 0:
            images_batch_eval, label_set_eval, seq_len_eval = data_test.nextBatch()
            images_batch_eval = np.expand_dims(images_batch_eval, axis=-1)

            cost_eval, raw_pred = sess.run([ctc.cost, crnn.rawPred],
                                feed_dict={
                                            x: images_batch_eval,
                                            keep_prob: 1.0,
                                            is_training: False,
                                            # rnn_seq_len: test_seq_len,
                                            input_ctc_seq_len: test_seq_len,
                                            target_seq_len: seq_len_eval,
                                            labels: label_set_eval[1],
                                           })

            str_pred = simpleDecoder(raw_pred)
            # acc = eval_accuracy(str_pred, label_set_eval[0])
            # wer = eval_WER(str_pred, label_set_eval[0])
            # cer = eval_CER(str_pred, label_set_eval[0])
            acc, wer, cer = evaluation_metrics(str_pred, label_set_eval[0])

            print('step: {}, cost: {}, eval accuracy: {}, cer: {}'.format(step, cost_eval, acc, cer))

            # for i in range(5):
            #     print('original: {}, predicted(no decode): {}, predicted: {}'.format(label_set_eval[0][i],
            #                                                                          str_pred[i]))

            t = time.time()

            summary, step = sess.run([merged, global_step],
                                     feed_dict={
                                         x: images_batch,
                                         keep_prob: 1.0,
                                         # rnn_seq_len: train_seq_len,
                                         input_ctc_seq_len: train_seq_len,
                                         target_seq_len: seq_len,
                                         labels: label_set[1],
                                         is_training: False,
                                         accuracy: acc,
                                         WER: wer,
                                         CER: cer
                                     })

            train_writer.add_summary(summary, step)

        if step != 0 and step % conf.saveInterval == 0:
            crnn.saveModel(conf.modelDir, step)

        if step >= conf.maxIteration:
            print('{} training has completed'.format(conf.maxIteration))
            crnn.saveModel(conf.modelDir, step)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--learning_rate', type=float, help='Starting learning rate', default=0.001)
    parser.add_argument('-d', '--decay_rate', type=float, help='Decay rate for learning rate', default=0.9)
    parser.add_argument('-s', '--decay_steps', type=int, help='Decay steps for learning rate', default=1000)
    parser.add_argument('-e', '--eval_interval', type=float, help='Evaluation interval (steps)', default=500)
    parser.add_argument('-o', '--optimizer', type=str, help='Optimizer (ada, adam or rms)', default='rms')

    args = parser.parse_args()

    config = Conf(n_classes=37,
                  train_batch_size=128,
                  # eval_batch_size=64,
                  learning_rate=args.learning_rate,
                  decay_rate=args.decay_rate,
                  decay_steps=args.decay_steps,
                  optimizer=args.optimizer,
                  max_iteration=3000000,
                  max_epochs=100,
                  eval_interval=args.eval_interval,
                  save_interval=2500,
                  file_writer='../{}_d{}-l{}'.format(args.optimizer, args.decay_rate, args.learning_rate),
                  data_set='/home/soliveir/NAS-DHProcessing/mnt/ramdisk/max/90kDICT32px/',
                  model_dir='../model-crnn-{}_d{}-l{}'.format(args.optimizer, args.decay_rate, args.learning_rate),
                  input_shape=[32, 100],
                  list_n_hidden=[256, 256],
                  max_len=24)

    crnn_train(conf=config)
