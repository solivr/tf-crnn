#!/usr/bin/env python
__author__ = 'solivr'

import argparse
import os

import tensorflow as tf
from .src.model_estimator import crnn_fn, data_loader

from .src.config import Conf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--output_dir', type=str, help='Directory for output', default='./estimator')
    parser.add_argument('-g', '--gpu', type=str, help='GPU 0,1 or '' ', default='')
    parser.add_argument('-t', '--isTraining', type=bool, help='1 for training, 0 for evaluation', default=True)
    parser.add_argument('-o', '--optimizer', type=str, help='Optimizer (rms, ada, adam)', default='rms')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    config_sess = tf.ConfigProto()
    config_sess.gpu_options.per_process_gpu_memory_fraction = 0.3

    conf = Conf(n_classes=37,
                train_batch_size=128,
                eval_batch_size=32,
                learning_rate=0.001,  # 0.001 for adadelta
                decay_rate=0.9,
                max_iteration=300000,
                eval_interval=200,
                save_interval=2500,
                # data_set='/home/soliveir/NAS-DHProcessing/mnt/ramdisk/max/90kDICT32px/',
                # data_set='/scratch/sofia/synth-data/90kDICT32px/',
                data_set='/scratch/sofia/synth-data/IIIT-data/IIIT-HWS-Dataset/groundtruth/',
                input_shape=[32, 100],
                list_n_hidden=[256, 256],
                max_len=24)

    # filename_train = os.path.join(conf.dataSet, 'new_annotation_train.csv')
    # filename_eval = os.path.join(conf.dataSet, 'new_annotation_val.csv')
    filename_train = os.path.abspath(os.path.join(conf.dataSet, 'new_iiit_hw_train.csv'))
    filename_eval = os.path.abspath(os.path.join(conf.dataSet, 'new_iiit_hw_val.csv'))

    model_params = {
        'input_shape': conf.inputShape,
        'starting_learning_rate': conf.learning_rate,
        'optimizer': args.optimizer,
        'decay_rate': conf.decay_rate,
        'decay_steps': 10000,
        'max_length': conf.maxLength
    }

    estimator = tf.estimator.Estimator(model_fn=crnn_fn,
                                       params=model_params,
                                       model_dir=args.output_dir,
                                       # config=
                                       )

    # if args.isTraining:
    #     estimator.train(input_fn=data_loader(filename_train, 128, num_epochs=20))
    # else:
    #     estimator.evaluate(input_fn=data_loader(filename_eval, 128, num_epochs=1))

    try:
        while True:
            # Train for 10K steps and then evaluate
            estimator.train(input_fn=data_loader(csv_filename=filename_train, batch_size=128, num_epochs=20), steps=10e3)

            estimator.evaluate(input_fn=data_loader(csv_filename=filename_eval, batch_size=2000), steps=3)
    except KeyboardInterrupt:
        print('Interrupted')
