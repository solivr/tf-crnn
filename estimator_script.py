#!/usr/bin/env python
__author__ = 'solivr'

import argparse
import os
import csv
import numpy as np
import better_exceptions

import tensorflow as tf
from src.model_estimator import crnn_fn
from src.data_handler import data_loader
from src.data_handler import preprocess_image_for_prediction

from src.config import Conf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--output_dir', type=str, help='Directory for output', default='./estimator')
    parser.add_argument('-g', '--gpu', type=str, help='GPU 0,1 or '' ', default='')
    parser.add_argument('-o', '--optimizer', type=str, help='Optimizer (rms, ada, adam)', default='adam')
    parser.add_argument('-ft', '--csv_files_train', type=str, help='CSV filename for training',
                        nargs='*', default=None)
    parser.add_argument('-fe', '--csv_files_eval', type=str, help='CSV filename for evaluation',
                        nargs='*', default=None)
    parser.add_argument('-e', '--export_dir', type=str, help='Export model directoy', default=None)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # config_sess.gpu_options.visible_devices
    config_sess = tf.ConfigProto()
    config_sess.gpu_options.per_process_gpu_memory_fraction = 0.6

    conf = Conf(n_classes=37,
                train_batch_size=128,
                eval_batch_size=1000,
                learning_rate=0.001,  # 0.001 recommended
                decay_rate=0.9,
                max_epochs=50,
                eval_interval=1000,
                save_interval=10000,
                input_shape=(32, 100),
                )

    filenames_train = args.csv_files_train
    filenames_eval = args.csv_files_eval

    model_params = {
        'input_shape': conf.inputShape,
        'starting_learning_rate': conf.learning_rate,
        'optimizer': args.optimizer,
        'decay_rate': conf.decay_rate,
        'decay_steps': 10000,
        'digits_only': True         # change to false to use all alphabet a-z
    }

    # Config estimator
    est_config = tf.estimator.RunConfig()
    # use replace instead
    est_config._keep_checkpoint_max = 10
    est_config._save_checkpoints_steps = conf.saveInterval
    est_config._session_config = config_sess
    est_config._save_checkpoints_secs = None
    est_config._save_summary_steps = 1000

    estimator = tf.estimator.Estimator(model_fn=crnn_fn,
                                       params=model_params,
                                       model_dir=args.output_dir,
                                       config=est_config
                                       )

    try:
        # Count number of image filenames in csv
        n_samples = 0
        for file in filenames_train:
            with open(file, 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter=' ')
                n_samples += len(list(reader))

        # train_steps = conf.evalInterval
        # glb_step = 0
        while True:
            # Train for approximately 1 epoch and then evaluate
            estimator.train(input_fn=data_loader(csv_filename=filenames_train,
                                                 # cursor=(glb_step * conf.trainBatchSize) % n_samples,
                                                 batch_size=conf.trainBatchSize,
                                                 input_shape=model_params['input_shape'],
                                                 num_epochs=conf.maxEpochs,
                                                 data_augmentation=True),
                            steps=np.floor(n_samples/conf.trainBatchSize))
            # glb_step += train_steps
            estimator.evaluate(input_fn=data_loader(csv_filename=filenames_eval,
                                                    batch_size=conf.evalBatchSize,
                                                    input_shape=model_params['input_shape'],
                                                    num_epochs=1),
                               steps=None)
    except KeyboardInterrupt:
        print('Interrupted')
        if args.export:
            estimator.export_savedmodel(args.export,
                                        serving_input_receiver_fn=preprocess_image_for_prediction(min_width=10))
            print('Exported model to {}'.format(args.export))

    if args.export:
        estimator.export_savedmodel(args.export,
                                    serving_input_receiver_fn=preprocess_image_for_prediction(min_width=10))
        print('Exported model to {}'.format(args.export))