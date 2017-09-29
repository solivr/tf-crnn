#!/usr/bin/env python
__author__ = 'solivr'

import argparse
import os
import csv
import numpy as np
try:
    import better_exceptions
except ImportError:
    pass

import tensorflow as tf
from src.model_estimator import crnn_fn
from src.data_handler import data_loader
from src.data_handler import preprocess_image_for_prediction

from src.config import Params

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ft', '--csv_files_train', required=True, type=str, help='CSV filename for training',
                        nargs='*', default=None)
    parser.add_argument('-fe', '--csv_files_eval', type=str, help='CSV filename for evaluation',
                        nargs='*', default=None)
    parser.add_argument('-o', '--output_model_dir', required=True, type=str, help='Directory for output', default='./estimator')
    parser.add_argument('-g', '--gpu', type=str, help='GPU 0,1 or '' ', default='')
    args = vars(parser.parse_args())

    parameters = Params(n_classes=37,
                        train_batch_size=128,
                        eval_batch_size=1e3,
                        learning_rate=1e-3,  # 0.001 recommended
                        decay_rate=0.9,
                        decay_steps=10000,
                        max_epochs=50,
                        eval_interval=1e3,
                        save_interval=1e4,
                        input_shape=(32, 100),
                        # optimizer=args.get('optimizer')
                        optimizer='adam',
                        digits_only=False,
                        csv_files_eval=args.get('csv_files_eval'),
                        csv_files_train=args.get('csv_files_train'),
                        output_model_dir=args.get('output_model_dir'),
                        gpu=args.get('gpu')
                        )

    model_params = {
        'Params': parameters,
    }

    parameters.export_experiment_params()

    os.environ['CUDA_VISIBLE_DEVICES'] = parameters.gpu
    # config_sess.gpu_options.visible_devices
    config_sess = tf.ConfigProto()
    config_sess.gpu_options.per_process_gpu_memory_fraction = 0.6

    # Config estimator
    est_config = tf.estimator.RunConfig()
    # use replace instead
    est_config._keep_checkpoint_max = 10
    est_config._save_checkpoints_steps = parameters.save_interval
    est_config._session_config = config_sess
    est_config._save_checkpoints_secs = None
    est_config._save_summary_steps = 1000

    estimator = tf.estimator.Estimator(model_fn=crnn_fn,
                                       params=model_params,
                                       model_dir=parameters.output_model_dir,
                                       config=est_config
                                       )

    try:
        # Count number of image filenames in csv
        n_samples = 0
        for file in parameters.csv_files_train:
            with open(file, 'r', encoding='utf8') as csvfile:
                reader = csv.reader(csvfile, delimiter=parameters.csv_delimiter)
                n_samples += len(list(reader))

        # train_steps = conf.evalInterval
        # glb_step = 0
        while True:
            # Train for approximately 1 epoch and then evaluate
            estimator.train(input_fn=data_loader(csv_filename=parameters.csv_files_train,
                                                 # cursor=(glb_step * conf.trainBatchSize) % n_samples,
                                                 batch_size=parameters.train_batch_size,
                                                 input_shape=parameters.input_shape,
                                                 num_epochs=parameters.max_epochs,
                                                 data_augmentation=True),
                            steps=np.floor(n_samples / parameters.train_batch_size))
            # glb_step += train_steps
            estimator.evaluate(input_fn=data_loader(csv_filename=parameters.csv_files_eval,
                                                    batch_size=parameters.eval_batch_size,
                                                    input_shape=model_params['input_shape'],
                                                    num_epochs=1),
                               steps=None)
    except KeyboardInterrupt:
        print('Interrupted')
        estimator.export_savedmodel(os.path.join(parameters.output_model_dir, 'export'),
                                    serving_input_receiver_fn=preprocess_image_for_prediction(min_width=10))
        print('Exported model to {}'.format(os.path.join(parameters.output_model_dir, 'export')))

    estimator.export_savedmodel(os.path.join(parameters.output_model_dir, 'export'),
                                serving_input_receiver_fn=preprocess_image_for_prediction(min_width=10))
    print('Exported model to {}'.format(os.path.join(parameters.output_model_dir, 'export')))
