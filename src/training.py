#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

from .config import Params
from src.model import get_model_train
from src.preprocessing import data_preprocessing
from src.data_handler import dataset_generator
import tensorflow as tf
import numpy as np
import time
import os


def training(parameters: Params):
    # parameters = Params(input_shape=[128, 1800],
    #                     lookup_alphabet_file='/home/soliveir/crnn_benchmark/tf-crnn/data/alphabet/lookup_iam.json',
    #                     csv_files_train='/scratch/sofia/DT_IAM/csv_experiments/iam_lines_train_tf_format.csv',
    #                     csv_files_eval='/scratch/sofia/DT_IAM/csv_experiments/iam_lines_validation1_tf_format.csv',
    #                     num_beam_paths=1,
    #                     cnn_batch_norm=5 * [True],
    #                     max_chars_per_string=80,
    #                     learning_rate=1e-4,
    #                     train_batch_size=128,
    #                     eval_batch_size=128)

    # check if output folder already exists
    assert parameters.output_model_dir, \
        '{} already exists, you cannot use it as output directory.'.format(parameters.output_model_dir)
        # 'Set "restore_model=True" to continue training, or delete dir "rm -r {0}"'.format(parameters.output_model_dir)
    os.makedirs(parameters.output_model_dir)

    # data and csv preprocessing
    csv_train_file, csv_eval_file, \
    n_samples_train, n_samples_eval = data_preprocessing(parameters)

    # Get model
    model = get_model_train(parameters)

    # Get datasets
    dataset_train = dataset_generator([csv_train_file],
                                      parameters,
                                      batch_size=parameters.train_batch_size,
                                      data_augmentation=parameters.data_augmentation,
                                      num_epochs=parameters.n_epochs)

    dataset_eval = dataset_generator([csv_eval_file],
                                     parameters,
                                     batch_size=parameters.eval_batch_size,
                                     data_augmentation=False,
                                     num_epochs=parameters.n_epochs)

    # Create callbacks
    logdir = os.path.join(parameters.output_model_dir, 'logs')
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    # Train model
    model.fit(dataset_train,
              epochs=parameters.n_epochs,
              steps_per_epoch=np.floor(n_samples_train / parameters.train_batch_size),
              validation_data=dataset_eval,
              validation_steps=np.floor(n_samples_eval / parameters.eval_batch_size),
              callbacks=[tb_callback])

    # Save weights
    save_dir = os.path.join(parameters.output_model_dir, 'saved_weights', int(time.time()))
    model.save_weights(save_dir, save_format='tf')

