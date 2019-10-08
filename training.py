#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

import logging
logging.getLogger("tensorflow").setLevel(logging.INFO)

from src.config import Params
from src.model import get_model_train
from src.preprocessing import data_preprocessing
from src.data_handler import dataset_generator
from src.callbacks import CustomLoaderCallback, CustomSavingCallback, LRTensorBoard, EPOCH_FILENAME, FOLDER_SAVED_MODEL
import tensorflow as tf
import numpy as np
import os
import json
import pickle
from glob import glob
from sacred import Experiment, SETTINGS

SETTINGS.CONFIG.READ_ONLY_CONFIG = False

ex = Experiment('crnn')

ex.add_config('config.json')

@ex.automain
def training(_config: dict):
    parameters = Params(**_config)

    export_config_filename =  os.path.join(parameters.output_model_dir, 'config.json')
    saving_dir = os.path.join(parameters.output_model_dir, FOLDER_SAVED_MODEL)

    if not parameters.restore_model:
        # check if output folder already exists
        assert not os.path.isdir(parameters.output_model_dir), \
            '{} already exists, you cannot use it as output directory.'.format(parameters.output_model_dir)
            # 'Set "restore_model=True" to continue training, or delete dir "rm -r {0}"'.format(parameters.output_model_dir)
        os.makedirs(parameters.output_model_dir)

    # data and csv preprocessing
    csv_train_file, csv_eval_file, \
    n_samples_train, n_samples_eval = data_preprocessing(parameters)

    # export config file in model output dir
    with open(export_config_filename, 'w') as file:
        json.dump(parameters.to_dict(), file)

    # Create callbacks
    logdir = os.path.join(parameters.output_model_dir, 'logs')
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir,
                                                 profile_batch=0)

    lrtb_callback = LRTensorBoard(log_dir=logdir,
                                  profile_batch=0)

    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5,
                                                       patience=10,
                                                       cooldown=0,
                                                       min_lr=1e-8,
                                                       verbose=1)

    es_callback = tf.keras.callbacks.EarlyStopping(min_delta=0.005,
                                                   patience=20,
                                                   verbose=1)

    sv_callback = CustomSavingCallback(saving_dir,
                                       saving_freq=parameters.save_interval,
                                       save_best_only=True,
                                       keep_max_models=3)

    list_callbacks = [tb_callback, lrtb_callback, lr_callback, es_callback, sv_callback]

    if parameters.restore_model:
        last_time_stamp = max([int(p.split(os.path.sep)[-1].split('-')[0])
                               for p in glob(os.path.join(saving_dir, '*'))])

        loading_dir = os.path.join(saving_dir, str(last_time_stamp))
        ld_callback = CustomLoaderCallback(loading_dir)

        list_callbacks.append(ld_callback)

        with open(os.path.join(loading_dir, EPOCH_FILENAME), 'rb') as f:
            initial_epoch = pickle.load(f)

        epochs = initial_epoch + parameters.n_epochs
    else:
        initial_epoch = 0
        epochs = parameters.n_epochs

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

    # Train model
    model.fit(dataset_train,
              epochs=epochs,
              initial_epoch=initial_epoch,
              steps_per_epoch=np.floor(n_samples_train / parameters.train_batch_size),
              validation_data=dataset_eval,
              validation_steps=np.floor(n_samples_eval / parameters.eval_batch_size),
              callbacks=list_callbacks)
