#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

import logging
logging.getLogger("tensorflow").setLevel(logging.INFO)

from src.config import Params
from src.model import get_model_train
from src.preprocessing import data_preprocessing
from src.data_handler import dataset_generator
import tensorflow as tf
import numpy as np
import time
import os
import json
from glob import glob
from sacred import Experiment

ex = Experiment('crnn')

ex.add_config('config.json')

@ex.automain
def training(_config: dict):
    parameters = Params(**_config)

    export_config_filename =  os.path.join(parameters.output_model_dir, 'config.json')
    export_architecture_filename = os.path.join(parameters.output_model_dir, 'architecture.json')

    # saveweights_dir = os.path.join(parameters.output_model_dir, 'saved_weights')
    # savemodel_dir = os.path.join(parameters.output_model_dir, 'saved_model')
    saved5model_dir = os.path.join(parameters.output_model_dir, 'saved_h5_model')

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
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    checkpoint_filepath = os.path.join(parameters.output_model_dir, 'model_checkpoint',
                                       'cp-{epoch:03d}-{val_loss:.2f}')
    mc_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                     save_best_only=True,
                                                     save_weights_only=True,
                                                     period=parameters.save_interval)

    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5,
                                                       patience=10,
                                                       cooldown=0,
                                                       min_lr=1e-8,
                                                       verbose=1)

    if parameters.restore_model:
        last_time_stamp = max([int(p.split(os.path.sep)[-1].split('-')[0])
                               for p in glob(os.path.join(saved5model_dir, '*'))])
        model_file = os.path.join(saved5model_dir, '{}-model.h5'.format(last_time_stamp))
        assert os.path.isfile(model_file)
        model = get_model_train(parameters, model_path=model_file)
        # TODO update this line once load_model can load custom loss / metric / layers...
        # model = tf.keras.models.load_model(os.path.join(savemodel_dir, '{}-model.h5'.format(last_time_stamp)))

    else:
        # Get model
        model = get_model_train(parameters)

        # Save architecture
        model_json = model.to_json()
        with open(export_architecture_filename, 'w') as f:
            json.dump(model_json, f)

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
              epochs=parameters.n_epochs,
              steps_per_epoch=np.floor(n_samples_train / parameters.train_batch_size),
              validation_data=dataset_eval,
              validation_steps=np.floor(n_samples_eval / parameters.eval_batch_size),
              callbacks=[tb_callback, mc_callback, lr_callback])

    timestamp = str(int(time.time()))
    # # Save weights
    # os.makedirs(saveweights_dir, exist_ok=True)
    # model.save_weights(os.path.join(saveweights_dir, timestamp, 'weights'),
    #                    save_format='tf')

    # Save all model
    os.makedirs(saved5model_dir, exist_ok=True)
    model.save(os.path.join(saved5model_dir, '{}-model.h5'.format(timestamp)))

    # TODO save with savedmodel, (restore is not working at the moment...)
    # os.makedirs(savemodel_dir, exist_ok=True)
    # tf.keras.models.save_model(model,
    #                            os.path.join(savemodel_dir, timestamp),
    #                            include_optimizer=True,
    #                            save_format="tf")


