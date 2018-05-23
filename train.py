#!/usr/bin/env python
__author__ = 'solivr'
__license__ = "GPL"

import csv
import os
import json
import numpy as np
from sacred import Experiment
from tqdm import trange
import tensorflow as tf
from typing import List
try:
    import better_exceptions
except ImportError:
    pass
from tf_crnn.model import crnn_fn
from tf_crnn.data_handler import data_loader, preprocess_image_for_prediction
from tf_crnn.config import Params, TrainingParams

ex = Experiment('CRNN_experiment')


@ex.config
def default_config():
    csv_files_train = None
    csv_files_eval = None
    output_model_dir = None
    gpu = ''
    lookup_alphabet_file = ''
    input_shape = (32, 100)
    training_params = TrainingParams().to_dict()


@ex.automain
def run(csv_files_train: List[str], csv_files_eval: List[str], output_model_dir: str,
        gpu: str, training_params: dict, _config):

    parameters = Params(**_config)
    training_params = TrainingParams(**training_params)

    model_params = {
        'Params': parameters,
        'TrainingParams': training_params
    }

    with open(os.path.join(output_model_dir, 'config.json'), 'w') as f:
        json.dump(_config, f, indent=4, sort_keys=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    config_sess = tf.ConfigProto()
    config_sess.gpu_options.per_process_gpu_memory_fraction = 0.8
    config_sess.gpu_options.allow_growth = True

    # Config estimator
    est_config = tf.estimator.RunConfig()
    est_config.replace(keep_checkpoint_max=10,
                       save_checkpoints_steps=training_params.save_interval,
                       session_config=config_sess,
                       save_checkpoints_secs=None,
                       save_summary_steps=1000,
                       model_dir=output_model_dir)

    estimator = tf.estimator.Estimator(model_fn=crnn_fn,
                                       params=model_params,
                                       model_dir=output_model_dir,
                                       config=est_config
                                       )

    # Count number of image filenames in csv
    n_samples = 0
    for file in csv_files_eval:
        with open(file, 'r', encoding='utf8') as csvfile:
            reader = csv.reader(csvfile, delimiter=parameters.csv_delimiter)
            n_samples += len(list(reader))

    try:
        for e in trange(0, training_params.n_epochs, training_params.evaluate_every_epoch):

            estimator.train(input_fn=data_loader(csv_filename=csv_files_train,
                                                 params=parameters,
                                                 batch_size=training_params.train_batch_size,
                                                 num_epochs=training_params.evaluate_every_epoch,
                                                 data_augmentation=True,
                                                 image_summaries=True))

            estimator.export_savedmodel(os.path.join(output_model_dir, 'export'),
                                        serving_input_receiver_fn=preprocess_image_for_prediction(min_width=10))

            estimator.evaluate(input_fn=data_loader(csv_filename=csv_files_eval,
                                                    params=parameters,
                                                    batch_size=training_params.eval_batch_size,
                                                    num_epochs=1),
                               steps=np.floor(n_samples / training_params.eval_batch_size)
                               )

    except KeyboardInterrupt:
        print('Interrupted')
        estimator.export_savedmodel(os.path.join(output_model_dir, 'export'),
                                    preprocess_image_for_prediction(min_width=10))
        print('Exported model to {}'.format(os.path.join(output_model_dir, 'export')))

    estimator.export_savedmodel(os.path.join(output_model_dir, 'export'),
                                preprocess_image_for_prediction(min_width=10))
    print('Exported model to {}'.format(os.path.join(output_model_dir, 'export')))
