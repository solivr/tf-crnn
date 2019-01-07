#!/usr/bin/env python
__author__ = 'solivr'
__license__ = "GPL"

import os
import json
from distutils.version import LooseVersion
from sacred import Experiment
from tqdm import trange
import tensorflow as tf
from typing import List
import string
try:
    import better_exceptions
except ImportError:
    pass
from tf_crnn.model import crnn_fn
from tf_crnn.data_handler import data_loader, serving_single_input, serving_batch_filenames_fn
from tf_crnn.config import Params, TrainingParams

ex = Experiment('CRNN_experiment')


def distribution_gpus(num_gpus):
    if num_gpus == 1:
        return tf.contrib.distribute.OneDeviceStrategy(device=tf.DeviceSpec(device_type="GPU", device_index=0))
    elif num_gpus > 1:
        return tf.contrib.distribute.MirroredStrategy(num_gpus=num_gpus)
    else:
        return None


@ex.config
def default_config():
    csv_files_train = None
    csv_files_eval = None
    output_model_dir = None
    num_gpus = 1
    lookup_alphabet_file = ''
    input_shape = (32, 100)
    num_beam_paths = 2
    training_params = TrainingParams().to_dict()
    restore_model = False
    csv_delimiter = ';'
    string_split_delimiter = '|'
    data_augmentation = True
    data_augmentation_max_rotation = 0.05
    input_data_n_parallel_calls = 4


@ex.automain
def run(csv_files_train: List[str], csv_files_eval: List[str], output_model_dir: str,
        training_params: dict, _config):

    # Save config
    if not os.path.isdir(output_model_dir):
        os.makedirs(output_model_dir)
    else:
        assert _config.get('restore_model'), \
            '{0} already exists, you cannot use it as output directory. ' \
            'Set "restore_model=True" to continue training, or delete dir "rm -r {0}"'.format(output_model_dir)

    with open(os.path.join(output_model_dir, 'config.json'), 'w') as f:
        json.dump(_config, f, indent=4, sort_keys=True)

    parameters = Params(**_config)
    training_params = TrainingParams(**training_params)

    model_params = {
        'Params': parameters,
        'TrainingParams': training_params
    }

    # Create export directory
    export_dir = os.path.join(output_model_dir, 'export')
    if not os.path.isdir(export_dir):
        os.makedirs(export_dir)

    # Check if alphabet contains all chars in csv input files
    discarded_chars = parameters.string_split_delimiter + parameters.csv_delimiter + string.whitespace[1:]
    parameters.alphabet.check_input_file_alphabet(
        parameters.csv_files_train + parameters.csv_files_eval,
        discarded_chars=discarded_chars,
        csv_delimiter=parameters.csv_delimiter)

    config_sess = tf.ConfigProto()
    # config_sess.gpu_options.per_process_gpu_memory_fraction = 0.9
    # config_sess.gpu_options.allow_growth = True

    # Config estimator
    est_config = tf.estimator.RunConfig()
    if LooseVersion(tf.__version__) < LooseVersion('1.8'):
        est_config.replace(keep_checkpoint_max=10,
                           save_checkpoints_steps=training_params.save_interval,
                           session_config=config_sess,
                           save_checkpoints_secs=None,
                           save_summary_steps=1000,
                           model_dir=output_model_dir)
    else:
        est_config.replace(keep_checkpoint_max=10,
                           save_checkpoints_steps=training_params.save_interval,
                           session_config=config_sess,
                           save_checkpoints_secs=None,
                           save_summary_steps=1000,
                           model_dir=output_model_dir,
                           train_distribute=distribution_gpus(parameters.num_gpus))

    estimator = tf.estimator.Estimator(model_fn=crnn_fn,
                                       params=model_params,
                                       model_dir=output_model_dir,
                                       config=est_config
                                       )

    for e in trange(0, training_params.n_epochs, training_params.evaluate_every_epoch):

        estimator.train(input_fn=data_loader(csv_filename=csv_files_train,
                                             params=parameters,
                                             batch_size=training_params.train_batch_size,
                                             num_epochs=training_params.evaluate_every_epoch,
                                             data_augmentation=parameters.data_augmentation,
                                             image_summaries=True))

        estimator.export_savedmodel(export_dir,
                                    serving_input_receiver_fn=serving_single_input(
                                        fixed_height=parameters.input_shape[0], min_width=10))

        estimator.evaluate(input_fn=data_loader(csv_filename=csv_files_eval,
                                                params=parameters,
                                                batch_size=training_params.eval_batch_size,
                                                num_epochs=1))
