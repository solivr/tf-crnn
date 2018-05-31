#!/usr/bin/env python
__author__ = 'solivr'
__license__ = "GPL"

import os
import tensorflow as tf
import click
from tf_crnn.model import crnn_fn
from tf_crnn.data_handler import serving_single_input
from tf_crnn.config import Params, TrainingParams, import_params_from_json
try:
    import better_exceptions
except ImportError:
    pass


@click.command()
@click.argument('--model-directory', help='Path to model directory')
@click.argument('--output-dir', help='Output directory')
@click.option('gpu', default='0', help='Which GPU to use')
def export_model(model_directory: str, output_dir: str, gpu: str):

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    config_sess = tf.ConfigProto()
    config_sess.gpu_options.per_process_gpu_memory_fraction = 0.6

    # Import parameters from the json file
    params_json = import_params_from_json(json_filename=os.path.join(model_directory, 'config.json'))
    training_params = TrainingParams(**params_json['training_params'])
    parameters = Params(**params_json)

    model_params = {
        'Params': parameters,
        'TrainingParams': training_params
    }

    # Config
    est_config = tf.estimator.RunConfig()
    est_config.replace(keep_checkpoint_max=10,
                       save_checkpoints_steps=training_params.save_interval,
                       session_config=config_sess,
                       save_checkpoints_secs=None,
                       save_summary_steps=1000)

    estimator = tf.estimator.Estimator(model_fn=crnn_fn,
                                       params=model_params,
                                       model_dir=model_directory,
                                       config=est_config,
                                       )

    estimator.export_savedmodel(output_dir,
                                serving_input_receiver_fn=serving_single_input(
                                    fixed_height=parameters.input_shape[0], min_width=10))
