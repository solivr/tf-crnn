#!/usr/bin/env python
__author__ = 'solivr'

import argparse
import os
import csv
import better_exceptions

import tensorflow as tf
from src.model_estimator import crnn_fn
from src.data_handler import data_loader

from src.config import Conf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--output_dir', type=str, help='Directory for output', default='./estimator')
    parser.add_argument('-g', '--gpu', type=str, help='GPU 0,1 or '' ', default='')
    parser.add_argument('-o', '--optimizer', type=str, help='Optimizer (rms, ada, adam)', default='rms')
    parser.add_argument('-f', '--csv_file', type=str, help='CSV filename (without _{train, val, test}.csv extension)', default=None)
    parser.add_argument('-s', '--dataset_dir', type=str, help='Dataset directory ', default=None)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # config_sess.gpu_options.visible_devices
    config_sess = tf.ConfigProto()
    config_sess.gpu_options.per_process_gpu_memory_fraction = 0.6

    conf = Conf(n_classes=37,
                train_batch_size=128,
                eval_batch_size=1000,
                learning_rate=0.001,  # 0.001 for adadelta
                decay_rate=0.9,
                max_epochs=50,
                eval_interval=1000,
                save_interval=5000,
                # data_set='/scratch/sofia/synth-data/90kDICT32px/',
                # data_set='/scratch/sofia/synth-data/IIIT-data/IIIT-HWS-Dataset/groundtruth/',
                data_set='/scratch/sofia/vtm_data/',
                # data_set=args.dataset_dir,
                input_shape=(32, 100),
                # max_len=24
                )

    # filename_train = os.path.join(conf.dataSet, 'new_annotation_train.csv')
    # filename_eval = os.path.join(conf.dataSet, 'new_annotation_val.csv')
    # filename_train = os.path.abspath(os.path.join(conf.dataSet, 'new_iiit_hw_train.csv'))
    # filename_eval = os.path.abspath(os.path.join(conf.dataSet, 'new_iiit_hw_val.csv'))
    filename_train = os.path.abspath(os.path.join(conf.dataSet, 'numbers_train.csv'))
    filename_eval = os.path.abspath(os.path.join(conf.dataSet, 'numbers_val.csv'))

    model_params = {
        'input_shape': conf.inputShape,
        'starting_learning_rate': conf.learning_rate,
        'optimizer': args.optimizer,
        'decay_rate': conf.decay_rate,
        'decay_steps': 10000,
        # 'max_length': conf.maxLength,
        'digits_only': True
    }

    # Config estimator
    est_config = tf.estimator.RunConfig()
    # use replace instead
    est_config._keep_checkpoint_max = 10
    est_config._save_checkpoints_steps = conf.saveInterval
    est_config._session_config = config_sess
    est_config._save_checkpoints_secs = None
    est_config._save_summary_steps = 500

    estimator = tf.estimator.Estimator(model_fn=crnn_fn,
                                       params=model_params,
                                       model_dir=args.output_dir,
                                       config=est_config
                                       )

    try:
        # Count number of filenames in csv
        with open(filename_train, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            n_samples = len(list(reader))

        train_steps = conf.evalInterval
        glb_step = 0
        while True:
            # Train for 10K steps and then evaluate
            estimator.train(input_fn=data_loader(csv_filename=filename_train,
                                                 cursor=(glb_step * conf.trainBatchSize) % n_samples,
                                                 batch_size=conf.trainBatchSize,
                                                 input_shape=model_params['input_shape'],
                                                 num_epochs=conf.maxEpochs,
                                                 data_augmentation=True),
                            steps=train_steps)
            glb_step += train_steps
            estimator.evaluate(input_fn=data_loader(csv_filename=filename_eval,
                                                    batch_size=conf.evalBatchSize,
                                                    input_shape=model_params['input_shape']),
                               steps=3)
    except KeyboardInterrupt:
        print('Interrupted')

# Export model
# estimator.export_savedmodel('./exported_models/',
#                            serving_input_receiver_fn= preprocess_image_for_prediction())


# To get input and output dicts when 'restoring' saved_model
# def _signature_def_to_tensors(signature_def):
#     g = tf.get_default_graph()
#     return {k: g.get_tensor_by_name(v.name) for k,v in signature_def.inputs.items()}, \
#            {k: g.get_tensor_by_name(v.name) for k,v in signature_def.outputs.items()}

# to see all : saved_model_cli show --dir . --all