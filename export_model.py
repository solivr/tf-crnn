#!/usr/bin/env python
__author__ = 'solivr'

from src.data_handler import preprocess_image_for_prediction
from src.model import crnn_fn
import tensorflow as tf
import os
import argparse
try:
    import better_exceptions
except ImportError:
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_dir', type=str, help='Directory of model to be exported', default='./model')
    parser.add_argument('-e', '--export_dir', type=str, help='Directory for exported model', default='./exported_model')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # config_sess.gpu_options.visible_devices
    config_sess = tf.ConfigProto()
    config_sess.gpu_options.per_process_gpu_memory_fraction = 0.6

    model_params = {
            'input_shape': (32, 100),
            'starting_learning_rate': 0.0001,
            'optimizer': 'adam',
            'decay_rate': 0.9,
            'decay_steps': 10000,
            'digits_only': True
        }

    est_config = tf.estimator.RunConfig()
    est_config._keep_checkpoint_max = 10
    est_config._save_checkpoints_steps = 100
    est_config._session_config = config_sess
    est_config._save_checkpoints_secs = None
    est_config._save_summary_steps = 1

    estimator = tf.estimator.Estimator(model_fn=crnn_fn, params=model_params,
                                       model_dir=args.model_dir,
                                       config=est_config,
                                       )

    estimator.export_savedmodel(args.export_dir,
                                serving_input_receiver_fn=preprocess_image_for_prediction(min_width=10))


#
# def _signature_def_to_tensors(signature_def):
#     g = tf.get_default_graph()
#     return {k: g.get_tensor_by_name(v.name) for k,v in signature_def.inputs.items()}, \
#            {k: g.get_tensor_by_name(v.name) for k,v in signature_def.outputs.items()}
#
# with tf.Session(graph=tf.Graph()) as sess:
#     loaded_model = tf.saved_model.loader.load(sess, ["serve"], './exported_models/1499264748/')
#     input_dict, output_dict =_signature_def_to_tensors(loaded_model.signature_def['predictions'])
#     out = sess.run(output_dict, feed_dict={input_dict['images']: img_test[:,:,None]})

# to see all : saved_model_cli show --dir . --all