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
from src.config import Params, import_params_from_json


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_dir', type=str, help='Directory of model to be exported', default='./model')
    parser.add_argument('-e', '--output_dir', type=str, help='Output directory (for exported model)', default='./exported_model')
    parser.add_argument('-g', '--gpu', type=str, help='GPU 1, 0 or '' for CPU', default='')
    args = vars(parser.parse_args())

    os.environ['CUDA_VISIBLE_DEVICES'] = args.get('gpu')
    config_sess = tf.ConfigProto()
    config_sess.gpu_options.per_process_gpu_memory_fraction = 0.6

    # Import parameters from the json file
    params_json = import_params_from_json(args.get('model_dir'))
    params = Params(**params_json)

    # Config
    est_config = tf.estimator.RunConfig()
    est_config.replace(keep_checkpoint_max=10,
                       save_checkpoints_steps=params.save_interval,
                       session_config=config_sess,
                       save_checkpoints_secs=None,
                       save_summary_steps=1000)

    estimator = tf.estimator.Estimator(model_fn=crnn_fn, params=params,
                                       model_dir=args.get('model_dir'),
                                       config=est_config,
                                       )

    estimator.export_savedmodel(args.get('export_dir'),
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