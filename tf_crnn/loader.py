#!/usr/bin/env python
__author__ = 'solivr'
__license__ = "GPL"

import tensorflow as tf
from typing import Union
import numpy as np


class PredictionModel:

    def __init__(self, model_dir: str, session: tf.Session=None, signature: str= 'predictions'):
        if session:
            self.session = session
        else:
            self.session = tf.get_default_session()
        self.model = tf.saved_model.loader.load(self.session, ['serve'], model_dir)

        if signature == 'predictions':
            self._input_dict, self._output_dict = _signature_def_to_tensors(self.model.signature_def['predictions'])
            input_dict_key = 'images'
        elif signature == 'rgb_images':
            self._input_dict, self._output_dict = \
                _signature_def_to_tensors(self.model.signature_def['input_rgb:serving_default'])
            input_dict_key = 'rgb_images'
        elif signature == 'filename':
            self._input_dict, self._output_dict = \
                _signature_def_to_tensors(self.model.signature_def['input_filename:serving_default'])
            input_dict_key = 'filename'
        elif signature == 'default':
            self._input_dict, self._output_dict = \
                _signature_def_to_tensors(self.model.signature_def['serving_default'])
            input_dict_key = 'images'
        else:
            raise NotImplementedError

        assert input_dict_key in self._input_dict.keys(), \
            'There is no {} key in input dictionnary. Try {}'.format(input_dict_key, self._input_dict.keys())
        self._input_tensor = self._input_dict[input_dict_key]

    def predict(self, input_to_predict: Union[np.ndarray, str]) -> dict:
        output = self._output_dict
        input_tensor = self._input_tensor
        return self.session.run(output, feed_dict={input_tensor: input_to_predict})


def _signature_def_to_tensors(signature_def):  # from SeguinBe
    g = tf.get_default_graph()
    return {k: g.get_tensor_by_name(v.name) for k, v in signature_def.inputs.items()}, \
           {k: g.get_tensor_by_name(v.name) for k, v in signature_def.outputs.items()}