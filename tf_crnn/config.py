#!/usr/bin/env python
__author__ = 'solivr'
__license__ = "GPL"

import os
import json
from .hlp.alphabet_helpers import load_lookup_from_json
from glob import glob


class CONST:
    DIMENSION_REDUCTION_W_POOLING = 2*2  # 2x2 pooling in dimension W on layer 1 and 2


class Alphabet:
    def __init__(self, lookup_alphabet_file: str=None, blank_symbol: str='$'):
        self._blank_symbol = blank_symbol

        if lookup_alphabet_file:
            lookup_alphabet = load_lookup_from_json(lookup_alphabet_file)
            # Blank symbol must have the largest value
            if self._blank_symbol in lookup_alphabet.keys():
                lookup_alphabet[self._blank_symbol] = max(lookup_alphabet.values()) + 1
            else:
                lookup_alphabet.update({self._blank_symbol: max(lookup_alphabet.values()) + 1})

            self._alphabet_units = list(lookup_alphabet.keys())
            self._codes = list(lookup_alphabet.values())
            self._nclasses = max(self.codes) + 1  # n_classes should be + 1 of labels codes

    @property
    def n_classes(self):
        return self._nclasses

    @property
    def blank_symbol(self):
        return self._blank_symbol

    @property
    def codes(self):
        return self._codes

    @property
    def alphabet_units(self):
        return self._alphabet_units


class TrainingParams:
    def __init__(self, **kwargs):
        self.n_epochs = kwargs.get('n_epochs', 50)
        self.train_batch_size = kwargs.get('train_batch_size', 64)
        self.eval_batch_size = kwargs.get('eval_batch_size', 128)
        # Initial value of learining rate (exponential learning rate is used)
        self.learning_rate = kwargs.get('learning_rate', 1e-4)
        # Learning rate decay for exponential learning rate
        self.learning_decay_rate = kwargs.get('learning_decay_rate', 0.96)
        # Decay steps for exponential learning rate
        self.learning_decay_steps = kwargs.get('learning_decay_steps', 1000)
        self.evaluate_every_epoch = kwargs.get('evaluate_every_epoch', 5)
        self.save_interval = kwargs.get('save_interval', 1e3)
        self.optimizer = kwargs.get('optimizer', 'adam')

        assert self.optimizer in ['adam', 'rms', 'ada'], 'Unknown optimizer {}'.format(self.optimizer)

    def to_dict(self):
        return self.__dict__


class Params:
    def __init__(self, **kwargs):
        # Shape of the image to be processed. The original with either be resized or pad depending on its original size
        self.input_shape = kwargs.get('input_shape', (32, 100))
        self.input_channels = kwargs.get('input_channels', 1)
        # Either decode with the same alphabet or map capitals and lowercase letters to the same symbol (lowercase)
        self.csv_delimiter = kwargs.get('csv_delimiter', ';')
        self.gpu = kwargs.get('gpu', '')
        self.lookup_alphabet_file = kwargs.get('lookup_alphabet_file')
        self.csv_files_train = kwargs.get('csv_files_train')
        self.csv_files_eval = kwargs.get('csv_files_eval')
        self.output_model_dir = kwargs.get('output_model_dir')
        self._keep_prob_dropout = kwargs.get('keep_prob')

        self._assign_alphabet()

    def show_experiment_params(self):
        return vars(self)

    def _assign_alphabet(self):
        self.alphabet = Alphabet(lookup_alphabet_file=self.lookup_alphabet_file)

    @property
    def keep_prob_dropout(self):
        return self._keep_prob_dropout

    @keep_prob_dropout.setter
    def keep_prob_dropout(self, value):
        assert (0.0 < value <= 1.0), 'Must be 0.0 < value <= 1.0'
        self._keep_prob_dropout = value


def import_params_from_json(model_directory: str=None, json_filename: str=None) -> dict:

    assert not all(p is None for p in [model_directory, json_filename]), 'One argument at least should not be None'

    if model_directory:
        # Import parameters from the json file
        try:
            json_filename = glob(os.path.join(model_directory, 'model_params*.json'))[-1]
        except IndexError:
            print('No json found in dir {}'.format(model_directory))
            raise FileNotFoundError
    else:
        if not os.path.isfile(json_filename):
            print('No json found with filename {}'.format(json_filename))
            raise FileNotFoundError

    with open(json_filename, 'r') as data_json:
        params_json = json.load(data_json)

    # Remove 'private' keys
    keys = list(params_json.keys())
    for key in keys:
        if key[0] == '_':
            params_json.pop(key)

    return params_json
