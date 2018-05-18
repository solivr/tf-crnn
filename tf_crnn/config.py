#!/usr/bin/env python
__author__ = 'solivr'
__license__ = "GPL"

import os
import json
import time
from .hlp.alphabet_helpers import make_json_lookup_alphabet
from glob import glob


class CONST:
    DIMENSION_REDUCTION_W_POOLING = 2*2  # 2x2 pooling in dimension W on layer 1 and 2


class Alphabet:
    # TODO Updating
    def __init__(self, lookup_alphabet_file: str=None, blank_symbol: str='$'):
        self._blank_symbol = blank_symbol

        if lookup_alphabet_file:
            lookup_alphabet = make_json_lookup_alphabet(csv_filenames=lookup_alphabet_file)
            # Blank symbol must have the largest value
            if self._blank_symbol in lookup_alphabet.keys():
                lookup_alphabet[self._blank_symbol] = max(lookup_alphabet.values()) + 1
            else:
                lookup_alphabet.update({self._blank_symbol: max(lookup_alphabet.values()) + 1})

            self._alphabet_units = list(lookup_alphabet.keys())
            self._codes = list(lookup_alphabet.values())
            self._nclasses = len(self.codes)

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

    # LettersLowercase = string.ascii_lowercase  # 26
    # LettersCapitals = string.ascii_uppercase  # 26
    # Digits = string.digits  # 10
    # Symbols = " '.,:;-_=()[]{}/"  # 16
    # Abbreviations = get_abbreviations_from_csv()  # TODO
    # # Symbols = " '.,:-="  # 7
    # DecodingList = ['same', 'lowercase']
    #
    # BLANK_SYMBOL = '$'
    # DIGITS_ONLY = Digits + BLANK_SYMBOL
    # LETTERS_DIGITS = Digits + LettersCapitals + LettersLowercase + BLANK_SYMBOL
    # LETTERS_DIGITS_LOWERCASE = Digits + LettersLowercase + BLANK_SYMBOL
    # LETTERS_ONLY = LettersCapitals + LettersLowercase + BLANK_SYMBOL
    # LETTERS_ONLY_LOWERCASE = LettersLowercase + BLANK_SYMBOL
    # LETTERS_EXTENDED = LettersCapitals + LettersLowercase + Symbols + BLANK_SYMBOL
    # LETTERS_EXTENDED_LOWERCASE = LettersLowercase + Symbols + BLANK_SYMBOL
    # LETTERS_DIGITS_EXTENDED = Digits + LettersCapitals + LettersLowercase + Symbols + BLANK_SYMBOL
    # LETTERS_DIGITS_EXTENDED_LOWERCASE = Digits + LettersLowercase + Symbols + BLANK_SYMBOL
    # # TODO : Maybe add a unique code (unicode?) to each character and add mask
    #
    # LabelMapping = {
    #     'digits_only': DIGITS_ONLY,
    #     'letters_only': LETTERS_ONLY,
    #     'letters_digits': LETTERS_DIGITS,
    #     'letters_extended': LETTERS_EXTENDED,
    #     'letters_digits_extended': LETTERS_DIGITS_EXTENDED
    # }
    # AlphabetsList = [DIGITS_ONLY, LETTERS_DIGITS, LETTERS_DIGITS_LOWERCASE, LETTERS_ONLY, LETTERS_ONLY_LOWERCASE,
    #                  LETTERS_EXTENDED, LETTERS_EXTENDED_LOWERCASE, LETTERS_DIGITS_EXTENDED,
    #                  LETTERS_DIGITS_EXTENDED_LOWERCASE]
    # LowercaseAlphabetsList = [LETTERS_DIGITS_LOWERCASE, LETTERS_ONLY_LOWERCASE,
    #                           LETTERS_EXTENDED_LOWERCASE, LETTERS_DIGITS_EXTENDED_LOWERCASE]
    # FullAlphabetList = [DIGITS_ONLY, LETTERS_DIGITS, LETTERS_ONLY,
    #                     LETTERS_EXTENDED, LETTERS_DIGITS_EXTENDED]
    #
    # # This are codes for the case DecodingList = 'lowercase'
    # CODES_DIGITS_ONLY = list(range(len(Digits) + 1))
    # CODES_LETTERS_DIGITS = list(range(len(Digits))) + \
    #                        list(range(len(Digits), len(Digits) + len(LettersCapitals))) + \
    #                        list(range(len(Digits), len(Digits) + len(LettersLowercase) + 1))
    # CODES_LETTERS_DIGITS_LOWERCASE = list(range(len(Digits))) + \
    #                                  list(range(len(Digits), len(Digits) + len(LettersLowercase) + 1))
    # CODES_LETTERS_ONLY = list(range(len(LettersCapitals))) + \
    #                      list(range(len(LettersLowercase) + 1))
    # CODES_LETTERS_ONLY_LOWERCASE = list(range(len(LettersLowercase) + 1))
    # CODES_LETTERS_EXTENDED = list(range(len(LettersCapitals))) + \
    #                          list(range(len(LettersLowercase))) + \
    #                          list(range(len(LettersCapitals), len(LettersCapitals) + len(Symbols) + 1))
    # CODES_LETTERS_EXTENDED_LOWERCASE = list(range(len(LettersLowercase))) + \
    #                                    list(range(len(LettersLowercase), len(LettersLowercase) + len(Symbols) + 1))
    # CODES_LETTERS_DIGITS_EXTENDED = list(range(len(Digits))) + \
    #                                 list(range(len(Digits), len(Digits) + len(LettersCapitals))) + \
    #                                 list(range(len(Digits), len(Digits) + len(LettersLowercase))) + \
    #                                 list(range(len(Digits) + len(LettersCapitals),
    #                                            len(Digits) + len(LettersCapitals) +
    #                                            len(Symbols) + 1))
    # CODES_LETTERS_DIGITS_EXTENDED_LOWERCASE = list(range(len(Digits))) + \
    #                                           list(range(len(Digits), len(Digits) + len(LettersLowercase))) + \
    #                                           list(range(len(Digits) + len(LettersLowercase),
    #                                                      len(Digits) + len(LettersLowercase) +
    #                                                      len(Symbols) + 1))


class Params:
    def __init__(self, **kwargs):
        self.train_batch_size = kwargs.get('train_batch_size', 100)
        self.eval_batch_size = kwargs.get('eval_batch_size', 200)
        # Initial value of learining rate (exponential learning rate is used)
        self.learning_rate = kwargs.get('learning_rate', 1e-4)
        # Learning rate decay for exponential learning rate
        self.learning_decay_rate = kwargs.get('learning_decay_rate', 0.96)
        # Decay steps for exponential learning rate
        self.learning_decay_steps = kwargs.get('learning_decay_steps', 1000)
        self.optimizer = kwargs.get('optimizer', 'adam')
        self.n_epochs = kwargs.get('n_epochs', 50)
        self.evaluate_every_epoch = kwargs.get('evaluate_every_epoch', 5)
        self.save_interval = kwargs.get('save_interval', 1e3)
        # Shape of the image to be processed. The original with either be resized or pad depending on its original size
        self.input_shape = kwargs.get('input_shape', (32, 100))
        # Either decode with the same alphabet or map capitals and lowercase letters to the same symbol (lowercase)
        # self.alphabet_decoding = kwargs.get('alphabet_decoding', 'same')
        self.csv_delimiter = kwargs.get('csv_delimiter', ';')
        self.gpu = kwargs.get('gpu', '')
        # Alphabet to use (from class Alphabet)
        # self.alphabet = kwargs.get('alphabet')
        self.lookup_alphabet_file = kwargs.get('alphabet_file')
        self.csv_files_train = kwargs.get('csv_files_train')
        self.csv_files_eval = kwargs.get('csv_files_eval')
        self.output_model_dir = kwargs.get('output_model_dir')
        self._keep_prob_dropout = kwargs.get('keep_prob')

        assert self.optimizer in ['adam', 'rms', 'ada'], 'Unknown optimizer {}'.format(self.optimizer)

        # self._assign_alphabet(alphabet_decoding_list=Alphabet.DecodingList)
        self._assign_alphabet()

    def export_experiment_params(self):
        if not os.path.isdir(self.output_model_dir):
            os.mkdir(self.output_model_dir)
        filename = os.path.join(self.output_model_dir, 'model_params_{}.json'.format(round(time.time())))
        with open(filename, 'w') as f:
            json.dump(vars(self), f)

    def show_experiment_params(self):
        return vars(self)

    # def _assign_alphabet(self, alphabet_decoding_list):
    #     assert (self.alphabet in Alphabet.LabelMapping.keys() or self.alphabet in Alphabet.LabelMapping.values()), \
    #         'Unknown alphabet {}'.format(self.alphabet)
    #     assert (self.alphabet_decoding in alphabet_decoding_list) or (self.alphabet in Alphabet.AlphabetsList), \
    #         'Unknown alphabet decoding {}'.format(self.alphabet_decoding)
    #
    #     if self.alphabet in Alphabet.LabelMapping.keys():
    #         self.alphabet = Alphabet.LabelMapping[self.alphabet]
    #
    #     if self.alphabet_decoding == 'lowercase' or self.alphabet_decoding in Alphabet.LowercaseAlphabetsList:
    #         if self.alphabet == Alphabet.LETTERS_DIGITS:
    #             self.alphabet_decoding = Alphabet.LETTERS_DIGITS_LOWERCASE
    #             self._alphabet_codes = Alphabet.CODES_LETTERS_DIGITS
    #             self._alphabet_decoding_codes = Alphabet.CODES_LETTERS_DIGITS_LOWERCASE
    #             self.blank_label_code = self._alphabet_codes[-1]
    #
    #         elif self.alphabet == Alphabet.LETTERS_ONLY:
    #             self.alphabet_decoding = Alphabet.LETTERS_ONLY_LOWERCASE
    #             self._alphabet_codes = Alphabet.CODES_LETTERS_ONLY
    #             self._alphabet_decoding_codes = Alphabet.CODES_LETTERS_ONLY_LOWERCASE
    #             self.blank_label_code = self._alphabet_codes[-1]
    #
    #         elif self.alphabet == Alphabet.LETTERS_EXTENDED:
    #             self.alphabet_decoding = Alphabet.LETTERS_EXTENDED_LOWERCASE
    #             self._alphabet_codes = Alphabet.CODES_LETTERS_EXTENDED
    #             self._alphabet_decoding_codes = Alphabet.CODES_LETTERS_EXTENDED_LOWERCASE
    #             self.blank_label_code = self._alphabet_codes[-1]
    #
    #         elif self.alphabet == Alphabet.LETTERS_DIGITS_EXTENDED:
    #             self.alphabet_decoding = Alphabet.LETTERS_DIGITS_EXTENDED_LOWERCASE
    #             self._alphabet_codes = Alphabet.CODES_LETTERS_DIGITS_EXTENDED
    #             self._alphabet_decoding_codes = Alphabet.CODES_LETTERS_DIGITS_EXTENDED_LOWERCASE
    #
    #     elif self.alphabet_decoding == 'same' or self.alphabet_decoding in Alphabet.FullAlphabetList:
    #         self.alphabet_decoding = self.alphabet
    #         self._alphabet_codes = list(range(len(self.alphabet)))
    #         self.blank_label_code = self._alphabet_codes[-1]
    #         self._alphabet_decoding_codes = self._alphabet_codes
    #
    #     self._nclasses = self._alphabet_codes[-1] + 1
    #     self._blank_label_symbol = Alphabet.BLANK_SYMBOL
    def _assign_alphabet(self):
        self.alphabet = Alphabet(lookup_alphabet_file=self.lookup_alphabet_file)

    @property
    def keep_prob_dropout(self):
        return self._keep_prob_dropout

    @keep_prob_dropout.setter
    def keep_prob_dropout(self, value):
        assert (0.0 < value <= 1.0), 'Must be 0.0 < value <= 1.0'
        self._keep_prob_dropout = value

    # @property
    # def n_classes(self):
    #     return self._nclasses

    # @property
    # def blank_label_symbol(self):
    #     return self._blank_label_symbol

    # @property
    # def alphabet_codes(self):
    #     return self._alphabet_codes

    # @property
    # def alphabet_decoding_codes(self):
    #     return self._alphabet_decoding_codes


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
