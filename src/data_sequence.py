#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

from tensorflow.keras.utils import Sequence
import tensorflow as tf
import pandas as pd
import numpy as np
import re
import os
from functools import reduce
from typing import Tuple
from imageio import imread
from skimage.transform import resize
from .config import Params


class SequenceDataset(Sequence):
    def __init__(self,
                 csv_filename: str,
                 csv_separator: str,
                 input_shape: Tuple[int, int],
                 batch_size: int,
                 parameters: Params):
        super(SequenceDataset, self).__init__()

        self.batch_size = batch_size
        self.input_shape = input_shape
        self.table_str2int = dict(zip(parameters.alphabet.alphabet_units, parameters.alphabet.codes))
        cnn_params = zip(parameters.cnn_pool_size, parameters.cnn_pool_strides, parameters.cnn_stride_size)
        self.n_pool = reduce(lambda i, j: i + j, map(lambda k: k[0][1] * k[1][1] * k[2][1], cnn_params))

        dataframe = pd.read_csv(csv_filename,
                                sep=csv_separator,
                                header=None,
                                names=['paths', 'labels'],
                                encoding='utf8',
                                escapechar="\\",
                                quoting=0)

        self.paths = dataframe.paths.to_list()
        self.labels = dataframe.labels.to_list()
        # Remove long labels
        self.labels = self._discard_long_labels(parameters.max_chars_per_string, parameters.string_split_delimiter)

        self.label_dense_codes, self.sequence_len = self._convert_label_to_dense_codes(parameters.string_split_delimiter,
                                                                                       parameters.max_chars_per_string)

    def __len__(self):
        return int(len(self.paths) / self.batch_size)

    def __getitem__(self, idx):

        batch_path = self.paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_labels_dense_codes = self.label_dense_codes[idx * self.batch_size:(idx + 1) * self.batch_size, :]
        batch_seq_len = self.sequence_len[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_img = list()
        batch_width = list()
        for filename in batch_path:
            img, width = self._load_image_and_resize(filename)
            batch_img.append(img)
            batch_width.append(width)

        return {'input_images': np.array(batch_img),
                'input_seq_length': np.floor(np.array(batch_width) / self.n_pool),
                'label_codes': np.array(batch_labels_dense_codes),
                'label_seq_length': np.array(batch_seq_len)}, np.zeros([self.batch_size])

    def _load_image_and_resize(self, img_filename: str):
        img = imread(img_filename, pilmode='L')
        img = resize(img, self.input_shape)
        img = np.expand_dims(img, axis=-1)

        img_width = img.shape[1]

        return img, img_width

    def _convert_label_to_dense_codes(self, split_char: str, max_width: int):
        labels_chars = [[c for c in label.split(split_char) if c] for label in self.labels]
        codes_list = [[self.table_str2int[c] for c in list_char] for list_char in labels_chars]

        seq_lengths = [len(cl) for cl in codes_list]

        dense_codes = list()
        for ls in codes_list:
            dense_codes.append(ls + (max_width - len(ls)) * [0])

        return np.array(dense_codes), np.array(seq_lengths)

    def _discard_long_labels(self,
                             maximum_length: int,
                             string_split_delimiter: str) -> list:
        """
        Discard samples that have a longer labels than ``maximum_length``

        :param maximum_length: maximum characters per string
        :param csv_separator:
        :param string_split_delimiter:
        :return: updated csv_filename, same type as input
        """
        # Remove lables that are longer than maximum_length
        updated_labels = map(lambda x: re.sub(re.escape(string_split_delimiter), '', x), self.labels)
        updated_labels = [lb for lb, upd_lb in zip(self.labels, updated_labels) if len(upd_lb) <= maximum_length]

        n_removed = len(self.labels) - len(updated_labels)
        if n_removed > 0:
            print('-- Removed {} samples ({:.2f} %) which label '
                  'is longer than {} '.format(n_removed,
                                              100 * n_removed / len(self.labels),
                                              maximum_length))

        return updated_labels




