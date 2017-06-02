#!/usr/bin/env python
__author__ = 'solivr'

import os
import sys
import numpy as np
import cv2
from tqdm import tqdm
from itertools import cycle


def format_mjsynth_txtfile(path, file_split):
    with open(os.path.join(path, file_split), 'r') as f:
        lines = f.readlines()
    with open(os.path.join(path, 'lexicon.txt'), 'r') as f:
        lexicon = f.readlines()

    # Split lines into path and label
    linesplit = [l[:-1].split(' ') for l in lines]

    label_index = [int(s[1]) for s in linesplit]
    img_paths = [s[0] for s in linesplit]

    labels_string = [lexicon[ind][:-1] for ind in label_index]

    return img_paths, labels_string
# -------------------------------------------------


def ascii2label(ascii):
    """
    Offsets the ASCII code to have continuous labelling
    :param ascii: ascii code (int)
    :return: offset label (int)
    """
    n_digits = 10
    if 48 <= ascii <= 57:  # 0-9
        c = ascii - 48
    elif 65 <= ascii <= 90:  # A-Z
        c = ascii - 65 + n_digits
    elif 97 <= ascii <= 122:  # a-z
        c = ascii - 97 + n_digits
    return c
# -------------------------------------------------


# def str2int_labels(labels_list):
#
#     assert type(labels_list) is list
#
#     n_labels = len(labels_list)
#     maxLength = 0
#     indices = []
#     values = []
#     seqLengths = []
#
#     for i in range(n_labels):
#         length_word = len(labels_list[i])
#         if length_word > maxLength:
#             maxLength = length_word
#
#         for j in range(length_word):
#             indices.append([i, j])
#             values.append(ascii2label(ord(labels_list[i][j])))
#         seqLengths.append(length_word)
#
#     dense_shape = [n_labels, maxLength]
#     indices = np.asarray(indices, dtype=np.int32)
#     values = np.asarray(values, dtype=np.int32)
#     dense_shape = np.asarray(dense_shape, dtype=np.int32)
#
#     # return Sparse Tensor
#     return (indices, values, dense_shape), seqLengths
# -------------------------------------------------


def str2int_label(str_label):
    values = []
    for c in str_label:
        values.append(ascii2label(ord(c)))

    return values
# -------------------------------------------------


class Dataset:
    def __init__(self, config, path, mode):
        self.imgH = config.imgH
        self.imgW = config.imgW
        self.imgC = config.imgC
        self.datapath = path
        self.mode = mode  # test, train, val
        self.count = 0
        # self.reset = False
        self.img_paths_cycle, self.labels_string_cycle = self.make_iters()

    def make_iters(self):
        img_paths_list, labels_string_list = format_mjsynth_txtfile(self.datapath,
                                                                    'annotation_{}.txt'.format(self.mode))
        self.nSamples = len(img_paths_list)

        return cycle(iter(img_paths_list)), cycle(iter(labels_string_list))

    def nextBatch(self, batch_size):
        """

        :param batch_size:
        :return: image batch,
                 label_set : tuple (sparse tensor, list string labels)
                 seqLength : length of the sequence
        """

        images = list()
        labels_int = list()  # ascii-like code
        labels_str = list()  # strings
        seqLengths = list()  # lengths of words
        max_length = 0  # length of the longest word
        while len(images) < batch_size:
            p = next(self.img_paths_cycle)
            l = next(self.labels_string_cycle)

            img_path = os.path.abspath(os.path.join(self.datapath, p))
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            try:
                if not img.data:
                    print('Error when reading image {}. Ignoring it.'.format(p))
                else:
                    # Resize and append image to list
                    resized = cv2.resize(img, (self.imgW, self.imgH), interpolation=cv2.INTER_CUBIC)
                    images.append(resized)

                    # Labels
                    labels_str.append(l)
                    labels_int.append(str2int_label(l))
                    seqLengths.append(len(l))
                    if len(l) > max_length:
                        max_length = len(l)
            except AttributeError:
                print('Error when reading image {}. Ignoring it.'.format(p))

        labels_flatten = np.array([char_code for word in labels_int for char_code in word], dtype=np.int32)
        dense_shape = [len(labels_str), max_length]
        label_set = (labels_str, labels_flatten, dense_shape)  # strings, flattened code_label,[n_labels, max_length]

        images = np.asarray(images)
        self.count += batch_size

        return images, label_set, seqLengths

    # def check_validity_img(self):
    #     """
    #     Loads all images of the dataset and removes the ones that rise errors
    #     :return:
    #     """
    #     print('Checking validity of dataset')
    #     for p, l in tqdm(zip(self.img_paths_list, self.labels_string_list), total=len(self.img_paths_list)):
    #         img_path = os.path.abspath(os.path.join(self.datapath, p))
    #         img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    #         if img is None:
    #             print('Error with image {} : removing from dataset.'.format(p))
    #             self.img_paths_list.remove(p)
    #             self.labels_string_list.remove(l)

