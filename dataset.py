#!/usr/bin/env python
__author__ = 'solivr'

import os
import numpy as np

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

def ascii2Label(ascii):
    if ascii >= 48 and ascii <=57:  # 0-9
        c = ascii - 48
    elif ascii >= 65 and ascii <=90: # A-Z
        c = ascii - 65 +10
    elif ascii >=97 and ascii <=122: # a-z
        c = ascii - 97 +10
    return c
# -------------------------------------------------


def str2int_labels(labels_list):

    assert type(labels_list) is list

    n_labels = len(labels_list)
    maxLength = 0
    indices = []
    values = []
    seqLengths = []

    for i in range(n_labels):
        length_word = len(labels_list[i])
        if length_word > maxLength:
            maxLength = length_word

        for j in range(length_word):
            indices.append([i, j])
            values.append(ascii2Label(ord(labels_list[i][j])))
        seqLengths.append(length_word)
        
    dense_shape = [n_labels, maxLength]
    indices = np.asarray(indices, dtype=np.int32)
    values = np.asarray(values, dtype=np.int32)
    dense_shape = np.asarray(dense_shape, dtype=np.int32)
    return (indices, values, dense_shape), seqLengths
# -------------------------------------------------


class Dataset:
    def __init__(self, config, path):
        self.imgH = config.imgH
        self.imgW = config.imgW
        self.imgC = config.imgC
        self.datapath = path
        self.nSamples

    def nextBatch(self, batch_size):
        #
