#!/usr/bin/env python
__author__ = 'solivr'

import os
from itertools import cycle

import cv2
import numpy as np
from tqdm import tqdm

from crnn.src.decoding import str2int_label


def load_paths_labels(path, file_split):
    filename = os.path.join(path, file_split)
    root = os.path.split(filename)[0]

    with open(filename, 'r') as f:
        lines = f.readlines()
    with open(os.path.join(path, 'lexicon.txt'), 'r') as f:
        lexicon = f.readlines()

    linesplit = [l[:-1].split(' ') for l in lines]

    # Absolute path
    img_paths = [os.path.abspath(os.path.join(root, s[0])) for s in linesplit]

    # Label
    label_index = [int(s[1]) for s in linesplit]
    labels_string = [lexicon[ind][:-1] for ind in label_index]

    return img_paths, labels_string
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
        img_paths_list, labels_string_list = load_paths_labels(self.datapath,
                                                                    'new_annotation_{}.txt'.format(self.mode))
        self.nSamples = len(img_paths_list)

        return cycle(iter(img_paths_list)), cycle(iter(labels_string_list))

    def nextBatch(self, batch_size=None):
        """

        :param batch_size:
        :return: image batch,
                 label_set : tuple (sparse tensor, list string labels)
                 seqLength : length of the sequence
        """
        if batch_size is None:
            batch_size = self.nSamples

        images = list()
        labels_int = list()  # ascii-like code
        labels_str = list()  # strings
        seqLengths = list()  # lengths of words
        max_length = 0  # length of the longest word
        while len(images) < batch_size:
            p = next(self.img_paths_cycle)
            l = next(self.labels_string_cycle)

            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            try:
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
                return None, None, None

        labels_flatten = np.array([char_code for word in labels_int for char_code in word], dtype=np.int32)
        dense_shape = [len(labels_str), max_length]
        label_set = (labels_str, labels_flatten, dense_shape)  # strings, flattened code_label, [n_labels, max_length]

        images = np.asarray(images)
        self.count += batch_size

        return images, label_set, seqLengths
# -------------------------------------------------------------


def verify_list_paths(list_paths, new_filename):
    updated_list = list_paths.copy()
    for p in tqdm(list_paths, total=len(list_paths)):
        try:
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        except FileNotFoundError:
            print('File already deleted')
            continue
        try:
            if img is None:
                print('Error when reading image {}. Removing it.'.format(p))
            elif not img.data:
                print('Error when reading image {}. Ignoring it.'.format(p))
            else:
                resized = cv2.resize(img, (10, 10), interpolation=cv2.INTER_CUBIC)
                continue
        except:
            print('Error when reading image {}. Ignoring it.'.format(p))

        try:
            # If problem delete img from disk and from list
            updated_list.remove(p)
            os.remove(p)
        except FileNotFoundError:
            print('File already deleted')
            continue

    print('Writing updated list of paths in {}'.format(new_filename))
    with open(new_filename, 'w') as handle:
        for p in tqdm(updated_list, total=len(updated_list)):
            # Reconstruct relative path
            root, file = os.path.split(p)
            dirs = tuple(['.'] + root.split('/')[-2:] + [file])
            relative_p = os.path.join(*dirs)

            # Label lexicon
            lex = file.split('.')[0].split('_')[-1]

            file_line = '{} {}{}'.format(relative_p, lex, os.linesep)

            handle.write(file_line)