#!/usr/bin/env python
__author__ = 'solivr'

"""
list =

ALLlabels: [1x8817036 single]
 ALLnames: {1x8817036 cell}
  ALLtext: {1x88172 cell}
   TRNind: [6612779x1 single]
   VALind: [1322551x1 single]

remaining left: 881706 -> What to do with it ???
"""

import h5py
from tqdm import tqdm
import random
import csv
import argparse
import os
import numpy as np
# from decoding import str2int_label


def shuffle_set(names, labels):
    """
    Shuffles the set (names, labels) and returns the shuffled version of each
    :param names:
    :param labels:
    :return:
    """
    # Zip two list and shuffle it
    tup = list(zip(names, labels))
    random.shuffle(tup)
    shuffled_zip = np.array(tup)
    shuffled_names = list(shuffled_zip[:, 0])
    shuffled_labels = list(shuffled_zip[:, 1])
    return shuffled_names, shuffled_labels
# ------------------------------------------------


def dump2csv(csv_filename, names, labels):
    """
    Writes the content of names and labels into csv file
    :param csv_filename:
    :param names:
    :param labels:
    :return:
    """
    assert len(names) == len(labels)
    n = len(names)

    with open(csv_filename, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ')
        for i in tqdm(range(n), total=n):
            writer.writerow([names[i], labels[i]])
# ------------------------------------------------


def make_set_from_index(indexes, names, labels):
    """
    Construct the list of names and labels corresponding to the indexes (TRAIN or VAL indexes for instance)
    :param indexes:
    :param names:
    :param labels:
    :return:
    """

    names_set = list()
    label_set = list()
    # Get train list
    for ind in indexes:
        names_set.append(names[ind - 1])
        label_set.append(labels[ind - 1])

    return names_set, label_set
# ------------------------------------------------


def dereferencing(hdf_data, field):
    """
    DEreferences string elements from a HDF5 reference list
    :param hdf_data: HDF5 file object
    :param field: string corresponding to the field within dhf data structure
    :return:
    """
    hdf_reference_list = hdf_data[field]
    dereferenced_list = list()

    for i in tqdm(range(len(hdf_reference_list)), total=len(hdf_reference_list)):
        ref = hdf_reference_list[i][0]

        # dereference and get string
        object = hdf_data[ref]
        dereferenced_list.append(''.join(chr(i) for i in object[:]))  # obj has ascii code of chars

    return dereferenced_list
# ------------------------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, help='Mat file')
    parser.add_argument('-o', '--output_folder', type=str, help='Output folder fore csv files')

    args = parser.parse_args()

    # Load file
    data = h5py.File(args.file, 'r')

    # Matlab list equivalent structures
    VAL_ind = list(map(int, list(data['list/VALind'][0])))
    TRN_ind = list(map(int, list(data['list/TRNind'][0])))
    ALL_labels = data['list/ALLlabels']

    # Dereferencing objects
    print('Dereferencing vocabulary list')
    list_vocabulary = dereferencing(data, 'list/ALLtext')
    print('Dereferencing filenames list')
    list_names = dereferencing(data, 'list/ALLnames')

    # >>>>>>>>> Absolute path for filenames ?
    # list_names = [os.path.abspath(n) for n in list_names]

    # Mapping labels (code) to vocabulary words (strings)
    print('Mapping labels code to vocabulary words')
    list_labels_str = list()
    for i in tqdm(range(len(ALL_labels)), total=len(ALL_labels)):
        word = list_vocabulary[int(ALL_labels[i][0] - 1)]
        list_labels_str.append(word)

    # Make training and validation sets
    train_names, train_labels = make_set_from_index(TRN_ind, list_names, list_labels_str)
    val_names, val_labels = make_set_from_index(VAL_ind, list_names, list_labels_str)

    # Shuffle sets
    train_names_shuffled, train_labels_shuffled = shuffle_set(train_names, train_labels)
    val_names_shuffled, val_labels_shuffled = shuffle_set(val_names, val_labels)

    # # Get codes labels
    # train_codes = [str2int_label(l) for l in train_labels_shuffled]
    # val_codes = [str2int_label(l) for l in val_labels_shuffled]

    # Dump to csv file
    dump2csv(os.path.join(args.output_folder, 'iiit_hw_train.csv'), train_names_shuffled, train_labels_shuffled)
    dump2csv(os.path.join(args.output_folder, 'iiit_hw_val.csv'), val_names_shuffled, val_labels_shuffled)
