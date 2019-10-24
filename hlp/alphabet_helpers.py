#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

from typing import List, Union
import csv
import json
import numpy as np
import pandas as pd


def get_alphabet_units_from_input_data(csv_filename: str,
                                       split_char: str='|'):
    """
    Get alphabet units from the input_data csv file (which contains in each row the tuple
    (filename image segment, transcription formatted))

    :param csv_filename: csv file containing the input data
    :param split_char: splitting character in input_data separting the alphabet units
    :return:
    """
    df = pd.read_csv(csv_filename, sep=';', header=None, names=['image', 'labels'],
                     encoding='utf8', escapechar="\\", quoting=3)
    transcriptions = list(df.labels.apply(lambda x: x.split(split_char)))

    unique_units = np.unique([chars for list_chars in transcriptions for chars in list_chars])

    return unique_units


def generate_alphabet_file(csv_filenames: List[str],
                           alphabet_filename: str):
    """

    :param csv_filenames:
    :param alphabet_filename:
    :return:
    """
    symbols = list()
    for file in csv_filenames:
        symbols.append(get_alphabet_units_from_input_data(file))

    alphabet_units = np.unique(np.concatenate(symbols))

    alphabet_lookup = dict([(au, i+1)for i, au in enumerate(alphabet_units)])

    with open(alphabet_filename, 'w') as f:
        json.dump(alphabet_lookup, f)


def get_abbreviations_from_csv(csv_filename: str) -> List[str]:
    with open(csv_filename, 'r', encoding='utf8') as f:
        csvreader = csv.reader(f, delimiter='\n')
        alphabet_units = [row[0] for row in csvreader]
    return alphabet_units


# def make_json_lookup_alphabet(string_chars: str=None) -> dict:
#     """
#
#     :param string_chars: for example string.ascii_letters, string.digits
#     :return:
#     """
#     lookup = dict()
#     if string_chars:
#         # Add characters to lookup table
#         lookup.update({char: ord(char) for char in string_chars})
#
#     return map_lookup(lookup)


# def load_lookup_from_json(json_filenames: Union[List[str], str])-> dict:
#     """
#     Load a lookup table from a json file to a dictionnary
#     :param json_filenames: either a filename or a list of filenames
#     :return:
#     """
#
#     lookup = dict()
#     if isinstance(json_filenames, list):
#         for file in json_filenames:
#             with open(file, 'r', encoding='utf8') as f:
#                 data_dict = json.load(f)
#             lookup.update(data_dict)
#
#     elif isinstance(json_filenames, str):
#         with open(json_filenames, 'r', encoding='utf8') as f:
#             lookup = json.load(f)
#
#     return map_lookup(lookup)


# def map_lookup(lookup_table: dict, unique_entry: bool=True)-> dict:
#     """
#     Converts an existing lookup table with minimal range code ([1, len(lookup_table)-1])
#     and avoids multiple instances of the same code label (bijectivity)
#
#     :param lookup_table: dictionary to be mapped {alphabet_unit : code label}
#     :param unique_entry: If each alphabet unit has a unique code and each code a unique alphabet unique ('bijective'),
#                         only True is implemented for now
#     :return: a mapped dictionary
#     """
#
#     # Create tuple (alphabet unit, code)
#     tuple_char_code = list(zip(list(lookup_table.keys()), list(lookup_table.values())))
#     # Sort by code
#     tuple_char_code.sort(key=lambda x: x[1])
#
#     # If each alphabet unit has a unique code and each code a unique alphabet unique ('bijective')
#     if unique_entry:
#         mapped_lookup = [[tp[0], i + 1] for i, tp in enumerate(tuple_char_code)]
#     else:
#         raise NotImplementedError
#         # Todo
#
#     return dict(mapped_lookup)
