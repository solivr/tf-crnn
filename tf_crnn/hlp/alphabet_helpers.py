#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

from typing import List, Union
import csv
import json


def get_alphabet_units_form_csv(csv_filename: str) -> List[str]:
    with open(csv_filename, 'r', encoding='utf8') as f:
        csvreader = csv.reader(f, delimiter='\n')
        alphabet_units = [row[0] for row in csvreader]
    return alphabet_units


def get_abbreviations_from_csv(csv_filename: str = './data/selected_abbreviations_n200.csv') -> List[str]:
    return get_alphabet_units_form_csv(csv_filename)


def make_json_lookup_alphabet(string_chars: str=None, csv_filenames: Union[List[str], str]=None) -> dict:
    """

    :param string_chars: for example string.ascii_letters, string.digits
    :param csv_filenames: csv files containing chars or words in each line.
                    Each line will be considered as a unit in the alphabet
    :return:
    """
    lookup = dict()
    offset = 0
    if string_chars:
        # Add characters to lookup table
        lookup.update({char: ord(char) for char in string_chars})
        # Add offset to the codes of alphabets units
        offset = max(lookup.values()) + 1

    if isinstance(csv_filenames, list):
        for file in csv_filenames:
            # Update lookup table with alphabets units from csv file
            alphabet_units = get_alphabet_units_form_csv(file)
            lookup.update({abbrev: offset + i for i, abbrev in enumerate(alphabet_units)})

            # Update offset
            offset = max(lookup.values()) + 1

    elif isinstance(csv_filenames, str):
        alphabet_units = get_alphabet_units_form_csv(csv_filenames)
        lookup.update({abbrev: offset + i for i, abbrev in enumerate(alphabet_units)})

    return lookup


def load_lookup_from_json(json_filenames: Union[List[str], str])-> dict:

    lookup = dict()
    if isinstance(json_filenames, list):
        for file in json_filenames:
            with open(file, 'r', encoding='utf8') as f:
                data_dict = json.load(f)
            lookup.update(data_dict)

    elif isinstance(json_filenames, str):
        with open(json_filenames, 'r', encoding='utf8') as f:
            lookup = json.load(f)

    return lookup
