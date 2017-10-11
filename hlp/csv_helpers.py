#!/usr/bin/env python
__author__ = 'solivr'

import csv
import os
import argparse
from tqdm import tqdm, trange


def csv_rel2abs_path_convertor(csv_filenames: str, delimiter: str=' ', encoding='utf8') -> None:
    """
    Convert relative paths into absolute paths
    :param csv_filenames: filename of csv
    :param delimiter: character to delimit felds in csv
    :param encoding: encoding format of csv file
    :return:
    """

    for filename in tqdm(csv_filenames):
        absolute_path, basename = os.path.split(os.path.abspath(filename))
        relative_paths = list()
        labels = list()
        # Reading CSV
        with open(filename, 'r', encoding=encoding) as f:
            csvreader = csv.reader(f, delimiter=delimiter)
            for row in csvreader:
                relative_paths.append(row[0])
                labels.append(row[1])

        # Writing converted_paths CSV
        export_filename = os.path.join(absolute_path, '{}_abs{}'.format(*os.path.splitext(basename)))
        with open(export_filename, 'w', encoding=encoding) as f:
            csvwriter = csv.writer(f, delimiter=delimiter)
            for i in trange(0, len(relative_paths)):
                csvwriter.writerow([os.path.abspath(os.path.join(absolute_path, relative_paths[i])), labels[i]])


def csv_filtering_chars_from_labels(csv_filename: str, chars_to_remove: str,
                                    delimiter: str=' ', encoding='utf8') -> int:
    """
    Remove labels containing chars_to_remove in csv_filename
    :param chars_to_remove: string (or list) with the undesired characters
    :param csv_filename: filenmae of csv
    :param delimiter: delimiter character
    :param encoding: encoding format of csv file
    :return: number of deleted labels
    """

    if not isinstance(chars_to_remove, list):
        chars_to_remove = list(chars_to_remove)

    paths = list()
    labels = list()
    n_deleted = 0
    with open(csv_filename, 'r', encoding=encoding) as file:
        csvreader = csv.reader(file, delimiter=delimiter)
        for row in csvreader:
            if not any((d in chars_to_remove) for d in row[1]):
                paths.append(row[0])
                labels.append(row[1])
            else:
                n_deleted += 1

    with open(csv_filename, 'w', encoding=encoding) as file:
        csvwriter = csv.writer(file, delimiter=delimiter)
        for i in tqdm(range(len(paths)), total=len(paths)):
            csvwriter.writerow([paths[i], labels[i]])

    return n_deleted


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_files', type=str, required=True, help='CSV filename to convert', nargs='*')
    parser.add_argument('-d', '--delimiter_char', type=str, help='CSV delimiter character', default=' ')

    args = vars(parser.parse_args())

    csv_filenames = args.get('input_files')

    csv_rel2abs_path_convertor(csv_filenames, delimiter=args.get('delimiter_char'))

