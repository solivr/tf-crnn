#!/usr/bin/env python
__author__ = 'solivr'

import csv
import os
import argparse
from tqdm import tqdm, trange


def csv_rel2abs_path_convertor(csv_filenames, delimiter=' ', encoding='utf8'):

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
        with open(export_filename, 'w', encoding='utf8') as f:
            csvwriter = csv.writer(f, delimiter=';')  # TODO change to delimiter
            for i in trange(0, len(relative_paths)):
                csvwriter.writerow([os.path.abspath(os.path.join(absolute_path, relative_paths[i])), labels[i]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_files', type=str, required=True, help='CSV filename to convert', nargs='*')
    parser.add_argument('-d', '--delimiter_char', type=str, help='CSV delimiter character', default=' ')

    args = vars(parser.parse_args())

    csv_filenames = args.get('input_files')

    csv_rel2abs_path_convertor(csv_filenames, delimiter=args.get('delimiter_char'))

