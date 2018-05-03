#!/usr/bin/env python
__author__ = 'solivr'

import cv2
import argparse
import csv
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--files', required=True, type=str, help='CSV filenames to check',
                        nargs='*')
    parser.add_argument('-d', '--delimiter', type=str, help='Delimiter in CSV file', default=' ')
    parser.add_argument('-r', '--min-ratio', type=float, help='Minimum ratio', default=0.04)

    args = vars(parser.parse_args())

    for file in tqdm(args.get('files')):
        with open(file, 'r', encoding='utf8') as f:
            csvreader = csv.reader(f, delimiter=args.get('delimiter'))
            for row in tqdm(csvreader):
                try:
                    path = row[0]
                    img = cv2.imread(path)
                    if img.size == 0:
                        print(path)
                    height, width = img.shape[:2]
                    ratio = width/height
                    if ratio < args.get('min_ratio'):
                        print(path)
                except AttributeError:
                    print(row)