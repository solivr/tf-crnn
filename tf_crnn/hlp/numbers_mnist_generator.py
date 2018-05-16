#!/usr/bin/env python
__author__ = 'solivr'
__license__ = "GPL"

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os
import csv
from imageio import imsave
from tqdm import tqdm
import random
import argparse


def generate_random_image_numbers(mnist_dir, dataset, output_dir, csv_filename, n_numbers):

    mnist = input_data.read_data_sets(mnist_dir, one_hot=False)

    output_dir_img = os.path.join(output_dir, 'images')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(output_dir_img):
        os.mkdir(output_dir_img)

    if dataset == 'train':
        dataset = mnist.train
    elif dataset == 'validation':
        dataset = mnist.validation
    elif dataset == 'test':
        dataset = mnist.test

    list_paths = list()
    list_labels = list()

    for i in tqdm(range(n_numbers), total=n_numbers):
        n_digits = random.randint(3, 8)
        digits, labels = dataset.next_batch(n_digits)
        # Reshape to have 28x28 image
        square_digits = np.reshape(digits, [-1, 28, 28])
        # White background
        square_digits = -(square_digits - 1) * 255
        stacked_number = np.hstack(square_digits[:, :, 4:-4])
        stacked_label = ''.join(map(str, labels))
        # chans3 = np.dstack([stacked_number]*3)

        # Save image number
        img_filename = '{:09}_{}.jpg'.format(i, stacked_label)
        img_path = os.path.join(output_dir_img, img_filename)
        imsave(img_path, stacked_number)

        # Add to list of paths and list of labels
        list_paths.append(img_filename)
        list_labels.append(stacked_label)

    root = './images'
    csv_path = os.path.join(output_dir, csv_filename)
    with open(csv_path, 'w') as csvfile:
        for i in tqdm(range(len(list_paths)), total=len(list_paths)):
            csvwriter = csv.writer(csvfile, delimiter=' ')
            csvwriter.writerow([os.path.join(root, list_paths[i]), list_labels[i]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-md', '--mnist_dir', type=str, help='Directory for MNIST data', default='./MNIST_data')
    parser.add_argument('-d', '--dataset', type=str, help='Dataset wanted (train, test, validation)', default='train')
    parser.add_argument('-csv', '--csv_filename', type=str, help='CSV filename to output paths and labels')
    parser.add_argument('-od', '--output_dir', type=str, help='Directory to output images and csv files', default='./output_numbers')
    parser.add_argument('-n', '--n_samples', type=int, help='Desired numbers of generated samples', default=1000)

    args = parser.parse_args()

    generate_random_image_numbers(args.mnist_dir, args.dataset, args.output_dir, args.csv_filename, args.n_samples)


