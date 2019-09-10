#!/usr/bin/env python
__author__ = 'solivr'
__license__ = "GPL"

import tensorflow as tf
import numpy as np
from .config import Params, CONST
from typing import Tuple, Union, List
import collections


def random_rotation(img: tf.Tensor, max_rotation: float=0.1, crop: bool=True) -> tf.Tensor:  # adapted from SeguinBe
    """
    Rotates an image with a random angle.
    See https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders for formulae

    :param img: Tensor
    :param max_rotation: maximum angle to rotate (radians)
    :param crop: boolean to crop or not the image after rotation
    :return:
    """
    with tf.name_scope('RandomRotation'):
        rotation = tf.random_uniform([], -max_rotation, max_rotation, name='pick_random_angle')
        rotated_image = tf.contrib.image.rotate(img, rotation, interpolation='BILINEAR')
        if crop:
            rotation = tf.abs(rotation)
            original_shape = tf.shape(rotated_image)[:2]
            h, w = original_shape[0], original_shape[1]
            old_l, old_s = tf.cond(h > w, lambda: [h, w], lambda: [w, h])
            old_l, old_s = tf.cast(old_l, tf.float32), tf.cast(old_s, tf.float32)
            new_l = (old_l * tf.cos(rotation) - old_s * tf.sin(rotation)) / tf.cos(2*rotation)
            new_s = (old_s - tf.sin(rotation) * new_l) / tf.cos(rotation)
            new_h, new_w = tf.cond(h > w, lambda: [new_l, new_s], lambda: [new_s, new_l])
            new_h, new_w = tf.cast(new_h, tf.int32), tf.cast(new_w, tf.int32)
            bb_begin = tf.cast(tf.math.ceil((h-new_h)/2), tf.int32), tf.cast(tf.math.ceil((w-new_w)/2), tf.int32)
            # Test sliced
            rotated_image_crop = tf.cond(
                tf.logical_and(bb_begin[0] < h - bb_begin[0], bb_begin[1] < w - bb_begin[1]),
                true_fn=lambda: rotated_image[bb_begin[0]:h - bb_begin[0], bb_begin[1]:w - bb_begin[1], :],
                false_fn=lambda: img,
                name='check_slices_indices'
            )
            # rotated_image_crop = rotated_image[bb_begin[0]:h - bb_begin[0], bb_begin[1]:w - bb_begin[1], :]

            # If crop removes the entire image, keep the original image
            rotated_image = tf.cond(tf.equal(tf.size(rotated_image_crop), 0),
                                    true_fn=lambda: img,
                                    false_fn=lambda: rotated_image_crop,
                                    name='check_size_crop')

        return rotated_image


def random_padding(image: tf.Tensor, max_pad_w: int=5, max_pad_h: int=10) -> tf.Tensor:
    """
    Given an image will pad its border adding a random number of rows and columns

    :param image: image to pad
    :param max_pad_w: maximum padding in width
    :param max_pad_h: maximum padding in height
    :return: a padded image
    """
    # TODO specify image shape in doc

    w_pad = list(np.random.randint(0, max_pad_w, size=[2]))
    h_pad = list(np.random.randint(0, max_pad_h, size=[2]))
    paddings = [h_pad, w_pad, [0, 0]]

    return tf.pad(image, paddings, mode='REFLECT', name='random_padding')


def augment_data(image: tf.Tensor, max_rotation: float=0.1) -> tf.Tensor:
    """
    Data augmentation on an image (padding, brightness, contrast, rotation)

    :param image: Tensor
    :param max_rotation: float, maximum permitted rotation (in radians)
    :return: Tensor
    """
    with tf.name_scope('DataAugmentation'):

        # Random padding
        image = random_padding(image)

        # TODO : add random scaling
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, 0.5, 1.5)
        image = random_rotation(image, max_rotation, crop=True)

        if image.shape[-1] >= 3:
            image = tf.image.random_hue(image, 0.2)
            image = tf.image.random_saturation(image, 0.5, 1.5)

        return image


def get_resized_width(image: tf.Tensor,
                      target_height: int,
                      increment: int=CONST.DIMENSION_REDUCTION_W_POOLING):

    image_shape = tf.shape(image)
    image_ratio = tf.divide(image_shape[1], image_shape[0], name='ratio')

    new_width = tf.cast(tf.round((image_ratio * target_height) / increment) * increment, tf.int32)
    f1 = lambda: (new_width, image_ratio)
    f2 = lambda: (target_height, tf.constant(1.0, dtype=tf.float64))
    new_width, image_ratio = tf.case({tf.greater(new_width, 0): f1,
                                      tf.less_equal(new_width, 0): f2},
                                     default=f1, exclusive=True)
    return new_width, image_ratio


def padding_inputs_width(image: tf.Tensor,
                         target_shape: Tuple[int, int],
                         increment: int) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Given an input image, will pad it to return a target_shape size padded image.
    There are 3 cases:
         - image width > target width : simple resizing to shrink the image
         - image width >= 0.5*target width : pad the image
         - image width < 0.5*target width : replicates the image segment and appends it

    :param image: Tensor of shape [H,W,C]
    :param target_shape: final shape after padding [H, W]
    :param increment: reduction factor due to pooling between input width and output width,
                        this makes sure that the final width will be a multiple of increment
    :return: (image padded, output width)
    """

    target_ratio = target_shape[1]/target_shape[0]
    target_w = target_shape[1]
    # Compute ratio to keep the same ratio in new image and get the size of padding
    # necessary to have the final desired shape
    new_h = target_shape[0]
    new_w, ratio = get_resized_width(image, new_h, increment)

    # Definitions for cases
    def pad_fn():
        with tf.name_scope('mirror_padding'):
            pad = tf.subtract(target_w, new_w)

            img_resized = tf.image.resize_images(image, [new_h, new_w])

            # Padding to have the desired width
            paddings = [[0, 0], [0, pad], [0, 0]]
            pad_image = tf.pad(img_resized, paddings, mode='SYMMETRIC', name=None)

            # Set manually the shape
            pad_image.set_shape([target_shape[0], target_shape[1], img_resized.get_shape()[2]])

            return pad_image, (new_h, new_w)

    def replicate_fn():
        with tf.name_scope('replication_padding'):
            img_resized = tf.image.resize_images(image, [new_h, new_w])

            # If one symmetry is not enough to have a full width
            # Count number of replications needed
            n_replication = tf.cast(tf.math.ceil(target_shape[1]/new_w), tf.int32)
            img_replicated = tf.tile(img_resized, tf.stack([1, n_replication, 1]))
            pad_image = tf.image.crop_to_bounding_box(image=img_replicated, offset_height=0, offset_width=0,
                                                      target_height=target_shape[0], target_width=target_shape[1])

            # Set manually the shape
            pad_image.set_shape([target_shape[0], target_shape[1], img_resized.get_shape()[2]])

            return pad_image, (new_h, new_w)

    def simple_resize():
        with tf.name_scope('simple_resize'):
            img_resized = tf.image.resize_images(image, target_shape)

            img_resized.set_shape([target_shape[0], target_shape[1], img_resized.get_shape()[2]])

            return img_resized, tuple(target_shape)

    # 3 cases
    pad_image, (new_h, new_w) = tf.case(
        {  # case 1 : new_w >= target_w
            tf.logical_and(tf.greater_equal(ratio, target_ratio),
                           tf.greater_equal(new_w, target_w)): simple_resize,
            # case 2 : new_w >= target_w/2 & new_w < target_w & ratio < target_ratio
            tf.logical_and(tf.less(ratio, target_ratio),
                           tf.logical_and(tf.greater_equal(new_w, tf.cast(tf.divide(target_w, 2), tf.int32)),
                                          tf.less(new_w, target_w))): pad_fn,
            # case 3 : new_w < target_w/2 & new_w < target_w & ratio < target_ratio
            tf.logical_and(tf.less(ratio, target_ratio),
                           tf.logical_and(tf.less(new_w, target_w),
                                          tf.less(new_w, tf.cast(tf.divide(target_w, 2), tf.int32)))): replicate_fn
        },
        default=simple_resize, exclusive=True)

    return pad_image, new_w  # new_w = image width used for computing sequence lengths


def dataset_generator(csv_filename: Union[List[str], str],
                       params: Params,
                       labels: bool=True,
                       batch_size: int=64,
                       data_augmentation: bool=False,
                       num_epochs: int=None):
    do_padding = True

    if labels:
        column_defaults = [['None'], ['None'], tf.int32]
        column_names = ['paths', 'label_codes', 'label_seq_length']
        label_name = 'label_codes'
    else:
        column_defaults = [['None']]
        column_names = ['paths']
        label_name = None

    num_parallel_reads = 1

    # ----- from data.experimental.make_csv_dataset
    def filename_to_dataset(filename):
        dataset = tf.data.experimental.CsvDataset(filename,
                                                  record_defaults=column_defaults,
                                                  field_delim=params.csv_delimiter,
                                                  header=False)
        return dataset

    def map_fn(*columns):
        """Organizes columns into a features dictionary.
        Args:
          *columns: list of `Tensor`s corresponding to one csv record.
        Returns:
          An OrderedDict of feature names to values for that particular record. If
          label_name is provided, extracts the label feature to be returned as the
          second element of the tuple.
        """
        features = collections.OrderedDict(zip(column_names, columns))
        if label_name is not None:
            label = features.pop(label_name)
            return features, label

        return features

    dataset = tf.data.Dataset.from_tensor_slices(csv_filename)
    # Read files sequentially (if num_parallel_reads=1) or in parallel
    dataset = dataset.apply(tf.data.experimental.parallel_interleave(filename_to_dataset,
                                                                     cycle_length=num_parallel_reads))
    dataset = dataset.map(map_fn)
    # -----

    def _load_image_and_pad_or_resize(features, labels):
        path = features['paths']
        image_content = tf.io.read_file(path)
        image = tf.io.decode_jpeg(image_content, channels=params.input_channels,
                                  try_recover_truncated=True, name='image_decoding_op')

        if do_padding:
            with tf.name_scope('do_padding'):
                image, img_width = padding_inputs_width(image, target_shape=params.input_shape,
                                                        increment=CONST.DIMENSION_REDUCTION_W_POOLING)
        # Resize
        else:
            image = tf.image.resize_images(image, size=params.input_shape)
            img_width = tf.shape(image)[1]

        input_seq_length = tf.cast(tf.floor(tf.divide(img_width, params.n_pool)), tf.int32)

        assert_op = tf.debugging.assert_greater_equal(input_seq_length,
                                                      features['label_seq_length'])
        with tf.control_dependencies([assert_op]):
            return {'input_images': image,
                    'label_seq_length': features['label_seq_length'],
                    'input_seq_length': input_seq_length}, labels

    def _format_label_codes(features, string_label_codes):
        splits = tf.sparse.to_dense(tf.strings.split([string_label_codes], sep=' '))
        label_codes = tf.squeeze(tf.strings.to_number(splits, out_type=tf.int32), axis=0)

        features.update({'label_codes': label_codes})
        return features, [0]

    def _data_augment_fn(features: dict, label) -> tf.data.Dataset:

        image = features['input_images']
        image = augment_data(image, params.data_augmentation_max_rotation)

        features.update({'input_images': image})
        return features, label

    dataset = dataset.map(_load_image_and_pad_or_resize)
    dataset = dataset.map(_format_label_codes)
    if data_augmentation:
        dataset = dataset.map(_data_augment_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(1024, reshuffle_each_iteration=False).repeat(num_epochs)

    return dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)


def dataset_prediction(image_filenames: Union[List[str], str],
                       params: Params,
                       batch_size: int = 64):
    do_padding = True

    dataset = tf.data.Dataset.from_tensor_slices(image_filenames)

    # -----

    def _load_image_and_pad_or_resize(path):
        image_content = tf.io.read_file(path)
        image = tf.io.decode_jpeg(image_content, channels=params.input_channels,
                                  try_recover_truncated=True, name='image_decoding_op')

        if do_padding:
            with tf.name_scope('do_padding'):
                image, img_width = padding_inputs_width(image, target_shape=params.input_shape,
                                                        increment=CONST.DIMENSION_REDUCTION_W_POOLING)
        # Resize
        else:
            image = tf.image.resize_images(image, size=params.input_shape)
            img_width = tf.shape(image)[1]

        input_seq_length = tf.cast(tf.floor(tf.divide(img_width, params.n_pool)), tf.int32)

        return {'input_images': image,
                'input_seq_length': input_seq_length}

    dataset = dataset.map(_load_image_and_pad_or_resize)

    return dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
