#!/usr/bin/env python
__author__ = 'solivr'

import tensorflow as tf
import numpy as np
from .config import Params, CONST
from typing import Tuple


def data_loader(csv_filename: str, params: Params, batch_size: int=128, data_augmentation: bool=False,
                num_epochs: int=None, image_summaries: bool=False):

    def input_fn():
        # Choose case one csv file or list of csv files
        if not isinstance(csv_filename, list):
            filename_queue = tf.train.string_input_producer([csv_filename], num_epochs=num_epochs, name='filename_queue')
        elif isinstance(csv_filename, list):
            filename_queue = tf.train.string_input_producer(csv_filename, num_epochs=num_epochs, name='filename_queue')

        # Skip lines that have already been processed
        reader = tf.TextLineReader(name='CSV_Reader', skip_header_lines=0)
        key, value = reader.read(filename_queue, name='file_reading_op')

        default_line = [['None'], ['None']]
        path, label = tf.decode_csv(value, record_defaults=default_line, field_delim=params.csv_delimiter,
                                    name='csv_reading_op')

        image, img_width = image_reading(path, resized_size=params.input_shape,
                                         data_augmentation=data_augmentation, padding=True)

        to_batch = {'images': image, 'images_widths': img_width, 'filenames': path, 'labels': label}
        prepared_batch = tf.train.shuffle_batch(to_batch,
                                                batch_size=batch_size,
                                                min_after_dequeue=500,
                                                num_threads=15, capacity=4000,
                                                allow_smaller_final_batch=False,
                                                name='prepared_batch_queue')

        if image_summaries:
            tf.summary.image('input/image', prepared_batch.get('images'), max_outputs=1)
        tf.summary.text('input/labels', prepared_batch.get('labels')[:10])
        tf.summary.text('input/widths', tf.as_string(prepared_batch.get('images_widths')))

        return prepared_batch, prepared_batch.get('labels')

    return input_fn


def image_reading(path: str, resized_size: Tuple[int, int]=None, data_augmentation: bool=False,
                  padding: bool=False) -> Tuple[tf.Tensor, tf.Tensor]:
    # Read image
    image_content = tf.read_file(path, name='image_reader')
    image = tf.cond(tf.equal(tf.string_split([path], '.').values[1], tf.constant('jpg', dtype=tf.string)),
                    true_fn=lambda: tf.image.decode_jpeg(image_content, channels=1, try_recover_truncated=True), # TODO channels = 3 ?
                    false_fn=lambda: tf.image.decode_png(image_content, channels=1), name='image_decoding')

    # Data augmentation
    if data_augmentation:
        image = augment_data(image)

    # Padding
    if padding:
        with tf.name_scope('padding'):
            image, img_width = padding_inputs_width(image, resized_size, increment=CONST.DIMENSION_REDUCTION_W_POOLING)
    # Resize
    else:
        image = tf.image.resize_images(image, size=resized_size)
        img_width = tf.shape(image)[1]

    with tf.control_dependencies([tf.assert_equal(image.shape[:2], resized_size)]):
        return image, img_width


def random_rotation(img: tf.Tensor, max_rotation: float=0.1, crop: bool=True) -> tf.Tensor:  # from SeguinBe
    with tf.name_scope('RandomRotation'):
        rotation = tf.random_uniform([], -max_rotation, max_rotation)
        rotated_image = tf.contrib.image.rotate(img, rotation, interpolation='BILINEAR')
        if crop:
            rotation = tf.abs(rotation)
            original_shape = tf.shape(rotated_image)[:2]
            h, w = original_shape[0], original_shape[1]
            # see https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders for formulae
            old_l, old_s = tf.cond(h > w, lambda: [h, w], lambda: [w, h])
            old_l, old_s = tf.cast(old_l, tf.float32), tf.cast(old_s, tf.float32)
            new_l = (old_l * tf.cos(rotation) - old_s * tf.sin(rotation)) / tf.cos(2*rotation)
            new_s = (old_s - tf.sin(rotation) * new_l) / tf.cos(rotation)
            new_h, new_w = tf.cond(h > w, lambda: [new_l, new_s], lambda: [new_s, new_l])
            new_h, new_w = tf.cast(new_h, tf.int32), tf.cast(new_w, tf.int32)
            bb_begin = tf.cast(tf.ceil((h-new_h)/2), tf.int32), tf.cast(tf.ceil((w-new_w)/2), tf.int32)
            rotated_image_crop = rotated_image[bb_begin[0]:h - bb_begin[0], bb_begin[1]:w - bb_begin[1], :]

            # If crop removes the entire image, keep the original image
            rotated_image = tf.cond(tf.equal(tf.size(rotated_image_crop), 0),
                                    true_fn=lambda: img,
                                    false_fn=lambda: rotated_image_crop)

        return rotated_image


def random_padding(image: tf.Tensor, max_pad_w: int=5, max_pad_h: int=10) -> tf.Tensor:
    w_pad = list(np.random.randint(0, max_pad_w, size=[2]))
    h_pad = list(np.random.randint(0, max_pad_h, size=[2]))
    paddings = [h_pad, w_pad, [0, 0]]

    return tf.pad(image, paddings, mode='REFLECT', name='random_padding')


def augment_data(image: tf.Tensor) -> tf.Tensor:
    with tf.name_scope('DataAugmentation'):

        # Random padding
        image = random_padding(image)

        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, 0.5, 1.5)
        image = random_rotation(image, 0.05, crop=True)

        if image.shape[-1] >= 3:
            image = tf.image.random_hue(image, 0.2)
            image = tf.image.random_saturation(image, 0.5, 1.5)

        return image


def padding_inputs_width(image: tf.Tensor, target_shape: Tuple[int, int], increment: int) -> Tuple[tf.Tensor, tf.Tensor]:

    target_ratio = target_shape[1]/target_shape[0]
    # Compute ratio to keep the same ratio in new image and get the size of padding
    # necessary to have the final desired shape
    shape = tf.shape(image)
    ratio = tf.divide(shape[1], shape[0], name='ratio')

    new_h = target_shape[0]
    new_w = tf.cast(tf.round((ratio * new_h) / increment) * increment, tf.int32)
    f1 = lambda: (new_w, ratio)
    f2 = lambda: (new_h, tf.constant(1.0, dtype=tf.float64))
    new_w, ratio = tf.case({tf.greater(new_w, 0): f1,
                            tf.less_equal(new_w, 0): f2},
                           default=f1, exclusive=True)
    target_w = target_shape[1]

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
            n_replication = tf.cast(tf.ceil(target_shape[1]/new_w), tf.int32)
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

            return img_resized, target_shape

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


def preprocess_image_for_prediction(fixed_height: int=32, min_width: int=8):
    """
    Input function to use when exporting the model for making predictions (see estimator.export_savedmodel)
    :param fixed_height: height of the input image after resizing
    :param min_width: minimum width of image after resizing
    :return:
    """

    def serving_input_fn():
        # define placeholder for input image
        image = tf.placeholder(dtype=tf.float32, shape=[None, None, 1])

        shape = tf.shape(image)
        # Assert shape is h x w x c with c = 1

        ratio = tf.divide(shape[1], shape[0])
        increment = CONST.DIMENSION_REDUCTION_W_POOLING
        new_width = tf.cast(tf.round((ratio * fixed_height) / increment) * increment, tf.int32)

        resized_image = tf.cond(new_width < tf.constant(min_width, dtype=tf.int32),
                                true_fn=lambda: tf.image.resize_images(image, size=(fixed_height, min_width)),
                                false_fn=lambda: tf.image.resize_images(image, size=(fixed_height, new_width))
                                )

        # Features to serve
        features = {'images': resized_image[None],  # cast to 1 x h x w x c
                    'images_widths': new_width[None]  # cast to tensor
                    }

        # Inputs received
        receiver_inputs = {'images': image}

        return tf.estimator.export.ServingInputReceiver(features, receiver_inputs)

    return serving_input_fn
