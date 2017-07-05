#!/usr/bin/env python
__author__ = 'solivr'

import os
import tensorflow as tf
import numpy as np


def random_rotation(img, max_rotation=0.1, crop=True):
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
            rotated_image = rotated_image[bb_begin[0]:h-bb_begin[0], bb_begin[1]:w-bb_begin[1], :]
        return rotated_image


def random_padding(image, max_pad_w=5, max_pad_h=10):
    w_pad = list(np.random.randint(0, max_pad_w, size=[2]))
    h_pad = list(np.random.randint(0, max_pad_h, size=[2]))
    paddings = [h_pad, w_pad, [0, 0]]

    return tf.pad(image, paddings, mode='REFLECT', name='random_padding')


def augment_data(image):
    with tf.name_scope('DataAugmentation'):

        # Random padding
        image = random_padding(image)

        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, 0.5, 1.5)
        image = random_rotation(image, 0.1, crop=True)

        if image.shape[-1] >= 3:
            image = tf.image.random_hue(image, 0.2)
            image = tf.image.random_saturation(image, 0.5, 1.5)

        return image


def padding_inputs_width(image, target_shape, increment=2):

    target_ratio = target_shape[1]/target_shape[0]
    # Compute ratio to keep the same ratio in new image and get the size of padding
    # necessary to have the final desired shape
    shape = tf.shape(image)
    ratio = tf.divide(shape[1], shape[0], name='ratio')

    new_h = target_shape[0]
    new_w = tf.cast(tf.round((ratio * new_h) / increment) * increment, tf.int32)
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
            pad_image = tf.image.crop_to_bounding_box(img_replicated, 0, 0, target_shape[0], target_shape[1])

            # Set manually the shape
            pad_image.set_shape([target_shape[0], target_shape[1], img_resized.get_shape()[2]])

            return pad_image, (new_h, new_w)

    def simple_resize():
        with tf.name_scope('simple_resize'):
            img_resized = tf.image.resize_images(image, target_shape)

            img_resized.set_shape([target_shape[0], target_shape[1], img_resized.get_shape()[2]])

            return img_resized, target_shape

    # 3 cases
    pad_image, (new_h, new_w) = tf.case({  # new_w >= target_w
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


def image_reading(path, resized_size=None, data_augmentation=False, padding=False):
    # Read image
    image_content = tf.read_file(path, name='image_reader')
    image = tf.cond(tf.equal(tf.string_split([path], '.').values[1], tf.constant('jpg', dtype=tf.string)),
                    true_fn=lambda: tf.image.decode_jpeg(image_content, channels=1, try_recover_truncated=True),
                    false_fn=lambda: tf.image.decode_png(image_content, channels=1))

    # tf.Assert(tf.equal(tf.size(image), 0), [image])

    # Data augmentation
    if data_augmentation:
        image = augment_data(image)

    # Padding
    if padding:
        with tf.name_scope('padding'):
            image, img_width = padding_inputs_width(image, resized_size)
    # Resize
    elif resized_size:
        image = tf.image.resize_images(image, size=resized_size)
        img_width = tf.shape(image)[1]

    return image, img_width


def data_loader(csv_filename, cursor=0, batch_size=128, input_shape=(32, 100), data_augmentation=False, num_epochs=None):

    def input_fn():
        # Choose case one csv file or list of csv files
        if not isinstance(csv_filename, list):
            dirname = os.path.dirname(csv_filename)
            filename_queue = tf.train.string_input_producer([csv_filename], num_epochs=num_epochs)
        elif isinstance(csv_filename, list):
            dirname = os.path.dirname(csv_filename[0])
            filename_queue = tf.train.string_input_producer(csv_filename, num_epochs=num_epochs)
        else:
            raise TypeError

        # Skip lines that have already been processed
        reader = tf.TextLineReader(name='CSV_Reader', skip_header_lines=cursor)
        key, value = reader.read(filename_queue, name='file_reading_op')

        default_line = [['None'], ['None']]
        path, label = tf.decode_csv(value, record_defaults=default_line, field_delim=' ', name='csv_reading_op')

        # Get full path
        full_dir = dirname
        full_path = tf.string_join([full_dir, path], separator=os.path.sep)

        image, img_width = image_reading(full_path, resized_size=input_shape,
                                         data_augmentation=data_augmentation, padding=True)

        # Batch
        img_batch, label_batch, filenames_batch, img_width_batch = tf.train.batch([image, label, full_path, img_width],
                                                                                  batch_size=batch_size,
                                                                                  num_threads=15, capacity=3000,
                                                                                  dynamic_pad=False)

        return {'images': img_batch, 'images_widths': img_width_batch, 'filenames': filenames_batch}, \
               label_batch

    return input_fn


# def data_loader_from_list_filenames(list_filenames, batch_size=128, input_shape=[32, 100], data_augmentation=False,
#                                     num_epochs=None):
#
#     def input_fn():
#         # Choose case one csv file or list of csv files
#         if not isinstance(list_filenames, list):
#             filename_queue = tf.train.string_input_producer([list_filenames], num_epochs=num_epochs)
#         elif isinstance(list_filenames, list):
#             filename_queue = tf.train.string_input_producer(list_filenames, num_epochs=num_epochs)
#         else:
#             raise TypeError
#
#         full_path = filename_queue.dequeue()
#
#         image = image_reading(full_path, resized_size=input_shape, data_augmentation=data_augmentation)
#
#         # Batch
#         img_batch, filenames_batch = tf.train.batch([image, full_path], batch_size=batch_size, num_threads=15,
#                                                     capacity=3000, dynamic_pad=False, allow_smaller_final_batch=True)
#         # img_batch, filenames_batch = tf.train.shuffle_batch([image, full_path], batch_size=batch_size,
#         #                                                     num_threads=15, capacity=3000, dynamic_pad=False,
#         #                                                     allow_smaller_final_batch=True)
#
#         return {'images': img_batch, 'filenames': filenames_batch}
#
#     return input_fn


def preprocess_image_for_prediction(fixed_height=32):

    def serving_input_fn():
        # define placeholder for input image
        image = tf.placeholder(dtype=tf.float32, shape=[None, None, 1])

        shape = tf.shape(image)
        # Assert shape is h x w x c with c = 1

        ratio = tf.divide(shape[1], shape[0])
        increment = 2
        new_width = tf.cast(tf.round((ratio * fixed_height) / increment) * increment, tf.int32)

        resized_image = tf.image.resize_images(image, size=(fixed_height, new_width))

        # Features to serve
        features = {'images': resized_image[None],  # cast to 1 x h x w x c
                    'images_widths': new_width[None] # cast to
                    }

        # Inputs received
        receiver_inputs = {'images': image}

        return tf.estimator.export.ServingInputReceiver(features, receiver_inputs)

    return serving_input_fn
