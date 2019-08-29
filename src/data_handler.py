#!/usr/bin/env python
__author__ = 'solivr'
__license__ = "GPL"

import tensorflow as tf
import numpy as np
from .config import Params, CONST
from typing import Tuple, Union, List
from functools import reduce
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


# def dataset_generator(csv_filename: Union[List[str], str],
#                       params: Params,
#                       labels: bool=True,
#                       batch_size: int=64,
#                       data_augmentation: bool=False,
#                       num_epochs: int=None):
#
#     do_padding = True
#
#     cnn_params = zip(params.cnn_pool_size, params.cnn_pool_strides, params.cnn_stride_size)
#     n_pool = reduce(lambda i, j: i + j, map(lambda k: k[0][1] * k[1][1] * k[2][1], cnn_params))
#
#     if labels:
#         csv_types = [['None'], ['None'], tf.int32]
#         column_names = ['paths', 'label_codes', 'label_seq_length']
#     else:
#         csv_types = [['None']]
#         column_names = ['input_images']
#
#     # Helper to read content of files
#     def _read_content(path):
#         image_content = tf.io.read_file(path)
#         image = tf.io.decode_jpeg(image_content, channels=params.input_channels,
#                                   try_recover_truncated=True, name='image_decoding_op')
#         return image
#
#     def _padding_or_resize(image) -> tf.data.Dataset:
#         # Padding
#         if do_padding:
#             with tf.name_scope('do_padding'):
#                 image, img_width = padding_inputs_width(image, target_shape=params.input_shape,
#                                                         increment=CONST.DIMENSION_REDUCTION_W_POOLING)
#         # Resize
#         else:
#             image = tf.image.resize_images(image, size=params.input_shape)
#             img_width = tf.shape(image)[1]
#
#         input_seq_length = tf.cast(tf.floor(tf.divide(img_width, n_pool)), tf.int32)
#
#         return {'input_images': image,
#                 'input_seq_length': input_seq_length}
#
#     def _load_paths(features, labels):
#         """
#         Load images from string filename, and pad/resize it
#         """
#         path_ds = tf.data.Dataset.from_tensor_slices(features['paths'])
#         image_ds = path_ds.map(_read_content)
#         image_ds = image_ds.map(_padding_or_resize)
#         return image_ds
#
#     def _get_features(features, string_label_codes):
#         splits = tf.sparse.to_dense(tf.strings.split(string_label_codes, sep=' '))
#         label_codes = tf.strings.to_number(splits, out_type=tf.int32)
#
#         return {'label_codes': label_codes,
#                 'label_seq_length': features['label_seq_length']}
#
#     def _format_extended_dataset(label_features, images_features):
#         assert_op = tf.debugging.assert_greater_equal(images_features['input_seq_length'],
#                                                       label_features['label_seq_length'])
#         with tf.control_dependencies([assert_op]):
#             return {'input_images': images_features['input_images'],
#                     'input_seq_length': images_features['input_seq_length'],
#                     'label_codes': label_features['label_codes'],
#                     'label_seq_length': label_features['label_seq_length']}, np.zeros(batch_size)
#
#     # this returns a dataset with structure ({'paths': paths, 'label_seq_length': label_seq_length}, label_codes)
#     dataset = tf.data.experimental.make_csv_dataset(csv_filename,
#                                                     batch_size=batch_size,
#                                                     column_names=column_names,
#                                                     label_name='label_codes',
#                                                     column_defaults=csv_types,
#                                                     field_delim=params.csv_delimiter,
#                                                     use_quote_delim=True,
#                                                     header=False,
#                                                     num_epochs=num_epochs)
#
#     # First create a dataset with the loaded images (int32)
#     ds_images_features = dataset.flat_map(_load_paths)
#     # Then create a dataset with the original features (except paths)
#     ds_label_features = dataset.map(_get_features)
#     # Create a dataset combining images and original features
#     # ds_images is already batched in order to have the same dimensions as ds_original_features
#     extended_ds = tf.data.Dataset.zip((ds_label_features, ds_images_features.batch(batch_size)))
#
#     # Get the dataset formatted as we need it {(features dict}, dummy label)
#     dataset = extended_ds.map(_format_extended_dataset)
#
#     def _data_augment_fn(features: dict, label) -> tf.data.Dataset:
#
#         image = features['input_images']
#         image = augment_data(image, params.data_augmentation_max_rotation)
#
#         features.update({'input_images': image})
#         return features, label
#
#     if data_augmentation:
#         dataset = dataset.map(_data_augment_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#
#     return dataset.prefetch(tf.data.experimental.AUTOTUNE)


def dataset_generator(csv_filename: Union[List[str], str],
                       params: Params,
                       labels: bool=True,
                       batch_size: int=64,
                       data_augmentation: bool=False,
                       num_epochs: int=None):
    do_padding = True

    cnn_params = zip(params.cnn_pool_size, params.cnn_pool_strides, params.cnn_stride_size)
    n_pool = reduce(lambda i, j: i + j, map(lambda k: k[0][1] * k[1][1] * k[2][1], cnn_params))

    if labels:
        column_defaults = [['None'], ['None'], tf.int32]
        column_names = ['paths', 'label_codes', 'label_seq_length']
        label_name = 'label_codes'
    else:
        column_defaults = [['None']]
        column_names = ['input_images']
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

        input_seq_length = tf.cast(tf.floor(tf.divide(img_width, n_pool)), tf.int32)

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
    dataset = dataset.shuffle(1024, reshuffle_each_iteration=False).repeat(1)

    return dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)


def serving_single_input(input_shape: Tuple[int, int]):
    """
    Serving input function needed for export (in TensorFlow).
    Features to serve :
        - `images` : greyscale image
        - `input_filename` : filename of image segment
        - `input_rgb`: RGB image segment

    :param fixed_height: height  of the image to format the input data with
    :param min_width: minimum width to resize the image
    :return: serving_input_fn
    """

    def serving_input_fn():

        # define placeholder for filename
        filename = tf.placeholder(dtype=tf.string)
        decoded_image = tf.cast(tf.image.decode_jpeg(tf.io.read_file(filename), channels=3,
                                                     try_recover_truncated=True), tf.float32)

        image = tf.image.rgb_to_grayscale(decoded_image, name='rgb2gray')
        # define placeholder for input image
        # image = tf.placeholder(dtype=tf.float32, shape=[None, None, 1])

        # shape = tf.shape(image)
        # # Assert shape is h x w x c with c = 1
        #
        # ratio = tf.divide(shape[1], shape[0])
        # increment = CONST.DIMENSION_REDUCTION_W_POOLING
        # new_width = tf.cast(tf.round((ratio * fixed_height) / increment) * increment, tf.int32)
        #
        # resized_image = tf.cond(new_width < tf.constant(min_width, dtype=tf.int32),
        #                         true_fn=lambda: tf.image.resize_images(image, size=(fixed_height, min_width)),
        #                         false_fn=lambda: tf.image.resize_images(image, size=(fixed_height, new_width))
        #                         )

        with tf.name_scope('padding'):
            padded_image, img_width = padding_inputs_width(image, target_shape=input_shape,
                                                           increment=CONST.DIMENSION_REDUCTION_W_POOLING)

        # Features to serve
        features = {'images': padded_image[None],  # cast to 1 x h x w x c
                    'images_widths': img_width[None]  # cast to tensor
                    }

        # Inputs received
        receiver_inputs = {'images': image}
        alternative_receivers = {'input_filename': {'filename': filename}, 'input_rgb': {'rgb_images': decoded_image}}

        return tf.estimator.export.ServingInputReceiver(features, receiver_tensors=receiver_inputs,
                                                        receiver_tensors_alternatives=alternative_receivers)

    return serving_input_fn


# TODO serving function for batches
def serving_batch_filenames_fn(input_shape=(32, 100), n_channels: int=1, padding=True):
    """
    Serving input function for batch inference using filenames as inputs

    :param input_shape: shape of the input after resizing/padding
    :param n_channels: number of channels of images
    :param padding: if True, keeps the image ratio and pads it to get to 'input_shape' shape,
        if False will resize the image using bilinear interpolation
    :param batch_size: batch_size for inference
    :return: serving input function
    """

    def serving_input_fn():

        # Define placeholder for batch size and filename
        batch_size = tf.placeholder(dtype=tf.int64, name='batch_size')
        image_filenames = tf.placeholder(dtype=tf.string, shape=[None], name='list_image_filenames')

        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices(image_filenames)

        # -- Read image
        def _image_reading_preprocessing(image_filename) -> dict():

            # Load
            image_content = tf.io.read_file(image_filename, name='filename_reader')
            # Decode image is not used because it seems the shape is not set...
            # image = tf.image.decode_jpeg(image_content, channels=params.input_channels,
            #                              try_recover_truncated=True,name='image_decoding_op')
            # tensorflow v1.8 change to :
            image = tf.cond(
                tf.image.is_jpeg(image_content),
                lambda: tf.image.decode_jpeg(image_content, channels=n_channels, name='image_decoding_op',
                                             try_recover_truncated=True),
                lambda: tf.image.decode_png(image_content, channels=n_channels, name='image_decoding_op'))

            # Padding
            if padding:
                with tf.name_scope('padding'):
                    image, img_width = padding_inputs_width(image, target_shape=input_shape,
                                                            increment=CONST.DIMENSION_REDUCTION_W_POOLING)
            # Resize
            else:
                image = tf.image.resize_images(image, size=input_shape)
                img_width = tf.shape(image)[1]

            return image, img_width
        dataset = dataset.map(_image_reading_preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = dataset.batch(batch_size).prefetch(32)

        # Build the Iterator this way in order to be able to initialize it when the saved_model will be loaded
        # From http://vict0rsch.github.io/2018/05/17/restore-tf-model-dataset/
        iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
        dataset_init_op = iterator.make_initializer(dataset, name='dataset_init')
        features_images, features_widths = iterator.get_next()

        # Features to serve:  'images', images_width'
        features = {'images': features_images, 'images_widths': features_widths}

        return tf.estimator.export.ServingInputReceiver(features, receiver_tensors={'list_filenames': image_filenames,
                                                                                    'batch_size': batch_size})

    return serving_input_fn


# TODO serving function from url...