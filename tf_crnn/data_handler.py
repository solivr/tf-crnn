#!/usr/bin/env python
__author__ = 'solivr'
__license__ = "GPL"

import tensorflow as tf
import numpy as np
import csv
from .config import Params, CONST
from typing import Tuple, Union, List


def random_rotation(img: tf.Tensor, max_rotation: float=0.1, crop: bool=True) -> tf.Tensor:  # adapted from SeguinBe
    """
    Rotates an image with a random angle
    see https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders for formulae
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
            bb_begin = tf.cast(tf.ceil((h-new_h)/2), tf.int32), tf.cast(tf.ceil((w-new_w)/2), tf.int32)
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


def augment_data(image: tf.Tensor) -> tf.Tensor:
    """
    Data augmentation on an image (padding, brightness, contrast, rotation)
    :param image: Tensor
    :return: Tensor
    """
    with tf.name_scope('DataAugmentation'):

        # Random padding
        image = random_padding(image)

        # TODO : add random scaling
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, 0.5, 1.5)
        image = random_rotation(image, 0.05, crop=True)

        if image.shape[-1] >= 3:
            image = tf.image.random_hue(image, 0.2)
            image = tf.image.random_saturation(image, 0.5, 1.5)

        return image


def padding_inputs_width(image: tf.Tensor, target_shape: Tuple[int, int], increment: int) \
        -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Given an input image, will pad it to return a target_shape size padded image.
    There is 3 cases:
         - image width > target width : simple resizing to shrink the image
         -
    :param image: Tensor of shape [H,W,C]
    :param target_shape: final shape after padding [H, W]
    :param increment: reduction factor due to pooling between input width and output width,
                        this makes sure that the final width will be a multiple of increment
    :return: (image padded, output width)
    """

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


# def preprocess_image_for_prediction(fixed_height: int=32, min_width: int=8):
#     """
#     Input function to use when exporting the model for making predictions (see estimator.export_savedmodel)
#     :param fixed_height: height of the input image after resizing
#     :param min_width: minimum width of image after resizing
#     :return:
#     """
#
#     def serving_input_fn():
#         # define placeholder for input image
#         image = tf.placeholder(dtype=tf.float32, shape=[None, None, 1])
#
#         shape = tf.shape(image)
#         # Assert shape is h x w x c with c = 1
#
#         ratio = tf.divide(shape[1], shape[0])
#         increment = CONST.DIMENSION_REDUCTION_W_POOLING
#         new_width = tf.cast(tf.round((ratio * fixed_height) / increment) * increment, tf.int32)
#
#         resized_image = tf.cond(new_width < tf.constant(min_width, dtype=tf.int32),
#                                 true_fn=lambda: tf.image.resize_images(image, size=(fixed_height, min_width)),
#                                 false_fn=lambda: tf.image.resize_images(image, size=(fixed_height, new_width))
#                                 )
#
#         # Features to serve
#         features = {'images': resized_image[None],  # cast to 1 x h x w x c
#                     'images_widths': new_width[None]  # cast to tensor
#                     }
#
#         # Inputs received
#         receiver_inputs = {'images': image}
#
#         return tf.estimator.export.ServingInputReceiver(features, receiver_inputs)
#
#     return serving_input_fn


def data_loader(csv_filename: Union[List[str], str], params: Params, labels=True, batch_size: int=64,
                data_augmentation: bool=False, num_epochs: int=None, image_summaries: bool=False):
    """
    Loads, preprocesses (data augmentation, padding) and feeds the data
    :param csv_filename: filename or list of filenames
    :param params: Params object containing all the parameters
    :param labels: transcription labels
    :param batch_size: batch_size
    :param data_augmentation: flag to select or not data augmentation
    :param num_epochs: feeds the data 'num_epochs' times
    :param image_summaries: floag to show image summaries or not
    :return: data_loader function
    """

    if labels:
        csv_types = [['None'], ['None']]
        csv_column_names = ['filenames', 'labels']
    else:
        csv_types = [['None']]
        csv_column_names = ['filenames']
    padding = True

    def input_fn():
        dataset = tf.data.TextLineDataset(csv_filename)
        with tf.name_scope('CSV_reading'):

            # -- Parse each line.
            def _parse_csv_line(line):
                # Decode the line into its fields
                fields = tf.decode_csv(line, record_defaults=csv_types,
                                       field_delim=params.csv_delimiter, name='csv_reading_op')
                # Pack the result into a dictionary
                features = dict(zip(csv_column_names, fields))

                return features

            dataset = dataset.map(_parse_csv_line)

        # -- Read image
        def _image_reading_preprocessing(features: dict) -> dict():

            # Load
            image_content = tf.read_file(features['filenames'], name='filename_reader')
            # decode image is not used because it seems the shape is not set...
            # image = tf.image.decode_jpeg(image_content, channels=params.input_channels,
            #                              try_recover_truncated=True,name='image_decoding_op')
            # tensorflow v1.8 change to :
            image = tf.cond(
                tf.image.is_jpeg(image_content),
                lambda: tf.image.decode_jpeg(image_content, channels=params.input_channels, name='image_decoding_op',
                                             try_recover_truncated=True),
                lambda: tf.image.decode_png(image_content, channels=params.input_channels, name='image_decoding_op'))

            # Data augmentation
            if data_augmentation:
                image = augment_data(image)

            # Padding
            if padding:
                with tf.name_scope('padding'):
                    image, img_width = padding_inputs_width(image, target_shape=params.input_shape,
                                                            increment=CONST.DIMENSION_REDUCTION_W_POOLING)
            # Resize
            else:
                image = tf.image.resize_images(image, size=params.input_shape)
                img_width = tf.shape(image)[1]

            # Update features
            features.update({'images': image, 'images_widths': img_width})

            return features

        dataset = dataset.map(_image_reading_preprocessing)

        # -- Shuffle, repeat, and batch features
        dataset = dataset.shuffle(2048).batch(batch_size).repeat(num_epochs).prefetch(4)
        dataset_iterator = dataset.make_one_shot_iterator()
        prepared_batch = dataset_iterator.get_next()

        if image_summaries:
            tf.summary.image('input/image', prepared_batch['images'], max_outputs=1)
        if labels:
            tf.summary.text('input/labels', prepared_batch.get('labels')[:10])

        return prepared_batch, prepared_batch.get('labels')

    return input_fn


def serving_single_input(fixed_height: int=32, min_width: int=8):

    def serving_input_fn():

        # define placeholder for filename
        filename = tf.placeholder(dtype=tf.string)
        decoded_image = tf.to_float(tf.image.decode_jpeg(tf.read_file(filename), channels=3,
                                                         try_recover_truncated=True))

        image = tf.image.rgb_to_grayscale(decoded_image, name='rgb2gray')
        # define placeholder for input image
        # image = tf.placeholder(dtype=tf.float32, shape=[None, None, 1])

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
            image_content = tf.read_file(image_filename, name='filename_reader')
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
        dataset = dataset.map(_image_reading_preprocessing)

        dataset = dataset.batch(batch_size)

        # Build the Iterator this way in order to be able to initialize it when the saved_model will be loaded
        # From http://vict0rsch.github.io/2018/05/17/restore-tf-model-dataset/
        iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
        dataset_init_op = iterator.make_initializer(dataset, name='dataset_init')
        features_images, features_widths = iterator.get_next()

        # Features to serve 'images', images_width'
        features = {'images': features_images, 'images_widths': features_widths}

        return tf.estimator.export.ServingInputReceiver(features, receiver_tensors={'list_filenames': image_filenames,
                                                                                    'batch_size': batch_size})

    return serving_input_fn


# TODO serving function from url...