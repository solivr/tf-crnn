#!/usr/bin/env python
__author__ = 'solivr'
__license__ = "GPL"

import tensorflow as tf
from tensorflow_addons.image.transform_ops import rotate, transform
from .config import Params, CONST
from typing import Tuple, Union, List
import collections


@tf.function
def random_rotation(img: tf.Tensor,
                    max_rotation: float=0.1,
                    crop: bool=True,
                    minimum_width: int=0) -> tf.Tensor:  # adapted from SeguinBe
    """
    Rotates an image with a random angle.
    See https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders for formulae

    :param img: Tensor
    :param max_rotation: maximum angle to rotate (radians)
    :param crop: boolean to crop or not the image after rotation
    :param minimum_width: minimum width of image after data augmentation
    :return:
    """
    with tf.name_scope('RandomRotation'):
        rotation = tf.random.uniform([], -max_rotation, max_rotation, name='pick_random_angle')
        # rotated_image = tf.contrib.image.rotate(img, rotation, interpolation='BILINEAR')
        rotated_image = rotate(tf.expand_dims(img, axis=0), rotation, interpolation='BILINEAR')
        rotated_image = tf.squeeze(rotated_image, axis=0)
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
            rotated_image = tf.cond(tf.less_equal(tf.shape(rotated_image_crop)[1], minimum_width),
                                    true_fn=lambda: img,
                                    false_fn=lambda: rotated_image_crop,
                                    name='check_size_crop')

        return rotated_image


# def random_padding(image: tf.Tensor, max_pad_w: int=5, max_pad_h: int=10) -> tf.Tensor:
#     """
#     Given an image will pad its border adding a random number of rows and columns
#
#     :param image: image to pad
#     :param max_pad_w: maximum padding in width
#     :param max_pad_h: maximum padding in height
#     :return: a padded image
#     """
#     # TODO specify image shape in doc
#
#     w_pad = list(np.random.randint(0, max_pad_w, size=[2]))
#     h_pad = list(np.random.randint(0, max_pad_h, size=[2]))
#     paddings = [h_pad, w_pad, [0, 0]]
#
#     return tf.pad(image, paddings, mode='REFLECT', name='random_padding')

@tf.function
def augment_data(image: tf.Tensor,
                 max_rotation: float=0.1,
                 minimum_width: int=0) -> tf.Tensor:
    """
    Data augmentation on an image (padding, brightness, contrast, rotation)

    :param image: Tensor
    :param max_rotation: float, maximum permitted rotation (in radians)
    :param minimum_width: minimum width of image after data augmentation
    :return: Tensor
    """
    with tf.name_scope('DataAugmentation'):

        # Random padding
        # image = random_padding(image)

        # TODO : add random scaling
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, 0.5, 1.5)
        image = random_rotation(image, max_rotation, crop=True, minimum_width=minimum_width)

        if image.shape[-1] >= 3:
            image = tf.image.random_hue(image, 0.2)
            image = tf.image.random_saturation(image, 0.5, 1.5)

        return image

@tf.function
def get_resized_width(image: tf.Tensor,
                      target_height: int,
                      increment: int=CONST.DIMENSION_REDUCTION_W_POOLING):

    image_shape = tf.shape(image)
    image_ratio = tf.divide(image_shape[1], image_shape[0], name='ratio')

    new_width = tf.cast(tf.round((image_ratio * target_height) / increment) * increment, tf.int32)
    f1 = lambda: (new_width, image_ratio)
    f2 = lambda: (target_height, tf.constant(1.0, dtype=tf.float64))
    if tf.math.less_equal(new_width, 0):
        return f2()
    else:
        return f1()


@tf.function
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

            img_resized = tf.image.resize(image, [new_h, new_w])

            # Padding to have the desired width
            paddings = [[0, 0], [0, pad], [0, 0]]
            pad_image = tf.pad(img_resized, paddings, mode='SYMMETRIC', name=None)

            # Set manually the shape
            pad_image.set_shape([target_shape[0], target_shape[1], img_resized.get_shape()[2]])

            return pad_image, (new_h, new_w)

    def replicate_fn():
        with tf.name_scope('replication_padding'):
            img_resized = tf.image.resize(image, [new_h, new_w])

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
            img_resized = tf.image.resize(image, target_shape)

            img_resized.set_shape([target_shape[0], target_shape[1], img_resized.get_shape()[2]])

            return img_resized, tuple(target_shape)

    # case 1 : new_w >= target_w
    if tf.logical_and(tf.greater_equal(ratio, target_ratio), tf.greater_equal(new_w, target_w)):
            pad_image, (new_h, new_w) =  simple_resize()
    # case 2 : new_w >= target_w/2 & new_w < target_w & ratio < target_ratio
    elif tf.logical_and(tf.less(ratio, target_ratio),
                        tf.logical_and(tf.greater_equal(new_w, tf.cast(tf.divide(target_w, 2), tf.int32)),
                                       tf.less(new_w, target_w))):
        pad_image, (new_h, new_w) = pad_fn()
    # case 3 : new_w < target_w/2 & new_w < target_w & ratio < target_ratio
    elif tf.logical_and(tf.less(ratio, target_ratio),
                        tf.logical_and(tf.less(new_w, target_w),
                                       tf.less(new_w, tf.cast(tf.divide(target_w, 2), tf.int32)))):
        pad_image, (new_h, new_w) = replicate_fn()
    else:
        pad_image, (new_h, new_w) = simple_resize()

    return pad_image, new_w


# def apply_slant(image: np.ndarray, alpha: np.ndarray) -> (np.ndarray, np.ndarray):
#     alpha = alpha[0]
#
#     def _find_background_color(image: np.ndarray) -> int:
#         """
#         Given a grayscale image, finds the background color value
#         :param image: grayscale image
#         :return: background color value (int)
#         """
#         # Otsu's thresholding after Gaussian filtering
#         blur = cv2.GaussianBlur(image[:, :, 0].astype(np.uint8), (5, 5), 0)
#         thresh_value, thresholded_image = cv2.threshold(blur.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#
#         # Find which is the background (0 or 255). Supposing that the background color occurrence is higher
#         # than the writing color
#         counts, bin_edges = np.histogram(thresholded_image, bins=2)
#         background_color = int(np.median(image[thresholded_image == 255 * np.argmax(counts)]))
#
#         return background_color
#
#     shape_image = image.shape
#     shift = max(-alpha * shape_image[0], 0)
#     output_size = (int(shape_image[1] + np.ceil(abs(alpha * shape_image[0]))), int(shape_image[0]))
#
#     warpM = np.array([[1, alpha, shift], [0, 1, 0]])
#
#     # Find color of background in order to replicate it in the borders
#     border_value = _find_background_color(image)
#
#     image_warp = cv2.warpAffine(image, np.array(warpM), output_size, borderValue=border_value)
#
#     return image_warp, np.array(output_size)


def dataset_generator(csv_filename: Union[List[str], str],
                      params: Params,
                      use_labels: bool=True,
                      batch_size: int=64,
                      data_augmentation: bool=False,
                      num_epochs: int=None,
                      shuffle: bool=True):
    do_padding = True

    if use_labels:
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
    # dataset = dataset.apply(tf.data.experimental.parallel_interleave(filename_to_dataset,
    #                                                                  cycle_length=num_parallel_reads))
    dataset = dataset.interleave(filename_to_dataset, cycle_length=num_parallel_reads,
                                 num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(map_fn)
    # -----

    def _load_image(features: dict, labels=None):
        path = features['paths']
        image_content = tf.io.read_file(path)
        image = tf.io.decode_jpeg(image_content, channels=params.input_channels,
                                  try_recover_truncated=True, name='image_decoding_op')

        if use_labels:
            return {'input_images': image,
                    'label_seq_length': features['label_seq_length']}, labels
        else:
            return {'input_images': image,
                    'filename_images': path}

    def _apply_slant(features: dict, labels=None):
        image = features['input_images']
        height_image = tf.cast(tf.shape(image)[0], dtype=tf.float32)

        with tf.name_scope('add_slant'):
            alpha = tf.random.uniform([],
                                      -params.data_augmentation_max_slant,
                                      params.data_augmentation_max_slant,
                                      name='pick_random_slant_angle')

            shiftx = tf.math.maximum(tf.math.multiply(-alpha, height_image), 0)

            # Pad in order not to loose image info when transformation is applied
            x_pad = 0
            y_pad = tf.math.round(tf.math.ceil(tf.math.abs(tf.math.multiply(alpha, height_image))))
            y_pad = tf.cast(y_pad, dtype=tf.int32)
            paddings = [[x_pad, x_pad], [y_pad, 0], [0, 0]]
            transform_matrix = [1, alpha, shiftx, 0, 1, 0, 0, 0]

            # Apply transformation to image
            image_pad = tf.pad(image, paddings)
            image_transformed = transform(image_pad, transform_matrix, interpolation='BILINEAR')

            # Apply transformation to mask. The mask will be used to retrieve the pixels that have been filled
            # with zero during transformation and update their value with background value
            # TODO : Would be better to have some kind of binarization (i.e Otsu) and get the mean background value
            background_pixel_value = 255
            empty = background_pixel_value * tf.ones(tf.shape(image))
            empty_pad = tf.pad(empty, paddings)
            empty_transformed = tf.subtract(
                tf.cast(background_pixel_value, dtype=tf.int32),
                tf.cast(transform(empty_pad, transform_matrix, interpolation='NEAREST'), dtype=tf.int32)
            )

            # Update additional zeros values with background_pixel_value and cast result to uint8
            image = tf.add(tf.cast(image_transformed, dtype=tf.int32), empty_transformed)
            image = tf.cast(image, tf.uint8)

        features['input_images'] = image
        return features, labels if use_labels else features

    def _data_augment_fn(features: dict, labels=None) -> tf.data.Dataset:
        image = features['input_images']
        image = augment_data(image, params.data_augmentation_max_rotation, minimum_width=params.max_chars_per_string)

        features.update({'input_images': image})
        return features, labels if use_labels else features

    def _pad_image_or_resize(features: dict, labels=None):
        image = features['input_images']
        if do_padding:
            with tf.name_scope('padding'):
                image, img_width = padding_inputs_width(image, target_shape=params.input_shape,
                                                        increment=CONST.DIMENSION_REDUCTION_W_POOLING)
        # Resize
        else:
            image = tf.image.resize(image, size=params.input_shape)
            img_width = tf.shape(image)[1]

        input_seq_length = tf.cast(tf.floor(tf.divide(img_width, params.downscale_factor)), tf.int32)
        if use_labels:
            assert_op = tf.debugging.assert_greater_equal(input_seq_length,
                                                          features['label_seq_length'])
            with tf.control_dependencies([assert_op]):
                return {'input_images': image,
                        'label_seq_length': features['label_seq_length'],
                        'input_seq_length': input_seq_length}, labels
        else:
            return {'input_images': image,
                    'input_seq_length': input_seq_length,
                    'filename_images': features['filename_images']}

    def _normalize_image(features: dict, labels=None):
        image = tf.cast(features['input_images'], tf.float32)
        image = tf.image.per_image_standardization(image)

        features['input_images'] = image
        return features, labels if use_labels else features

    def _format_label_codes(features: dict, string_label_codes):
        splits = tf.strings.split([string_label_codes], sep=' ')
        label_codes = tf.squeeze(tf.strings.to_number(splits, out_type=tf.int32), axis=0)

        features.update({'label_codes': label_codes})
        return features, [0]


    num_parallel_calls = tf.data.experimental.AUTOTUNE
    #  1. load image 2. data augmentation 3. padding
    dataset = dataset.map(_load_image, num_parallel_calls=num_parallel_calls)
    # this causes problems when using the same cache for training, validation and prediction data...
    # dataset = dataset.cache(filename=os.path.join(params.output_model_dir, 'cache.tf-data'))
    if data_augmentation and params.data_augmentation_max_slant != 0:
        dataset = dataset.map(_apply_slant, num_parallel_calls=num_parallel_calls)
    if data_augmentation:
        dataset = dataset.map(_data_augment_fn, num_parallel_calls=num_parallel_calls)
    dataset = dataset.map(_normalize_image, num_parallel_calls=num_parallel_calls)
    dataset = dataset.map(_pad_image_or_resize, num_parallel_calls=num_parallel_calls)
    dataset = dataset.map(_format_label_codes, num_parallel_calls=num_parallel_calls) if use_labels else dataset
    dataset = dataset.shuffle(10 * batch_size, reshuffle_each_iteration=False) if shuffle else dataset
    dataset = dataset.repeat(num_epochs) if num_epochs is not None else dataset

    return dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)


# def dataset_prediction(image_filenames: Union[List[str], str]=None,
#                        csv_filename: str=None,
#                        params: Params=None,
#                        batch_size: int=64):
#
#     assert params, 'params cannot be None'
#     assert image_filenames or csv_filename, 'You need to feed an input (image_filenames or csv_filename)'
#
#     do_padding = True
#
#     def _load_image(path):
#         image_content = tf.io.read_file(path)
#         image = tf.io.decode_jpeg(image_content, channels=params.input_channels,
#                                   try_recover_truncated=True, name='image_decoding_op')
#
#         return {'input_images': image}
#
#     def _normalize_image(features: dict):
#         image = tf.cast(features['input_images'], tf.float32)
#         image = tf.image.per_image_standardization(image)
#
#         features['input_images'] = image
#         return features
#
#     def _pad_image_or_resize(features: dict):
#         image = features['input_images']
#         if do_padding:
#             with tf.name_scope('padding'):
#                 image, img_width = padding_inputs_width(image, target_shape=params.input_shape,
#                                                         increment=CONST.DIMENSION_REDUCTION_W_POOLING)
#         # Resize
#         else:
#             image = tf.image.resize(image, size=params.input_shape)
#             img_width = tf.shape(image)[1]
#
#         input_seq_length = tf.cast(tf.floor(tf.math.divide(img_width, params.n_pool)), tf.int32)
#
#         return {'input_images': image,
#                 'input_seq_length': input_seq_length}
#     if image_filenames is not None:
#         dataset = tf.data.Dataset.from_tensor_slices(image_filenames)
#     elif csv_filename is not None:
#         column_defaults = [['None']]
#         dataset = tf.data.experimental.CsvDataset(csv_filename,
#                                                   record_defaults=column_defaults,
#                                                   field_delim=params.csv_delimiter,
#                                                   header=False)
#         # dataset = dataset.map(map_fn)
#     dataset = dataset.map(_load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#     dataset = dataset.map(_normalize_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#     dataset = dataset.map(_pad_image_or_resize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#
#     return dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
