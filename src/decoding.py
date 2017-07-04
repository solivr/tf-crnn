#!/usr/bin/env python
__author__ = 'solivr'

import tensorflow as tf


def get_words_from_chars(characters_list, sequence_lengths, name='chars_conversion'):
    with tf.name_scope(name=name):
        def my_func(coords):
            return tf.reduce_join(characters_list[coords[0]:coords[1]])

        end_coords = tf.cumsum(sequence_lengths)
        start_coords = tf.concat([[0], end_coords[:-1]], axis=0)
        coords = tf.stack([start_coords, end_coords], axis=1)
        coords = tf.cast(coords, dtype=tf.int32)

        return tf.map_fn(my_func, coords, dtype=tf.string)
