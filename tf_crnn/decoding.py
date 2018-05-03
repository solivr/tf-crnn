#!/usr/bin/env python
__author__ = 'solivr'
__license__ = "GPL"

import tensorflow as tf
from typing import List


def get_words_from_chars(characters_list: List[str], sequence_lengths: List[int], name='chars_conversion'):
    with tf.name_scope(name=name):
        def join_charcaters_fn(coords):
            return tf.reduce_join(characters_list[coords[0]:coords[1]])

        def coords_several_sequences():
            end_coords = tf.cumsum(sequence_lengths)
            start_coords = tf.concat([[0], end_coords[:-1]], axis=0)
            coords = tf.stack([start_coords, end_coords], axis=1)
            coords = tf.cast(coords, dtype=tf.int32)
            return tf.map_fn(join_charcaters_fn, coords, dtype=tf.string)

        def coords_single_sequence():
            return tf.reduce_join(characters_list, keep_dims=True)

        words = tf.cond(tf.shape(sequence_lengths)[0] > 1,
                        true_fn=lambda: coords_several_sequences(),
                        false_fn=lambda: coords_single_sequence())

        return words