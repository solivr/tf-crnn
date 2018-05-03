#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

from setuptools import setup, find_packages

setup(name='tf_crnn',
      version='0.4',
      url='https://github.com/solivr/tf-crnn',
      description='TensorFlow Convolutional Recurrent Neural Network (CRNN)',
      packages=find_packages(where='.'),
      zip_safe=False)
