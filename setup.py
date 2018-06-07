#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

from setuptools import setup, find_packages

setup(name='tf_crnn',
      version='0.5.0',
      license='GPL',
      author='Sofia Ares Oliveira',
      url='https://github.com/solivr/tf-crnn',
      description='TensorFlow Convolutional Recurrent Neural Network (CRNN)',
      install_requires=[
            'tensorflow-gpu',
            'imageio',
            'tqdm',
            'sacred',
            'tensorflow-tensorboard',
            'better_exceptions',
            'opencv-python'
      ],
      packages=find_packages(where='.'),
      zip_safe=False)
