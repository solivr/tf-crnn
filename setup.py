#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

from setuptools import setup, find_packages

setup(name='tf_crnn',
      version='0.4.1',
      license='GPL',
      author='Sofia Ares Oliveira',
      url='https://github.com/solivr/tf-crnn',
      description='TensorFlow Convolutional Recurrent Neural Network (CRNN)',
      install_requires=[
            'tensorflow-gpu==1.4.1',
            'imageio',
            'tqdm',
            'sacred',
            'tensorflow-tensorboard==0.4.0',
            'better_exceptions',
            'opencv-python'
            # missing opencv2 that needs to be installed via conda
      ],
      packages=find_packages(where='.'),
      zip_safe=False)
