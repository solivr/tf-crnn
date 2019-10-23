#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

from setuptools import setup, find_packages

setup(name='tf_crnn',
      version='0.6.0',
      license='GPL',
      author='Sofia Ares Oliveira',
      url='https://github.com/solivr/tf-crnn',
      description='TensorFlow Convolutional Recurrent Neural Network (CRNN)',
      install_requires=[
            'imageio',
            'numpy',
            'tqdm',
            'sacred',
            'opencv-python',
            'pandas',
            'click',
            'tensorflow-addons',
            'tensorflow-gpu',
            'taputapu'
      ],
      dependency_links=['https://github.com/solivr/taputapu/tarball/master#egg=taputapu-1.0'],
      extras_require={
            'doc': [
                  'sphinx',
                  'sphinx-autodoc-typehints',
                  'sphinx-rtd-theme',
                  'sphinxcontrib-bibtex',
                  'sphinxcontrib-websupport'
            ],
      },
      packages=find_packages(where='.'),
      zip_safe=False)
