# Convolutional Recurrent Neural Network in Tensorflow (tf.crnn)

[![Documentation Status](https://readthedocs.org/projects/tf-crnn/badge/?version=latest)](https://tf-crnn.readthedocs.io/en/latest/?badge=latest)

CRNN model in Tensorflow using Estimators

Implementation of the Convolutional Recurrent Neural Network (CRNN) for image-based sequence recognition tasks, such as scene text recognition and OCR. 
Original paper http://arxiv.org/abs/1507.05717 and code https://github.com/bgshih/crnn

This version uses the `tf.estimator.Estimator` to build the model and `tf.data` module to handle input data.


## Installation
`tf_crnn` make use of `tensorflow-gpu` package (so CUDA and cuDNN are needed). 

You can install it using the `environment.yml` file provided and use it within an environment, or install and run it with Docker.

See the [docs](https://tf-crnn.readthedocs.io/en/latest/start/index.html#) for the installation procedures and how to use it.