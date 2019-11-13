# Text recognition with Convolutional Recurrent Neural Network and TensorFlow 2.0 (tf2-crnn)

[![Documentation Status](https://readthedocs.org/projects/tf-crnn/badge/?version=latest)](https://tf-crnn.readthedocs.io/en/latest/?badge=latest)

Implementation of a Convolutional Recurrent Neural Network (CRNN) for image-based sequence recognition tasks, such as scene text recognition and OCR. 

This implementation is based on Tensorflow 2.0 and uses `tf.keras` and `tf.data` modules to build the model and to handle input data.


## Installation
`tf_crnn` makes use of `tensorflow-gpu` package (so CUDA and cuDNN are needed). 

You can install it using the `environment.yml` file provided and use it within an environment.
    
    conda env create -f environment.yml

See also the [docs](https://tf-crnn.readthedocs.io/en/latest/start/index.html#) for more information.


## Try it
 
 Train a model with [IAM dataset](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database).
 
 **Generate the data in the correct format**

    cd hlp
    python prepare_iam.py --download_dir ../data/iam --generated_data_dir ../data/iam/generated
    cd ..
    
**Train the model**

    python training.py with config.json

More details in the [documentation](https://tf-crnn.readthedocs.io/en/latest/start/training.html).