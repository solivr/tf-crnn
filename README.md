# Convolutional Recurrent Neural Network in Tensorflow (tf.crnn)
CRNN model in Tensorflow using Estimators

Implementation of the Convolutional Recurrent Neural Network (CRNN) for image-based sequence recognition tasks, such as scene text recognition and OCR. 
Original paper http://arxiv.org/abs/1507.05717 and code https://github.com/bgshih/crnn

This version uses the `tf.estimator.Estimator` to build the model.


### Installation
`tf_crnn` uses tensorflow-gpu v1.4.1 (so CUDA 8.0 and cuDNN v6.0 should be installed).

Before using `tf_crnn` it is recommended to create a virtual environment (python 3.5).
Then, run `python setup.py install` to install the package and its dependencies.

### Contents
* `tf_crnn/model.py` : definition of the model
* `tf_crnn/data_handler.py` : functions for data loading, preprocessing and data augmentation
* `tf_crnn/config.py` : `class TrainingParams` and `class Params` manage parameters of training, model and experiments
* `tf_crnn/decoding.py` : helper fucntion to transform characters to words
* `train.py` : script to launch to train the model, more info on the parameters and options inside
* `export_model.py`: script to export a model once trained, i.e for serving
* Extra : `tf_crnn/hlp/numbers_mnist_generator.py` : generates a sequence of digits to form a number using the MNIST database
* Extra : `tf_crnn/hlp/alphabet_helpers.py` : helpers to generate a lookup table alphabet


## TODO
### How to train a model
The main script to launch is `train.py`. 
To train the model, you should input a csv file with each row containing the filename of the image (full path) and its label (plain text) separated by a delimiting character (let's say ';') :

```
/full/path/to/image1.{jpg,png};string_label1
/full/path/to/image2.{jpg,png};string_label2
...
```

For example launch the script using :
```
python train.py -g 1 -ft train_data.csv -fe val_data.csv -o ./export_model_dir
```
See `train.py` for more details on the options.



