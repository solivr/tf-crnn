# Convolutional Recurrent Neural Network in Tensorflow (tf.crnn)
CRNN model in Tensorflow using Estimators

Implementation of the Convolutional Recurrent Neural Network (CRNN) for image-based sequence recognition tasks, such as scene text recognition and OCR. 
Original paper http://arxiv.org/abs/1507.05717 and code https://github.com/bgshih/crnn

This version uses the `tf.estimator.Estimator` to build the model and `tf.data` module to handle input data.


## Installation
`tf_crnn` uses `tensorflow-gpu` package (so CUDA and cuDNN are needed). 

Before using `tf_crnn` it is recommended to create a virtual environment (python 3.5).
Then, run `python setup.py install` to install the package and its dependencies. 
If you are using `anaconda` you can run `conda env create -f environment.yml`.

## Contents
* `tf_crnn/model.py` : definition of the model
* `tf_crnn/data_handler.py` : data loading, preprocessing and data augmentation
* `tf_crnn/config.py` : `class TrainingParams` and `class Params` manage parameters of training, model and experiments
* `tf_crnn/decoding.py` : helper function to transform characters to words
* `tf_crnn/loader.py` : to load a `saved_model` and run the model on new data
* `train.py` : script to launch to train the model, more info on the parameters and options inside
* `config_template.json` : configuration file used by `sacred` to run the experiment
* `tf_crnn/data/` : contains 3 'lookup tables' for alphabets in the form {'char', <int_code>}. 
* Extra : `tf_crnn/hlp/export_model.py`: script to export a model once trained, i.e for serving
* Extra : `tf_crnn/hlp/numbers_mnist_generator.py` : generates a sequence of digits to form a number using the MNIST database
* Extra : `tf_crnn/hlp/alphabet_helpers.py` : helpers to generate an alphabet lookup table


## How to train a model
`sacred` package is used to deal with experiments. If you are not yet familiar with it, have a quick look at the [documentation](https://sacred.readthedocs.io/en/latest/).

#### Input data
In order to train a model, you should input a csv file with each row containing the filename of the image (full path) and its label (plain text) separated by a delimiting character (let's say `;`). 
Also, each character should be separated by a splitting character (let's say `|`), this in order to deal with arbitrary alphabets (especially characters that cannot be encoded with `utf-8` format).
An example of such csv file would look like : 

```
/full/path/to/image1.{jpg,png};|s|t|r|i|n|g|_|l|a|b|e|l|1|
/full/path/to/image2.{jpg,png};|s|t|r|i|n|g|_|l|a|b|e|l|2| |w|i|t|h| |special_char|
...
```

#### Config file (with `sacred`)
Set the parameters of the experiment in `config_template.json`, especially `csv_files_train` and `csv_files_eval`. 

All the configurable parameters can be found in classes `tf_crnn.config.Params` and `tf_crnn.config.TrainingParams`, which can be added to the `config_template.json` file if needed.

#### Training
Once you have your csv file completed and the parameters set in `config_template.json`, run :
```
python train.py with config_template.json
```


## Dependencies 
All dependencies should be installed if you run `python setup.py install` or use the `environment.yml` file.
* `tensorflow` >= 1.3
* `sacred`
* `tensorflow-tensorboard` >= 0.1.7 (not mandatory but useful to visualise loss, accuracy and inputs / outputs)
* `tqdm` for progress bars
* `json`

