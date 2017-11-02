# Convolutional Recurrent Neural Network in Tensorflow (tf.crnn)
CRNN model in Tensorflow using Estimators

Implementation of the Convolutional Recurrent Neural Network (CRNN) for image-based sequence recognition tasks, such as scene text recognition and OCR. 
Original paper http://arxiv.org/abs/1507.05717 and code https://github.com/bgshih/crnn

This version uses the `tf.estimator.Estimator` to build the model.

### Contents
* `src/model.py` : definition of the model
* `src/data_handler.py` : functions for data loading, preprocessing and data augmentation
* `src/config.py` : `class Params` manages parameters of model and experiments
* `src/decoding.py` : helper fucntion to transform characters to words
* `train.py` : script to launch for training the model, more info on the parameters and options inside
* `export_model.py`: script to export a model once trained, i.e for serving
* Extra : `hlp/numbers_mnist_generator.py` : generates a sequence of digits to form a number using the MNIST database
* Extra : `hlp/csv_path_convertor.py` : converts a csv file with relative paths to a csv file with absolute paths

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

### Dependencies 
* `tensorflow` (1.3)
* `tensorflow-tensorboard` (0.1.7) (not mandatory but useful to visualise loss, accuracy and inputs / outputs)
* `tqdm` for progress bars
* `json`
* `glob`



