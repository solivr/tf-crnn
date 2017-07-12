# Convolutional Recurrent Neural Network in Tensorflow (tf.crnn)
CRNN model in Tensorflow using Estimators

Implementation of the Convolutional Recurrent Neural Network (CRNN) for image-based sequence recognition tasks, such as scene text recognition and OCR. 
Original paper http://arxiv.org/abs/1507.05717 and code https://github.com/bgshih/crnn

This version uses the `tf.estimator.Estimator` to build the model.

### Contents
* `src/model.py` : the definition of the model
* `src/data_handler.py` : the function for data loading, preprocessing and data augmentation
* `hlp/numbers_mnist_generator.py` : generates a sequence of digits to form a number using the MNIST database
* `train.py` : script to launch for training the model, more info on the parameters and options inside
* `export_model.py`: script to export a model once trained, i.e for serving

### How to

The main script to launch is `train.py`.
To train the model, you should input a csv file with each row containing the filename of the image (full path) and its label (plain text) separated by a space :

```
/full/path/to/image1.{jpg,png} string_label1
/full/path/to/image2.{jpg,png} string_label2
...
```

For example launch the script using :
```
python train.py -g 1 -ft train_data.csv -fe val_data.csv -o ./export_model_dir
```
See `train.py` for more details on the options.

### Dependencies 
* `tensorflow` (1.2)
* `warpctc_tensorflow` (from Baidu's warp-CTC)
* `tqdm` for progress bars



