#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

import tensorflow as tf
import os
import pickle
import json
import time

MODEL_WEIGHTS_FILENAME = 'weights.h5'
OPTIMIZER_WEIGHTS_FILENAME = 'optimizer_weights.pkl'
LEARNING_RATE_FILENAME = 'learning_rate.pkl'
LAYERS_FILENAME = 'architecture.json'
EPOCH_FILENAME = 'epoch.pkl'


class CustomSavingCallback(tf.keras.callbacks.Callback):
    """
    Callback to save weights, architecture, and optimizer at the end of training
    """
    def __init__(self,
                 output_dir: str):
        super(CustomSavingCallback, self).__init__()

        self.saving_dir = output_dir

    def on_epoch_begin(self, epoch, logs=None):
        self._current_epoch = epoch

    def on_train_end(self, logs=None):
        timestamp = str(int(time.time()))
        folder = os.path.join(self.saving_dir, timestamp)
        os.makedirs(folder)

        # save architecture
        model_json = self.model.to_json()
        with open(os.path.join(folder, LAYERS_FILENAME), 'w') as f:
            json.dump(model_json, f)

        # model weights
        self.model.save_weights(os.path.join(folder, MODEL_WEIGHTS_FILENAME))

        # optimizer weights
        optimizer_weights = tf.keras.backend.batch_get_value(self.model.optimizer.weights)
        with open(os.path.join(folder, OPTIMIZER_WEIGHTS_FILENAME), 'wb') as f:
            pickle.dump(optimizer_weights, f)

        # learning rate
        learning_rate = self.model.optimizer.learning_rate
        with open(os.path.join(folder, LEARNING_RATE_FILENAME), 'wb') as f:
            pickle.dump(learning_rate, f)

        # n epochs
        epoch = self._current_epoch + 1
        with open(os.path.join(folder, EPOCH_FILENAME), 'wb') as f:
            pickle.dump(epoch, f)


class CustomLoaderCallback(tf.keras.callbacks.Callback):
    """
    Callback to load necessary weight and parameters for training, evaluation and prediction
    """
    def __init__(self,
                 loading_dir: str):
        super(CustomLoaderCallback, self).__init__()

        self.loading_dir = loading_dir


    def on_train_begin(self, logs=None):
        # Load model weights
        self.model.load_weights(os.path.join(self.loading_dir, MODEL_WEIGHTS_FILENAME))

        # Load optimizer params
        with open(os.path.join(self.loading_dir, OPTIMIZER_WEIGHTS_FILENAME), 'rb') as f:
            optimizer_weights = pickle.load(f)
        with open(os.path.join(self.loading_dir, LEARNING_RATE_FILENAME), 'rb') as f:
            learning_rate = pickle.load(f)

        # Set optimizer params
        self.model.optimizer.learning_rate.assign(learning_rate)
        self.model._make_train_function()
        self.model.optimizer.set_weights(optimizer_weights)

    def on_test_begin(self, logs=None):
        # Load model weights
        self.model.load_weights(os.path.join(self.loading_dir, MODEL_WEIGHTS_FILENAME))

        # Load optimizer params
        with open(os.path.join(self.loading_dir, OPTIMIZER_WEIGHTS_FILENAME), 'rb') as f:
            optimizer_weights = pickle.load(f)
        with open(os.path.join(self.loading_dir, LEARNING_RATE_FILENAME), 'rb') as f:
            learning_rate = pickle.load(f)

        # Set optimizer params
        self.model.optimizer.learning_rate.assign(learning_rate)
        self.model._make_train_function()
        self.model.optimizer.set_weights(optimizer_weights)

    def on_predict_begin(self, logs=None):
        # Load model weights

        self.model.load_weights(os.path.join(self.loading_dir, MODEL_WEIGHTS_FILENAME))


