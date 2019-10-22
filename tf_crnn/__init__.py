r"""


Data handling for input function
--------------------------------
.. currentmodule:: tf_crnn.data_handler

.. autosummary::
    dataset_generator
    padding_inputs_width
    augment_data
    random_rotation


Model definitions
-----------------
.. currentmodule:: tf_crnn.model

.. autosummary::
    ConvBlock
    get_model_train
    get_model_inference
    get_crnn_output


Config for training
-------------------
.. currentmodule:: tf_crnn.config

.. autosummary::
    Alphabet
    Params
    import_params_from_json


Custom Callbacks
----------------
.. currentmodule:: tf_crnn.callbacks

.. autosummary::
    CustomSavingCallback
    LRTensorBoard
    CustomLoaderCallback
    CustomPredictionSaverCallback


Preprocessing data
------------------
.. currentmodule:: tf_crnn.preprocessing

.. autosummary::
    data_preprocessing
    preprocess_csv


----

"""

_DATA_HANDLING = [
    'dataset_generator',
    'padding_inputs_width',
    'augment_data',
    'random_rotation'
]

_CONFIG = [
    'Alphabet',
    'Params',
    'import_params_from_json'

]

_MODEL = [
    'ConvBlock',
    'get_model_train',
    'get_model_inference'
    'get_crnn_output'
]

_CALLBACKS = [
    'CustomSavingCallback',
    'CustomLoaderCallback',
    'CustomPredictionSaverCallback',
    'LRTensorBoard'
]

_PREPROCESSING = [
    'data_preprocessing',
    'preprocess_csv'
]

__all__ = _DATA_HANDLING + _CONFIG + _MODEL + _CALLBACKS + _PREPROCESSING

from tf_crnn.config import *
from tf_crnn.model import *
from tf_crnn.callbacks import *
from tf_crnn.preprocessing import *
from tf_crnn.data_handler import *