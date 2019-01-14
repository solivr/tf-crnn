r"""

The :mod:`tf_crnn.data_handler`

Data handling for input function
--------------------------------
.. autosummary::
    data_loader
    padding_inputs_width
    augment_data
    random_rotation
    random_padding
    serving_single_input


Config for training
-------------------

.. autosummary::
    Alphabet
    TrainingParams
    Params
    import_params_from_json


Model
-----

.. autosummary::
    deep_cnn
    deep_bidirectional_lstm
    crnn_fn
    get_words_from_chars

Loading exported model
----------------------

.. autosummary::
    PredictionModel

----

"""

_DATA_HANDLING = [
    'data_loader',
    'padding_inputs_width',
    'augment_data',
    'random_rotation',
    'random_padding',
    'serving_single_input',
    'serving_batch_filenames_fn'
]

_CONFIG = [
    'Alphabet',
    'TrainingParams',
    'Params',
    'import_params_from_json'

]

_MODEL = [
    'deep_cnn',
    'deep_bidirectional_lstm',
    'crnn_fn',
    'get_words_from_chars'
]

_LOADER = [
    'PredictionModel',
    'PredictionModelBatch'
]

__all__ = _DATA_HANDLING + _CONFIG + _MODEL + _LOADER

from tf_crnn.data_handler import *
from tf_crnn.config import *
from tf_crnn.model import *
from tf_crnn.decoding import *
from tf_crnn.loader import *