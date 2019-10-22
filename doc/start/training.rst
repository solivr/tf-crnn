How to train a model
--------------------

``sacred`` package is used to deal with experiments.
If you are not yet familiar with it, have a quick look at the `documentation <https://sacred.readthedocs.io/en/latest/>`_.

Input data
^^^^^^^^^^

In order to train a model, you should input a csv file with each row containing the filename of the image (full path)
and its label (plain text) separated by a delimiting character (let's say ``;``).
Also, each character should be separated by a splitting character (let's say ``|``), this in order to deal with arbitrary
alphabets (especially characters that cannot be encoded with ``utf-8`` format).

An example of such csv file would look like : ::

    /full/path/to/image1.{jpg,png};|s|t|r|i|n|g|_|l|a|b|e|l|1|
    /full/path/to/image2.{jpg,png};|s|t|r|i|n|g|_|l|a|b|e|l|2| |w|i|t|h| |special_char|
    ...

Input lookup alphabet file
^^^^^^^^^^^^^^^^^^^^^^^^^^

You also need to provide a lookup table for the *alphabet* that will be used. The term *alphabet* refers to all the
symbols you want the network to learn, whether they are characters, digits, symbols, abbreviations, or any other graphical element.

The lookup table is a dictionary mapping alphabet units to integer codes (i.e {'char' : <int_code>}).
Some lookup tables are already provided as examples in ``data/alphabet/``.

For example to transcribe words that contain only the characters *'abcdefg'*, one possible lookup table would be : ::

    {'a': 1, 'b': 2, 'c': 3, 'd': 4. 'e': 5, 'f': 6, 'g': 7}

The lookup table / dictionary needs to be saved in a json file.

Config file (with ``sacred``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set the parameters of the experiment in ``config.json``. The file looks like this : ::

    {
      "lookup_alphabet_file" : "./data/alphabet/lookup.json",
      "csv_files_train" : "./data/csv_experiments/lines_train_tf_format.csv",
      "csv_files_eval" : "./data/csv_experiments/lines_validation1_tf_format.csv",
      "output_model_dir" : "./output_model",
      "num_beam_paths" : 1,
      "cnn_batch_norm" : [true, true, true, true, true],
      "max_chars_per_string" : 80,
      "n_epochs" : 50,
      "train_batch_size" : 128,
      "eval_batch_size" : 128,
      "learning_rate": 1e-4,
      "input_shape" : [128, 1400],
      "rnn_units" : [256, 256, 256],
      "restore_model" : false
    }

In order to use your data, you should change the parameters ``csv_files_train``, ``csv_files_eval`` and probably ``lookup_alphabet_file``.

All the configurable parameters can be found in class ``tf_crnn.config.Params``, which can be added to the config file if needed.

Training
^^^^^^^^

Once you have your input csv and alphabet file completed, and the parameters set in ``config.json``,
we will use ``sacred`` syntax to launch the training : ::

    python train.py with config.json

The saved model will then be exported to the folder specified in the config file (``output_model_dir``).
