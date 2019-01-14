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

    {'a': 0, 'b': 1, 'c': 2, 'd': 3. 'e': 4, 'f': 5, 'g': 6}

The lookup table / dictionary needs to be saved in a json file.

Config file (with ``sacred``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set the parameters of the experiment in ``config_template.json``. The file looks like this : ::

    {
      "training_params" : {
        "learning_rate" : 1e-3,
        "learning_decay_rate" : 0.95,
        "learning_decay_steps" : 5000,
        "save_interval" : 1e3,
        "n_epochs" : 50,
        "train_batch_size" : 128,
        "eval_batch_size" : 128
      },
      "input_shape" : [32, 304],
      "string_split_delimiter" : "|",
      "csv_delimiter" : ";",
      "data_augmentation_max_rotation" : 0.1,
      "input_data_n_parallel_calls" : 4,
      "lookup_alphabet_file" : "./data/alphabet/lookup_letters_digits_symbols.json",
      "csv_files_train" : ["./data/csv/train_sample.csv"],
      "csv_files_eval" : ["./data/csv/eval_sample.csv"],
      "output_model_dir" : "./output/"
    }


In order to use your data, you should change the parameters ``csv_files_train``, ``csv_files_eval`` and probably ``lookup_alphabet_file``.

All the configurable parameters can be found in classes ``tf_crnn.config.Params`` and ``tf_crnn.config.TrainingParams``,
which can be added to the config file if needed.

Training
^^^^^^^^

Once you have your input csv and alphabet file completed, and the parameters set in ``config_template.json``,
we will use ``sacred`` syntax to launch the training : ::

    python train.py with config_template.json

The saved model will then be exported to the folder specified in the config file (``output_model_dir``).
