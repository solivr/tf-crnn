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
      "csv_files_train" : "./data/csv_experiments/train_data.csv",
      "csv_files_eval" : "./data/csv_experiments/validation_data.csv",
      "output_model_dir" : "./output_model",
      "num_beam_paths" : 1,
      "max_chars_per_string" : 80,
      "n_epochs" : 50,
      "train_batch_size" : 64,
      "eval_batch_size" : 64,
      "learning_rate": 1e-4,
      "input_shape" : [128, 1400],
      "restore_model" : false
    }

In order to use your data, you should change the parameters ``csv_files_train``, ``csv_files_eval`` and ``lookup_alphabet_file``.

All the configurable parameters can be found in class ``tf_crnn.config.Params``, which can be added to the config file if needed.

Training
^^^^^^^^

Once you have your input csv and alphabet file completed, and the parameters set in ``config.json``,
we will use ``sacred`` syntax to launch the training : ::

    python training.py with config.json

The saved model and logs will then be exported to the folder specified in the config file (``output_model_dir``).


Example of training
-------------------

We will use the `IAM Database <http://www.fki.inf.unibe.ch/databases/iam-handwriting-database>`_ :cite:`marti2002iam`
as an example to generate the data in the correct input data and train a model.


Generating data
^^^^^^^^^^^^^^^

Run the script ``hlp/prepare_iam.py`` in order to download the data, extract it and format it correctly to train a model. ::

    cd hlp
    python prepare_iam.py --download_dir ../data/iam --generated_data_dir ../data/iam/generated
    cd ..

The images of the lines are extracted in ``data/iam/lines/`` and the folder ``data/generated/`` contains all the
additional files necessary to run the experiment. The csv files are saved in ``data/generated/generated_csv`` and
the alphabet is placed in ``data/generated/generated_alphabet``.

Training the model
^^^^^^^^^^^^^^^^^^

Make sure the ``config.json`` file has the correct paths for training and validation data, as well as for the
alphabet lookup file and run: ::

    python training.py with config.json

