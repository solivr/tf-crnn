Using a saved model for prediction
----------------------------------

During the training, the model is exported every *n* epochs (you can set *n* in the config file, by default *n=5*).
The exported models are ``SavedModel`` TensorFlow objects, which need to be loaded in order to be used.

Assuming that the output folder is named ``output_dir``, the exported models will be saved in ``output_dir/export/<timestamp>``
with different timestamps for each export. Each ``<timestamp>`` folder contains a ``saved_model.pb``
file and a ``variables`` folder.

The ``saved_model.pb`` contains the graph definition of your model and the ``variables`` folder contains the
saved variables (where the weights are stored). You can find more information about SavedModel
on the `TensorFlow dedicated page <https://www.tensorflow.org/guide/saved_model>`_.


In order to easily handle the loading of the exported models, a ``PredictionModel`` class is provided and
you can use the trained model to transcribe new image segments in the following way :

.. code-block:: python

    import tensorflow as tf
    from tf_crnn.loader import PredictionModel

    model_directory = 'output/export/<timestamp>/'
    image_filename = 'data/images/b04-034-04-04.png'

    with tf.Session() as session:
        model = PredictionModel(model_directory, signature='filename')
        prediction = model.predict(image_filename)

