Installation
------------

``tf_crnn`` uses ``tensorflow-gpu`` package, which needs CUDA and CuDNN libraries for GPU support. Tensorflow
`GPU support page <https://www.tensorflow.org/install/gpu>`_ lists the requirements.

Using Anaconda
^^^^^^^^^^^^^^

When using Anaconda (or Miniconda), conda will install automatically the compatible versions of CUDA and CuDNN ::

    conda env create -f environment.yml


From `this page <https://www.anaconda.com/blog/developer-blog/tensorflow-in-anaconda/>`_:

    When the GPU accelerated version of TensorFlow is installed using conda, by the command
    “conda install tensorflow-gpu”, these libraries are installed automatically, with versions
    known to be compatible with the tensorflow-gpu package. Furthermore, conda installs these libraries
    into a location where they will not interfere with other instances of these libraries that may have
    been installed via another method. Regardless of using pip or conda-installed tensorflow-gpu,
    the NVIDIA driver must be installed separately.

..    Using ``pip``
    ^^^^^^^^^^^^^

    Before using ``tf_crnn`` we recommend creating a virtual environment (python 3.5).
    Then, install the dependencies using Github repository's ``setup.py`` file. ::

        pip install git+https://github.com/solivr/tf-crnn

    You will then need to install CUDA and CuDNN libraries manually.


..    Using Docker
    ^^^^^^^^^^^^
    (thanks to `PonteIneptique <https://github.com/PonteIneptique>`_)

    The ``Dockerfile`` in the root directory allows you to run the whole program as a Docker Nvidia Tensorflow GPU container.
    This is potentially helpful to deal with external dependencies like CUDA and the likes.

    You can follow installations processes here :

    - docker-ce : `Ubuntu <https://docs.docker.com/install/linux/docker-ce/ubuntu/#os-requirements>`_
    - nvidia-docker : `Ubuntu <https://nvidia.github.io/nvidia-docker/>`_

    Once this is installed, we will need to build the image of the container by doing : ::

        nvidia-docker build . --tag tf-crnn


    Our container model is now named ``tf-crnn``.
    We will be able to run it from ``nvidia-docker run -it tf-crnn:latest bash``
    which will open a bash directory exactly where you are. Although, we recommend using ::

        nvidia-docker run -it -p 8888:8888 -p 6006:6006 -v /absolute/path/to/here/config:./config -v $INPUT_DATA:/sources  tf-crnn:latest bash

    where ``$INPUT_DATA`` should be replaced by the directory where you have your training and testing data.
    This will get mounted on the ``sources`` folder. We propose to mount by default ``./config`` to the current ``./config`` directory.
    Path need to be absolute path. We also recommend to change ::

        //...
        "output_model_dir" : "/.output/"


    to ::

        //...
        "output_model_dir" : "/config/output"


    **Do not forget** to rename your training and testing file path, as well as renaming the path to their
    image by ``/sources/.../file.{png,jpg}``


    .. note:: if you are uncomfortable with bash, you can always replace bash by ``ipython3 notebook --allow-root``
        and go to your browser on ``http://localhost:8888/`` . A token will be shown in the terminal