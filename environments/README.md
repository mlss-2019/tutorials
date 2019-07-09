README
======

This directory contains conda environments for running the tutorials.
It\'s meant as an aid to all the participants and to avoid technical
errors. To use this, please install the
[anaconda](https://docs.anaconda.com/anaconda/install/) package manager,
or alternatively the more minimal
[miniconda](https://docs.conda.io/en/latest/miniconda.html)
distribution. I would recommend the `python3` version since `python2` is
deprecated, but you can install environments for `pytho3` or `python2`
with either.

Note that the environments that has tensorflow installed only runs on
the CPU. If you have access to a GPU, you will have to set this up
yourself.

Loading an environment
----------------------

### Linux / OSX

To create an environment, `cd` into the directory
`$TUTORIAL_DIR/environments` and from the command line, run
`conda env create -f $ENV_NAME.yml`. To activate the environment run
`conda activate $ENV_NAME`.

### Windows

Jupyter notebooks
-----------------

In order to run jupyter notebooks, `cd` into the directory of the
notebooks and from the command line run `jupyter notebook`. You can then
change the kernel to your conda environment by clicking on the notebook
you want to run and then go to the tab **Kernel** and choose **Change
kernel**, you will be presented with the available kernels on your
machine.

Tutorials notes
---------------

### GP

1.  `Lab1 - Sampling a GP.ipynb`

    If you find that the first cell

    ``` {.python}
    import numpy as np
    import matplotlib
    matplotlib.rcParams['figure.figsize'] = (14, 6)
    plt = matplotlib.pyplot
    ```

    doesn\'t work, change this cell to

    ``` {.python}
    import numpy as np
    import matplotlib
    matplotlib.rcParams['figure.figsize'] = (14, 6)
    import matplotlib.pyplot as plt
    ```

2.  `Lab2 - GPflow and GP Regression.ipynb`

    See the above.

### RL

For the `minerl` to be installed properly you will need to set your
system java to `jdk8`. which will depend on your OS.

### MCMC

### VI

Best to run this in notebook. Upload the jupyter notebook
`$TUTORIAL_DIR/MLSS2019_VAEs_exercise.ipynb` to [Google
Colab](https://colab.research.google.com/) and run it there. There is an
environment which *should work* but it\'s on CPU and will be
excruciatingly slow compared to using a GPU.
