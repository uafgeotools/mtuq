Installation
============

We recommend installing MTUQ under Anaconda or Miniconda, which can be obtained following these `instructions <https://conda.io/projects/conda/en/latest/user-guide/install/index.html>`_, if not already present.


To install MTUQ, first create a conda virtual environment:

.. code::

   conda config --prepend channels conda-forge
   conda create -n mtuq python=3


Download the MTUQ source code:

.. code::

   git clone https://github.com/uafgeotools/mtuq.git
   cd mtuq


Install in editable mode:

.. code::

   conda activate mtuq
   conda env update --file env.yaml


Unpack seismic waveforms used by examples:

.. code::

    bash ./data/examples/unpack.bash
    bash ./data/tests/unpack.bash


Finally, install PyGMT:

.. code::

    conda install 'pygmt>=0.9'

.. note::

    PyGMT installation has been finicky at times, so we include it as a separate step after the main env.yaml-based installation. 


.. note::

    If PyGMT installation fails, most MTUQ functions will still work, but some plotting functions will fall back to GMT or ObsPy.  We note that some versions of GMT and ObsPy do not plot `full moment tensors <https://github.com/obspy/obspy/issues/2388>`_ correctly.



**Alternative method**

For installation under `virtualenv` instead, see `here <https://uafgeotools.github.io/mtuq/install/env_virtualenv.html>`_.


