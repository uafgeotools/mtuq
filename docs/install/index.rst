Installation
============

We recommend installing MTUQ under Anaconda or Miniconda, which if not already present, can be installed following these `instructions <https://conda.io/projects/conda/en/latest/user-guide/install/index.html>`_.


To install MTUQ, first create a conda virtual environment:

.. code::

   conda config --prepend channels conda-forge
   conda create -n mtuq python=3


Install MTUQ:

.. code::

   git clone https://github.com/uafgeotools/mtuq.git
   cd mtuq

   conda activate mtuq
   conda env update --file env.yml
   conda develop .


Unpack seismic waveforms used by examples:

.. code::

    bash ./data/examples/unpack.bash
    bash ./data/tests/unpack.bash


Finally, install PyGMT:

.. code::

    conda install pygmt

.. note::

    PyGMT is only used for generating figures. If `conda install pygmt` fails, MTUQ will attempt to fall back to GMT or ObsPy.  We note that some versions of GMT and ObsPy do not plot full moment tensors `correctly <https://github.com/obspy/obspy/issues/2388>`_.



Alternative method
------------------

For installation under `virtualenv` instead, see these instructions:

.. toctree::

   env_virtualenv

