Installation
============

We recommend installing MTUQ under Anaconda or Miniconda, which, if not already present, can be obtained following these `instructions <https://conda.io/projects/conda/en/latest/user-guide/install/index.html>`_.


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


**Troubleshooting**

For a discussion of common issues, please see our `troubleshooting page <https://uafgeotools.github.io/mtuq/install/issues.html>`_.


