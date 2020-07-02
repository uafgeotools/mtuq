Installation under `conda`
==========================

First, download and install `miniconda` following these `instructions <https://conda.io/projects/conda/en/latest/user-guide/install/index.html>`_.


Create a `conda` virtual environment:

.. code::

   conda config --prepend channels conda-forge
   conda create -n mtuq python=3


Install `MTUQ`: 

.. code::

   conda activate mtuq
   conda install numpy obspy instaseis
   git clone https://github.com/uafgeotools/mtuq.git
   cd mtuq
   pip install -e .


Unpack seismic waveforms used by examples:

.. code::

    bash ./data/examples/unpack.bash
    bash ./data/tests/unpack.bash


Finally, install `GMT`:

.. code::

    conda install gmt

