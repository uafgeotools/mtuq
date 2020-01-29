Installation under conda
========================

Download and install `miniconda <https://docs.conda.io/en/latest/miniconda.html>`_.


Create a conda virtual environment:

.. code::

   conda config --prepend channels conda-forge
   conda create -n mtuq python=3


Install MTUQ: 

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

