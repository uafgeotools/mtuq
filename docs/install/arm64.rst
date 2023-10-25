
Installation on ARM64 systems, including Apple M1 and Apple M2
==============================================================

First create a conda virtual environment:

.. code::

   conda config --prepend channels conda-forge
   conda create -n mtuq python=3


Download the MTUQ source code:

.. code::

   git clone https://github.com/uafgeotools/mtuq.git
   cd mtuq


Install MTUQ in editable mode:

.. code::

   conda activate mtuq
   conda env update --file env_arm64.yaml


Install Instaseis:


.. code::

    git clone https://github.com/sstaehler/instaseis.git
    cd instaseis
    pip install -e .

Unpack seismic waveforms used by examples:

.. code::

    bash ./data/examples/unpack.bash
    bash ./data/tests/unpack.bash


Finally, install PyGMT:

.. code::

    conda install 'pygmt>=0.9'


.. note::

    Once linux-aarch64 and osx-arm64 builds are contributed to the Instaseis `conda-forge <https://anaconda.org/conda-forge/instaseis>`_ channel, the above workaround should be unnecessary.
