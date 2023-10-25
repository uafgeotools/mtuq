
Installation on ARM64 systems
=============================

To install MTUQ on ARM64 systems, create a conda virtual environment:

.. code::

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

    Users may receive the following Instaseis warning message:

    `UserWarning: Please don't install from a tarball. Use the proper pypi release or install from git.`

    Now that Instaseis PyPI releases are no longer maintained, the above message can be ignored.


.. note::

    If `linux-aarch64` or `osx-arm64` builds are ever added to the Instaseis `conda-forge channel <https://anaconda.org/conda-forge/instaseis>`_, the above workaround may become unnecessary.

