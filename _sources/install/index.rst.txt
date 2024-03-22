Installation
============

We recommend installing MTUQ under Miniconda, which can be obtained following `these instructions <https://conda.io/projects/conda/en/latest/user-guide/install/index.html>`_.


To install MTUQ, create a conda virtual environment:

.. code::

   conda create -n mtuq


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

    conda install 'pygmt=0.9'

.. note::

    PyGMT installation has been finicky at times, so we include it as a separate step after the main env.yaml-based installation. Most MTUQ functions will continue to work without PyGMT. (`See here <https://uafgeotools.github.io/mtuq/install/issues.html#pygmt-installation>`_ for more information.)


**Troubleshooting**

For troubleshooting common installation issues, please `see here <https://uafgeotools.github.io/mtuq/install/issues.html>`_.


