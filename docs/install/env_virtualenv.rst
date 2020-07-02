Installation under virtualenv
=============================

Install the `virtualenv <https://virtualenv.pypa.io/en/latest/installation.html>`_ program, if not already present.

Create a new virtual environment:

.. code::

   virtualenv --python=python3 mtuq_env


Download `MTUQ` source code:

.. code::

   git clone https://github.com/uafgeotools/mtuq.git mtuq_src


Install `MTUQ`:

.. code::

   source mtuq_env/bin/activate
   cd mtuq_src
   pip install numpy
   pip install obspy instaseis
   pip install -e .


Unpack seismic waveforms used by examples:

.. code::

    bash ./data/examples/unpack.bash
    bash ./data/tests/unpack.bash


Finally, install `GMT`:

.. code::

    conda install gmt

