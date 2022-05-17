Installation
============

If you're familiar with Python virtual environments, feel free to install MTUQ as follows within an environment of your choice::

   git clone https://github.com/uafgeotools/mtuq.git
   cd mtuq
   pip install -e .


Otherwise, follow these instructions for installation under `conda` (preferred) or `virtualenv`:

.. toctree::

   env_conda
   env_virtualenv



Graphics dependencies
---------------------

.. warning::

   To avoid graphics problems, we recommend installing PyGMT.

MTUQ uses `PyGMT <https://www.pygmt.org/latest/>`_ if present.  PyGMT is not installed by default, but requires an extra `pip install pygmt` or `conda install pygmt` command.  

If PyGMT is not present, MTUQ may fall back to GMT or ObsPy in some cases.  We note that some versions of GMT and ObsPy do not plot full moment tensors correctly (`see here <https://github.com/obspy/obspy/issues/2388>`_).

