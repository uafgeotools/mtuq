Installation
============

If you are familiar with virtual environment management in Python, then feel free to install MTUQ as follows within an environment of your choice::

   git clone https://github.com/uafseismo/mtuq.git
   cd mtuq
   pip install -e .


Otherwise, follow the instructions below for installation under `conda` (preferred) or `virtualenv`:

.. toctree::

   env_conda
   env_virtualenv



Graphics dependencies
=====================

.. warning::

   To avoid graphics problems, we recommend users install `GMT` (>`v6.0.0`). 

MTUQ uses `GMT <https://www.soest.hawaii.edu/gmt>` graphics utilities if present on the system path.  Otherwise, MTUQ falls back to ObsPy, which does not attempt to plot full moment tensors correctly (see `issue <https://github.com/obspy/obspy/issues/2388>`_).  For information about installing `GMT`, see the `conda` or `virtualenv` links above.

