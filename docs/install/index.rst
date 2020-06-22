Installation
============

To install MTUQ, follow the instructions for conda (preferred) or one of the other environments below:

.. toctree::

   env_conda
   env_virtualenv
   env_custom



.. warning::

  To avoid problems plotting moment tensor first motions (e.g. beachballs), *we urge users to install Generic Mapping Tools (GMT) version 6.0.1 or greater*.  

  MTUQ automatically uses GMT executables if present on the system path.  Otherwise, MTUQ falls back to ObsPy, which does not attempt to plot full moment tensors correctly (see `issue <https://github.com/obspy/obspy/issues/2388>`_).  To install GMT under conda, users can simply type ``conda install gmt`` from within an active virtual environment.


