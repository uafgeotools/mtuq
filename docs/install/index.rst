Installation instructions
=========================

To install MTUQ, follow the instructions for conda (preferred) or one of the other environments below:

.. toctree::

   env_conda
   env_virtualenv
   env_custom



.. warning::

  To avoid problems plotting beachballs, *we urge users to install GMT version 6.0.1 or greater*.  

  MTUQ automatically uses GMT executables if present on the system path.  Otherwise, MTUQ falls back to ObsPy, which has `known bugs <https://github.com/obspy/obspy/issues/2388>`_.  To install GMT under conda, users can simply type ``conda install gmt`` from within an active virtual environment.

  Beachballs, like the one shown in the upper-left corner of the `output <https://github.com/rmodrak/mtbench/blob/master/output/SilwalTape2016/figures_syngine/20090407201255351.png>`_ from `this example <https://github.com/uafgeotools/mtuq/blob/master/examples/SerialGridSearch.DoubleCouple.py>`_, are commonly used to visualize seismic moment tensors.   Plotting beachballs correctly in all cases is actually very difficult.   Many seismology packages, including ObsPy, plot beachballs incorrectly.

