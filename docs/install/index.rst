Installation
============

To install MTUQ, follow the instructions for conda (preferred) or one of the other environments below:

.. toctree::

   env_conda
   env_virtualenv
   env_custom



Important note on beachballs
----------------------------

`Beachballs`, like the one shown in the `output <https://github.com/rmodrak/mtbench/blob/master/output/SilwalTape2016/figures_syngine/20090407201255351.png>`_ from `this example <https::/uafgeotools.github.io/mtuq/quick_start.html#serial-example>`_, are commonly used to visualize seismic moment tensors.   In all but simple cases, plotting beachballs is actually difficult.   Many seismology packages, including ObsPy, plot beachballs incorrectly.

To avoid beachball problems, *we urge users to install GMT version 6.0.1 or greater*.  MTUQ automatically uses GMT executables if present on the system.  Otherwise, MTUQ falls back to ObsPy.  To install GMT under conda, users can simply type ``conda install gmt`` from within the MTUQ virtual environment.
