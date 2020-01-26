Quick start information
=======================

Serial example
--------------

One of the easiest ways to get started is to try out `examples/SerialGridSearch.Double.py <https://github.com/uafgeotools/mtuq/blob/master/examples/SerialGridSearch.DoubleCouple.py>`_.  This script estimates the focal mechanism of an event from `Silwal2016`, using raw seismic data included within the MTUQ repository. 

Before running the example, users must install MTUQ following `these instructions <https://uafgeotools.github.io/mtuq/install/index.html>`_.

After running the example, users can check their result against this `expected output <https://github.com/rmodrak/mtbench/blob/master/output/SilwalTape2016/figures_syngine/20090407201255351.png>`_.

.. warning::

  At the time this documentation was written, there were ObsPy graphics bugs.  `Please see here <https://uafgeotools.github.io/mtuq/install/index.html>`_ for more information.

.. note::

  The first time the example runs, Green's functions will be downloaded, which can take several minutes.  Any subsequent runs will be much faster.



Parallel examples
-----------------

All the other scripts in `examples/` use MPI parallelization.  To see if MPI executables and libraries are present in your environment, start by checking if `mpirun` is on the system path.  If necessary, ask a system administrator for assistance configuring MPI.



Modifying examples
------------------

To obtain moment tensor estimates for events other than the `Silwal2016` event, users can download waveforms using PySEP (github.com/uafgeotools/pysep) and modify the examples as follows:

- Change `path_data` to the new PySEP download directory

- Change `path_weights` to the new weight file included in the PySEP download

- Optionally, adjust the Earth model, data processing and misfit settings.  (Default choices match Silwal2016.)

- Change `origin` and `magnitude` to desired values for the new event. (Alternatively, if you which to include these variables as search parameters, create lists of origins and magnitudes at the desired spacing.)



References
----------

[Silwal2016] Vipul Silwal and Carl Tape (2016), Seismic moment tensors and
estimated uncertainties in southern Alaska, JGR, https://doi.org/10.1002/2015JB012588


