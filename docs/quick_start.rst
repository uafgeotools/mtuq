Quick start
===========

An easy way to get started is to try out any of these `examples <https://github.com/uafgeotools/mtuq/blob/master/examples>`_.

Before running examples, users must first `install MTUQ <https://uafgeotools.github.io/mtuq/install/index.html>`_.

After installation finishes, examples can be invoked from within the `examples/` directory by

.. code::

   python ./name_of_example


.. note::

  The first time an example runs, Green's functions will be downloaded, which can take several minutes.  Any subsequent runs will be much faster.


Serial example
--------------

The simplest example is `SerialGridSearch.DoubleCouple.py <https://github.com/uafgeotools/mtuq/blob/master/examples/SerialGridSearch.DoubleCouple.py>`_, which estimates the focal mechanism of an event from `Silwal2016` using seismic waveforms included with the `MTUQ` repository. 

Users can check their output against this `expected result <https://github.com/rmodrak/mtbench/blob/master/output/SilwalTape2016/figures_syngine/20090407201255351.png>`_.

Because the  example lacks MPI directives, it is more readable more than the other examples.  A typical runtime is about 60 seconds.  For faster results, `GridSearch.DoubleCouple.py <https://github.com/uafgeotools/mtuq/blob/master/examples/GridSearch.DoubleCouple.py>`_ runs the same inversion in parallel.


Parallel examples
-----------------

All the other examples contain MPI directives for faster evaluation on parallel systems.

To see if MPI executables and libraries are present in your environment, start by checking if `mpirun` is on the system path.  If necessary, ask a system administrator for assistance installing or configuring MPI.

If MPI is present, run the parallel examples as follows:

.. code::

   mpirun -n <NPROC> python ./name_of_example


If MPI is not present, users can still run any example by invoking it directly, but any parallelization speedup will be lost.




References
----------

[Silwal2016] Vipul Silwal and Carl Tape (2016), Seismic moment tensors and
estimated uncertainties in southern Alaska, JGR, https://doi.org/10.1002/2015JB012588


