=================
Library Reference
=================

.. warning::

   MTUQ is still in under development.  This documentation is incomplete and describes planned rather than current functionality.


.. rubric:: Data and synthetics structures

=================================================================================================  =================================================================================================
`mtuq.Dataset <generated/mtuq.Dataset.html>`_                                                       Container for oberved or synthetic seismic data
`mtuq.GreenTensor <generated/mtuq.GreensTensor.html>`_                                              Green's tensor object
`mtuq.GreenTensorList <generated/mtuq.GreensTensorList.html>`_                                      Container for Green's tensor objects
=================================================================================================  =================================================================================================


.. rubric:: Station and event structures

=================================================================================================  =================================================================================================
`mtuq.Station <generated/mtuq.station.Station.html>`_                                               Station metdata object
`mtuq.Origin <generated/mtuq.event.Origin.html>`_                                                   Origin metdata object
`mtuq.Wavelet <generated/mtuq.wavelet.Wavelet.html>`_                                               Source wavelet or source-time function object
`mtuq.Force <generated/mtuq.event.Force.html>`_                                                     Force source
`mtuq.MomentTensor <generated/mtuq.event.MomentTensor.html>`_                                       Moment tensor source
=================================================================================================  =================================================================================================


.. rubric:: Data I/O functions

=================================================================================================  =================================================================================================
`mtuq.get_greens_tensors <generated/mtuq.get_greens_tensors.html>`_                                 Downloads Green's tensors from syngine
`mtuq.open_db <generated/mtuq.open_db.html>`_                                                       Opens Green's tensor database from disk
`mtuq.read <generated/mtuq.read.html>`_                                                             Reads seismic data from disk
=================================================================================================  =================================================================================================



.. rubric:: Data processing and inversion functions

=================================================================================================  =================================================================================================
`mtuq.cap.process_data.ProcessData <generated/mtuq.cap.process_data.ProcessData.html>`_             Data processing function
`mtuq.misfit.Misfit <generated/mtuq.misfit.Misfit.html>`_                                           Misfit function
`mtuq.grid_search <generated/mtuq.grid_search.mpi.grid_search.html>`_                               Grid search iterator
=================================================================================================  =================================================================================================


.. rubric:: Grid classes and utilities

=================================================================================================  =================================================================================================
`mtuq.grid.MomentTensorGridRandom <generated/mtuq.grid.MomentTensorGridRandom.html>`_               Grid with randomly-drawn moment tensors
`mtuq.grid.MomentTensorGridRegular <generated/mtuq.grid.MomentTensorGridRegular.html>`_             Grid with uniformly-spaced moment tensors
`mtuq.grid.DoubleCoupleGridRandom <generated/mtuq.grid.DoubleCoupleGridRandom.html>`_               Grid with randomly-drawn double couples
`mtuq.grid.DoubleCoupleGridRegular <generated/mtuq.grid.DoubleCouleGridRegular.html>`_              Grid with uniformly-spaced double couples
=================================================================================================  =================================================================================================


.. rubric:: Database and web service clients

.. note::
    These classes provide low-level database and web service access.  Normally it is easier to use high-level functions ``mtuq.get_greens_tensors`` or ``mtuq.open_db`` instead.

=================================================================================================  ================================================================================================= 
`mtuq.io.clients.axisem_netcdf.Client <generated/mtuq.io.clients.axisem_netcdf.html>`_              AxiSEM NetCDF database client (based on instaseis)
`mtuq.io.clients.fk_sac.Client <generated/mtuq.io.clients.fk_sac.Client.html>`_                     FK database client
`mtuq.io.clients.syngine.Client <generated/mtuq.io.clients.syngine.Client.html>`_                   Syngine web service client
=================================================================================================  ================================================================================================= 



.. rubric::  Low-level misfit evalutation functions

.. note::
    Misfit evaluation is the most computationally expensive task performed by this package, and the only part that has been heavily optimized.  These classes provide low-level access to optimized and non-optimized versions of the misfit function  (the former exists to check correctness of the latter).  Normally users can avoid this complexity by simply calling ``mtuq.misfit.Misft``.

=================================================================================================  =================================================================================================
`mtuq.misfit.O1.misft <generated/mtuq.misfit.O1.misfit.html>`_                                      Easy-to-read pure Python implementation
`mtuq.greens.fk_sac.Client <generated/mtuq.greens.fk.Client.html>`_                                 Fast pure Python implementation
`mtuq.greens.syngine.Client <generated/mtuq.greens.syngine.Client.html>`_                           Very fast Python/C implementation
=================================================================================================  =================================================================================================

