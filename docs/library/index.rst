=================
Library Reference
=================

.. warning::

   MTUQ is still in under development.  This documentation is incomplete and describes planned rather than current functionality.


.. rubric:: Data structures

=================================================================================================  =================================================================================================
`mtuq.Dataset <generated/mtuq.Dataset.html>`_                                                       Seismic data container
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


.. rubric:: Data acquisition functions

=================================================================================================  =================================================================================================
`mtuq.get_greens_tensors <generated/mtuq.get_greens_tensors.html>`_                                 Downloads Green's tensors from syngine
`mtuq.get_waveforms <generated/mtuq.get_waveforms.html>`_                                           Downloads seismic data from IRIS
`mtuq.open_db <generated/mtuq.open_db.html>`_                                                       Opens Green's tensor database from disk
`mtuq.read <generated/mtuq.read.html>`_                                                             Reads seismic data from disk
=================================================================================================  =================================================================================================



.. rubric:: CAP-style functions

=================================================================================================  =================================================================================================
`mtuq.cap.process_data.ProcessData <generated/mtuq.cap.process_data.ProcessData.html>`_             Data processing function
`mtuq.cap.misfit.Misfit <generated/mtuq.cap.misfit.Misfit.html>`_                                   Misfit function
=================================================================================================  =================================================================================================


.. rubric:: Grid classes and utilities

=================================================================================================  =================================================================================================
`mtuq.grid.Grid <generated/mtuq.grid.Grid.html>`_                                                   Structured grid
`mtuq.grid.UnstructuredGrid <generated/mtuq.grid.UnstructuredGrid.html>`_                           Fully-unstructured grid
`mtuq.grid.DoubleCoupleGridRandom <generated/mtuq.grid.DoubleCoupleGridRandom.html>`_               Grid with randomly-drawn double couples
`mtuq.grid.DoubleCoupleGridRegular <generated/mtuq.grid.DoubleCouleGridRegular.html>`_              Grid with uniformly-spaced double couples
`mtuq.grid.MomentTensorGridRandom <generated/mtuq.grid.MomentTensorGridRandom.html>`_               Grid with randomly-drawn moment tensors
`mtuq.grid.MomentTensorGridRegular <generated/mtuq.grid.MomentTensorGridRegular.html>`_             Grid with uniformly-spaced moment tensors
`mtuq.grid.OriginGrid <generated/mtuq.grid.OriginGrid.html>`_                                       Grid with uniformly-spaced origins
`mtuq.grid.cross <generated/mtuq.grid.cross.html>`_                                                 Cartesian product utility
=================================================================================================  =================================================================================================


.. rubric:: Database and web service clients

.. note::
    These classes provide low-level database and web service access.  Normally it is easier to use high-level functions ``mtuq.get_greens_tensors`` or ``mtuq.open_db`` instead.

=================================================================================================  ================================================================================================= 
`mtuq.greens.axisem_netcdf.Client <generated/mtuq.greens.axisem_netcdf.html>`_                      AxiSEM NetCDF database client (based on Instaseis)
`mtuq.greens.fk_sac.Client <generated/mtuq.greens.fk.Client.html>`_                                 FK database client
`mtuq.greens.syngine.Client <generated/mtuq.greens.syngine.Client.html>`_                           Syngine web service client
=================================================================================================  ================================================================================================= 


