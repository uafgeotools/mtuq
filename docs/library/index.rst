=================
Library Reference
=================

.. rubric:: Data structures

=============================================================================  =============================================================================
`mtuq.Dataset <generated/mtuq.Dataset.html>`_                                  Seismic data container
`mtuq.GreenTensor <generated/mtuq.GreensTensor.html>`_                         Green's tensor object
`mtuq.GreenTensorList <generated/mtuq.GreensTensorList.html>`_                 Container for Green's tensor objects
=============================================================================  =============================================================================


.. rubric:: Station and event structures

=============================================================================  =============================================================================
`mtuq.Station <generated/mtuq.MomentTensor>`_                                  Station object (based on obspy.core.trace.Stats)
`mtuq.Origin <generated/mtuq.MomentTensor>`_                                   Event origin (baded on obspy.core.event.Origin)
`mtuq.Wavelet <generated/mtuq.MomentTensor>`_                                  Source wavelet object
`mtuq.Force <generated/mtuq.Force>`_                                           Force object
`mtuq.MomentTensor <generated/mtuq.MomentTensor>`_                             Moment tensor object
=============================================================================  =============================================================================


.. rubric:: Data acquisition functions

=============================================================================  =============================================================================
`mtuq.get_greens_tensors <generated/mtuq.get_greens_tensors>`_                 Downloads Green's tensors from syngine
`mtuq.get_waveforms <generated/mtuq.get_waveforms>`_                           Downloads seismic data from IRIS
`mtuq.open_db <generated/mtuq.open_db>`_                                       Opens Green's tensor database from disk
`mtuq.read <generated/mtuq.read.html>`_                                        Reads seismic data from disk
=============================================================================  =============================================================================



.. rubric:: CAP-style functions

=============================================================================  =============================================================================
`mtuq.process_data.cap.ProcessData <generated/mtuq.read.html>`_                Data processing function
`mtuq.misfit.cap.Misfit <generated/mtuq.read.html>`_                           Misfit function
=============================================================================  =============================================================================


.. rubric:: Grid classes and utilities

=============================================================================  =============================================================================
`mtuq.grid.Grid <generated/mtuq.read.html>`_                                   Structured grid
`mtuq.grid.UnstructuredGrid <generated/mtuq.read.html>`_                       Fully-unstructured grid
`mtuq.grid.DoubleCoupleGridRandom <generated/mtuq.read.html>`_                 Grid with randomly-drawn double couples
`mtuq.grid.DoubleCoupleGridRegular <generated/mtuq.read.html>`_                Grid with uniformly-spaced double couples
`mtuq.grid.MomentTensorGridRandom <generated/mtuq.read.html>`_                 Grid with randomly-drawn moment tensors
`mtuq.grid.MomentTensorGridRegular <generated/mtuq.read.html>`_                Grid with uniformly-spaced moment tensors
`mtuq.grid.OriginGrid <generated/mtuq.read.html>`_                             Grid with uniformly-spaced origins
`mtuq.grid.cross <generated/mtuq.read.html>`_                                  Cartesian product utility
=============================================================================  =============================================================================


.. rubric:: Database and web service clients

.. note::
    These classes provide low-level database and web service access.  Normally it is easier to use high-level functions ``mtuq.get_greens_tensors`` or ``mtuq.open_db`` instead.

=============================================================================  ============================================================================= 
`mtuq.greens.axisem_netcdf.Client <generated/mtuq.greens.axisem_netcdf>`_      AxiSEM NetCDF database client (based on Instaseis)
`mtuq.greens.fk_sac.Client <generated/mtuq.greens.fk.Client>`_                 FK database client
`mtuq.greens.syngine.Client <generated/mtuq.greens.syngine.Client>`_           Syngine web service client
=============================================================================  ============================================================================= 



