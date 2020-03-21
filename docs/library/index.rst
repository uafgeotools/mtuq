Library reference
=================

.. warning::

   MTUQ is still in under development.  Some documentation may be incomplete or describe planned rather than current functionality.


.. rubric:: Seismic data structures

=================================================================================================  =================================================================================================
`mtuq.Dataset <generated/mtuq.Dataset.html>`_                                                       Container for observed or synthetic seismic data
`mtuq.GreenTensor <generated/mtuq.GreensTensor.html>`_                                              Green's tensor object
`mtuq.GreenTensorList <generated/mtuq.GreensTensorList.html>`_                                      Container for Green's tensor objects
=================================================================================================  =================================================================================================


.. rubric:: Station and event structures

=================================================================================================  =================================================================================================
`mtuq.Station <generated/mtuq.Station.html>`_                                                       Station metdata object
`mtuq.Origin <generated/mtuq.Origin.html>`_                                                         Origin metdata object
`mtuq.Wavelet <generated/mtuq.wavelet.Wavelet.html>`_                                               Source wavelet or source-time function object
`mtuq.Force <generated/mtuq.Force.html>`_                                                           Force source
`mtuq.MomentTensor <generated/mtuq.MomentTensor.html>`_                                             Moment tensor source
=================================================================================================  =================================================================================================


.. rubric:: Data I/O functions

=================================================================================================  =================================================================================================
`mtuq.download_greens_tensors <generated/mtuq.download_greens_tensors.html>`_                       Downloads Green's tensors from syngine
`mtuq.open_db <generated/mtuq.open_db.html>`_                                                       Opens Green's tensor database from disk
`mtuq.read <generated/mtuq.read.html>`_                                                             Reads seismic data from disk
=================================================================================================  =================================================================================================



.. rubric:: Data processing and inversion functions

=================================================================================================  =================================================================================================
`mtuq.process_data.ProcessData <generated/mtuq.process_data.ProcessData.html>`_                     Data processing function
`mtuq.misfit.Misfit <generated/mtuq.misfit.Misfit.html>`_                                           Misfit function
`mtuq.grid_search.grid_search <generated/mtuq.grid_search.grid_search.html>`_                       Grid search wrapper
=================================================================================================  =================================================================================================


.. rubric:: Data visualization

=================================================================================================  =================================================================================================
mtuq.graphics.plot_data_synthetics                                                                 Compares data and synthetics
mtuq.graphics.plot_misfit_vs_depth                                                                 Plots misfit vs depth
mtuq.graphics.plot_beachball                                                                       Plots focal mechansim (requires GMT)
=================================================================================================  =================================================================================================


.. rubric:: Uncertainty visualization

=================================================================================================  =================================================================================================
mtuq.graphics.plot_likelihood                                                                      Plots maximum likelihoods on lune (requires GMT)
mtuq.graphics.plot_marginal                                                                        Plots marginal likelihoods on lune (requires GMT)
mtuq.graphics.plot_misfit                                                                          Plots misfit on lune (requires GMT)
mtuq.graphics.plot_likelihood_vw                                                                   Plots maximum likelihoods on `vw` rectangle
mtuq.graphics.plot_marginal_vw                                                                     Plots marginal likelihoods on `vw` rectangle
mtuq.graphics.plot_misfit_vw                                                                       Plots misfit on `vw` rectangle
mtuq.graphics.plot_likelihood_dc                                                                   Plots maximum likelihoods over strike, dip, and slip
mtuq.graphics.plot_marginal_dc                                                                     Plots marginal likelihoods over strike, dip, and slip
mtuq.graphics.plot_misfit_dc                                                                       Plots misfit over strike, dip, and slip

=================================================================================================  =================================================================================================



.. rubric:: Moment tensor grids

=================================================================================================  =================================================================================================
`mtuq.grid.FullMomentTensorGridRandom <generated/mtuq.grid.FullMomentTensorGridRandom.html>`_       Grid with randomly-drawn full moment tensors
`mtuq.grid.FullMomentTensorGridRegular <generated/mtuq.grid.FullMomentTensorGridRegular.html>`_     Grid with uniformly-spaced full moment tensors
`mtuq.grid.DoubleCoupleGridRandom <generated/mtuq.grid.DoubleCoupleGridRandom.html>`_               Grid with randomly-drawn double couples
`mtuq.grid.DoubleCoupleGridRegular <generated/mtuq.grid.DoubleCoupleGridRegular.html>`_              Grid with uniformly-spaced double couples
=================================================================================================  =================================================================================================


.. rubric:: Grid base classes
.. note::
   These base classes can be used for specialized uses cases not covered by any of the above grids

=================================================================================================  =================================================================================================
`mtuq.grid.Grid <generated/mtuq.grid.Grid.html>`_                                                   Grid defined by values on axes
`mtuq.grid.UnstructuredGrid <generated/mtuq.grid.UnstructuredGrid.html>`_                           Grid defined by lists of coordinates points
=================================================================================================  =================================================================================================



.. rubric:: Database and web service clients

.. note::
    These classes provide low-level database and web service access.  Normally it is easier to use high-level functions ``mtuq.download_greens_tensors`` or ``mtuq.open_db`` instead.

=================================================================================================  ================================================================================================= 
`mtuq.io.clients.AxiSEM_NetCDF.Client <generated/mtuq.io.clients.AxiSEM_NetCDF.Client.html>`_       AxiSEM NetCDF database client (based on instaseis)
`mtuq.io.clients.FK_SAC.Client <generated/mtuq.io.clients.FK_SAC.Client.html>`_                     FK database client
`mtuq.io.clients.syngine.Client <generated/mtuq.io.clients.syngine.Client.html>`_                   Syngine web service client
=================================================================================================  ================================================================================================= 



.. rubric::  Low-level misfit modules

.. note::
    Misfit evaluation is the most computationally intensive task performed by this package, and the only task that has been heavily optimized.  These classes provide low-level access to non-optimized and optimized versions of the misfit function  (the former exists to check correctness of the latter).  Normally users can avoid this complexity by simply calling ``mtuq.misfit.Misfit``.

=================================================================================================  =================================================================================================
`mtuq.misfit.level0.misfit <generated/mtuq.misfit.level0.html>`_                                    Easy-to-read pure Python implementation
`mtuq.misfit.level1.misfit <generated/mtuq.misfit.level1.html>`_                                      Fast pure Python implementation
`mtuq.misfit.level2.misfit <generated/mtuq.misfit.level2.html>`_                                      Fast Python/C implementation
=================================================================================================  =================================================================================================

