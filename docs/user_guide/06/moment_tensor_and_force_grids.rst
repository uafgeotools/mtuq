
Moment tensor and force grids
=============================


Currently implemented moment tensor grids
-----------------------------------------

============================================================================================================  ============================================================================================================
`mtuq.grid.FullMomentTensorGridRandom <generated/mtuq.grid.FullMomentTensorGridRandom.html>`_                  Grid with randomly-drawn full moment tensors
`mtuq.grid.FullMomentTensorGridSemiregular <generated/mtuq.grid.FullMomentTensorGridSemiregular.html>`_        Grid with regularly-spaced full moment tensors
`mtuq.grid.DoubleCoupleGridRandom <generated/mtuq.grid.DoubleCoupleGridRandom.html>`_                          Grid with randomly-drawn double couples
`mtuq.grid.DoubleCoupleGridRegular <generated/mtuq.grid.DoubleCoupleGridRegular.html>`_                        Grid with regularly-spaced double couples
`mtuq.grid.DeviatoricGridRandom <generated/mtuq.grid.DeviatoricGridRandom.html>`_                              Grid with randomly-drawn deviatoric moment tensors
`mtuq.grid.DeviatoricGridSeimiregular <generated/mtuq.grid.DeviatoricGridSemiregular.html>`_                   Grid with regularly-spaced deviatoric moment tensors
============================================================================================================  ============================================================================================================


Currently implemented force grids
---------------------------------

============================================================================================================  ============================================================================================================
`mtuq.grid.ForceGridRandom <generated/mtuq.grid.ForceGridRandom.html>`_                                        Grid with randomly-drawn forces
`mtuq.grid.ForceGridRegular <generated/mtuq.grid.ForceGridRegular.html>`_                                      Grid with regularly-spaced forces
============================================================================================================  ============================================================================================================


Using moment tensor grids
-------------------------

Moment tensor grids created in the above manner can be iterated over like a NumPy array or list, or individual grid points can be accessed by the following methods:

- Use ``get(i)`` to return the `i`-th moment tensor as a `MomentTensor` object
- Use ``get(i).as_vector()`` to return the `i`-th moment tensor as a vector `Mrr, Mtt, Mpp, Mrp, Mrt, Mtp`
- Use ``get_dict(i)`` to return the `i`-th moment tensor as dictionary of Tape2015 parameters `rho, v, w, kappa, sigma, h`


Using force grids
-----------------

Force grids created in the above manner can be iterated over like a NumPy array or list, or individual grid points can be accessed by the following methods:

- Use ``get(i)`` to return the `i`-th force as a `Force` object
- Use ``get(i).as_vector()`` to return the `i`-th force as a vector `Fr, Ft, Fp`
- Use ``get_dict(i)`` to return the `i`-th force as dictionary of parameters `F0, phi, h`


More details
------------

For additional usage information, see the `Grid` or `UnstructuredGrid` documentation.
