
Moment tensor and force grids
=============================


Currently implemented moment tensor grids
-----------------------------------------

====================================================================================================================================================  ====================================================================================================================================================
`mtuq.grid.FullMomentTensorGridRandom <https://uafgeotools.github.io/mtuq/library/generated/mtuq.grid.FullMomentTensorGridRandom.html>`_                  Grid with randomly-drawn full moment tensors
`mtuq.grid.FullMomentTensorGridSemiregular <https://uafgeotools.github.io/mtuq/library/generated/mtuq.grid.FullMomentTensorGridSemiregular.html>`_        Grid with regularly-spaced full moment tensors
`mtuq.grid.DoubleCoupleGridRandom <https://uafgeotools.github.io/mtuq/library/generated/mtuq.grid.DoubleCoupleGridRandom.html>`_                          Grid with randomly-drawn double couples
`mtuq.grid.DoubleCoupleGridRegular <https://uafgeotools.github.io/mtuq/library/generated/mtuq.grid.DoubleCoupleGridRegular.html>`_                        Grid with regularly-spaced double couples
`mtuq.grid.DeviatoricGridRandom <https://uafgeotools.github.io/mtuq/library/generated/mtuq.grid.DeviatoricGridRandom.html>`_                              Grid with randomly-drawn deviatoric moment tensors
`mtuq.grid.DeviatoricGridSeimiregular <https://uafgeotools.github.io/mtuq/library/generated/mtuq.grid.DeviatoricGridSemiregular.html>`_                   Grid with regularly-spaced deviatoric moment tensors
====================================================================================================================================================  ====================================================================================================================================================


Currently implemented force grids
---------------------------------

====================================================================================================================================================  ====================================================================================================================================================
`mtuq.grid.ForceGridRandom <https://uafgeotools.github.io/mtuq/library/generated/mtuq.grid.ForceGridRandom.html>`_                                        Grid with randomly-drawn forces
`mtuq.grid.ForceGridRegular <https://uafgeotools.github.io/mtuq/library/generated/mtuq.grid.ForceGridRegular.html>`_                                      Grid with regularly-spaced forces
====================================================================================================================================================  ====================================================================================================================================================


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

For additional usage information, see `Grid <https://uafgeotools.github.io/mtuq/library/generated/mtuq.grid.Grid.html>`_ for regularly-spaced grids or `UnstructuredGrid <https://uafgeotools.github.io/mtuq/library/generated/mtuq.grid.UnstructuredGrid.html>`_ for randomly-drawn grids.

Moment tensor grids are implemented using the `rho, v, w, kappa, sigma, h` parameterization from `Tape215 <https://uafgeotools.github.io/mtuq/references.html>`_.  

Force grids are implemented using parameters `F0, phi, h`, which are related to `r, phi, theta` spherical coordinates (physics convention) by `F0 = r`, `phi = phi`, `h = cos(theta)`.  In addition, `F0, phi, h` are related to geographic directions by these `formulas <https://uafgeotools.github.io/mtuq/_modules/mtuq/util/math.html#to_rtp>`_.

Random moment tensors are drawn from the uniform distributions described by `Tape215 <https://uafgeotools.github.io/mtuq/references.html>`_.


