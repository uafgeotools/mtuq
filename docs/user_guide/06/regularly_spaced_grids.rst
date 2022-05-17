

Working with regularly-spaced grids
===================================

Regularly-spaced grids of moment tensors, forces, or other sources parameters can be defined using the `Grid` class.

A grid search over a reguarly-spaced grid of `sources`

.. code::

    da = grid_search(data, greens, misfit, stations, origins, sources)

returns an xarray DataArray `da`, which contains misfit values associated with each grid point.

To write the grid search output `da` to disk as a NetCDF file:

.. code::

    da.save('output.nc')


To read the NetCDF file back from disk:

.. code::

    from xarray import open_dataarray
    da = open_dataarray('output.nc')


Before or after writing to disk, `da` can be passed to visualization utilities or manipulated using xarray methods.

