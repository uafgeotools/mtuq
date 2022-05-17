
Reading and writing grid search results
=======================================

Regularly-spaced grids
----------------------

Regularly-spaced grids of moment tensors, forces, or other source parameters can be defined using the `Grid` class.

A grid search over a reguarly-spaced grid of `sources`

.. code::

    da = grid_search(data, greens, misfit, stations, origins, sources)

returns an xarray DataArray `da`, which contains misfit values for all grid points.

To write the grid search output `da` to disk as a NetCDF file:

.. code::

    da.save('output.nc')


To read the NetCDF file back from disk:

.. code::

    from xarray import open_dataarray
    da = open_dataarray('output.nc')


Before or after writing to disk, `da` can be passed to visualization utilities or manipulated using `xarray methods <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html>`_.



Randomly-drawn grids
--------------------

Randomly-drawn grids of moment tensors, forces, or other source parameters can be defined using the `UnstructredGrid` class.

A grid search over a randomly-drawn grid of `sources`

.. code::

    df = grid_search(data, greens, misfit, stations, origins, sources)

returns a pandas DataFrame `df`, which contains misfit values for all grid points.

To write the grid search output `df` to disk as an HDF5 file:

.. code::

    df.save('output.hf5')


To read the HDF5 file back from disk:

.. code::

    from pandas import read_hdf
    df = read_hdf('output.hf5')


Before or after writing to disk, `df` can be passed to visualization utilities or manipulated using `pandas methods <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_.


