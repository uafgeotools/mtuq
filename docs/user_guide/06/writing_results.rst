
Reading and writing grid search results
=======================================

A grid search 

.. code::

    ds = grid_search(data, greens, misfit, stations, origins, sources)

returns a data structure `ds` that contains both misfit values and grid points.

To write these results to disk:

.. code::

    ds.save(filename)

Depending on whether the grid points are regularly- or irregularly spaced, the result will be either a NetCDF or HDF5 file.

The results can be read back in using the I/O function `open_ds`, which attempts to automatically determine the file type:

.. code::

    from mtuq import open_ds
    ds = open_ds(filename)

If for some reason file format determination fails, users can pass a keyword argument `format`, as in the following example:


.. code::

    from mtuq import open_ds
    ds = open_ds('output.nc', 'format='NetCDF'')




More details
------------

Regularly-spaced grids
""""""""""""""""""""""

A grid search over a reguarly-spaced grid of `sources`

.. code::

    da = grid_search(data, greens, misfit, stations, origins, sources)

returns an xarray DataArray `da`.

To write the grid search output `da` to disk as a NetCDF file:

.. code::

    da.save('output.nc')


To read the NetCDF file back from disk:

.. code::

    from mtuq import open_ds
    da = open_ds('output.nc', format='NetCDF')


Before or after writing to disk, `da` can be passed to visualization utilities or manipulated using `xarray methods <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html>`_.



Irregularly-spaced grids
""""""""""""""""""""""""

A grid search over a irregularly-spaced grid of `sources`

.. code::

    df = grid_search(data, greens, misfit, stations, origins, sources)

returns a pandas DataFrame `df`.

To write the grid search output `df` to disk as an HDF5 file:

.. code::

    df.save('output.hf5')


To read the HDF5 file back from disk:

.. code::

    from mtuq import open_ds
    df = open_ds('output.hf5', format='HDF5')


Before or after writing to disk, `df` can be passed to visualization utilities or manipulated using `pandas methods <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_.


