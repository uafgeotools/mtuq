
Grid search data structures
===========================

During a grid search

.. code::

    ds = grid_search(data, greens, misfit, stations, origins, sources)


the misfit function is evaluated over all `origins` and `sources`.

The result is a data strucutre `ds` containing both misfit values and grid points.


Input data structures
---------------------

The `origins` data structure

- Is simply a list of `Origin` objects
- Allows for searching over depth, latitude, longitude and/or origin time, as specified by the `Origin` objects in the list

The `sources` data structure

- Can be either a regularly-spaced `Grid` or an irregularly-spaced `UnstructuredGrid`
- Allows for searching over any subset of moment tensor space and/or force space, including these currently-implemented `moment tensor` and `force` grids


Output data structures
----------------------

If `sources` is regularly spaced, the output `ds` will be a type of xarray DataArray.

If `sources` is irregularly spaced, the output `ds` will be a type of pandas DataFrame.

See here for information about reading or writing grid search results.


