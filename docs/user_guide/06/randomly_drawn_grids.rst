

Working with randomly-drawn grids
=================================

Randomly-drawn grids of moment tensors, forces, or other sources parameters can be defined using the `UnstructredGrid` class.

A grid search over a randomly-drawn grid of `sources`

.. code::

    df = grid_search(data, greens, misfit, stations, origins, sources)

returns a pandas DataFrame `df`, which contains misfit values associated with each grid point.

To write the grid search output `df` to disk as an HDF5 file:

.. code::

    df.save('output.hf5')


To read the HDF5 file back from disk:

.. code::

    from pandas import read_hdf
    df = read_hdf('output.hf5')


Before or after writing to disk, `df` can be passed to visualization utilities or manipulated using pandas methods.

