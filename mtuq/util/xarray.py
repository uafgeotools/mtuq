
import numpy as np

from mtuq.grid import UnstructuredGrid


def parse_regular(origins, sources, values):
    from mtuq.grid import Grid

    origin_dims = ('latitude', 'longitude', 'depth_in_m')

    ni = len(origins)
    nj = len(origin_dims)

    array = np.empty((ni, nj))
    for _i, origin in enumerate(origins):
        for _j, dim in enumerate(origin_dims):
            array[_i,_j] = origin[dim]

    coords_uniq = []
    shape = ()
    for _j, dim in enumerate(origin_dims):
        coords_uniq += [np.unique(array[:,_j])]
        shape += (len(coords_uniq[-1]),)

    if np.product(shape)==ni:
        origin_coords, origin_shape = coords_uniq, shape
    else:
        raise TypeError

    if issubclass(type(sources), Grid):
        source_dims, source_coords, source_shape =\
            sources.dims, sources.coords, sources.shape
    else:
        raise TypeError

    attrs = {
        'origins': origins,
        'sources': sources,
        'origin_dims': origin_dims,
        'origin_coords': origin_coords,
        'origin_shape': origin_shape,
        'source_dims': source_dims,
        'source_coords': source_coords,
        'source_shape': source_shape,
        }

    return {
        'data': np.reshape(values, source_shape + origin_shape),
        'coords': source_coords + origin_coords,
        'dims': source_dims + origin_dims,
        'attrs': attrs,
        }


def parse_irregular(origins, sources, values):
    if issubclass(type(sources), UnstructuredGrid):
        source_dims, source_coords = sources.dims, sources.coords
    else:
        raise TypeError

    for _i, coords in enumerate(source_coords):
        source_coords[_i] = np.tile(coords, len(origins))

    origin_idx = np.arange(len(origins), dtype='int')
    origin_idx = np.repeat(origin_idx, len(sources))

    dims = ('origin_idx',) + source_dims + ('values',)
    coords = [origin_idx] + source_coords + [values.flatten()]

    ndim = len(dims)
    data_vars = {dims[_i]: coords[_i] for _i in range(ndim)}

    return data_vars


