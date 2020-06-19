
import numpy as np


def parse_regular(origins, sources, values):
    from mtuq.grid import Grid

    origin_dims, origin_coords, origin_shape = _parse(origins)
    if origin_shape is None:
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
    origin_dims, origin_coords, _ = _parse(origins)

    if issubclass(type(sources), UnstructuredGrid):
        source_dims, source_coords = sources.dims, sources.coords
    else:
        raise TypeError

    for _i, coords in enumerate(source_coords):
        source_coords[_i] = repmat(coords, len(origins))

    for _j, coords in enumerate(origin_coords):
        origin_coords[_j] = repmat(coords, len(sources))

    attrs = {
        'origins': origins,
        'sources': sources,
        'origin_dims': origin_dims,
        'origin_coords': origin_coords,
        'source_dims': source_dims,
        'source_coords': source_coords,
        }


    return {
        'data': values.flatten(),
        'coords': source_coords + origin_coords,
        'dims': source_dims + origin_dims,
        'attrs': attrs}


def _parse(origins, dims=('latitude', 'longitude', 'depth_in_m')):
    ni = len(origins)
    nj = len(dims)

    array = np.empty((ni, nj))
    for _i, origin in enumerate(origins):
        for _j, dim in enumerate(dims):
            array[_i,_j] = origin[dim]

    coords = []
    coords_uniq = []
    shape = ()
    for _j, dim in enumerate(dims):
        coords += [np.unique(array[:,_j])]
        coords_uniq += [np.unique(array[:,_j])]
        shape += (len(coords_uniq[-1]),)

    if np.product(shape)==ni:
        return dims, coords_uniq, shape
    else:
        return dims, coords, None


