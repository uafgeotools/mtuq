
import numpy as np

from mtuq.graphics.uq.vw import vw_area
from mtuq.grid_search import DataArray, DataFrame, MTUQDataArray, MTUQDataFrame
from mtuq.util import dataarray_idxmin, dataarray_idxmax, product


def likelihood_analysis(*args):
    """ Converts misfit to likelihood and multiplies together contributions 
    from different data categories 
    """

    arrays = [arg[0] for arg in args]
    covs = [arg[1] for arg in args]

    values = []
    for _i, array in enumerate(arrays):
        values += [np.exp(-array.values**2/covs[_i])]

    dims = arrays[0].dims
    coords = arrays[0].coords
    values = product(*values)

    likelihoods = MTUQDataArray(**{
        'data': values,
        'dims': dims,
        'coords': coords,
        })

    marginals = likelihoods.sum(
        dim=('origin_idx', 'rho', 'kappa', 'sigma', 'h'))
    marginals.values /= marginals.values.sum()
    marginals /= vw_area

    # extract maximum likelihood estimate
    idxmax = dataarray_idxmax(likelihoods)
    mle = {key: float(idxmax[key].values)
        for key in dims}

    # extract marginal vw estimate
    idxmax = dataarray_idxmax(marginals)
    marginal_vw = {key: float(idxmax[key].values)
        for key in ('v', 'w')}

    return likelihoods, mle, marginal_vw

