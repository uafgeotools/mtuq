
import numpy as np

from mtuq.grid_search import DataArray, DataFrame, MTUQDataArray, MTUQDataFrame
from mtuq.util import dataarray_idxmin, dataarray_idxmax, product, warn


def _nothing_to_plot(values):
    """ Sanity check - are all values identical in 2-D array?
    """
    mask = np.isnan(values)
    if np.all(mask):
        warn(
            "Nothing to plot: all values are NaN",
            Warning)
        return True

    masked = np.ma.array(values, mask=mask)
    minval = masked.min()
    maxval = masked.max()

    if minval==maxval:
        warn(
            "Nothing to plot: all values are identical",
            Warning)
        return True



def likelihood_analysis(*args):
    """ Converts misfit to likelihood and multiplies together contributions 
    from different data categories 
    """
    from mtuq.graphics.uq.vw import vw_area


    arrays = [arg[0] for arg in args]
    vars = [arg[1] for arg in args]

    values = []
    for _i, array in enumerate(arrays):
        values += [np.exp(-0.5*array.values/vars[_i])]

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



