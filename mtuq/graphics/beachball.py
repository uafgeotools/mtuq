#
# utilities for plotting
#

import numpy as np
import matplotlib.pyplot as pyplot
from mtuq.event import MomentTensor
from obspy.imaging.beachball import beach, beachball



def plot_beachball(filename, mt):
    """ Plots source mechanism
    """
    beachball(mt, size=200, linewidth=2, facecolor='b')
    pyplot.savefig(filename)
    pyplot.close()


def misfit_vs_depth(filename, data, misfit, origins, sources, results):
    """ Plots misfit versus depth from grid search results

    Creates a scatter plot in which the the placment of each marker shows the 
    misfit of the best-fitting source for a given depth. 

    Following SilwalTape2016, the marker itself shows the focal mechanism and
    moment magnitude of the best-fitting source.
    """
    # requires some further work
    # - "publication quality" figures
    # - double check correctness of variance reduction formulas

    fig = pyplot.figure()
    ax = pyplot.gca()

    # normalize results
    norm = 0
    for stream in data:
        for trace in stream:
            if misfit.norm=='L1':
                norm += np.sum(np.abs(trace.data))
            elif misfit.norm=='L2':
                norm += np.sum(trace.data**2)
            elif misfit.norm=='hybrid':
                norm += np.sum(trace.data**2)**0.5
    results /= norm

    # what is the minimum result for each depth?
    indices = results.argmin(axis=0)
    results = results.min(axis=0)

    # optional further normalization
    results = transform1(results)
    #results = transform2(results)
    yrange = results.max() - results.min()

    for _i, origin in enumerate(origins):
        result = results[_i]
        source = sources.get(indices[_i])

        xp = origin.depth_in_m/1000.
        yp = result
        pyplot.plot(xp, yp)

        # add beachball
        marker = beach(source, xy=(xp, yp), width=20., linewidth=0.5, axes=ax)
        ax.add_collection(marker)

        # add magnitude label
        label = '%2.1f' % MomentTensor(source).magnitude()
        _text(xp, yp-0.075*yrange, label)

    pyplot.ylim([-0.15*yrange, 1.15*yrange])

    pyplot.xlabel('Depth (km)')
    pyplot.ylabel('Normalized misfit')
    pyplot.savefig(filename)
    pyplot.close()


def _hide_axis(ax):
    # hides axes lines, ticks, and labels
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])


def _text(xp, yp, label):
    pyplot.text(xp, yp, label, 
        fontsize=8,
        horizontalalignment='center',
        )

def transform1(v):
    return (v - v.min())/v.min()


def transform2(v):
    return np.log((1. - v.min())/(1. - v))


gray = [0.667, 0.667, 0.667]

