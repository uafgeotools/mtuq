
import numpy as np
import matplotlib.pyplot as pyplot
from mtuq.graphics.waveform import _hide_axes
from obspy.imaging.beachball import beach, beachball



def plot_beachball(filename, mt):
    """ Plots source mechanism
    """
    beachball(mt, size=200, linewidth=2, facecolor='b')
    pyplot.savefig(filename)


def misfit_vs_depth(filename, misfit, sources, origins):
    nn=len(origins)
    fig = pyplot.figure(figsize=(nn+1, 1))
    ax = pyplot.gca()

    for _i, origin in enumerate(origins):
        best_misfit = misfit[_i, :].min()
        best_source = sources.get(misfit[_i, :].argmin())

        # add beachball
        ax.add_collection(
            beach(best_source, xy=(_i+1, 0.125), width=0.5))

        # add depth label
        label = '%d km' % (origin.depth_in_m/1000.)
        x, y = _i+1, -0.5

        pyplot.text(x, y, label,
            fontsize=8,
            horizontalalignment='center')

        # add magnitude label
        label = '%2.1f' % _magnitude(best_source)
        x, y = _i+1, -0.33

        pyplot.text(x, y, label,
            fontsize=8,
            horizontalalignment='center')

    ax.set_aspect("equal")
    ax.set_xlim((0, nn+1))
    ax.set_ylim((-0.5, +0.5))
    _hide_axes(ax)

    pyplot.savefig(filename)
    pyplot.close()


def _magnitude(mt):
    M = _asmatrix(mt)
    M0 = (np.tensordot(M,M)/2.)**0.5
    Mw = 2./3.*(np.log10(M0) - 9.1)
    return Mw


def _asmatrix(m):
    return np.array([
        [m[0], m[3], m[4]],
        [m[3], m[1], m[5]],
        [m[4], m[5], m[2]]])

