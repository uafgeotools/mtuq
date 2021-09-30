
#
# JUST A QUICK SKETCH, REQUIRES MORE WORK
#

import numpy as np
import os

from glob import glob
from matplotlib import pyplot
from mtuq.graphics.beachball import plot_beachball
from mtuq.graphics.uq.lune import plot_misfit_lune
from mtuq.graphics.uq.vw import plot_marginal_vw


def plot_summary1(filename, misfit, stations, origin, mt):

    tmp1 = 'tmp_variance_reduction.png'

    plot_misfit_lune(tmp1, misfit, colorbar_type=3)

    tmp2 = 'tmp_beachball.png'

    plot_beachball(tmp2, mt, None)

    _create_summary1(filename, tmp1, tmp2, cleanup=True)



def _create_summary1(filename, tmp1, tmp2, cleanup=False):

    height = 6.
    width = 6.
    aspect = 1.

    fig = pyplot.figure(figsize=(width, height))
    ax = _get_axis(fig, width, height)

    img = pyplot.imread(tmp1)
    ax.imshow(img, extent=(0.60,0.95,0.025,0.975))

    img = pyplot.imread(tmp2)
    ax.imshow(img, extent=(0.05,0.55,0.05,0.55))

    pyplot.savefig(filename)


    if cleanup:
        for filename in glob('tmp*'):
            os.remove(filename)


def plot_summary2(filename, misfit, var, stations, origin, mt):

    tmp1 = 'tmp_misfit.png'
    plot_misfit_lune(tmp1, misfit, colorbar_type=3)

    tmp2 = 'tmp_marginal.png'
    plot_marginal_vw(tmp2, misfit, var, colorbar_type=3)

    tmp3 = 'tmp_beachball.png'

    # stations is not yet implemented in plot_beachball
    #plot_beachball('tmp_beachball.ps', mt, stations)

    plot_beachball(tmp3, mt, None)

    _create_summary2(filename, tmp1, tmp2, tmp3, cleanup=True)



def _create_summary2(filename, tmp1, tmp2, tmp3, cleanup=False):

    height = 6.
    width = 8.
    aspect = width/height

    fig = pyplot.figure(figsize=(width, height))
    ax = _get_axis(fig, width, height)

    img = pyplot.imread(tmp1)
    ax.imshow(img, extent=(0.450*aspect,0.700*aspect,0.025,0.975))

    img = pyplot.imread(tmp2)
    ax.imshow(img, extent=(0.745*aspect,0.975*aspect,0.025,0.975))

    img = pyplot.imread(tmp3)
    ax.imshow(img, extent=(0.05,0.55,0.05,0.55))

    pyplot.savefig(filename)


    if cleanup:
        for filename in glob('tmp*'):
            os.remove(filename)



def _get_axis(fig, width, height):

    ax = fig.add_axes([0., 0., 1., 1.])

    ax.set_xlim([0., width/height])
    ax.set_ylim([0., 1.])

    # hides axes lines, ticks, and labels
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)
    #ax.spines['left'].set_visible(False)
    #ax.get_xaxis().set_ticks([])
    #ax.get_yaxis().set_ticks([])

    return ax

