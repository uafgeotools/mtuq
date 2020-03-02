
import obspy.imaging.beachball
import os
import matplotlib.pyplot as pyplot
import numpy as np
import shutil
import subprocess
import warnings
from mtuq.event import MomentTensor


# To correctly plot focal mechanims, MTUQ uses Generic Mapping Tools (GMT).
# Users must install this package by themselves, since it is not available
# through the Python Package Index.

# If GMT >=6.0.0 executables are not found on the system path, MTUQ falls 
# back to ObsPy. As described in the following GitHub issue, ObsPy 
# focal mechanism plots suffer from  plotting artifacts:

# https://github.com/obspy/obspy/issues/2388


def plot_beachball(filename, mt):
    """ Plots focal mechanism of given moment tensor as PNG image
    """
    from mtuq.graphics.gmt import gmt_major_version

    if type(mt)!=MomentTensor:
        mt = MomentTensor(mt)

    try:
        assert gmt_major_version() >= 6
        beachball_gmt(filename, mt)

    except:
        beachball_obspy(filename, mt)


def beachball_gmt(filename, mt):
    """ Plots focal mechanism using GMT
    """
    # check file extension
    if filename.endswith('.png'):
        filename = filename[:-4]

    if filename.endswith('.ps'):
        filename = filename[:-3]

    # create Post Script image
    subprocess.call('\n'.join([
        ('gmt psmeca -R-5/5/-5/5 -JM5 -Sm1 -Ggrey50 -h1 << END > %s' % filename+'.ps'),
        'lat lon depth   mrr   mtt   mff   mrt    mrf    mtf',
        ('0.  0.  10.    %e     %e    %e    %e     %e     %e 25 0 0' % tuple(mt.as_vector())),
        'END']), shell=True)

    # create PNG image
    subprocess.call('gmt psconvert %s -A -Tg' % (filename+'.ps'),
        shell=True)


def beachball_obspy(filename, mt):
    """ Plots focal mechanism using ObsPy
    """
    warnings.warn("""
        WARNING

        Generic Mapping Tools (>=6.0.0) executables not found on system path.
        Falling back to ObsPy.

        As described in the following GitHub issue, ObsPy focal mechanism
        plots suffer from artifacts:

        https://github.com/obspy/obspy/issues/2388
        """)

    obspy.imaging.beachball.beachball(
        mt.as_vector(), size=200, linewidth=2, facecolor=gray)

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

    depths = []
    for origin in origins:
        depths += [origin.depth_in_m/1000.]

    xr = max(depths) - min(depths)
    yr = results.max() - results.min()

    for _i, origin in enumerate(origins):

        source = sources.get(indices[_i])
        result = results[_i]

        xp = origin.depth_in_m/1000.
        yp = result
        pyplot.plot(xp, yp)

        # add beachball
        plot_beachball('tmp.png', source)
        img = pyplot.imread('tmp.png')
        os.remove('tmp.png')
        os.remove('tmp.ps')

        xw = 0.1*xr
        yw = 0.1*yr
        #ax.imshow(img, extent=(xp-xw,xp+xw,yp-yw,yp+yw), transform=ax.transAxes)

        # add magnitude label
        label = '%2.1f' % MomentTensor(source).magnitude()
        _text(xp, yp-0.075*yr, label)

    pyplot.xlim((-0.1*xr + min(depths), 0.1*xr + max(depths)))
    pyplot.ylim((-0.1*yr + results.min(), 0.1*yr + results.max()))

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

