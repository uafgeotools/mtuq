
import numpy as np
import matplotlib.pyplot as pyplot
from obspy.imaging.beachball import beach, beachball



def plot_beachball(filename, mt):
    """ Plots source mechanism
    """
    beachball(mt, size=200, linewidth=2, facecolor='b')
    pyplot.savefig(filename)


def beachball_vs_depth(filename, mt_dict):
    n=len(mt_dict)

    fig = pyplot.figure(figsize=(n+1, 1))
    ax = pyplot.gca()

    depths = mt_dict.keys()
    mt_list = mt_dict.values()
    magnitudes =  [_magnitude(mt) for mt in mt_list]

    # create iterator
    zipped = zip(depths, mt_list, magnitudes)
    zipped = sorted(zipped, key=lambda x: x[0])

    # plot beachballs
    for _i, item in enumerate(zipped):
        depth_in_m, mt, magnitude = item

        # add beachball
        ax.add_collection(
            beach(mt, xy=(_i+1, 0.125), width=0.5))

        # add depth label
        label = '%d km' % (depth_in_m/1000.)
        x, y = _i+1, -0.5

        pyplot.text(x, y, label,
            fontsize=8,
            horizontalalignment='center')

        # add magnitude label
        label = '%2.1f' % magnitude
        x, y = _i+1, -0.33

        pyplot.text(x, y, label,
            fontsize=8,
            horizontalalignment='center')

    ax.set_aspect("equal")
    ax.set_xlim((0, n+1))
    ax.set_ylim((-0.5, +0.5))
    hide_axes(ax)

    pyplot.savefig(filename)
    pyplot.close()


def misfit_vs_depth(filename, misfit_dict):
    fig = pyplot.figure()
    ax = pyplot.gca()

    pyplot.plot(misfit_dict.keys(), misfit_dict.values(), '.')

    pyplot.xlabel('Depth (m)')
    pyplot.ylabel('Misfit')

    pyplot.savefig(filename)
    pyplot.close()


