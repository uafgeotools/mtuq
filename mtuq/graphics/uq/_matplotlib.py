
import numpy as np

from matplotlib import pyplot
from os.path import exists
from xarray import DataArray

from mtuq.graphics.uq import _nothing_to_plot
from mtuq.graphics._gmt import read_cpt, _cpt_path


def _plot_dc_matplotlib(filename, coords, 
    values_h_kappa, values_sigma_kappa, values_sigma_h,
    best_dc=None, colormap='viridis',  figsize=(8., 8.), fontsize=14):

    # prepare axes
    fig, axes = pyplot.subplots(2, 2,
        figsize=figsize,
        )

    pyplot.subplots_adjust(
        wspace=0.4,
        hspace=0.4,
        )

    # parse colormap
    if exists(_cpt_path(colormap)):
       colormap = read_cpt(_cpt_path(colormap))

    # note the following parameterization details
    #     kappa = strike
    #     sigma = slip
    #     h = cos(dip)

    # plot surfaces
    _pcolor(axes[0][0], coords['h'], coords['kappa'], values_h_kappa, colormap)

    _pcolor(axes[0][1], coords['sigma'], coords['kappa'], values_sigma_kappa, colormap)

    _pcolor(axes[1][1], coords['sigma'], coords['h'], values_sigma_h, colormap)

    # optional markers
    if best_dc:
        _kappa, _sigma, _h = best_dc
        _add_marker(axes[0][0], (_h, _kappa))
        _add_marker(axes[0][1], (_sigma, _kappa))
        _add_marker(axes[1][1], (_sigma, _h))

    _set_dc_labels(axes, fontsize=fontsize)

    pyplot.savefig(filename)



def _plot_vw_matplotlib(filename, v, w, values, best_vw=None, lune_array=None, 
    colormap='viridis', title='', figsize=(3., 8.)):

    if _nothing_to_plot(values):
        return

    fig, ax = pyplot.subplots(figsize=figsize, constrained_layout=True)

    # pcolor requires corners of pixels
    corners_v = _centers_to_edges(v)
    corners_w = _centers_to_edges(w)

    # `values` gets mapped to pixel colors
    pyplot.pcolor(corners_v, corners_w, values, cmap=colormap)

    # v and w have the following bounds
    # (see https://doi.org/10.1093/gji/ggv262)
    pyplot.xlim([-1./3., 1./3.])
    pyplot.ylim([-3./8.*np.pi, 3./8.*np.pi])

    pyplot.xticks([], [])
    pyplot.yticks([], [])

    if exists(_cpt_path(colormap)):
       cmap = read_cpt(_cpt_path(colormap))

    if True:
        cbar = pyplot.colorbar(
            orientation='horizontal',
            pad=0.,
            )

        cbar.formatter.set_powerlimits((-2, 2))

    if title:
        fontdict = {'fontsize': 16}
        pyplot.title(title, fontdict=fontdict)


    if best_vw:
        pyplot.scatter(*best_vw, s=333,
            marker='o',
            facecolors='none',
            edgecolors=[0,1,0],
            linewidths=1.75,
            )

    pyplot.savefig(filename)
    pyplot.close()



def _plot_depth_matplotlib(filename, depths, values,
        magnitudes=None, lune_array=None,
        title=None, xlabel=None, ylabel=None, figsize=(6., 6.), fontsize=16.):

    pyplot.figure(figsize=figsize)
    pyplot.plot(depths, values, 'k-')

    if title:
        pyplot.title(title, fontsize=fontsize)

    if xlabel:
         pyplot.xlabel(xlabel, fontsize=fontsize)

    if ylabel:
         pyplot.ylabel(ylabel, fontsize=fontsize)

    pyplot.savefig(filename)



#
# utility functions
#

def _centers_to_edges(v):
    if issubclass(type(v), DataArray):
        v = v.values.copy()
    else:
        v = v.copy()

    dv = (v[1]-v[0])
    v -= dv/2
    v = np.pad(v, (0, 1))
    v[-1] = v[-2] + dv

    return v


def _add_marker(axis, coords):
    axis.scatter(*coords, s=250,
        marker='o',
        facecolors='none',
        edgecolors=[0,1,0],
        linewidths=1.75,
        clip_on=False,
        zorder=100,
        )


def _pcolor(axis, x, y, values, colormap, **kwargs):
    # workaround matplotlib compatibility issue
    try:
        axis.pcolor(x, y, values, cmap=colormap, shading='auto', **kwargs)
    except:
        axis.pcolor(x, y, values, cmap=colormap, **kwargs)


def _set_dc_labels(axes, **kwargs):

    # note the following parameterization details
    #     kappa = strike
    #     sigma = slip
    #     h = cos(dip)

    kappa_ticks = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    kappa_ticklabels = ['0', '', '90', '', '180', '', '270', '', '360']

    sigma_ticks = [-90, -67.5, -45, -22.5, 0, 22.5, 45, 67.5, 90]
    sigma_ticklabels = ['-90', '', '-45', '', '0', '', '45', '', '90']

    h_ticks = [np.cos(np.radians(tick)) for tick in [0, 15, 30, 45, 60, 75, 90]]
    h_ticklabels = ['0', '', '30', '', '60', '', '90']

    # upper left panel
    axis = axes[0][0]
    axis.set_xlabel('Dip', **kwargs)
    axis.set_xticks(h_ticks)
    axis.set_xticklabels(h_ticklabels)
    axis.set_ylabel('Strike', **kwargs)
    axis.set_yticks(kappa_ticks)
    axis.set_yticklabels(kappa_ticklabels)

    # upper right panel
    axis = axes[0][1]
    axis.set_xlabel('Slip', **kwargs)
    axis.set_xticks(sigma_ticks)
    axis.set_xticklabels(sigma_ticklabels)
    axis.set_ylabel('Strike', **kwargs)
    axis.set_yticks(kappa_ticks)
    axis.set_yticklabels(kappa_ticklabels)

    # lower right panel
    axis = axes[1][1]
    axis.set_xlabel('Slip', **kwargs)
    axis.set_xticks(sigma_ticks)
    axis.set_xticklabels(sigma_ticklabels)
    axis.set_ylabel('Dip', **kwargs)
    axis.set_yticks(h_ticks)
    axis.set_yticklabels(h_ticklabels)

    # lower left panel
    axes[1][0].axis('off')

