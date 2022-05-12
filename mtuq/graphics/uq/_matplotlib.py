
import numpy as np

from matplotlib import pyplot
from os.path import exists
from xarray import DataArray

from mtuq.graphics.uq import _nothing_to_plot
from mtuq.graphics._gmt import read_cpt, _cpt_path


#
# vw rectangle
#

def _plot_vw_matplotlib(filename, v, w, values, best_vw=None, lune_array=None, 
    colormap='viridis', title=''):

    if _nothing_to_plot(values):
        return

    fig, ax = pyplot.subplots(figsize=(3., 8.), constrained_layout=True)

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

