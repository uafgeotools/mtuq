
import numpy as np



def _plot_lune_pygmt(filename, lon, lat, values, best_vw=None, lune_array=None, 
    **kwargs):

    import pygmt

    if _nothing_to_plot(values):
        return

    raise NotImplementedError


def _plot_vw_pygmt(filename, lon, lat, values, best_vw=None, lune_array=None,
    **kwargs):

    import pygmt

    if _nothing_to_plot(values):
        return

    raise NotImplementedError


def _plot_force_pygmt(filename, phi, h, values, best_force=None, **kwargs):

    import pygmt

    if _nothing_to_plot(values):
        return

    raise NotImplementedError


def _plot_latlon_pygmt(filename, lon, lat, values, best_latlon=None, lune_array=None,
    **kwargs):

    import pygmt

    if _nothing_to_plot(values):
        return

    raise NotImplementedError


def _plot_depth_pygmt(filename,
        depths,
        values,
        magnitudes=None,
        lune_array=None,
        title='',
        xlabel='',
        ylabel='',
        fontsize=16.):

    import pygmt

    if _nothing_to_plot(values):
        return

    raise NotImplementedError


def _nothing_to_plot(values):
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


