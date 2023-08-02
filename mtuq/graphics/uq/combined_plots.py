from mtuq.graphics.uq.lune import plot_misfit_lune, _plot_lune
from mtuq.graphics.uq.double_couple import plot_misfit_dc, _plot_dc
from mtuq.graphics.uq._matplotlib import _plot_lune_matplotlib, _plot_dc_matplotlib

import matplotlib.gridspec as gridspec
import tempfile
import matplotlib.pyplot as plt

from mtuq.graphics.uq.vw import _misfit_vw_regular, _misfit_vw_random
from mtuq.graphics.uq.double_couple import _misfit_dc_regular, _misfit_dc_random
from mtuq.grid_search import DataArray, DataFrame

def plot_combined(filename, ds, **kwargs):
    """
    Creates a figure with two subplots, one for a lune plot and one for a DC plot,
    and saves it to the specified file.

    :param filename: The name of the file to save the figure to.
    :param ds_lune: A DataArray or DataFrame containing the data for the lune plot.
    :param ds_dc: A DataArray or DataFrame containing the data for the DC plot.
    """

    ds_lune = ds.copy()
    ds_dc = ds.copy()

    # Apply the necessary preprocessing to each dataset
    if issubclass(type(ds_lune), DataArray):
        misfit_lune = _misfit_vw_regular(ds_lune.copy())
    elif issubclass(type(ds_lune), DataFrame):
        misfit_lune = _misfit_vw_random(ds_lune.copy())
    else:
        raise Exception("ds_lune must be a DataArray or DataFrame")

    if issubclass(type(ds_dc), DataArray):
        misfit_dc = _misfit_dc_regular(ds_dc.copy())
    elif issubclass(type(ds_dc), DataFrame):
        misfit_dc = _misfit_dc_random(ds_dc.copy())
    else:
        raise Exception("ds_dc must be a DataArray or DataFrame")

    # Create a GridSpec with two columns, the second one being 20% smaller
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 0.5])
    # gs.update(wspace=-0.025)

    # Create a temporary file for the lune plot
    with tempfile.NamedTemporaryFile(suffix=".png") as tmpfile_lune:
        # Generate the lune plot
        _plot_lune(tmpfile_lune.name, misfit_lune, backend=_plot_lune_matplotlib, **kwargs)

        # Load the lune plot into an image
        img_lune = plt.imread(tmpfile_lune.name)

        # Display the lune plot in the first subplot
        ax0 = plt.subplot(gs[0])
        ax0.imshow(img_lune)
        ax0.axis("off")  # Hide the axes

    # Create a temporary file for the DC plot
    with tempfile.NamedTemporaryFile(suffix=".png") as tmpfile_dc:
        # Generate the DC plot
        _plot_dc(tmpfile_dc.name, misfit_dc, backend=_plot_dc_matplotlib, **kwargs)

        # Load the DC plot into an image
        img_dc = plt.imread(tmpfile_dc.name)

        # Display the DC plot in the second subplot
        ax1 = plt.subplot(gs[1])
        ax1.imshow(img_dc)
        ax1.axis("off")  # Hide the axes

        # Adjust the position of ax2
        pos1 = ax1.get_position() # get the original position 
        ax1.set_position([pos1.x0 + 0.14, pos1.y0, pos1.width, pos1.height]) # shift ax2 to the left

    gs.tight_layout(fig)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
