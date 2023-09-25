import numpy as np
import math

from matplotlib import pyplot
from os.path import exists
from xarray import DataArray

from mtuq.graphics.uq import _nothing_to_plot
from mtuq.graphics._gmt import read_cpt, _cpt_path
from mtuq.graphics.uq._gmt import _parse_vw, _parse_lune_array
from matplotlib.colors import BoundaryNorm

# Define a pure matplotlib backend as an alternative to GMT
# It should behave as similarly as possible to the GMT backend 
# and take the same input arguments
def _plot_lune_matplotlib(filename, longitude, latitude, values, 
    best_vw=None, lune_array=None, colormap='viridis', title=None, plot_type='contour', **kwargs):

    """ Plots DataArray values on the eigenvalue lune (requires matplotlib)

    .. rubric :: Keyword arguments
    filename : str
        Name of output image file

    longitude : array_like (xarray.DataArray or numpy.ndarray)
        Array of longitudes

    latitude : array_like (xarray.DataArray or numpy.ndarray)
        Array of latitudes

    values : array_like (xarray.DataArray or numpy.ndarray)
        Array of values

    best_vw : list
        List of two floats representing the best-fitting eigenvalues

    lune_array : array_like (xarray.DataArray or numpy.ndarray)
        Used to plot beachball tradeoffs on the lune

    colormap : str
        Name of colormap

    title : str
        Title of plot

    plot_type : str
        Type of plot. Can be either contour, colormesh or scatter

    """

    # Check plot_type. Can be either contour or colormesh
    if plot_type not in ['contour', 'colormesh', 'scatter']:
        raise Exception('plot_type must be either contour or colormesh')


    # # Check 
    # if 'contour' in kwargs:
    #     contour = kwargs['contour']
    # else:
    #     contour = False

    fig, ax = _generate_lune()

    # Transform data to Hammer projection
    # Create a grid for pcollormesh from longitude and latitude arrays
    x, y = np.meshgrid(longitude, latitude)
    x, y = _hammer_projection(x, y)

    # Plot data
    # Use the percentile method to filter out outliers (Will alway clip the 10% greater values)
    if plot_type == 'colormesh':
        vmin, vmax = np.nanpercentile(np.asarray(values), [0,75])
        im = ax.pcolormesh(x, y, values, cmap=colormap, vmin=vmin, vmax=vmax, shading='auto', zorder=10)
    elif plot_type == 'contour':
        # Plot using contourf
        levels = 20
        im = ax.contourf(x, y, values, cmap=colormap, levels=levels, zorder=10)
    elif plot_type == 'scatter':
        # Prepare colormap
        boundaries = np.linspace(0, 5, 6)
        norm = BoundaryNorm(boundaries, ncolors=256, clip=False)
        # Plot using scatter
        im = ax.scatter(x, y, c=values, cmap='Spectral_r', norm=norm, zorder=100)


    # Plot best-fitting moment tensor
    if best_vw is not None:
        best_vw = _parse_vw(best_vw)
        gamma, delta = _hammer_projection(
            best_vw[0], best_vw[1])
        if plot_type not in ['scatter']:
            _add_marker(ax, (gamma, delta))

    # Plot tradeoffs
    if lune_array is not None:
        gamma, delta = _hammer_projection(
            lune_array[:,0], lune_array[:,1])
        ax.plot(gamma, delta, 'w.', markersize=1)

    # Set axis limits
    ax.set_xlim(-30.5, 30.5)
    ax.set_ylim(-90, 90)

    # Set axis labels
    ax.set_xlabel('Gamma')
    ax.set_ylabel('Delta')

    # if plot type is colormesh or contour:
    if plot_type in ['colormesh', 'contour']:
        cb = pyplot.colorbar(im, location='bottom', ax=ax, pad=0.001, fraction=0.02)
        cb.set_label('l2-misfit')
    elif plot_type == 'scatter':
        cb = pyplot.colorbar(im, location='bottom', ax=ax, pad=0.001, fraction=0.02, ticks=boundaries, extend='max')
        cb.set_label('Mismatching polarities')

    # Set title
    if title is not None:
        ax.set_title(title)
        
    # Save figure
    pyplot.tight_layout()
    pyplot.savefig(filename, dpi=300, bbox_inches='tight')
    pyplot.close()



def _plot_dc_matplotlib(filename, coords, 
    values_h_kappa, values_sigma_kappa, values_sigma_h,
    title=None, best_dc=None,  figsize=(8., 8.), fontsize=14, **kwargs):

    colormap = kwargs.get('colormap', 'viridis')

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

    vals = np.append(np.append(values_sigma_kappa.ravel(), values_sigma_kappa.ravel()),(values_sigma_h.ravel()))
    # Plot data
    # Use the percentile method to filter out outliers (Will alway clip the 10% greater values)
    vmin, vmax = np.nanpercentile(vals, [0,75])

    # plot surfaces
    _pcolor(axes[0][0], coords['h'], coords['kappa'], values_h_kappa.T, colormap, vmin=vmin, vmax=vmax)

    _pcolor(axes[0][1], coords['sigma'], coords['kappa'], values_sigma_kappa.T, colormap, vmin=vmin, vmax=vmax)

    _pcolor(axes[1][1], coords['sigma'], coords['h'], values_sigma_h.T, colormap, vmin=vmin, vmax=vmax)

    # optional markers
    if best_dc:
        _kappa, _sigma, _h = best_dc
        _add_marker(axes[0][0], (_h, _kappa))
        _add_marker(axes[0][1], (_sigma, _kappa))
        _add_marker(axes[1][1], (_sigma, _h))

    _set_dc_labels(axes, fontsize=fontsize)

    pyplot.savefig(filename)
    pyplot.close()



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
    pyplot.close()


def _plot_omega_matplotlib(filename, omega, values,
    title=None, xlabel='Angular distance', ylabel=None, figsize=(6., 6.), fontsize=16.):

    pyplot.figure(figsize=figsize)
    pyplot.plot(omega, values, 'k-')

    pyplot.xlim([0., 180.])

    if title:
        pyplot.title(title, fontsize=fontsize)

    if xlabel:
         pyplot.xlabel(xlabel, fontsize=fontsize)

    if ylabel:
         pyplot.ylabel(ylabel, fontsize=fontsize)

    pyplot.savefig(filename)
    pyplot.close()


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

# Define the Hammer projection function for matplotlib backend
def _hammer_projection(lon, lat):
    lon = np.radians(lon)
    lat = np.radians(lat)
    alpha = np.sqrt(1 + np.cos(lat) * np.cos(lon / 2))
    x = (2 * np.sqrt(2) * np.cos(lat) * np.sin(lon / 2)) / alpha
    y = (np.sqrt(2) * np.sin(lat)) / alpha
    return np.degrees(x), np.degrees(y)


def _generate_lune(ax=None):
    if ax is None:
        fig = pyplot.figure(figsize=(2.5, 8))
        ax = fig.add_subplot(111)
        ax.set_axis_off()
    else:
        fig = ax.figure

    # Generate curved gridlines
    num_lines = 7
    lon_lines = np.linspace(-30, 30, num_lines * 2 + 1)
    lat_lines = np.linspace(-90, 90, num_lines)

    for lon_line in lon_lines:
        lat_line = np.linspace(-90, 90, 1000)
        x_line, y_line = _hammer_projection(np.full_like(lat_line, lon_line), lat_line)
        ax.plot(x_line, y_line, 'k--', linewidth=0.5, alpha=0.5)

    for lat_line in lat_lines:
        lon_line = np.linspace(-30, 30, 1000)
        x_line, y_line = _hammer_projection(lon_line, np.full_like(lon_line, lat_line))
        ax.plot(x_line, y_line, 'k--', linewidth=0.5, alpha=0.5)

    _plot_lune_arcs(ax, _compute_lune_arcs())
    # Transform data to Hammer projection

    return fig, ax

def _compute_center_of_minimum_distance(lon_a, lat_a, lon_b, lat_b, iterations):
    if iterations == 0:
        return [(lon_a, lat_a), (lon_b, lat_b)]
    
    # Convert coordinates to radians
    lon_a_rad = math.radians(lon_a)
    lat_a_rad = math.radians(lat_a)
    lon_b_rad = math.radians(lon_b)
    lat_b_rad = math.radians(lat_b)
    
    # Convert lat/lon to cartesian coordinates
    x_a = math.cos(lat_a_rad) * math.cos(lon_a_rad)
    y_a = math.cos(lat_a_rad) * math.sin(lon_a_rad)
    z_a = math.sin(lat_a_rad)
    
    x_b = math.cos(lat_b_rad) * math.cos(lon_b_rad)
    y_b = math.cos(lat_b_rad) * math.sin(lon_b_rad)
    z_b = math.sin(lat_b_rad)
    
    # Compute barycenter of the points in cartesian coordinates
    x_mid = (x_a + x_b) / 2
    y_mid = (y_a + y_b) / 2
    z_mid = (z_a + z_b) / 2
    
    # Convert cartesian coordinate to latitude and longitude for the midpoint
    lat_mid_rad = math.atan2(z_mid, math.sqrt(x_mid**2 + y_mid**2))
    lon_mid_rad = math.atan2(y_mid, x_mid)
    
    # Convert midpoint lat and lon from radians to degrees
    lat_mid = math.degrees(lat_mid_rad)
    lon_mid = math.degrees(lon_mid_rad)
    
    # Recursive calls
    coordinates = []
    coordinates.extend(_compute_center_of_minimum_distance(lon_a, lat_a, lon_mid, lat_mid, iterations - 1))
    coordinates.extend(_compute_center_of_minimum_distance(lon_mid, lat_mid, lon_b, lat_b, iterations - 1))
    
    return coordinates

def _compute_lune_arcs():
    arc_points = [
    [-30, 35.2644, 30, 54.7356],
    [-30, -54.7356, 30, -35.2644],
    [-30, 35.2644, 30, -35.2644]
]
    arcs = []
    for arc in arc_points:
        arcs.append(_compute_center_of_minimum_distance(arc[0], arc[1], arc[2], arc[3], 3))
    return arcs

def _plot_lune_arcs(axis, arcs):
    for arc in arcs:
        lon = [x[0] for x in arc]
        lat = [x[1] for x in arc]
        axis.plot(*_hammer_projection(lon, lat), color='lightgrey', linewidth=1, zorder=100)