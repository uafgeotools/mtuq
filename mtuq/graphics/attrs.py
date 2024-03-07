
import matplotlib
import os
import numpy as np

from matplotlib import pyplot
from os.path import join

from mtuq.util import defaults, warn
from mtuq.graphics._pygmt import exists_pygmt
from mtuq.event import MomentTensor


def plot_time_shifts(dirname, attrs, stations, origin, key='total_shift', 
    **kwargs):

    """ Plots how time shifts vary by location and component

    By default, total time shifts are plotted. To plot just static or
    cross-correlation time shifts, use ``key='static_shift'`` or 
    ``key='time_shift'``, respectively

    .. note ::

        MTUQ distinguishes between the following different types of 
        time shifts
    
        - `static_shift` is an initial user-supplied time shift applied during
        data processing
        
        - `time_shift` is a subsequent cross-correlation time shift applied 
        during misfit evaluation
        
        - `total_shift` is the total correction, or in other words the sum of
        static and cross-correlation time shifts
        

    .. rubric :: Required input arguments

    ``dirname`` (`str`):
    Directory in which figures will be written

    ``attrs`` (`list` of `AttribDict`):
    List returned by misfit function's `collect_attributes` method

    ``stations`` (`list` of `mtuq.Station` objects):
    Used to plot station locations

    ``origin`` (`mtuq.Origin` object):
    Used to plot origin location


    .. rubric :: Optional input arguments

    For optional argument descriptions, 
    `see here <mtuq.graphics._plot_attrs.html>`_

    """
    defaults(kwargs, {
        'label': 'Time shift (s)',
        })

    _plot_attrs(dirname, stations, origin, attrs, key, **kwargs)

def plot_cross_corr(dirname, attrs, stations, origin, key='normalized_cc_max', 
    **kwargs):

    """ Plots how cross-correlation values vary by location and component

    By default, maximum normalized cross-correlation values are plotted. To plot just
    maximum cross-correlation values, use ``key='cc_max'``

    .. note ::

        MTUQ distinguishes between the following different types of
        cross-correlation values

        - `cc_max` is the maximum cross-correlation value

        - `normalized_cc_max` is the maximum cross-correlation value normalized between 0 and 1

    .. rubric :: Required input arguments

    ``dirname`` (`str`):
    Directory in which figures will be written

    ``attrs`` (`list` of `AttribDict`):
    List returned by misfit function's `collect_attributes` method

    ``stations`` (`list` of `mtuq.Station` objects):
    Used to plot station locations

    ``origin`` (`mtuq.Origin` object):
    Used to plot origin location


    .. rubric :: Optional input arguments

    For optional argument descriptions, 
    `see here <mtuq.graphics._plot_attrs.html>`_

    """
    defaults(kwargs, {
        'label': 'Maximum normalized CC',
        'zero_centered': False,
        'colormap': 'inferno',
        'min_val': 0.0,
        'max_val': 1.0,
        })

    _plot_attrs(dirname, stations, origin, attrs, key, **kwargs)


def plot_amplitude_ratios(dirname, attrs, stations, origin, **kwargs):
    """ Plots how Aobs/Asyn varies by location and component

    .. rubric :: Required input arguments

    ``dirname`` (`str`):
    Directory in which figures will be written

    ``attrs`` (`list` of `AttribDict`):
    List returned by misfit function's `collect_attributes` method

    ``stations`` (`list` of `mtuq.Station` objects):
    Used to plot station locations

    ``origin`` (`mtuq.Origin` object):
    Used to plot origin location


    .. rubric :: Optional input arguments

    For optional argument descriptions, 
    `see here <mtuq.graphics._plot_attrs.html>`_


    """
    defaults(kwargs, {
        'colormap': 'inferno',
        'label': '$A_{obs}/A_{syn}$',
        'zero_centered': False,
        })

    _plot_attrs(dirname, stations, origin, attrs, 'amplitude_ratio', **kwargs)


def plot_log_amplitude_ratios(dirname, attrs, stations, origin, **kwargs):
    """ Plots how ln(Aobs/Asyn) varies by location and component

    .. rubric :: Required input arguments

    ``dirname`` (`str`):
    Directory in which figures will be written

    ``attrs`` (`list` of `AttribDict`):
    List returned by misfit function's `collect_attributes` method

    ``stations`` (`list` of `mtuq.Station` objects):
    Used to plot station locations

    ``origin`` (`mtuq.Origin` object):
    Used to plot origin location


    .. rubric :: Optional input arguments

    For optional argument descriptions, 
    `see here <mtuq.graphics._plot_attrs.html>`_

    """
    defaults(kwargs, {
        'label': 'ln($A_{obs}/A_{syn}$)',
        })

    _plot_attrs(dirname, stations, origin, attrs, 'log_amplitude_ratio', **kwargs)


def _plot_attrs(dirname, stations, origin, attrs, key,
    components=['Z', 'R', 'T'], format='png', backend=None,
    **kwargs):

    """ Reads the attribute given by `key` from the `attrs` data structure, and
    plots how this attribute varies

    Within the specified directory, a separate figure will be created for each
    component, e.g. `Z.png`, `R.png`, `T.png`.


    .. rubric ::  Keyword arguments

    ``components`` (`list`):
    Generate figures for the given components

    ``format`` (`str`):
    Image file format (defaults to `png`)

    ``backend`` (`function`):
    Backend function


    .. rubric :: Backend function

    To customize figure appearance, users can pass their own backend function.
    See `online documentation 
    <https://uafgeotools.github.io/mtuq/user_guide/06/customizing_figures.html>`_
    for details. Otherwise, defaults to a generic matplotlib `backend
    <mtuq.graphics.attrs._default_backend.html>`_.


    """

    if backend is None:
        backend = _default_backend
    elif backend == _pygmt_backend and not exists_pygmt(): 
        warn('PyGMT backend requested but PyGMT not found'); backend = _default_backend

    if not callable(backend):
        raise TypeError

    os.makedirs(dirname, exist_ok=True)

    for component in components:
        values = []
        active_stations_list = []

        for _i, station in enumerate(stations):
            if component not in attrs[_i]:
                continue

            values += [attrs[_i][component][key]]
            active_stations_list += [stations[_i]]

        if len(values) > 0:
            filename = join(dirname, component+'.'+format)
            backend(filename, values, active_stations_list, origin, stations_list = stations, **kwargs)


#
# low-level function for plotting trace attributes
#

def _default_backend(filename, values, stations, origin,
    colormap='coolwarm', zero_centered=True, colorbar=True,
    label='', width=5., height=5., **kwargs):

    """ Default backend for all other `mtuq.graphics.attrs` functions

    The frontend functions perform only data manipulation. All graphics library
    calls occur in the backend

    By isolating the graphics function calls in this way, users can completely
    interchange graphics libraries (matplotlib, GMT, PyGMT, and so on)

    .. rubric::  Keyword arguments

    ``colormap`` (`str`):
    Matplotlib color palette

    ``zero_centered`` (`bool`):
    Whether or not the colormap is centered on zero

    ``colorbar`` (`bool`):
    Whether or not to display a colorbar

    ``label`` (`str`):
    Optional colorbar label


    """

    fig = pyplot.figure(figsize=(width, height))


    # generate colormap
    cmap = matplotlib.cm.get_cmap(colormap)

    if zero_centered:
        min_val = -np.max(np.abs(values))
        max_val = +np.max(np.abs(values))
    else:
        min_val = np.min(values)
        max_val = np.max(values)

    # plot stations
    im = pyplot.scatter(
        [station.longitude for station in stations],
        [station.latitude for station in stations], 
        s=80.,
        c=values, 
        cmap=cmap, 
        vmin=min_val,
        vmax=max_val,
        marker='^',
        )

    # plot line segments
    for _i, station in enumerate(stations):

        scaled = (values[_i]-min_val)/(max_val-min_val)
        rgb = cmap(scaled)

        pyplot.plot(
            [origin.longitude, station.longitude],
            [origin.latitude, station.latitude],
            marker=None,
            color=rgb,
            linestyle='-',
            linewidth=0.5,
            )

    # plot origin
    pyplot.plot(
        origin.longitude,
        origin.latitude,
        marker='*',
        markersize=15.,
        color='black',
        )

    # adjust ticks
    pyplot.gca().tick_params(top=True, right=True,
        labeltop=True, labelright=True)

    pyplot.locator_params(nbins=3)

    # add colorbar
    if not label:
        label = ''

    fig.colorbar(im, orientation="horizontal", pad=0.2,
        label=label)

    pyplot.savefig(filename)
    pyplot.close()


def _pygmt_backend(filename, values, active_stations, origin,
                colormap='polar', zero_centered=True, display_topo=True,
                label='', width=5, moment_tensor=None, process=None,
                stations_list=None, station_labels=True, min_val=None, max_val=None, **kwargs):
    """
    PyGMT backend for plotting station attributes with hillshading using the
    Miller Cylindrical projection, with an azimuth of 0/90 and a normalization
    of t1 for the hillshade intensity.
    """
    import pygmt

    if not stations_list:
        stations_list = active_stations
        print('Complete station list not passed to pygmt plotting backend \nWill plot only active stations')
    # Collection of longitudes and latitudes from all available stations
    longitudes = [s.longitude for s in stations_list + [origin]]
    latitudes = [s.latitude for s in stations_list + [origin]]

    # Calculate the region to display with a buffer around the stations
    region, lat_buffer = PyGMTUtilities.calculate_plotting_region(stations_list, origin, buffer_percentage=0.1)
    
    # Setting up the figure
    fig = pygmt.Figure()

    # Dynamically determine the grid resolution for topography based on the range of longitudes and latitudes
    # (etopo topography file will be downloaded if not found)
    resolution = PyGMTUtilities.get_resolution(max(longitudes) - min(longitudes), max(latitudes) - min(latitudes))
    grid = pygmt.datasets.load_earth_relief(region=region, resolution=resolution)

    # Calculate the gradient (hillshade) grid with azimuth 0/300 and normalization t1
    # <https://www.pygmt.org/dev/gallery/images/grdgradient_shading.html>
    shade = pygmt.grdgradient(grid=grid, azimuth="0/300", normalize="t1")

    # Define a grayscale colormap for topography
    normal = pygmt.grdhisteq.equalize_grid(grid=grid, gaussian=True)
    gray_cmap = pygmt.makecpt(cmap='gray', series=[np.min(normal.values), np.max((normal.values))])

    # Plot the hillshade grid as an image
    if display_topo:
        fig.grdimage(grid=normal, shading=shade, projection=f'J{width}i', frame='a', cmap=gray_cmap, no_clip=True)

    # Overlay coastlines
    PyGMTUtilities.draw_coastlines(fig)

    # Configure the colormap for station values
    colormap, cmap_reverse_flag = PyGMTUtilities.configure_colormap(colormap)
    if zero_centered:
        pygmt.makecpt(cmap=colormap, series=[-np.max(np.abs(values))*1.01, np.max(np.abs(values))*1.01], reverse=cmap_reverse_flag)
    elif min_val is not None and max_val is not None:
        pygmt.makecpt(cmap=colormap, series=[min_val, max_val], continuous=True, reverse=cmap_reverse_flag)
    else:
        pygmt.makecpt(cmap=colormap, series=[np.min(values), np.max(values)], continuous=True, reverse=cmap_reverse_flag)


    # Plotting lines from origin to stations
    for station in stations_list:
        if station in active_stations:
            # Plot line for active station as colored line
            value = values[active_stations.index(station)] if station in active_stations else 0
            fig.plot(
                x=[origin.longitude, station.longitude],
                y=[origin.latitude, station.latitude],
                cmap=True, 
                zvalue=value, 
                pen="thick,+z,-"
            )

    # Plotting stations as triangles
    fig.plot(
        x=[station.longitude for station in active_stations],
        y=[station.latitude for station in active_stations],
        style='i0.8c',  # Triangle
        color=values,
        cmap=True,
        pen="0.5p,black"
    )

    # Plotting non-active stations as hollow triangles
    non_active_stations = [station for station in stations_list if station not in active_stations]
    if len(non_active_stations) > 0:
        fig.plot(
            x=[station.longitude for station in non_active_stations],
            y=[station.latitude for station in non_active_stations],
            style='i0.8c',  # Triangle
            color=None,  # Hollow (white) triangle
            pen="0.5p,black"  # Outline color
        )
        fig.plot(
            x=[station.longitude for station in non_active_stations],
            y=[station.latitude for station in non_active_stations],
            style='i0.6c',  # Triangle
            color=None,  # Hollow (white) triangle
            pen="0.5p,white"  # Outline color
        )

    # Plotting the origin as a star
    fig.plot(
        x=[origin.longitude],
        y=[origin.latitude],
        style='a0.6c',  # Star, size 0.5 cm
        color='yellow',
        pen="0.5p,black"
    )

    if moment_tensor is not None:
        # Normalize the moment tensor components to the desired exponent

        if type(moment_tensor) is MomentTensor:
            moment_tensor = moment_tensor.as_vector()

        moment_tensor = np.array(moment_tensor)/np.linalg.norm(moment_tensor)

        moment_tensor_spec = {
            'mrr': moment_tensor[0],
            'mtt': moment_tensor[1],
            'mff': moment_tensor[2],
            'mrt': moment_tensor[3],
            'mrf': moment_tensor[4],
            'mtf': moment_tensor[5],
            'exponent': 21  # Merely for size control, as the MT is normalized prior to plotting
        }

        # Plot the moment tensor as a beachball
        fig.meca(
            spec=moment_tensor_spec,
            scale="1c",  # Sets a fixed size for the beachball plot
            longitude=origin.longitude,
            latitude=origin.latitude,
            depth=10,  # Depth is required, even if not used, set to a small number
            convention="mt",  # Use GMT's mt convention <https://www.pygmt.org/dev/api/generated/pygmt.Figure.meca.html>
            compressionfill="gray15",
            extensionfill="white",
            pen="0.5p,black"
        )
        
    if station_labels is True:
        # Plotting station labels
        for station in stations_list:
            fig.text(
                x=station.longitude,
                y=station.latitude,
                text=station.station,
                font="5p,Helvetica-Bold,black",
                justify="LM",
                offset="-0.45c/0.125c",
                fill='white'
            )

    fig.colorbar(frame=f'+l"{PyGMTUtilities.prepare_latex_annotations(label)}"', position="JMR+o1.5c/0c+w7c/0.5c")

    fig.basemap(region=region, projection=f'J{width}i', frame=True)

    # Now starts the header text above the plot -- It is not a title and can be modified. 
    # Add an integer increment to the text_line_val bellow to add a new line above.
    text_line_val = 1
    header_lines = PyGMTUtilities.get_header(label, origin, filename, process)
    
    # Add the header text to the plot
    # Text spacing is based on longitude range and latitude buffer size.
    lon_mean = np.max(longitudes) - (np.max(longitudes) - np.min(longitudes)) / 2
    text_spacing = lat_buffer / 1.5
    for header_line in header_lines:
        fig.text(x=lon_mean,
                 y=max(latitudes) + lat_buffer + text_line_val*text_spacing,
                 text=header_line, font="14p,Helvetica-Bold,black", justify="MC", no_clip=True)
        text_line_val += 1

    # Saving the figure
    fig.savefig(filename, crop=True, dpi=300)
    
class PyGMTUtilities:
    @staticmethod
    def calculate_plotting_region(stations, origin, buffer_percentage=0.1):
        longitudes = [station.longitude for station in stations] + [origin.longitude]
        latitudes = [station.latitude for station in stations] + [origin.latitude]

        lon_buffer = (max(longitudes) - min(longitudes)) * buffer_percentage
        lat_buffer = (max(latitudes) - min(latitudes)) * buffer_percentage

        region = [min(longitudes) - lon_buffer, max(longitudes) + lon_buffer,
                min(latitudes) - lat_buffer, max(latitudes) + lat_buffer]
        return region, lat_buffer


    @staticmethod
    def get_resolution(lon_range, lat_range):
        """
        Determines the resolution based on the given longitude and latitude ranges.

        Args:
        lon_range (float): The range of longitudes.
        lat_range (float): The range of latitudes.

        Returns:
        str: pygmt etopo grid resolution based on the given ranges.
        """

        if lon_range > 10 or lat_range > 10:
            return '01m'
        elif lon_range > 5 or lat_range > 5:
            return '15s'
        elif lon_range > 2 or lat_range > 2:
            return '03s'
        elif lon_range > 1 or lat_range > 1:
            return '01s'
        else:
            return '05m'

    @staticmethod
    def configure_colormap(colormap):
        """
        Configures the colormap based on the given input - as conventions for matplotlib and pygmt can differ

        Args:
            colormap (str): The name of the colormap.

        Returns:
            tuple: A tuple containing the modified colormap name and a flag indicating
                   whether the colormap should be reversed.
        """
        cmap_reverse_flag = True if colormap.endswith('_r') else False
        colormap = colormap[:-2] if cmap_reverse_flag else colormap
        return colormap, cmap_reverse_flag

    @staticmethod
    def prepare_latex_annotations(label):
        """
        Prepares LaTeX annotations for plotting. Uses HTML for compatibility with PyGMT/GMT.

        Args:
            label (str): The LaTeX label to be prepared.

        Returns:
            str: The prepared label.

        """
        if label.startswith('$') and label.endswith('$'):
            # Convert LaTeX to HTML for compatibility with PyGMT/GMT
            return f"<math>{label[1:-1]}</math>"
        else:
            return label
        
    @staticmethod
    def get_header(label, origin, filename, process = None):
        """
        Generates a header for a plot based on the provided parameters.

        Args:
            label (str): The label for the plot. Defined in default kwargs.
            origin (Origin): mtuq.event.Origin object.
            filename (str): The filename of the plot. Defined by default the high-level function. Used to retrieve the component.
            process (Process, optional): mtuq.process_data.ProcessData object for appropriate dataset.

        Returns:
            list: A list containing two lines of the header.
        """
        if process is not None:
            # get type of waves used for the window
            window_type = process.window_type
            if window_type == 'surface_wave' or window_type == 'group_velocity':
                window_type = 'Surface wave'
            elif window_type == 'body_wave':
                window_type = 'Body wave'

        component = filename.split('/')[-1].split('.')[0]
        origin_time = str(origin.time)[0:19]
        origin_depth = origin.depth_in_m/1000

        label = PyGMTUtilities.prepare_latex_annotations(label)

        # if window_type exists, define Rayleigh or Love wave
        if process is not None:
            if window_type == 'Surface wave' and component == 'Z' or window_type == 'Surface wave' and component == 'R':
                # First line of the header defined as: label - Rayleigh wave (component)
                header_line_1 = f"{label} - Rayleigh wave ({component})"
            elif window_type == 'Surface wave' and component == 'T':
                # First line of the header defined as: label - Love wave (component)
                header_line_1 = f"{label} - Love wave ({component})"
            elif window_type == 'Body wave':
                # First line of the header defined as: label - (component)
                header_line_1 = f"{label} - Body wave ({component})"
        else:
            # First line of the header defined as: label - (component)
            header_line_1 = f"{label} - ({component})"

        header_line_2 = f"Event Time: {origin_time} UTC, Depth: {origin_depth:.1f} km"

        return [header_line_1, header_line_2]
    
    @staticmethod
    def draw_coastlines(fig, area_thresh=100, water_color='paleturquoise', water_transparency=55):
        fig.coast(shorelines=True, area_thresh=area_thresh)
        fig.coast(shorelines=False, water=water_color, transparency=water_transparency, area_thresh=area_thresh)