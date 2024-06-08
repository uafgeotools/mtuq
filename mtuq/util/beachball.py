import numpy as np
from mtuq.graphics.uq._matplotlib import _hammer_projection

def offset_fibonacci_sphere(samples=1000, epsilon=0.36, equator_points=180):
    equator_axis = 'y'
    total_points = samples + equator_points
    points = np.empty((total_points, 3))  # Pre-allocate array
    goldenRatio = (1 + 5**0.5) / 2

    for i in range(samples):
        theta = 2 * np.pi * i / goldenRatio
        phi = np.arccos(1 - 2 * (i + epsilon) / (samples - 1 + 2 * epsilon))
        x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
        points[i] = (x, y, z)

    if equator_axis == 'y':
        for i in range(equator_points):
            theta = 2 * np.pi * i / equator_points
            x, y, z = np.cos(theta), np.sin(theta), 0
            points[samples + i] = (x, z, y)

    elif equator_axis == 'x':
        for i in range(equator_points):
            theta = 2 * np.pi * i / equator_points
            x, y, z = np.cos(theta), np.sin(theta), 0
            points[samples + i] = (y, x, z)

    elif equator_axis == 'z':
        for i in range(equator_points):
            theta = 2 * np.pi * i / equator_points
            x, y, z = np.cos(theta), np.sin(theta), 0
            points[samples + i] = (x, y, z)

    return points

def convert_sphere_points_to_angles(points):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    azimuth = np.arctan2(y, x)
    azimuth = np.rad2deg(azimuth) % 360

    r = np.sqrt(x**2 + y**2 + z**2)
    takeoff_angle = np.arccos(z / r)
    takeoff_angle = 180 - np.rad2deg(takeoff_angle)

    return takeoff_angle, azimuth


def lambert_azimuthal_equal_area_projection(points, hemisphere='upper'):
    x, z, y = points[:, 0], points[:, 1], points[:, 2]
    if hemisphere == 'upper':
        z = -z
    x_proj = x * np.sqrt(1 / (1 - z))
    y_proj = y * np.sqrt(1 / (1 - z))
    return np.vstack((x_proj, y_proj)).T

def polarities_mt(mt_array, takeoff, azimuth):
    # Variation on the polarity function in polarity.py -- This one returns radiation patterns as well
    if mt_array.shape[1] != 6:
        raise Exception('Inconsistent dimensions')

    if len(takeoff) != len(azimuth):
        raise Exception('Inconsistent dimensions')

    # Define lambda functions for sine and cosine of degrees to radians
    sin_rad = lambda x: np.sin(np.deg2rad(x))
    cos_rad = lambda x: np.cos(np.deg2rad(x))

    # Pre-compute trig functions for efficiency
    sth = sin_rad(takeoff)
    cth = cos_rad(takeoff)
    sphi = sin_rad(azimuth)
    cphi = cos_rad(azimuth)

    drc = np.column_stack((sth * cphi, sth * sphi, cth))

    # Pre-compute squares for efficiency
    drc_sq = drc ** 2

    # Using pre-computed squares in the dot product calculation
    cth = mt_array[:, 0:1] * drc_sq[:, 2:3] +\
          mt_array[:, 1:2] * drc_sq[:, 0:1] +\
          mt_array[:, 2:3] * drc_sq[:, 1:2] +\
          2 * (mt_array[:, 3:4] * drc[:, 0:1] * drc[:, 2:3] -
               mt_array[:, 4:5] * drc[:, 1:2] * drc[:, 2:3] -
               mt_array[:, 5:6] * drc[:, 0:1] * drc[:, 1:2])

    cth = cth.squeeze()  # Remove the extra dimension

    polarities = np.where(cth > 0, 1, -1)
    radiations = cth

    return polarities, radiations

def rotate_tensor(mt_array):
    xx, yy, zz, xy, xz, yz = mt_array[0]
    return np.array([[yy, zz, xx, yz, xy, xz]])

def rotate_points(xi, zi, angle):
    xi_shape, zi_shape = xi.shape, zi.shape
    theta = np.deg2rad(angle)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta),  np.cos(theta)]])
    rotated_points = np.dot(rotation_matrix, np.vstack((xi.flatten(), zi.flatten())))
    return rotated_points[0].reshape(xi_shape), rotated_points[1].reshape(zi_shape)

def estimate_angle_on_lune(lon, lat):
    # Lon, Lat, coordinates of the point on the lune in degrees
    delta_lat = lat+0.1

    x1, y1 = _hammer_projection(lon, lat)
    x2, y2 = _hammer_projection(lon, delta_lat)

    dx = x2 - x1
    dy = y2 - y1
    # print(dx, dy)
    angle = np.rad2deg(np.arctan2(dx, dy))
    return angle

def _project_on_sphere(takeoff_angle, azimuth, scale=2.0):
    # Convert takeoff and azimuth angles to radians
    takeoff_angle += 180
    takeoff_angle = np.deg2rad(takeoff_angle)
    azimuth = np.deg2rad(azimuth)
    r = scale

    # Calculate the x, y, z coordinates of the point on the unit sphere
    x = r*np.sin(takeoff_angle)*np.cos(azimuth)
    y = r*np.sin(takeoff_angle)*np.sin(azimuth)
    z = r*np.cos(takeoff_angle)

    return -y,-z,-x

def _generate_sphere_points(mode):
    """Generates points on the unit sphere using the offset Fibonacci algorithm.
    This function returns only the half sphere in the XZ plane.

    :: mode :: str
        'MT_Only' : Ideal for a single, large beachball 
        'Scatter MT' : Reduced precision for a scatter plot of beachballs
    
    """
    if mode == 'MT_Only':
        points = offset_fibonacci_sphere(50000, 0, 360)
    elif mode == 'Scatter MT':
        points = offset_fibonacci_sphere(5000, 0, 360)
    upper_hemisphere_mask = points[:, 1] >= 0
    return points, upper_hemisphere_mask

def _adjust_scale_based_on_axes(ax, scale):
    """
    Adjusts the beachball scale based on the extent of the x and y axes of the given plot.
    This will ensure consistent beachball sizes across different plots regardless of the axis extent.

    Parameters:
        ax (matplotlib.axes.Axes): The axes object representing the plot.
        scale (float): The scale value to be adjusted.

    Returns:
        float: The adjusted scale value.
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    axis_extent = min(xlim[1] - xlim[0], ylim[1] - ylim[0])
    scale *= (axis_extent / 10.0)  # Adjust scale based on axis extent
    return scale