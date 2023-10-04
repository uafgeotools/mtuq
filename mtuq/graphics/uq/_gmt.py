
import numpy as np
import shutil
import subprocess

from mtuq.graphics._gmt import exists_gmt, gmt_not_found_warning, gmt_version,\
    gmt_formats, _parse_filetype, _safename
from mtuq.graphics.uq import _nothing_to_plot
from mtuq.util import fullpath, warn
from mtuq.util.math import wrap_180, to_delta, to_gamma, to_mij
from os.path import basename, exists, splitext
from six import string_types



def _plot_lune_gmt(filename, lon, lat, values, best_vw=None, lune_array=None,
    **kwargs):

    if _nothing_to_plot(values):
        return

    data = _parse_data(lon, lat, values)

    _call(fullpath('mtuq/graphics/uq/_gmt/plot_lune'),
        filename, data, supplemental_data=_parse_lune_array(lune_array),
        marker_coords=_parse_vw(best_vw), **kwargs)


def _plot_vw_gmt(filename, lon, lat, values, best_vw=None, lune_array=None,
    **kwargs):

    if _nothing_to_plot(values):
        return

    data = _parse_data(lon, lat, values)

    _call(fullpath('mtuq/graphics/uq/_gmt/plot_vw'),
        filename, data, supplemental_data=None,
        marker_coords=best_vw, **kwargs)


def _plot_force_gmt(filename, phi, h, values, best_force=None, **kwargs):

    if _nothing_to_plot(values):
        return

    lat = np.degrees(np.pi/2 - np.arccos(h))
    lon = wrap_180(phi + 90.)

    data =  _parse_data(lon, lat, values)

    _call(fullpath('mtuq/graphics/uq/_gmt/plot_force'),
       filename, data, supplemental_data=None,
       marker_coords=_parse_force(best_force), **kwargs)


def _plot_latlon_gmt(filename, lon, lat, values, best_latlon=None, lune_array=None,
    **kwargs):

    if _nothing_to_plot(values):
        return

    data = np.column_stack((lon, lat, values))

    _call(fullpath('mtuq/graphics/uq/_gmt/plot_latlon'),
        filename, data, supplemental_data=_parse_lune_array2(lon, lat, lune_array),
        marker_coords=best_latlon, **kwargs)


def _call(shell_script, filename, data, supplemental_data=None,
    title='', colormap='viridis', flip_cpt=False, colorbar_type=1,
    colorbar_label='', colorbar_limits=None, marker_coords=None, marker_type=0):

    #
    # Common wrapper for all GMT plotting functions involving 2D surfaces
    # (lune, vw, force, and hypocenter surfaces)
    #

    print('  calling GMT script: %s' % basename(shell_script))

    # parse filename and title
    filename, filetype = _parse_filetype(filename)
    title, subtitle = _parse_title(title)

    # parse color palette and label
    cpt_name = _parse_cpt_name(colormap)
    colorbar_label = _parse_label(colorbar_label)

    # parse colorbar limits
    try:
        minval, maxval = colorbar_limits
        exp = _exponent((minval, maxval))
        minval /= 10.**exp
        maxval /= 10.**exp
    except:
        minval, maxval, exp = _parse_limits(data[:,-1])

    data[:,-1] /= 10.**exp
    cpt_step=(maxval-minval)/20.

    # write values to be plotted as ASCII table
    ascii_file_1 = _safename('tmp_'+filename+'_ascii1.txt')
    _savetxt(ascii_file_1, data)

    # write supplementatal ASCII table, if given
    ascii_file_2 = _safename('tmp_'+filename+'_ascii2.txt')
    if supplemental_data is not None:
        _savetxt(ascii_file_2, supplemental_data)

    # write marker coordinates, if given
    marker_coords_file = _safename('tmp_'+filename+'_marker_coords.txt')
    if marker_coords is not None:
        _savetxt(marker_coords_file, *marker_coords)

    # call bash script
    if exists_gmt():
        subprocess.call("%s %s %s %s %s %f %f %d %s %s %d %d %s %s %d %s %s" %
           (shell_script,
            filename,
            filetype,
            ascii_file_1,
            ascii_file_2,
            minval,
            maxval,
            exp,
            # workaround GMT scientific notation parsing
            _float_to_str(cpt_step),
            cpt_name,
            int(bool(flip_cpt)),
            int(colorbar_type),
            colorbar_label,
            marker_coords_file,
            int(marker_type),
            title,
            subtitle,
            ),
            shell=True)
    else:
        gmt_not_found_warning(
            values_ascii)



def _plot_depth_gmt(filename, depths, values,
        magnitudes=None, lune_array=None,
        title='', xlabel='', ylabel='', fontsize=16.):

    # parse filenames
    filename, filetype = _parse_filetype(filename)

    ascii_file_1 = _safename('tmp_'+filename+'_ascii1.txt')
    ascii_file_2 = _safename('tmp_'+filename+'_ascii2.txt')
    ascii_file_3 = _safename('tmp_'+filename+'_ascii3.txt')


    # parase title and labels
    title, subtitle = _parse_title(title)

    xlabel = "'%s'" % xlabel
    ylabel = "'%s'" % ylabel


    data = np.column_stack((depths, values))
    minval, maxval, exp = _parse_limits(data[:,-1])
    data[:,-1] /= 10.**exp

    # write values to be plotted as ASCII table
    _savetxt(ascii_file_1, data)

    if lune_array is not None:
        data2 = _parse_lune_array2(data[:,0], data[:,1], lune_array)
        _savetxt(ascii_file_2, data2)

    if magnitudes is not None:
        data3 = np.column_stack((data[:,0], data[:,1], magnitudes))
        _savetxt(ascii_file_3, data3, fmt='%e %e %.2f')

    # call bash script
    if exists_gmt():
        subprocess.call("%s %s %s %s %s %s %f %f %d %s %s %s %s" %
           (fullpath('mtuq/graphics/uq/_gmt/plot_depth'),
            filename,
            filetype,
            ascii_file_1,
            ascii_file_2,
            ascii_file_3,
            minval,
            maxval,
            exp,
            title,
            subtitle,
            xlabel,
            ylabel,
            ),
            shell=True)
    else:
        gmt_not_found_warning(
            values_ascii)


#
# utility functions
#

def _parse_data(lon, lat, values):

    lon, lat = np.meshgrid(lon, lat)
    lat = lat.flatten()
    lon = lon.flatten()
    values = values.flatten()

    return np.column_stack((lon, lat, values))


def _parse_title(title):

    try:
        parts=title.split('\n')
    except:
        parts=[]

    if len(parts) >= 2:
        title = "'%s'" % parts[0]
        subtitle = "'%s'" % parts[1]
    elif len(parts) == 1:
        title = "'%s'" % parts[0]
        subtitle = "''"
    else:
        title = "''"
        subtitle = "''"

    return title, subtitle


def _parse_label(label):

    assert type(label) in string_types

    if len(label) > 0:
        return "'%s'" % label
    else:
        return "''"


def _parse_cpt_name(cpt_name):

    cpt_local = fullpath('mtuq/graphics/_gmt/cpt', cpt_name+'.cpt')
    if exists(cpt_local):
       return cpt_local
    else:
       return cpt_name


def _parse_limits(values):

    masked = np.ma.array(values, mask=np.isnan(values))

    minval = masked.min()
    maxval = masked.max()
    exp = _exponent(masked)

    if -1 <= exp <= 2:
        return minval, maxval, 0

    else:
        minval /= 10**exp
        maxval /= 10**exp
        return minval, maxval, exp


def _parse_force(force):
    if force is None:
        return None

    phi = force[0]
    if phi + 90 > 180.:
        lon = phi - 270.
    else:
        lon = phi + 90.

    h = force[1]
    lat = np.degrees(np.pi/2 - np.arccos(h))

    return [lon, lat]


def _parse_vw(vw):
    if vw is None:
        return None

    lon = to_gamma(vw[0])
    lat = to_delta(vw[1])

    return [lon, lat]


def _parse_lune_array(lune_array):
    if lune_array is None:
        return None

    # Convert from an (N x 6) table of lune parameters to the (N x 12) table
    # expected by psmeca
    N = lune_array.shape[0]
    gmt_array = np.empty((N, 12))

    for _i in range(N):
        rho,v,w,kappa,sigma,h = lune_array[_i,:]

        # adding a random negative perturbation to the dip, to avoid GMT plotting bug
        perturb = np.random.uniform(0.2,0.4)
        if sigma > (-90.0 + 0.4):
            sigma -= perturb

        mt = to_mij(rho, v, w, kappa, sigma, h)

        list_str =  [('%.2e' % mt[i]) for i in range(len(mt))]
        list_int = [int(string.split('e')[1]) for string in list_str]

        exponent = np.max(list_int)
        scaled_mt = mt/10**(exponent)
        dummy_value = 0.

        gmt_array[_i, 0] = to_gamma(v)
        gmt_array[_i, 1] = to_delta(w)
        gmt_array[_i, 2] = dummy_value
        gmt_array[_i, 3:9] = scaled_mt
        gmt_array[_i, 9] = exponent+7
        gmt_array[_i, 10:] = 0

        if N == 1:
            gmt_array[_i, 9] = exponent+20

    return gmt_array


def _parse_best_lune(best_vw, lune_array):
    if best_vw is None:
        return None
    if lune_array is None:
        return None

    # Scan to find the corresponding cell in lune_array for the coordinates in best_vw.
    best_vw_array = np.ones_like(lune_array[:,1:3]) * best_vw
    idx = np.where((best_vw_array == lune_array[:,1:3]).all(axis=1))

    # If idx is longer than 1, there is a problem...
    if np.shape(idx)[1] != 1:
        raise ValueError('There are more than one matching values!')
    # return the lune_array row corresponding to the best_vw coordinates only.
    return(lune_array[idx])


def _parse_lune_array2(lon, lat, lune_array):
    # TODO - reduce code duplcation with _parse_lune_array
    if lune_array is None:
        return None

    # Convert from an (N x 6) table of lune parameters to the (N x 12) table
    # expected by psmeca
    N = lune_array.shape[0]
    gmt_array = np.empty((N, 12))

    for _i in range(N):
        rho,v,w,kappa,sigma,h = lune_array[_i,:]

        # adding a random negative perturbation to the dip, to avoid GMT plotting bug
        perturb = np.random.uniform(0.2,0.4)
        if sigma > (-90.0 + 0.4):
            sigma -= perturb

        mt = to_mij(rho, v, w, kappa, sigma, h)

        list_str =  [('%.2e' % mt[i]) for i in range(len(mt))]
        list_int = [int(string.split('e')[1]) for string in list_str]

        exponent = np.max(list_int)
        scaled_mt = mt/10**(exponent)
        dummy_value = 0.

        gmt_array[_i, 0] = lon[_i]
        gmt_array[_i, 1] = lat[_i]
        gmt_array[_i, 2] = dummy_value
        gmt_array[_i, 3:9] = scaled_mt
        gmt_array[_i, 9] = exponent+7
        gmt_array[_i, 10:] = 0

    return gmt_array


def _float_to_str(val):
    # workaround GMT scientific notation parsing (problem with GMT >=6.1)
    if str(val).endswith('e+00'):
        return str(val).replace('e+00', '')
    else:
        return str(val).replace('e+', 'e')


def _savetxt(filename, *args, fmt='%.6e'):
    # TODO - can GMT accept virtual files?
    np.savetxt(filename, np.column_stack(args), fmt=fmt)


def _exponent(values):
    return np.floor(np.log10(np.max(np.abs(values))))

