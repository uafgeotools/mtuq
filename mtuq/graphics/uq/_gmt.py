
import numpy as np
import shutil
import subprocess

from mtuq.graphics._gmt import exists_gmt, gmt_not_found_warning, gmt_version,\
    gmt_formats
from mtuq.util import fullpath, warn
from mtuq.util.math import wrap_180, to_delta, to_gamma, to_mij
from os.path import basename, exists, splitext



def gmt_plot_lune(filename, lon, lat, values, best_vw=None, lune_array=None, 
    **kwargs):

    if _nothing_to_plot(values):
        return

    lon, lat =  _parse_coords(lon,lat)

    marker_coords = None
    supplemental_data = None
    if best_vw is not None:
        marker_coords = _parse_vw(best_vw)
    if lune_array is not None:
        supplemental_data = _parse_lune_array(lune_array)


    _call(fullpath('mtuq/graphics/uq/_gmt/plot_lune'),
        filename, lon, lat, values, supplemental_data=supplemental_data,
        marker_coords=marker_coords, **kwargs)


def gmt_plot_vw(filename, lon, lat, values, best_vw=None, lune_array=None,
    **kwargs):

    if _nothing_to_plot(values):
        return

    lon, lat =  _parse_coords(lon,lat)

    marker_coords = None
    supplemental_data = None
    if lune_array is not None:
        supplemental_data = _parse_lune_array(lune_array)


    _call(fullpath('mtuq/graphics/uq/_gmt/plot_vw'),
        filename, lon, lat, values, supplemental_data=supplemental_data,
        marker_coords=best_vw, **kwargs)


def gmt_plot_force(filename, phi, h, values, best_force=None, **kwargs):

    if _nothing_to_plot(values):
        return

    lat = np.degrees(np.pi/2 - np.arccos(h))
    lon = wrap_180(phi + 90.)

    lon, lat =  _parse_coords(lon,lat)

    marker_coords = None
    if best_force is not None:
        marker_coords = _parse_force(best_force)

    _call(fullpath('mtuq/graphics/uq/_gmt/plot_force'), 
        filename, lon, lat, values, supplemental_data=None,
       marker_coords=marker_coords, **kwargs)


def _call(shell_script, filename, lon, lat, values, supplemental_data=None,
    title='', colormap='viridis', flip_cpt=False, colorbar_type=1,
    marker_coords=None, marker_type=0):

    print('  calling GMT script: %s' % basename(shell_script))

    filename, filetype = _parse_filetype(filename)

    values, minval, maxval, exp = _parse_values(values)

    cpt_name = _parse_cpt(colormap)
    cpt_step=(maxval-minval)/20.

    title, subtitle = _parse_title(title)


    # write ASCII data
    ascii_file_1 = _safename('tmp_'+filename+'_ascii1.txt')
    ascii_file_2 = _safename('tmp_'+filename+'_ascii2.txt')
    marker_coords_file = _safename('tmp_'+filename+'_marker_coords.txt')

    _savetxt(ascii_file_1, lon, lat, values)

    if supplemental_data is not None:
        _savetxt(ascii_file_2, supplemental_data)

    if marker_coords is not None:
        _savetxt(marker_coords_file, *marker_coords)

    # call bash script
    if exists_gmt():
        subprocess.call("%s %s %s %s %s %f %f %d %s %s %d %d %s %d %s %s" %
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
            marker_coords_file,
            int(marker_type),
            title,
            subtitle,
            ),
            shell=True)
    else:
        gmt_not_found_warning(
            values_ascii)



#
# utility functions
#

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


def _parse_coords(lon, lat):

    lon, lat = np.meshgrid(lon, lat)
    lat = lat.flatten()
    lon = lon.flatten()
    return lon, lat


def _parse_values(values):

    values = values.flatten()
    masked = np.ma.array(values, mask=np.isnan(values))

    minval = masked.min()
    maxval = masked.max()
    exp = np.floor(np.log10(np.max(np.abs(masked))))

    if -1 <= exp <= 2:
        return masked, minval, maxval, 0

    else:
        masked /= 10**exp
        minval /= 10**exp
        maxval /= 10**exp
        return masked, minval, maxval, exp


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


def _parse_filetype(filename):

    parts = splitext(filename)
    name, ext = parts[0], parts[1].lstrip('.')

    if ext.upper() in ['PS']:
        return name, 'EPS'

    elif ext.upper() in ['JPG', 'JPEG']:
        return name, 'JPEG'

    elif ext.upper() in ['TIF', 'TIFF']:
        return name, 'TIFF'

    elif ext.upper() in gmt_formats:
        return name, ext.upper()

    else:
        warn('Unrecognized extension: defaulting to PNG')
        return name, 'PNG'


def _parse_cpt(cpt_name):
    cpt_local = fullpath('mtuq/graphics/_gmt/cpt', cpt_name+'.cpt')
    if exists(cpt_local):
       return cpt_local
    else:
       return cpt_name


def _parse_force(force):
    phi = force[0]
    if phi + 90 > 180.:
        lon = phi - 270.
    else:
        lon = phi + 90.

    h = force[1]
    lat = np.degrees(np.pi/2 - np.arccos(h))

    return [lon, lat]


def _parse_vw(vw):
    lon = to_gamma(vw[0])
    lat = to_delta(vw[1])

    return [lon, lat]


def _parse_lune_array(lune_array):
    # converts from a N x 6 table of lune parameters to the N x 12 format expected by psmeca

    dummy_value = 0.

    N = lune_array.shape[0]

    gmt_array = np.empty((N,12))
    for _i in range(N):

        rho,v,w,kappa,sigma,h = lune_array[_i,:]

        # adding a random negative perturbation to the dip, to avoid GMT plotting bug
        perturb = np.random.uniform(0.2,0.4)
        if sigma > (-90.0 + 0.4):
            sigma -= perturb

        mt = to_mij(rho, v, w, kappa, sigma, h)
        exponent = np.max([int('{:.2e}'.format(mt[i]).split('e+')[1]) for i in range(len(mt))])
        scaled_mt = mt/10**(exponent)

        gmt_array[_i, 0] = to_gamma(v)
        gmt_array[_i, 1] = to_delta(w)
        gmt_array[_i, 2] = dummy_value
        gmt_array[_i, 3:9] = scaled_mt
        gmt_array[_i, 9] = exponent+7
        gmt_array[_i, 10:] = 0

    return gmt_array


def _safename(filename):
    return filename.replace('/', '__')


def _float_to_str(val):
    # workaround GMT scientific notation parsing (problem with GMT >=6.1)
    if str(val).endswith('e+00'):
        return str(val).replace('e+00', '')
    else:
        return str(val).replace('e+', 'e')


def _savetxt(filename, *args):
    # FIXME: can GMT accept virtual files?
    np.savetxt(filename, np.column_stack(args), fmt='%.3e')


