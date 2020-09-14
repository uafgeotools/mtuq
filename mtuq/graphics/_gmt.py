
import numpy as np
import shutil
import subprocess

from mtuq.util import fullpath, warn
from mtuq.util.math import wrap_180
from os.path import splitext



def exists_gmt():
    return bool(shutil.which('gmt'))


def gmt_version():
    if exists_gmt():
        proc = subprocess.Popen('gmt --version',
            stdout=subprocess.PIPE, shell=True)

        bytes_string = proc.stdout.readline()
        string = str(bytes_string, "utf-8").strip()
        return string


def gmt_major_version():
    if gmt_version() is not None:
        return int(gmt_version().split('.')[0])


def gmt_not_found_warning(filename):
    warn("""
        WARNING

        Generic Mapping Tools executables not found on system path.
        PostScript output has not been written. 

        Misfit values have been saved to:
            %s
        """ % filename)


gmt_formats = [
    'BMP',
    'EPS',
    'JPG',
    'JPEG',
    'PDF',
    'PNG',
    'PPM',
    'SVG',
    'TIFF',
    ]


#
# bash wrappers
#

def gmt_plot_misfit_lune(filename, lon, lat, values, **kwargs):

    if _nothing_to_plot(values):
        return

    lon, lat, values = _parse_values(lon, lat, values)

    _call(fullpath('mtuq/graphics/_gmt/plot_misfit_lune'),
        filename, 
        lon, lat, values, 
        zmin=values.min(),
        zmax=values.max(), 
        dz=(values.max()-values.min())/20.,
        **kwargs)


def gmt_plot_likelihood_lune(filename, lon, lat, values, **kwargs):

    if _nothing_to_plot(values):
        return

    lon, lat, values = _parse_values(lon, lat, values)

    _call(fullpath('mtuq/graphics/_gmt/plot_likelihood_lune'),
        filename, 
        lon, lat, values,
        zmin=values.min(),
        zmax=values.max(), 
        dz=(values.max()-values.min())/10.,
        **kwargs)


def gmt_plot_misfit_force(filename, phi, h, values, **kwargs):

    if _nothing_to_plot(values):
        return

    lat = np.degrees(np.pi/2 - np.arccos(h))
    lon = wrap_180(phi + 90.)

    lon, lat, values = _parse_values(lon, lat, values)

    _call(fullpath('mtuq/graphics/_gmt/plot_misfit_force'),
        filename,
        lon, lat, values,
        zmin=values.min(),
        zmax=values.max(),
        dz=(values.max()-values.min())/20.,
        **kwargs)


def gmt_plot_likelihood_force(filename, phi, h, values, **kwargs):

    if _nothing_to_plot(values):
        return

    lat = np.degrees(np.pi/2 - np.arccos(h))
    lon = wrap_180(phi + 90.)

    lon, lat, values = _parse_values(lon, lat, values)

    _call(fullpath('mtuq/graphics/_gmt/plot_likelihood_force'),
        filename,
        lon, lat, values,
        zmin=values.min(),
        zmax=values.max(),
        dz=(values.max()-values.min())/10.,
        **kwargs)


def _call(shell_script, filename, lon, lat, values,
    zmin=None, zmax=None, dz=None,
    add_colorbar=False, add_marker=True, title=''):

    filetype = _parse_filetype(filename)
    title, subtitle = _parse_title(title)

    # write lon,lat,val ASCII table
    ascii_data = 'tmp_'+filename+'.txt'
    _savetxt(ascii_data, lon, lat, values)

    # call bash script
    if exists_gmt():
        subprocess.call("%s %s %s %s %e %e %e %s %s %s %s" %
           (shell_script,
            ascii_data,
            filename,
            filetype,
            zmin,
            zmax,
            dz,
            int(bool(add_colorbar)),
            int(bool(add_marker)),
            title,
            subtitle
            ),
            shell=True)
    else:
        gmt_not_found_warning(
            ascii_data)



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


def _parse_values(lon, lat, values, normalize=True):
    lon, lat = np.meshgrid(lon, lat)
    lat = lat.flatten()
    lon = lon.flatten()
    values = values.flatten()

    masked = np.ma.array(values, mask=np.isnan(values))
    minval = masked.min()
    maxval = masked.max()

    if normalize:
        values /= maxval

    elif maxval-minval < 1.e-6:
        exp = -np.fix(np.log10(maxval-minval))
        warn(
           "Multiplying by 10^%d to avoid GMT plotting errors" % exp,
           Warning)
        values *= 10.**exp

    return lon, lat, masked


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
        return 'EPS'
        
    elif ext.upper() in ['JPG', 'JPEG']:
        return 'JPEG'

    elif ext.upper() in ['TIF', 'TIFF']:
        return 'TIFF'
   
    elif ext.upper() in gmt_formats:
        return ext.upper()

    else:
        warn('Unrecognized extension: defaulting to PNG')
        return 'PNG'



def _savetxt(filename, gamma, delta, values):
    # FIXME: can GMT accept virtual files?
    np.savetxt(filename, np.column_stack([gamma, delta, values]))


