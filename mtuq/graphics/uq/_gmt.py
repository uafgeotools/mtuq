
import numpy as np
import shutil
import subprocess

from mtuq.graphics._gmt import exists_gmt, gmt_not_found_warning, gmt_version,\
    gmt_formats
from mtuq.util import fullpath, warn
from mtuq.util.math import wrap_180
from os.path import splitext



def gmt_plot_misfit_lune(filename, lon, lat, values, **kwargs):

    if _nothing_to_plot(values):
        return

    lon, lat =  _parse_lonlat(lon,lat)
    values, zmin, zmax, exp = _parse_values(values)

    _call(fullpath('mtuq/graphics/uq/_gmt/plot_misfit_lune'),
        filename, 
        lon, lat, values, 
        zmin=zmin,
        zmax=zmax,
        dz=(zmax-zmin)/20.,
        exp=exp,
        **kwargs)


def gmt_plot_likelihood_lune(filename, lon, lat, values, **kwargs):

    if _nothing_to_plot(values):
        return

    lon, lat =  _parse_lonlat(lon,lat)
    values, zmin, zmax, exp = _parse_values(values)

    _call(fullpath('mtuq/graphics/uq/_gmt/plot_likelihood_lune'),
        filename, 
        lon, lat, values,
        zmin=zmin,
        zmax=zmax,
        dz=(zmax-zmin)/10.,
        exp=exp,
        **kwargs)


def gmt_plot_misfit_force(filename, phi, h, values, **kwargs):

    if _nothing_to_plot(values):
        return

    lat = np.degrees(np.pi/2 - np.arccos(h))
    lon = wrap_180(phi + 90.)

    lon, lat =  _parse_lonlat(lon,lat)
    values, zmin, zmax, exp = _parse_values(values)

    _call(fullpath('mtuq/graphics/uq/_gmt/plot_misfit_force'),
        filename,
        lon, lat, values,
        zmin=zmin,
        zmax=zmax,
        dz=(zmax-zmin)/20.,
        exp=exp,
        **kwargs)


def gmt_plot_likelihood_force(filename, phi, h, values, **kwargs):

    if _nothing_to_plot(values):
        return

    lat = np.degrees(np.pi/2 - np.arccos(h))
    lon = wrap_180(phi + 90.)

    lon, lat =  _parse_lonlat(lon,lat)
    values, zmin, zmax, exp = _parse_values(values)

    _call(fullpath('mtuq/graphics/uq/_gmt/plot_likelihood_force'),
        filename,
        lon, lat, values,
        zmin=zmin,
        zmax=zmax,
        dz=(zmax-zmin)/10.,
        exp=exp,
        **kwargs)


def _call(shell_script, filename, lon, lat, values,
    zmin=None, zmax=None, dz=None, exp=0,
    colorbar_type=0, add_marker=True, title=''):

    filetype = _parse_filetype(filename)
    title, subtitle = _parse_title(title)

    # write lon,lat,val ASCII table
    ascii_data = 'tmp_'+filename+'.txt'
    _savetxt(ascii_data, lon, lat, values)

    # call bash script
    if exists_gmt():
        subprocess.call("%s %s %s %s %e %e %e %d %s %s %s %s" %
           (shell_script,
            ascii_data,
            filename,
            filetype,
            zmin,
            zmax,
            dz,
            exp,
            int(bool(colorbar_type)),
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


def _parse_lonlat(lon, lat):

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


